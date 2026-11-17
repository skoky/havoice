//! The audio frontend the Voice PE runs before its wake word models: 16 kHz 16-bit audio in,
//! one 40-channel feature slice out every 10 ms.
//!
//! A port of the fixed-point TFLite Micro frontend as the firmware builds it
//! (`esphome/esp-micro-speech-features`, configured by `micro_wake_word`): 30 ms Hann window,
//! 512-point fixed-point KISS FFT, 40 mel channels from 125 to 7500 Hz, noise reduction, PCAN
//! gain control and a fixed-point log. Integer arithmetic follows the C code step by step,
//! including its 16-bit wrap-arounds, so the features match what the device (and
//! microWakeWord training, via `pymicro-features`) computes.
//!
//! ```
//! use havoice::micro_frontend::{MicroFrontend, to_model_input};
//!
//! let mut frontend = MicroFrontend::new();
//! let audio = vec![0i16; 16_000]; // one second of silence
//! let slices = frontend.process(&audio);
//! assert_eq!(slices.len(), 98); // a slice per 10 ms once the first 30 ms window is full
//! let int8: [i8; 40] = to_model_input(&slices[0]);
//! assert!(int8.iter().all(|&v| v == -128));
//! ```

/// Feature channels per slice.
pub const CHANNELS: usize = 40;
/// Samples per window (30 ms).
const WINDOW: usize = 480;
/// Samples between slices (10 ms).
pub const STEP: usize = 160;
const FFT_SIZE: usize = 512;
const SPECTRUM: usize = FFT_SIZE / 2 + 1;

const WINDOW_BITS: u32 = 12;
const FILTERBANK_BITS: u32 = 12;
const NOISE_REDUCTION_BITS: u32 = 14;
const SMOOTHING_BITS: u32 = 10;
const PCAN_SNR_BITS: u32 = 12;
const PCAN_OUTPUT_BITS: u32 = 6;
const PCAN_GAIN_BITS: i32 = 21;
const WIDE_DYNAMIC_FUNCTION_BITS: usize = 32;
const LOG_SCALE_SHIFT: u32 = 6;
const LOG_SCALE_LOG2: u32 = 16;
const LOG_SEGMENTS_LOG2: u32 = 7;
const LOG_COEFF: u64 = 45426;

/// One 10 ms slice: 40 log-mel values, roughly 0..=670.
pub type Slice = [u16; CHANNELS];

/// Scale a slice the way `micro_wake_word.cpp` does before handing it to a model:
/// `feature * 256 / 666 - 128`, rounded and clamped to int8.
pub fn to_model_input(slice: &Slice) -> [i8; CHANNELS] {
    let mut out = [0i8; CHANNELS];
    for (o, &v) in out.iter_mut().zip(slice) {
        let value = (v as i32 * 256 + 333) / 666 - 128;
        *o = value.clamp(-128, 127) as i8;
    }
    out
}

/// Streaming frontend state. Feed it audio in any chunk size with [`MicroFrontend::process`].
#[derive(Debug, Clone)]
pub struct MicroFrontend {
    window: Vec<i16>,
    input: Vec<i16>,
    input_used: usize,
    fft: KissFftr,
    filterbank: Filterbank,
    noise_estimate: [u32; CHANNELS],
    even_smoothing: u32,
    odd_smoothing: u32,
    min_signal_remaining: u32,
    gain_lut: Vec<i16>,
    snr_shift: u32,
}

impl Default for MicroFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl MicroFrontend {
    /// A frontend with the Voice PE's settings (`preprocessor_settings.h`).
    pub fn new() -> Self {
        let arg = std::f32::consts::PI * 2.0 / WINDOW as f32;
        let window = (0..WINDOW)
            .map(|i| {
                // mul_add: clang contracts these C expressions into fused multiply-adds
                let v = (-0.5f32).mul_add((arg * (i as f32 + 0.5)).cos(), 0.5);
                v.mul_add((1 << WINDOW_BITS) as f32, 0.5).floor() as i16
            })
            .collect();
        // input_correction_bits = MostSignificantBit32(512) - 1 - FILTERBANK_BITS / 2 = 3
        let correction = msb32(FFT_SIZE as u32) as i32 - 1 - (FILTERBANK_BITS / 2) as i32;
        Self {
            window,
            input: vec![0; WINDOW],
            input_used: 0,
            fft: KissFftr::new(FFT_SIZE),
            filterbank: Filterbank::new(16_000.0, 125.0, 7500.0),
            noise_estimate: [0; CHANNELS],
            // the C code stores these float products in uint16_t, truncating
            even_smoothing: (0.025f32 * (1 << NOISE_REDUCTION_BITS) as f32) as u32,
            odd_smoothing: (0.06f32 * (1 << NOISE_REDUCTION_BITS) as f32) as u32,
            min_signal_remaining: (0.05f32 * (1 << NOISE_REDUCTION_BITS) as f32) as u32,
            gain_lut: pcan_gain_lut(SMOOTHING_BITS as i32 - correction),
            snr_shift: (PCAN_GAIN_BITS - correction - PCAN_SNR_BITS as i32) as u32,
        }
    }

    /// Back to the state of a fresh frontend.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Feed 16 kHz mono samples; returns every slice completed by them.
    pub fn process(&mut self, mut samples: &[i16]) -> Vec<Slice> {
        let mut out = Vec::new();
        while !samples.is_empty() {
            let take = (WINDOW - self.input_used).min(samples.len());
            self.input[self.input_used..self.input_used + take].copy_from_slice(&samples[..take]);
            self.input_used += take;
            samples = &samples[take..];
            if self.input_used == WINDOW {
                out.push(self.slice());
            }
        }
        out
    }

    fn slice(&mut self) -> Slice {
        // window (window.c)
        let mut windowed = [0i16; WINDOW];
        let mut max_abs: i16 = 0;
        for (i, w) in windowed.iter_mut().enumerate() {
            let v = ((self.input[i] as i32 * self.window[i] as i32) >> WINDOW_BITS) as i16;
            *w = v;
            max_abs = max_abs.max(v.wrapping_abs());
        }
        self.input.copy_within(STEP.., 0);
        self.input_used -= STEP;

        // FFT (fft.c), scaled up so the loudest sample uses the full 16 bits
        let input_shift = 15 - msb32(max_abs as u16 as u32);
        let mut fft_input = [0i16; FFT_SIZE];
        for (dst, &src) in fft_input.iter_mut().zip(&windowed) {
            *dst = ((src as u16 as u32) << input_shift) as u16 as i16;
        }
        let spectrum = self.fft.compute(&fft_input);

        // filterbank (filterbank.c)
        let mut energy = [0i32; SPECTRUM];
        for i in self.filterbank.start_index..self.filterbank.end_index {
            let (re, im) = (spectrum[i].r as i32, spectrum[i].i as i32);
            energy[i] = re.wrapping_mul(re).wrapping_add(im.wrapping_mul(im));
        }
        let mut signal = self.filterbank.accumulate(&energy, input_shift);

        // noise reduction (noise_reduction.c)
        for (i, s) in signal.iter_mut().enumerate() {
            let smoothing = if i & 1 == 0 {
                self.even_smoothing
            } else {
                self.odd_smoothing
            };
            let one_minus = (1u32 << NOISE_REDUCTION_BITS) - smoothing;
            let scaled_up = *s << SMOOTHING_BITS;
            let mut estimate = ((scaled_up as u64 * smoothing as u64
                + self.noise_estimate[i] as u64 * one_minus as u64)
                >> NOISE_REDUCTION_BITS) as u32;
            self.noise_estimate[i] = estimate;
            estimate = estimate.min(scaled_up);
            let floor =
                ((*s as u64 * self.min_signal_remaining as u64) >> NOISE_REDUCTION_BITS) as u32;
            let subtracted = (scaled_up - estimate) >> SMOOTHING_BITS;
            *s = subtracted.max(floor);
        }

        // PCAN gain control (pcan_gain_control.c)
        for (i, s) in signal.iter_mut().enumerate() {
            let gain = wide_dynamic_function(self.noise_estimate[i], &self.gain_lut) as i32 as u32;
            let snr = ((*s as u64 * gain as u64) >> self.snr_shift) as u32;
            *s = pcan_shrink(snr);
        }

        // log scale (log_scale.c)
        let correction = msb32(FFT_SIZE as u32) as i32 - 1 - (FILTERBANK_BITS / 2) as i32;
        let mut out = [0u16; CHANNELS];
        for (o, &s) in out.iter_mut().zip(&signal) {
            let mut value = if correction < 0 {
                s >> (-correction)
            } else {
                s.wrapping_shl(correction as u32)
            };
            value = if value > 1 { log_scaled(value) } else { 0 };
            *o = value.min(0xFFFF) as u16;
        }
        out
    }
}

fn msb32(n: u32) -> u32 {
    32 - n.leading_zeros()
}

fn msb64(n: u64) -> u32 {
    64 - n.leading_zeros()
}

// ----------------------------------------------------------------------- KISS FFT (16-bit)

#[derive(Debug, Clone, Copy, Default)]
struct Cpx {
    r: i16,
    i: i16,
}

const FRAC_BITS: u32 = 15;
const SAMP_MAX: i32 = 32767;

fn sround(x: i32) -> i16 {
    (x.wrapping_add(1 << (FRAC_BITS - 1)) >> FRAC_BITS) as i16
}

fn smul(a: i16, b: i16) -> i32 {
    a as i32 * b as i32
}

fn c_mul(a: Cpx, b: Cpx) -> Cpx {
    Cpx {
        r: sround(smul(a.r, b.r).wrapping_sub(smul(a.i, b.i))),
        i: sround(smul(a.r, b.i).wrapping_add(smul(a.i, b.r))),
    }
}

fn c_fixdiv(c: Cpx, div: i32) -> Cpx {
    let k = (SAMP_MAX / div) as i16;
    Cpx {
        r: sround(smul(c.r, k)),
        i: sround(smul(c.i, k)),
    }
}

fn c_add(a: Cpx, b: Cpx) -> Cpx {
    Cpx {
        r: a.r.wrapping_add(b.r),
        i: a.i.wrapping_add(b.i),
    }
}

fn c_sub(a: Cpx, b: Cpx) -> Cpx {
    Cpx {
        r: a.r.wrapping_sub(b.r),
        i: a.i.wrapping_sub(b.i),
    }
}

fn cexp(phase: f64) -> Cpx {
    Cpx {
        r: (0.5 + SAMP_MAX as f64 * phase.cos()).floor() as i16,
        i: (0.5 + SAMP_MAX as f64 * phase.sin()).floor() as i16,
    }
}

/// `kiss_fftr` for real input, forward only, `FIXED_POINT == 16`.
#[derive(Debug, Clone)]
struct KissFftr {
    n: usize,
    twiddles: Vec<Cpx>,
    factors: Vec<(usize, usize)>,
    super_twiddles: Vec<Cpx>,
}

impl KissFftr {
    fn new(nfft: usize) -> Self {
        let n = nfft / 2;
        let pi = std::f64::consts::PI;
        let twiddles = (0..n)
            .map(|i| cexp(-2.0 * pi * i as f64 / n as f64))
            .collect();
        let super_twiddles = (0..n / 2)
            .map(|i| cexp(-pi * ((i + 1) as f64 / n as f64 + 0.5)))
            .collect();
        Self {
            n,
            twiddles,
            factors: kf_factor(n),
            super_twiddles,
        }
    }

    fn compute(&self, time: &[i16; FFT_SIZE]) -> [Cpx; SPECTRUM] {
        let packed: Vec<Cpx> = time
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[r, i]| Cpx { r, i })
            .collect();
        let mut tmp = vec![Cpx::default(); self.n];
        self.work(&mut tmp, 0, &packed, 0, 1, &self.factors);

        let ncfft = self.n;
        let mut freq = [Cpx::default(); SPECTRUM];
        let tdc = c_fixdiv(tmp[0], 2);
        freq[0].r = tdc.r.wrapping_add(tdc.i);
        freq[ncfft].r = tdc.r.wrapping_sub(tdc.i);
        for k in 1..=ncfft / 2 {
            let fpk = c_fixdiv(tmp[k], 2);
            let fpnk = c_fixdiv(
                Cpx {
                    r: tmp[ncfft - k].r,
                    i: tmp[ncfft - k].i.wrapping_neg(),
                },
                2,
            );
            let f1k = c_add(fpk, fpnk);
            let f2k = c_sub(fpk, fpnk);
            let tw = c_mul(f2k, self.super_twiddles[k - 1]);
            freq[k].r = ((f1k.r as i32 + tw.r as i32) >> 1) as i16;
            freq[k].i = ((f1k.i as i32 + tw.i as i32) >> 1) as i16;
            freq[ncfft - k].r = ((f1k.r as i32 - tw.r as i32) >> 1) as i16;
            freq[ncfft - k].i = ((tw.i as i32 - f1k.i as i32) >> 1) as i16;
        }
        freq
    }

    fn work(
        &self,
        out: &mut [Cpx],
        out_at: usize,
        input: &[Cpx],
        in_at: usize,
        fstride: usize,
        factors: &[(usize, usize)],
    ) {
        let (p, m) = factors[0];
        if m == 1 {
            for k in 0..p {
                out[out_at + k] = input[in_at + k * fstride];
            }
        } else {
            for k in 0..p {
                self.work(
                    out,
                    out_at + k * m,
                    input,
                    in_at + k * fstride,
                    fstride * p,
                    &factors[1..],
                );
            }
        }
        match p {
            2 => self.bfly2(&mut out[out_at..], fstride, m),
            4 => self.bfly4(&mut out[out_at..], fstride, m),
            _ => unreachable!("FFT size {} has a factor {p}", self.n),
        }
    }

    fn bfly2(&self, f: &mut [Cpx], fstride: usize, m: usize) {
        for k in 0..m {
            let a = c_fixdiv(f[k], 2);
            let b = c_fixdiv(f[k + m], 2);
            let t = c_mul(b, self.twiddles[k * fstride]);
            f[k + m] = c_sub(a, t);
            f[k] = c_add(a, t);
        }
    }

    fn bfly4(&self, f: &mut [Cpx], fstride: usize, m: usize) {
        for k in 0..m {
            let f0 = c_fixdiv(f[k], 4);
            let f1 = c_fixdiv(f[k + m], 4);
            let f2 = c_fixdiv(f[k + 2 * m], 4);
            let f3 = c_fixdiv(f[k + 3 * m], 4);
            let s0 = c_mul(f1, self.twiddles[k * fstride]);
            let s1 = c_mul(f2, self.twiddles[k * fstride * 2]);
            let s2 = c_mul(f3, self.twiddles[k * fstride * 3]);
            let s5 = c_sub(f0, s1);
            let f0 = c_add(f0, s1);
            let s3 = c_add(s0, s2);
            let s4 = c_sub(s0, s2);
            f[k + 2 * m] = c_sub(f0, s3);
            f[k] = c_add(f0, s3);
            f[k + m] = Cpx {
                r: s5.r.wrapping_add(s4.i),
                i: s5.i.wrapping_sub(s4.r),
            };
            f[k + 3 * m] = Cpx {
                r: s5.r.wrapping_sub(s4.i),
                i: s5.i.wrapping_add(s4.r),
            };
        }
    }
}

fn kf_factor(mut n: usize) -> Vec<(usize, usize)> {
    let mut p = 4;
    let floor_sqrt = (n as f64).sqrt().floor() as usize;
    let mut out = Vec::new();
    loop {
        while !n.is_multiple_of(p) {
            p = match p {
                4 => 2,
                2 => 3,
                _ => p + 2,
            };
            if p > floor_sqrt {
                p = n;
            }
        }
        n /= p;
        out.push((p, n));
        if n <= 1 {
            return out;
        }
    }
}

// ----------------------------------------------------------------------------- filterbank

#[derive(Debug, Clone)]
struct Filterbank {
    start_index: usize,
    end_index: usize,
    frequency_starts: Vec<usize>,
    weight_starts: Vec<usize>,
    widths: Vec<usize>,
    weights: Vec<i16>,
    unweights: Vec<i16>,
}

fn freq_to_mel(freq: f32) -> f32 {
    1127.0f32 * (freq / 700.0f32).ln_1p()
}

impl Filterbank {
    /// `FilterbankPopulateState` for 40 channels.
    fn new(sample_rate: f32, lower: f32, upper: f32) -> Self {
        const ALIGN: usize = 2; // kFilterbankIndexAlignment / sizeof(int16_t)
        const BLOCK: usize = 4; // kFilterbankChannelBlockSize
        let n = CHANNELS + 1;
        let mel_low = freq_to_mel(lower);
        let mel_span = freq_to_mel(upper) - mel_low;
        let spacing = mel_span / n as f32;
        // mul_add here and below: the C code is built with fused multiply-adds, and a
        // weight that rounds the other way changes the features
        let centers: Vec<f32> = (0..n)
            .map(|i| spacing.mul_add((i + 1) as f32, mel_low))
            .collect();

        let hz_per_sbin = 0.5f32 * sample_rate / (SPECTRUM as f32 - 1.0);
        let start_index = (1.5f32 + lower / hz_per_sbin) as usize;
        let mut frequency_starts = vec![0; n];
        let mut weight_starts = vec![0; n];
        let mut widths = vec![0; n];
        let mut actual_starts = vec![0; n];
        let mut actual_widths = vec![0; n];
        let mut chan_freq_start = start_index;
        let mut weight_index = 0;
        let mut needs_zeros = false;
        for chan in 0..n {
            let mut freq_index = chan_freq_start;
            while freq_to_mel(freq_index as f32 * hz_per_sbin) <= centers[chan] {
                freq_index += 1;
            }
            let width = freq_index - chan_freq_start;
            actual_starts[chan] = chan_freq_start;
            actual_widths[chan] = width;
            if width == 0 {
                frequency_starts[chan] = 0;
                weight_starts[chan] = 0;
                widths[chan] = BLOCK;
                if !needs_zeros {
                    needs_zeros = true;
                    for ws in weight_starts.iter_mut().take(chan) {
                        *ws += BLOCK;
                    }
                    weight_index += BLOCK;
                }
            } else {
                let aligned_start = chan_freq_start / ALIGN * ALIGN;
                let aligned_width = chan_freq_start - aligned_start + width;
                let padded = ((aligned_width - 1) / BLOCK + 1) * BLOCK;
                frequency_starts[chan] = aligned_start;
                weight_starts[chan] = weight_index;
                widths[chan] = padded;
                weight_index += padded;
            }
            chan_freq_start = freq_index;
        }

        let mut weights = vec![0i16; weight_index];
        let mut unweights = vec![0i16; weight_index];
        let mut end_index = 0;
        for chan in 0..n {
            let mut frequency = actual_starts[chan];
            let offset = frequency - frequency_starts[chan];
            let denom = if chan == 0 {
                mel_low
            } else {
                centers[chan - 1]
            };
            for j in 0..actual_widths[chan] {
                let w = (centers[chan] - freq_to_mel(frequency as f32 * hz_per_sbin))
                    / (centers[chan] - denom);
                let at = weight_starts[chan] + offset + j;
                weights[at] = w.mul_add((1 << FILTERBANK_BITS) as f32, 0.5).floor() as i16;
                unweights[at] = (1.0 - w)
                    .mul_add((1 << FILTERBANK_BITS) as f32, 0.5)
                    .floor() as i16;
                frequency += 1;
            }
            end_index = end_index.max(frequency);
        }
        Self {
            start_index,
            end_index,
            frequency_starts,
            weight_starts,
            widths,
            weights,
            unweights,
        }
    }

    /// `FilterbankAccumulateChannels` + `FilterbankSqrt`.
    fn accumulate(&self, energy: &[i32; SPECTRUM], scale_down_shift: u32) -> [u32; CHANNELS] {
        let mut work = [0u64; CHANNELS + 1];
        let mut weight_acc: u64 = 0;
        let mut unweight_acc: u64 = 0;
        for (chan, slot) in work.iter_mut().enumerate() {
            let f0 = self.frequency_starts[chan];
            let w0 = self.weight_starts[chan];
            for j in 0..self.widths[chan] {
                // magnitudes past the spectrum only ever meet zero weights
                let mag = energy.get(f0 + j).copied().unwrap_or(0) as i64 as u64;
                weight_acc =
                    weight_acc.wrapping_add((self.weights[w0 + j] as i64 as u64).wrapping_mul(mag));
                unweight_acc = unweight_acc
                    .wrapping_add((self.unweights[w0 + j] as i64 as u64).wrapping_mul(mag));
            }
            *slot = weight_acc;
            weight_acc = unweight_acc;
            unweight_acc = 0;
        }
        let mut out = [0u32; CHANNELS];
        for (o, &w) in out.iter_mut().zip(&work[1..]) {
            *o = sqrt64(w) >> scale_down_shift;
        }
        out
    }
}

fn sqrt32(mut num: u32) -> u32 {
    if num == 0 {
        return 0;
    }
    let mut res: u32 = 0;
    let max_bit_number = (32 - msb32(num)) | 1;
    let mut bit = 1u32 << (31 - max_bit_number);
    let mut iterations = (31 - max_bit_number) / 2 + 1;
    while iterations > 0 {
        iterations -= 1;
        if num >= res.wrapping_add(bit) {
            num -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    if num > res && res != 0xFFFF {
        res += 1;
    }
    res as u16 as u32
}

fn sqrt64(mut num: u64) -> u32 {
    if num >> 32 == 0 {
        return sqrt32(num as u32);
    }
    let mut res: u64 = 0;
    let max_bit_number = (64 - msb64(num)) | 1;
    let mut bit = 1u64 << (63 - max_bit_number);
    let mut iterations = (63 - max_bit_number) / 2 + 1;
    while iterations > 0 {
        iterations -= 1;
        if num >= res.wrapping_add(bit) {
            num -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    if num > res && res != 0xFFFF_FFFF {
        res += 1;
    }
    res as u32
}

// ------------------------------------------------------------------------------------ PCAN

fn pcan_lookup(input_bits: i32, x: u32) -> i16 {
    let x_as_float = x as f32 / (1u32 << input_bits) as f32;
    let gain = (1u32 << PCAN_GAIN_BITS) as f32 * (x_as_float + 80.0f32).powf(-0.95f32);
    if gain > 0x7FFF as f32 {
        return 0x7FFF;
    }
    (gain + 0.5) as i16
}

/// `PcanGainControlPopulateState`: the LUT as `WideDynamicFunction` indexes it.
fn pcan_gain_lut(input_bits: i32) -> Vec<i16> {
    let mut lut = vec![0i16; 4 * WIDE_DYNAMIC_FUNCTION_BITS - 3];
    lut[0] = pcan_lookup(input_bits, 0);
    lut[1] = pcan_lookup(input_bits, 1);
    for interval in 2..=WIDE_DYNAMIC_FUNCTION_BITS {
        let x0 = 1u32 << (interval - 1);
        let x1 = x0 + (x0 >> 1);
        let x2 = if interval == WIDE_DYNAMIC_FUNCTION_BITS {
            x0 + (x0 - 1)
        } else {
            2 * x0
        };
        let y0 = pcan_lookup(input_bits, x0);
        let y1 = pcan_lookup(input_bits, x1);
        let y2 = pcan_lookup(input_bits, x2);
        let diff1 = y1 as i32 - y0 as i32;
        let diff2 = y2 as i32 - y0 as i32;
        let a1 = 4 * diff1 - diff2;
        let a2 = diff2 - a1;
        let at = 4 * interval - 6;
        lut[at] = y0;
        lut[at + 1] = a1 as i16;
        lut[at + 2] = a2 as i16;
    }
    lut
}

fn wide_dynamic_function(x: u32, lut: &[i16]) -> i16 {
    if x <= 2 {
        return lut[x as usize];
    }
    let interval = msb32(x) as i32;
    let lut = &lut[(4 * interval - 6) as usize..];
    let frac = (if interval < 11 {
        x << (11 - interval)
    } else {
        x >> (interval - 11)
    } & 0x3FF) as i16 as i32;
    let mut result = (lut[2] as i32 * frac) >> 5;
    result = result.wrapping_add(((lut[1] as i32 as u32) << 5) as i32);
    result = result.wrapping_mul(frac);
    result = (result.wrapping_add(1 << 14)) >> 15;
    result += lut[0] as i32;
    result as i16
}

fn pcan_shrink(x: u32) -> u32 {
    if x < (2 << PCAN_SNR_BITS) {
        (x * x) >> (2 + 2 * PCAN_SNR_BITS - PCAN_OUTPUT_BITS)
    } else {
        (x >> (PCAN_SNR_BITS - PCAN_OUTPUT_BITS)) - (1 << PCAN_OUTPUT_BITS)
    }
}

// ------------------------------------------------------------------------------------- log

const LOG_LUT: [u16; 130] = [
    0, 224, 442, 654, 861, 1063, 1259, 1450, 1636, 1817, 1992, 2163, 2329, 2490, 2646, 2797, 2944,
    3087, 3224, 3358, 3487, 3611, 3732, 3848, 3960, 4068, 4172, 4272, 4368, 4460, 4549, 4633, 4714,
    4791, 4864, 4934, 5001, 5063, 5123, 5178, 5231, 5280, 5326, 5368, 5408, 5444, 5477, 5507, 5533,
    5557, 5578, 5595, 5610, 5622, 5631, 5637, 5640, 5641, 5638, 5633, 5626, 5615, 5602, 5586, 5568,
    5547, 5524, 5498, 5470, 5439, 5406, 5370, 5332, 5291, 5249, 5203, 5156, 5106, 5054, 5000, 4944,
    4885, 4825, 4762, 4697, 4630, 4561, 4490, 4416, 4341, 4264, 4184, 4103, 4020, 3935, 3848, 3759,
    3668, 3575, 3481, 3384, 3286, 3186, 3084, 2981, 2875, 2768, 2659, 2549, 2437, 2323, 2207, 2090,
    1971, 1851, 1729, 1605, 1480, 1353, 1224, 1094, 963, 830, 695, 559, 421, 282, 142, 0, 0,
];

fn log2_fraction_part(x: u32, log2x: u32) -> u32 {
    let mut frac = (x as i64 - (1i64 << log2x)) as i32;
    if log2x < LOG_SCALE_LOG2 {
        frac <<= LOG_SCALE_LOG2 - log2x;
    } else {
        frac >>= log2x - LOG_SCALE_LOG2;
    }
    let base_seg = (frac as u32) >> (LOG_SCALE_LOG2 - LOG_SEGMENTS_LOG2);
    let seg_unit = (1u32 << LOG_SCALE_LOG2) >> LOG_SEGMENTS_LOG2;
    let c0 = LOG_LUT[base_seg as usize] as i32;
    let c1 = LOG_LUT[base_seg as usize + 1] as i32;
    let seg_base = (seg_unit * base_seg) as i32;
    let rel_pos = ((c1 - c0) * (frac - seg_base)) >> LOG_SCALE_LOG2;
    (frac + c0 + rel_pos) as u32
}

fn log_scaled(x: u32) -> u32 {
    let integer = msb32(x) - 1;
    let fraction = log2_fraction_part(x, integer);
    let log2 = (integer << LOG_SCALE_LOG2).wrapping_add(fraction);
    let round = 1u32 << (LOG_SCALE_LOG2 - 1);
    let loge = ((LOG_COEFF * log2 as u64 + round as u64) >> LOG_SCALE_LOG2) as u32;
    ((loge << LOG_SCALE_SHIFT) + round) >> LOG_SCALE_LOG2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_factors_like_kiss() {
        assert_eq!(kf_factor(256), vec![(4, 64), (4, 16), (4, 4), (4, 1)]);
    }

    #[test]
    fn silence_is_the_lowest_input() {
        let mut f = MicroFrontend::new();
        let slices = f.process(&[0; 4800]);
        assert_eq!(slices.len(), (4800 - WINDOW) / STEP + 1);
        assert!(slices.iter().all(|s| s.iter().all(|&v| v == 0)));
        assert_eq!(to_model_input(&slices[0]), [-128; CHANNELS]);
    }

    #[test]
    fn chunking_does_not_change_the_output() {
        let audio: Vec<i16> = (0..8000)
            .map(|i| ((i as f32 * 0.07).sin() * 9000.0 + (i as f32 * 0.31).sin() * 3000.0) as i16)
            .collect();
        let whole = MicroFrontend::new().process(&audio);
        let mut f = MicroFrontend::new();
        let pieces: Vec<Slice> = audio.chunks(37).flat_map(|c| f.process(c)).collect();
        assert_eq!(whole, pieces);
        assert!(whole.last().unwrap().iter().any(|&v| v > 100));
    }
}
