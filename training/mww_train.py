#!/usr/bin/env python3
"""Run microwakeword.model_train_eval with GPU memory fixes for small GPUs.

microWakeWord validates by calling model.evaluate() on the whole validation
set as one NumPy array. Keras turns that array into a single tensor and copies
it to the GPU in one piece, which fails on GPUs with little memory (e.g. 4 GB
laptop cards) with:

    InternalError: Failed copying input tensor from CPU:0 to GPU:0 in order to
    run _EagerConst: Dst tensor is not initialized.

This launcher feeds NumPy inputs to evaluate() batch by batch from host memory
instead, so only one batch lives on the GPU at a time, and enables GPU memory
growth. Results are the same, and RAM stays flat across validations.
Arguments are passed through unchanged:

    python3 training/mww_train.py --training_config=... --train 1 ... mixednet ...
"""

import runpy
import sys

import numpy as np
import tensorflow as tf


def enable_memory_growth() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # already initialized


def patch_evaluate() -> None:
    """Make evaluate() feed NumPy inputs to the GPU one batch at a time.

    This is Keras 3's evaluate loop with the input pipeline removed: no
    tf.data dataset is built, so no full-size tensor is created. That also
    avoids tf.data.Dataset.from_generator, whose global generator registry
    keeps every evaluated array alive (+3.8 GB RAM per validation run here).
    Metrics accumulate across batches exactly as in evaluate(), and
    self.reset_metrics() is still called through the instance, which
    microWakeWord temporarily swaps out to chain two evaluations.
    """
    import keras
    from keras.src.utils.python_utils import pythonify_logs

    original = keras.Model.evaluate

    def evaluate(self, x=None, y=None, batch_size=None, verbose="auto",
                 sample_weight=None, steps=None, callbacks=None,
                 return_dict=False, **kwargs):
        if not isinstance(x, np.ndarray) or sample_weight is not None or steps \
                or callbacks or kwargs:
            return original(self, x, y, batch_size, verbose, sample_weight, steps,
                            callbacks, return_dict, **kwargs)
        batch_size = batch_size or 32
        if y is not None:
            y = np.asarray(y)
        self._maybe_symbolic_build(data_batch=(x[:batch_size], None if y is None
                                               else y[:batch_size], None))
        self.make_test_function()
        self.reset_metrics()
        logs = {}
        for start in range(0, len(x), batch_size):
            stop = start + batch_size
            batch = (x[start:stop], None if y is None else y[start:stop], None)
            logs = self.test_function(iter([batch]))
        logs = pythonify_logs(self._get_metrics_result_or_logs(logs))
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    keras.Model.evaluate = evaluate


def main() -> None:
    enable_memory_growth()
    patch_evaluate()
    sys.argv[0] = "microwakeword.model_train_eval"
    runpy.run_module("microwakeword.model_train_eval", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
