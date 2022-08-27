# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A Deepmind-specific platform for running Experiments with Jaxline."""

from concurrent import futures
import os
from typing import Any, Mapping

from absl import flags
from absl import logging

import chex
import jax

from jaxline import base_config
import jaxline_train
from jaxline import utils
from ml_collections import config_dict
from ml_collections import config_flags
import numpy as np
import ray
import config

import tensorflow as tf


# TODO(tomhennigan) Add support for ipdb and pudb.
_CONFIG = config_flags.DEFINE_config_file(
    name="config",
    help_string="Training configuration file.",
)
# This flag is expected to be used only internally by jaxline.
# It is prefixed by "jaxline" to prevent a conflict with a "mode" flag defined
# by Monarch.
_JAXLINE_MODE = flags.DEFINE_string(
    name="jaxline_mode",
    default="train",
    help=("Execution mode. "
          " `train` will run training, `eval` will run evaluation."),
)
_JAXLINE_TPU_DRIVER = flags.DEFINE_string(
    name="jaxline_tpu_driver",
    default="",
    help="Whether to use tpu_driver.",
)
_JAXLINE_ENSURE_TPU = flags.DEFINE_bool(
    name="jaxline_ensure_tpu",
    default=False,
    help="Whether to ensure we have a TPU connected.",
)


def create_checkpointer(
    config: config_dict.ConfigDict,
    mode: str,
) -> utils.Checkpointer:
    """Creates an object to be used as a checkpointer."""
    return utils.InMemoryCheckpointer(config, mode)


@ray.remote
class TensorBoardLogger:
    """Writer to write experiment data to stdout."""

    def __init__(self, config, mode: str):
        """Initializes the writer."""
        log_dir = os.path.join(config.checkpoint_dir, mode)
        self._writer = tf.summary.create_file_writer(log_dir)

    def write_scalars(self, global_step: int, scalars: Mapping[str, Any]):
        """Writes scalars to stdout."""
        global_step = int(global_step)
        with self._writer.as_default():
            for k, v in scalars.items():
                tf.summary.scalar(k, v, step=global_step)
        self._writer.flush()

    def write_images(self, global_step: int, images: Mapping[str, np.ndarray]):
        """Writes images to writers that support it."""
        global_step = int(global_step)
        with self._writer.as_default():
            for k, v in images.items():
                # Tensorboard only accepts [B, H, W, C] but we support [H, W] also.
                if v.ndim == 2:
                    v = v[None, ..., None]
                tf.summary.image(k, v, step=global_step)
        self._writer.flush()


def create_writer(config: config_dict.ConfigDict, mode: str) -> Any:
    """Creates an object to be used as a writer."""
    return TensorBoardLogger.remote(config, mode)


@utils.debugger_fallback
def main(experiment_class, argv, checkpointer_factory=create_checkpointer):
    """Main potentially under a debugger."""
    del argv  # Unused.

    # Make sure the required fields are available in the config.
    jax_config = config.get_config()
    base_config.validate_config(jax_config)

    if _JAXLINE_TPU_DRIVER.value:
        jax.config.update("jax_xla_backend", "tpu_driver")
        jax.config.update("jax_backend_target", _JAXLINE_TPU_DRIVER.value)
        logging.info("Backend: %s %r",
                     _JAXLINE_TPU_DRIVER.value, jax.devices())

    if _JAXLINE_ENSURE_TPU.value:
        # JAX currently falls back to CPU if it cannot register the TPU platform.
        # In multi-host setups this can happen if we timeout waiting for hosts to
        # come back up at startup or after pre-emption. This test will crash the
        # task if TPU devices are not available. We have increased the number of
        # acceptable failures per-task to allow for this.
        # TODO(tomhennigan) This test will eventually be part of JAX itself.
        chex.assert_tpu_available()

    jaxline_mode = _JAXLINE_MODE.value
    print("CONFIG", jax_config)
    if jaxline_mode == "train":
        # Run training.
        checkpointer = checkpointer_factory(jax_config, jaxline_mode)
        writer = create_writer(jax_config, jaxline_mode)
        print("TRAIN", jaxline_train.train)
        return jaxline_train.train.remote(experiment_class, jax_config, checkpointer, writer)
    elif jaxline_mode.startswith("eval"):
        # Run evaluation.
        checkpointer = checkpointer_factory(jax_config, jaxline_mode)
        writer = create_writer(jax_config, jaxline_mode)
        return jaxline_train.evaluate.remote(experiment_class, jax_config, checkpointer, writer,
                                             jaxline_mode)
    elif jaxline_mode == "train_eval_multithreaded":
        pool = futures.ThreadPoolExecutor(1)

        # Run training in a background thread!
        pool.submit(jaxline_train.train, experiment_class, jax_config,
                    checkpointer_factory(jax_config, "train"),
                    create_writer(jax_config, "train"))

        # Run eval!
        jaxline_train.evaluate(experiment_class, jax_config,
                               checkpointer_factory(jax_config, "eval"),
                               create_writer(jax_config, "eval"))

        # If we're here, eval has finished. Wait for train to finish!
        pool.shutdown()
    else:
        raise ValueError(f"Mode {jaxline_mode} not recognized.")
