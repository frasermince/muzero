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
from utils import confirm_tpus
from jax.tree_util import register_pytree_node
from experience_replay import MuZeroMemory, SelfPlayMemory, GameMemory
import experience_replay

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


@ray.remote(max_restarts=-1, max_task_retries=-1)
class TensorBoardLogger:
    """Writer to write experiment data to stdout."""

    def __init__(self, config, mode: str):
        # confirm_tpus(head_node_id)
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


@ray.remote(resources={"PREEMPT_TPU": 1}, max_restarts=-1, max_task_retries=-1)
class JaxlineWorker:
    def __init__(self) -> None:
        jax.distributed.initialize()
        register_pytree_node(
            GameMemory,
            experience_replay.game_memory_flatten,
            experience_replay.game_memory_unflatten
        )

        register_pytree_node(
            SelfPlayMemory,
            experience_replay.self_play_flatten,
            experience_replay.self_play_unflatten
        )
        register_pytree_node(
            MuZeroMemory,
            experience_replay.muzero_flatten,
            experience_replay.muzero_unflatten
        )

    def run(self, experiment_class):
        """Main potentially under a debugger."""
        # Make sure the required fields are available in the config.
        chex.assert_tpu_available()

        jax_config = config.get_config()
        base_config.validate_config(jax_config)

        jaxline_mode = "train"
        print("CONFIG", jax_config)
        if jaxline_mode == "train":
            # Run training.
            writer = create_writer(jax_config, jaxline_mode)
            print("TRAIN", jaxline_train.train)
            jaxline_train.train(experiment_class, jax_config, writer)
        elif jaxline_mode.startswith("eval"):
            # Run evaluation.
            checkpointer = checkpointer_factory(jax_config, jaxline_mode)
            writer = create_writer(jax_config, jaxline_mode)
            jaxline_train.evaluate(experiment_class, jax_config, checkpointer, writer,
                                   jaxline_mode)
        else:
            raise ValueError(f"Mode {jaxline_mode} not recognized.")
