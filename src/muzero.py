from random import sample
from absl import flags, app
import ray
from experiment import get_experiment_class
from experience_replay import MuZeroMemory
from self_play import SelfPlayWorker
from jax import random
import jax
from model import MuZeroNet
import jaxline_platform
from utils import confirm_tpus
import time
from jaxline_platform import JaxlineWorker
from sample_queue import MemorySampler, RolloutWorker
from ray.util.queue import Queue
import config
from jaxline import base_config
from ml_collections import config_dict
from typing import Any, Mapping
import numpy as np
import tensorflow as tf
import os


@ray.remote(max_restarts=-1, max_task_retries=-1)
class GlobalParamsActor(object):
    def __init__(self, params):
        self.counter = 0
        self.params = params
        self.target_params = params

    def get_target_params(self):
        return self.target_params

    def sync_target_params(self):
        self.target_params = self.params

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params
        self.counter += 1
        if self.counter % 100 == 0:
            print("SYNCING TARGET PARAMS", self.counter)
            self.sync_target_params()


def create_writer(config: config_dict.ConfigDict, mode: str) -> Any:
    """Creates an object to be used as a writer."""
    return TensorBoardLogger.remote(config, mode)


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


# TODO confirm that target_params and params are being used in the right spot
def main(argv):
    ray.init(address='auto', include_dashboard=True)
    print("***RESOURCES", ray.nodes())

    worker_counts = {"self_play": 6, "train": 6, "sample": 7, "rollout": 8}
    rollout_size = 5

    key = random.PRNGKey(0)

    network_key, worker_key = random.split(key)
    memory_actor = MuZeroMemory.options(max_concurrency=16).remote(
        8000, rollout_size=rollout_size)
    network = MuZeroNet()
    network_key, representation_params, dynamics_params, prediction_params = network.initialize_networks_individual(
        network_key)
    params = (representation_params, dynamics_params, prediction_params)
    params_actor = GlobalParamsActor.remote(params)
    cpu = jax.devices("cpu")[0]

    with jax.default_device(cpu):
        sample_queue = Queue(25, actor_options={
            "resources": {"PREEMPT_TPU": 1, "TPU_VM_CPU": 1}, "num_cpus": 1, "max_restarts": -1, "max_task_retries": -1})
        rollout_queue = Queue(25, actor_options={
            "resources": {"PREEMPT_TPU": 1, "TPU_VM_CPU": 1}, "num_cpus": 1, "max_restarts": -1, "max_task_retries": -1})

    experiment_class = get_experiment_class(
        memory_actor, params_actor, rollout_queue, sample_queue)

    flags.mark_flag_as_required('config')

    workers = []
    print("WORKERS", workers)
    time.sleep(30)
    workers += [self_play_workers.remote(worker_key, params_actor, memory_actor,
                                         worker_counts["self_play"])]
    workers += [jaxline_workers.remote(experiment_class,
                                       worker_counts["train"])]
    workers += [sample_workers.remote(sample_queue, memory_actor,
                                      worker_counts["sample"], key)]

    workers += [rollout_workers.remote(sample_queue, rollout_queue,
                                       worker_counts["rollout"])]

    workers_to_run = []
    # for worker in self_play_workers:
    #     if ray.get(worker.has_tpus.remote()):
    #         workers_to_run.append(worker)

    ray.get(workers)
    ray.shutdown()
    print("AFTER WAIT")


@ray.remote(num_cpus=1)
def rollout_workers(sample_queue, rollout_queue, worker_count):
    workers = []
    i = 0
    while i < 10:
        workers = []

        for _ in range(worker_count):
            rollout_worker = RolloutWorker.remote(
                sample_queue, rollout_queue, 64)
            for _ in range(8):
                workers += [rollout_worker.rollout.remote()]
            i += 1
            try:
                ray.wait(workers)
            except ray.exceptions.WorkerCrashedError:
                print('ROLLOUT FAILURE ', i)


@ray.remote(num_cpus=1)
def sample_workers(sample_queue, memory_actor, worker_count, key):
    workers = []
    i = 0
    while i < 10:
        workers = []
        for _ in range(worker_count):
            memory_sampler = MemorySampler.remote(
                sample_queue, memory_actor, 64)
            for _ in range(8):
                key, sample_key = random.split(key)
                workers += [memory_sampler.run_queue.remote(
                    sample_key)]
        i += 1
        try:
            ray.wait(workers)
        except ray.exceptions.WorkerCrashedError:
            print('SAMPLE FAILURE ', i)


@ray.remote(num_cpus=1)
def jaxline_workers(experiment_class, worker_count):
    jaxline_workers = [JaxlineWorker.remote()
                       for _ in range(worker_count)]

    i = 0
    jax_config = config.get_config()
    base_config.validate_config(jax_config)
    writer = create_writer(jax_config, "train")

    while i < 10:
        workers = [jaxline_worker.run.remote(
            experiment_class, writer, jax_config) for jaxline_worker in jaxline_workers]
        i += 1
        try:
            ray.wait(workers)
        except ray.exceptions.WorkerCrashedError:
            print('TRAIN FAILURE ', i)


@ray.remote(num_cpus=1)
def self_play_workers(worker_key, params_actor, memory_actor, worker_count):
    multiple = 4
    num_envs = multiple * 16
    env_batch_size = int(num_envs / 4)
    self_play_workers = [SelfPlayWorker.remote(i, num_envs, env_batch_size,
                                               worker_key, params_actor, memory_actor) for i in range(worker_count)]
    workers_to_run = []
    for worker in self_play_workers:
        if ray.get(worker.has_tpus.remote()):
            workers_to_run.append(worker)

    i = 0
    while i < 10:
        workers = [self_play_worker.play.remote()
                   for self_play_worker in workers_to_run]
        i += 1
        try:
            ray.wait(workers)
        except ray.exceptions.WorkerCrashedError:
            print('SELF PLAY FAILURE ', i)


app.run(lambda argv: main(argv))
