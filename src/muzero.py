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
from sample_queue import MemorySampler
from ray.util.queue import Queue


@ray.remote(max_restarts=-1, max_task_retries=-1)
class GlobalParamsActor(object):
    def __init__(self, params, head_node_id):
        # confirm_tpus(head_node_id)
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


# @ray.remote(resources={"TPU": 1})
# class InitializeNode:
#     def initialize_custer(self):
#         print("REMOTE")
#         jax.distributed.initialize()


# def initialize_nodes():
#     num_nodes = len(ray.nodes()) - 2
#     bundles = [{"TPU": 1, "CPU": 1} for _ in range(num_nodes)]
#     pg = ray.util.placement_group(
#         bundles=bundles, strategy="STRICT_SPREAD")
#     ray.get(pg.ready())
#     executors = [InitializeNode.options(
#         placement_group=pg).remote() for n in range(num_nodes)]
#     ray.get([executor.initialize_custer.remote() for executor in executors])
#     ray.util.remove_placement_group(pg)


# TODO confirm that target_params and params are being used in the right spot
def main(argv):
    ray.init(address='auto')
    print("***RESOURCES", ray.nodes())
    # initialize_nodes()
    head_node_id = None
    for node in ray.nodes():
        if 'TPU' not in node["Resources"].keys():
            head_node_id = node["NodeID"]
            break

    rollout_size = 5

    multiple = 4
    num_envs = multiple * 16
    env_batch_size = int(num_envs / 4)
    key = random.PRNGKey(0)

    network_key, worker_key = random.split(key)
    memory_actor = MuZeroMemory.remote(
        400, rollout_size=rollout_size, head_node_id=head_node_id)
    network = MuZeroNet()
    network_key, representation_params, dynamics_params, prediction_params = network.initialize_networks_individual(
        network_key)
    params = (representation_params, dynamics_params, prediction_params)
    params_actor = GlobalParamsActor.remote(params, head_node_id)

    sample_actor = Queue(10)
    memory_sampler = MemorySampler.remote(sample_actor, memory_actor, 64)
    experiment_class = get_experiment_class(
        memory_actor, params_actor, sample_actor)

    flags.mark_flag_as_required('config')
    print("WORKERS")

    self_play_workers_count = 6
    training_worker_count = 2
    jaxline_workers = [JaxlineWorker.remote()
                       for _ in range(training_worker_count)]
    workers = []
    print("WORKERS", workers)
    time.sleep(30)
    self_play_workers = [SelfPlayWorker.remote(i, num_envs, env_batch_size,
                                               worker_key, params_actor, memory_actor, head_node_id) for i in range(self_play_workers_count)]
    workers_to_run = []
    for worker in self_play_workers:
        if ray.get(worker.has_tpus.remote()):
            workers_to_run.append(worker)

    workers += [self_play_worker.play.remote()
                for self_play_worker in workers_to_run]
    for _ in range(4):
        key, sample_key = random.split(key)
        workers += [memory_sampler.run_queue.remote(
            key)]
    workers += [jaxline_worker.run.remote(
        experiment_class) for jaxline_worker in jaxline_workers]

    print(workers)
    ray.get(workers)
    print("AFTER WAIT")


# @ray.remote(resources={"TPU": 1})
# def run_jaxline(experiment_class):
#     _NODE_IP = flags.DEFINE_string(
#         name="node-ip-address",
#         default="",
#         help="ray node ip",
#     )
#     _NODE_PORT = flags.DEFINE_string(
#         name="node-manager-port",
#         default="",
#         help="ray node ip",
#     )

#     _OBJECT_STORE = flags.DEFINE_string(
#         name="object-store-name",
#         default="",
#         help="ray node ip",
#     )

#     _RAYLET_NAME = flags.DEFINE_string(
#         name="raylet-name",
#         default="",
#         help="ray node ip",
#     )

#     _REDIS_ADDRESS = flags.DEFINE_string(
#         name="redis-address",
#         default="",
#         help="ray node ip",
#     )

#     def app_run(argv):
#         print("ARGV", argv)
#         flags.mark_flag_as_required('config')
#         platform.main(experiment_class, argv)
#     app.run(lambda argv: app_run(argv))


app.run(lambda argv: main(argv))
