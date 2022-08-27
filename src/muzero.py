from absl import flags, app
import sys
import ray
from experiment import get_experiment_class
from experience_replay import MuZeroMemory
from self_play import SelfPlayWorker
from jax import random
from model import MuZeroNet
import jaxline_platform


@ray.remote
class GlobalParamsActor(object):
    def __init__(self, params):
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


def main(argv):
    ray.init(address='auto')
    print("***RESOURCES", ray.nodes())
    rollout_size = 5

    multiple = 4
    num_envs = multiple * 16
    env_batch_size = int(num_envs / 4)
    key = random.PRNGKey(0)

    network_key, worker_key = random.split(key)
    memory_actor = MuZeroMemory.remote(5000, rollout_size=rollout_size)
    network = MuZeroNet()
    network_key, representation_params, dynamics_params, prediction_params = network.initialize_networks_individual(
        network_key)
    params = (representation_params, dynamics_params, prediction_params)
    params_actor = GlobalParamsActor.remote(params)

    self_play_worker = SelfPlayWorker.remote(num_envs, env_batch_size,
                                             worker_key, params_actor, memory_actor)
    experiment_class = get_experiment_class(memory_actor, params_actor)

    flags.mark_flag_as_required('config')
    jaxline_worker = jaxline_platform.main(experiment_class, argv)
    ray.get([jaxline_worker,
             self_play_worker.play.remote()])
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


main()
app.run(lambda argv: main(argv))
