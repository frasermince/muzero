from absl import flags, app
import sys
import ray
from experiment import get_experiment_class
from jaxline import platform
from experience_replay import MuZeroMemory
from self_play import SelfPlayWorker
from jax import random
from model import MuZeroNet


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
    flags.mark_flag_as_required('config')
    print(sys.argv[1])
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
    ray.get([run_jaxline.remote(experiment_class, argv),
             self_play_worker.play.remote()])
    print("AFTER WAIT")


@ray.remote(resources={"TPU": 1})
def run_jaxline(experiment_class, argv):
    platform.main(experiment_class, argv)


app.run(lambda argv: main(argv))
