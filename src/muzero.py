from absl import flags, app
import sys
import ray
from experiment import MuzeroExperiment
from jaxline import platform


def main(argv, experiment_class):
    ray.init()
    flags.mark_flag_as_required('config')
    print(sys.argv[1])
    platform.main(experiment_class, argv)


app.run(lambda argv: main(argv, MuzeroExperiment))
