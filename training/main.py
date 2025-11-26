from experiment import Experiment

# Before running, set your parameters in the config file in this directory. You can see all possible parameters in the schem.yaml file in this directory
config_path = "config.yaml"

exp = Experiment(config_path, name="exp2", overwrite=True)
exp.run()
