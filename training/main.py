experiment_name = "Simple_VQ"

from pathlib import Path
from experiment import Experiment, project_root

# Before running, set your parameters in the config file in this directory. You can see all possible parameters in the schem.yaml file in this directory
config_path = Path(__file__).resolve().parent / "config.yaml"

# exp = Experiment(config_path, name=experiment_name, overwrite=True)
# exp.run()


from experiment import ExperimentData
from cscg_helpers import Plotting

e = ExperimentData.load(experiment_name=experiment_name)
Plotting.plot_graph(e.chmm, e.x, e.a, output_file=project_root / "experiments" / experiment_name / "cscg.png")

from analysis import Analysis

# Use "a", "r" and arrow keys to move the agent while viewing the inferred state
Analysis.simulate_trained_model(experiment_name=experiment_name)
