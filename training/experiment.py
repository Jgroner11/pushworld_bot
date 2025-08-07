import yaml
from pathlib import Path
import shutil
import pickle

import numpy as np

from pushworld.gym_env import PushWorldEnv
from pushworld.puzzle import NUM_ACTIONS

from encoders import *
from chmm_actions import CHMM

project_root = Path(__file__).resolve().parents[1]

class ConfigWrapper(dict):
    """Dict with attribute-style access. So that config parameters can be accessed with config.parameter """
    def __getattr__(self, key):
        value = self.get(key)
        if isinstance(value, dict) and not isinstance(value, ConfigWrapper):
            return ConfigWrapper(value)
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def assign_name(name, dir, overwrite):
    if not (dir / name).exists():
        return name
    i = 1
    while (dir / f"{name}{i}").exists():
        i += 1
    if i == 1 and overwrite:
        return name
    if overwrite:
        i -= 1
    return f"{name}{i}"

def delete_all_experiments():
    for item in (project_root / "experiments").iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    

class Experiment:
    def __init__(self, config_path: str, schema_path: str = Path(__file__).resolve().parents[0] / "schema.yaml", name=None, overwrite=False):

        if name is None:
            name = "exp"
        self.name = assign_name(name, project_root / "experiments", overwrite)

        self.config_path = Path(config_path)
        self.schema_path = Path(schema_path)
        self.config, self._schema = Experiment.load_config(config_path, schema_path)
        self._validate_config()

    @staticmethod
    def load_config(config_path, schema_path):
        """Returns ConfigWrapper after loading schema and config files."""
        with open(schema_path, "r") as f:
            schema = yaml.safe_load(f)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

        def merge(schema_section, config_section):
            merged = {}
            for key, val in schema_section.items():
                # Nested section
                if isinstance(val, dict) and "default" not in val:
                    merged[key] = merge(
                        val,
                        config_section.get(key, {}) if isinstance(config_section.get(key), dict) else {}
                    )
                else:
                    # Use config override if available, else schema default
                    merged[key] = config_section.get(key, val["default"])
            return merged

        config = merge(schema, config)
        return ConfigWrapper(config), schema

    def _validate_config(self):
        """Check if options are respected."""
        def validate(config_section, schema_section, path=""):
            for key, val in schema_section.items():
                if isinstance(val, dict) and "default" not in val:
                    validate(config_section[key], val, path + key + ".")
                else:
                    if "options" in val:
                        if config_section[key] not in val["options"]:
                            raise ValueError(
                                f"{path+key} must be one of {val['options']}, got {config_section[key]}"
                            )
        validate(self.config, self._schema)

    def run(self):
        config = self.config 

        # Make puzzle path
        path = project_root / "benchmark/puzzles" / config.puzzle
        if not path.suffix:
            path = path.with_suffix(".pwp")
        puzzle_path = str(path)

        env = PushWorldEnv(puzzle_path, border_width=2, pixels_per_cell=20) # these are the defaults
        image, info = env.reset()

        encoder = globals()[config.encoder](image.shape, config.n_obs)
        o = np.zeros(config.seq_len, dtype=np.int64)
        a = np.zeros(config.seq_len, dtype=np.int64)
        input = np.zeros((config.seq_len,) + image.shape, dtype=np.int64)

        # Randomly take 10 actions and show observation
        for i in range(config.seq_len):
            action = np.random.randint(NUM_ACTIONS)
            
            input[i] = image
            o[i] = encoder(image)
            a[i] = action

            rets = env.step(action)

            image = rets[0]

        n_clones = np.ones(config.n_obs, dtype=np.int64) * 1

        chmm = CHMM(n_clones=n_clones, pseudocount=2e-3, x=o, a=a, seed=42)  # Initialize the model
        progression = chmm.learn_em_T(o, a, n_iter=100)  # Training

        chmm.pseudocount = 0.0
        chmm.learn_viterbi_T(o, a, n_iter=100)

        experiment_path = project_root / "experiments" / self.name
        experiment_path.mkdir(parents=True, exist_ok=True)
        
        with open(experiment_path / "config.yml", "w") as f:
            yaml.safe_dump(
                dict(config),
                f,
                default_flow_style=False,
                sort_keys=False,    # preserve the schema order
                indent=2
            )



        np.save(experiment_path / "actions.npy", a)
        np.save(experiment_path / "observations.npy", o)
        np.save(experiment_path / "input.npy", input)
        with open(experiment_path / "cscg.pkl", "wb") as f:
            pickle.dump(chmm, f)
        with open(experiment_path / "encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)




if __name__ == "__main__":
    config_path = Path(__file__).resolve().parents[0] / "config.yaml"

    exp = Experiment(config_path, overwrite=True)
    exp.run()
    # delete_all_experiments()
