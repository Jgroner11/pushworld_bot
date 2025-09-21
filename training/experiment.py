import yaml
from pathlib import Path
import os
import shutil
import stat
import time
import pickle
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pushworld.gym_env import PushWorldEnv
from pushworld.puzzle import NUM_ACTIONS

from encoders import *
from encoder_training import learn_encoder

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

@dataclass
class ExperimentData:
    config: ConfigWrapper
    input: NDArray[np.int64]
    x: NDArray[np.int64]
    a: NDArray[np.int64]
    encoder: nn.Module
    chmm: CHMM


    def save(self, experiment_name):
        experiment_path = project_root / "experiments" / experiment_name
        data_path  = experiment_path / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        with open(experiment_path / "config.yml", "w") as f:
            yaml.safe_dump(
                dict(self.config),
                f,
                default_flow_style=False,
                sort_keys=False,    # preserve the schema order
                indent=2
            )
        np.save(data_path / "actions.npy", self.a)
        np.save(data_path / "observations.npy", self.x)
        np.save(data_path / "input.npy", self.input)
        with open(data_path / "encoder.pkl", "wb") as f:
            pickle.dump(self.encoder, f)
        with open(data_path / "cscg.pkl", "wb") as f:
            pickle.dump(self.chmm, f)

    def get_all(self):
        """
        Returns (config, input, x, a, encoder, chmm)
        """
        return self.config, self.input, self.x, self.a, self.encoder, self.chmm

    @classmethod
    def load(cls, experiment_name):
        experiment_path = project_root / "experiments" / experiment_name

        if not experiment_path.exists():
            raise Exception(f"Experiment {experiment_name} could not be found.")

        data_path = experiment_path / "data"

        # Load config
        with open(experiment_path / "config.yml", "r") as f:
            config_dict = yaml.safe_load(f)
        config = ConfigWrapper(config_dict)

        # Load arrays
        a = np.load(data_path / "actions.npy")
        x = np.load(data_path / "observations.npy")
        input = np.load(data_path / "input.npy")

        # Load pickled objects
        with open(data_path / "encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        with open(data_path / "cscg.pkl", "rb") as f:
            chmm = pickle.load(f)

        return cls(
            config=config,
            input=input,
            x=x,
            a=a,
            encoder=encoder,
            chmm=chmm
        )
    

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

def _handle_remove_readonly(func, path, exc_info):
    """shutil.rmtree onerror handler: make path writable then retry."""
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    except Exception:
        pass
    try:
        func(path)
    except PermissionError:
        # brief wait in case a sync client releases the handle
        time.sleep(0.2)
        func(path)

def delete_all_experiments(root=None):
    base = (project_root / "experiments") if root is None else Path(root)
    if not base.exists():
        return

    for item in base.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item, onerror=_handle_remove_readonly)
            else:
                try:
                    os.chmod(item, stat.S_IWRITE | stat.S_IREAD)
                except Exception:
                    pass
                item.unlink()
        except PermissionError:
            # Last resort: rename to break open handles, then delete
            temp = item.with_name(item.name + ".__del__")
            try:
                item.rename(temp)
                if temp.is_dir():
                    shutil.rmtree(temp, onerror=_handle_remove_readonly)
                else:
                    try:
                        os.chmod(temp, stat.S_IWRITE | stat.S_IREAD)
                    except Exception:
                        pass
                    temp.unlink()
            except Exception as e:
                print(f"Failed to remove {item}: {e}")
    

class Experiment:
    cur_trial_length = -1
    def __init__(self, config_path: str, schema_path: str = Path(__file__).resolve().parents[0] / "schema.yaml", name=None, overwrite=False):
        if name is None:
            name = "exp"
        self.name = assign_name(name, project_root / "experiments", overwrite)

        self.config_path = Path(config_path)
        self.schema_path = Path(schema_path)
        self.config, self._schema = Experiment.load_config(config_path, schema_path)
        self._validate_config()

        Experiment.cur_trial_length = int(np.ceil(np.random.normal(loc=self.config.environment_reset.mean, scale=self.config.environment_reset.variance)))


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

    def check_reset(self, n_steps):
        
        match self.config.environment_reset.method:
            case "geometric":
                return np.random.rand() < self.config.environment_reset.p
            case "gaussian":
                if n_steps >= Experiment.cur_trial_length:
                    Experiment.cur_trial_length = int(np.ceil(np.random.normal(loc=self.config.environment_reset.mean, scale=self.config.environment_reset.variance)))
                    return True
        return False

    def run(self):
        config = self.config


        np.random.seed(config.seed)

        # Make puzzle path
        path = project_root / "benchmark/puzzles" / config.puzzle
        if not path.suffix:
            path = path.with_suffix(".pwp")
        puzzle_path = str(path)

        env = PushWorldEnv(puzzle_path, border_width=2, pixels_per_cell=20) # these are the defaults
        image, info = env.reset()

        if config.separate_cscg_train_encoder is not None:
            encoder = globals()[config.separate_cscg_train_encoder](image.shape, config.n_obs)
        else:
            encoder = globals()[config.encoder](image.shape, config.n_obs)

        x = np.zeros(config.seq_len, dtype=np.int64)
        a = np.zeros(config.seq_len, dtype=np.int64)
        input = np.zeros((config.seq_len,) + image.shape, dtype=np.int64)

        n_steps = 0
        for i in range(config.seq_len):
            action = np.random.randint(NUM_ACTIONS)
            input[i] = image
            x[i] = encoder.classify(image)
            a[i] = action
            if self.check_reset(n_steps):
                image, info = env.reset()
                n_steps = 0
            else:
                rets = env.step(action)
                image = rets[0]
                n_steps += 1

        n_clones = np.ones(config.n_obs, dtype=np.int64) * config.clones_per_obs

        chmm = CHMM(n_clones=n_clones, pseudocount=2e-3, x=x, a=a, seed=42)  # Initialize the model
        progression = chmm.learn_em_T(x, a, n_iter=config.training_procedure.n_iters_cscg)  # Training

        chmm.pseudocount = 0.0
        chmm.learn_viterbi_T(x, a, n_iter=100)

        # If you previously set the encoder to the separate_cscg_train_encoder, now use the actual encoder for encoder training
        if config.separate_cscg_train_encoder is not None:
            encoder = globals()[config.encoder](image.shape, config.n_obs)

        if not isinstance(encoder, IntEncoder): # Don't train encoder when using an IntEncoder (doesn't have weights)
            T = torch.tensor(chmm.T, dtype=torch.float32)
            E = torch.zeros((sum(chmm.n_clones), config.n_obs), dtype=torch.float32) # shape of E is n_latent_states x n_obs
            state_loc = np.hstack(([0], chmm.n_clones)).cumsum(0)
            for i in range(config.n_obs):
                E[state_loc[i]:state_loc[i+1], i] = 1.0
            pi = torch.tensor(chmm.Pi_x)

            learn_encoder(encoder, torch.as_tensor(input, dtype=torch.float32), a, T, E, pi, n_iters=config.training_procedure.n_iters_encoder)

        data = ExperimentData(config, input, x, a, encoder, chmm)
        data.save(self.name)

        
        
if __name__ == "__main__":
    config_path = Path(__file__).resolve().parents[0] / "config.yaml"

    exp = Experiment(config_path, overwrite=False)
    exp.run()
    # delete_all_experiments()
