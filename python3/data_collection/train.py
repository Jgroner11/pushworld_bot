import os
import argparse
import pickle
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple, Union, Optional, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


# ----------------------------
# Data shapes & conventions
# ----------------------------
# - Each state is a small "image" tensor with shape (C, H, W). For the starter,
#   C=2 channels: [agent_one_hot, goal_one_hot].
# - Actions are integer class labels in {0..A-1}. Default A=5: up, down, left, right, stay.
# - Episodes: list of dicts with keys like "observations": (T, C, H, W), "actions": (T,)
#   The dataset flattens to step-level samples.


@dataclass
class DataSpec:
    num_actions: int = 5           # up, down, left, right, stay
    channels: int = 2              # agent + goal channels
    grid_size: int = 11            # H=W
    # If you bring your own data, you can ignore grid_size so long as observations match (C, H, W).


def greedy_action_to_goal(agent_xy: Tuple[int, int], goal_xy: Tuple[int, int]) -> int:
    """
    Returns the greedy action index in {0:up, 1:down, 2:left, 3:right, 4:stay}
    """
    ax, ay = agent_xy
    gx, gy = goal_xy
    dx = gx - ax
    dy = gy - ay
    if abs(dx) >= abs(dy):
        if dx > 0:
            return 3  # right
        elif dx < 0:
            return 2  # left
        else:
            # dx=0, move in y
            if dy > 0:
                return 1  # down
            elif dy < 0:
                return 0  # up
            else:
                return 4  # stay
    else:
        if dy > 0:
            return 1  # down
        elif dy < 0:
            return 0  # up
        else:
            # dy=0, move in x
            if dx > 0:
                return 3
            elif dx < 0:
                return 2
            else:
                return 4


def render_state(agent_xy: Tuple[int, int], goal_xy: Tuple[int, int], grid: int) -> np.ndarray:
    """
    Create a 2-channel grid (C=2, H=grid, W=grid). Channel 0 is agent one-hot; channel 1 is goal one-hot.
    """
    c = 2
    state = np.zeros((c, grid, grid), dtype=np.float32)
    ax, ay = agent_xy
    gx, gy = goal_xy
    state[0, ay, ax] = 1.0
    state[1, gy, gx] = 1.0
    return state


def generate_dummy_episodes(n_episodes: int, max_steps: int, spec: DataSpec, rng: np.random.RandomState) -> List[Dict[str, np.ndarray]]:
    """
    Simple demonstration generator: agent starts random, goal random; labels are greedy moves.
    Produces a list of episodes, each with 'observations' (T, C, H, W) and 'actions' (T,).
    """
    episodes = []
    for _ in range(n_episodes):
        T = rng.randint(low=max(3, max_steps // 2), high=max_steps + 1)
        ax = rng.randint(0, spec.grid_size)
        ay = rng.randint(0, spec.grid_size)
        gx = rng.randint(0, spec.grid_size)
        gy = rng.randint(0, spec.grid_size)

        observations = np.zeros((T, spec.channels, spec.grid_size, spec.grid_size), dtype=np.float32)
        actions = np.zeros((T,), dtype=np.int64)

        x, y = ax, ay
        for t in range(T):
            observations[t] = render_state((x, y), (gx, gy), spec.grid_size)
            a = greedy_action_to_goal((x, y), (gx, gy))
            actions[t] = a

            # Step the agent according to the label (not random)
            if a == 0 and y > 0:
                y -= 1
            elif a == 1 and y < spec.grid_size - 1:
                y += 1
            elif a == 2 and x > 0:
                x -= 1
            elif a == 3 and x < spec.grid_size - 1:
                x += 1
            # a == 4 stay

        episodes.append({"observations": observations, "actions": actions})
    return episodes


def load_episodes_from_pickle(pickle_path: str) -> List[Dict[str, np.ndarray]]:
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    # Expect either:
    # - list of {"observations": (T,C,H,W), "actions": (T,)} dicts
    # - or a dict with keys "episodes": [...]
    if isinstance(data, dict) and "episodes" in data:
        return data["episodes"]
    if isinstance(data, list):
        return data
    raise ValueError("Unrecognized pickle data format; expected list of episodes or dict with 'episodes' key.")


class StepDataset(Dataset):
    """
    Adapts a list of episodes with 'observations' and 'actions' into step-level samples (state_t, action_t).
    """
    def __init__(self, episodes: List[Dict[str, np.ndarray]]):
        self.index = []  # (episode_idx, t)
        self.episodes = episodes
        for ei, ep in enumerate(episodes):
            T = ep["observations"].shape[0]
            for t in range(T):
                self.index.append((ei, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ei, t = self.index[i]
        obs_t = self.episodes[ei]["observations"][t]     # (C,H,W), float32
        act_t = self.episodes[ei]["actions"][t]    # (), int64
        x = torch.from_numpy(obs_t)             # (C,H,W)
        y = torch.tensor(act_t, dtype=torch.long)
        return x, y


class GridworldDataModule(pl.LightningDataModule):
    """
    Flexible DataModule:
    - If pickle_path is provided, loads episodes from it.
    - Else if generator is provided, calls it to produce episodes.
    - Else uses the dummy generator.
    """
    def __init__(
        self,
        batch_size: int,
        spec: DataSpec,
        train_episodes: int,
        val_episodes: int,
        max_steps: int,
        workers: int = 4,
        pickle_path: Optional[str] = None,
        generator: Optional[Callable[[], List[Dict[str, np.ndarray]]]] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.spec = spec
        self.train_episodes = train_episodes
        self.val_episodes = val_episodes
        self.max_steps = max_steps
        self.workers = workers
        self.pickle_path = pickle_path
        self.generator = generator
        self.seed = seed

        self._train_dataset = None
        self._val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        rng = np.random.RandomState(self.seed)

        if self.pickle_path:
            episodes = load_episodes_from_pickle(self.pickle_path)
            # Simple split
            n = len(episodes)
            n_train = max(1, int(0.9 * n))
            train_eps = episodes[:n_train]
            val_eps = episodes[n_train:]
            if not val_eps:
                val_eps = episodes[-1:]
        elif self.generator is not None:
            episodes = self.generator()
            n = len(episodes)
            n_train = max(1, int(0.9 * n))
            train_eps = episodes[:n_train]
            val_eps = episodes[n_train:]
            if not val_eps:
                val_eps = episodes[-1:]
        else:
            # Fallback: generate fresh dummy data
            train_eps = generate_dummy_episodes(self.train_episodes, self.max_steps, self.spec, rng)
            val_eps = generate_dummy_episodes(self.val_episodes, self.max_steps, self.spec, rng)

        self._train_dataset = StepDataset(train_eps)
        self._val_dataset = StepDataset(val_eps)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int, num_actions: int, grid_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/4
            nn.Flatten(),
        )
        # Compute flattened size with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, grid_size, grid_size)
            flat = self.net(dummy).shape[-1]
        self.head = nn.Sequential(
            nn.Linear(flat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        logits = self.head(z)
        return logits


class CNNPolicyLit(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_actions: int,
        grid_size: int,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SmallCNN(in_channels=in_channels, num_actions=num_actions, grid_size=grid_size)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--pickle_path", type=str, default=None, help="Optional path to a pickle of episodes.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--train_episodes", type=int, default=300)
    p.add_argument("--val_episodes", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=30)
    p.add_argument("--grid_size", type=int, default=11)
    p.add_argument("--num_actions", type=int, default=5)
    p.add_argument("--channels", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)

    # Model/opt
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--precision", type=str, default="32-true", choices=["32-true", "16-mixed", "bf16-mixed"])

    # Logging
    p.add_argument("--no_log", action="store_true", help="Disable logging to WandB.")

    return p.parse_args()


def main():
    args = parse_args()

    pl.seed_everything(args.seed, workers=True)

    spec = DataSpec(
        num_actions=args.num_actions,
        channels=args.channels,
        grid_size=args.grid_size,
    )

    # Optionally, you can plug in your own generator callable here.
    generator = None  # e.g., lambda: my_generator_that_returns_list_of_episodes()

    dm = GridworldDataModule(
        batch_size=args.batch_size,
        spec=spec,
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
        max_steps=args.max_steps,
        pickle_path=args.pickle_path,
        generator=generator,
        seed=args.seed,
    )

    model = CNNPolicyLit(
        in_channels=spec.channels,
        num_actions=spec.num_actions,
        grid_size=spec.grid_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    )

    config = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "grid_size": args.grid_size,
        "channels": args.channels,
        "num_actions": args.num_actions,
        "seed": args.seed,
        "max_epochs": args.max_epochs,
    }

    logger = WandbLogger(project='cscg-pushworld', entity='cscg-group', mode='disabled' if args.no_log else 'online')
    logger.experiment.config.update(config, allow_val_change=True)

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, save_last=True, filename="epoch{epoch:02d}-valacc{val_acc:.3f}"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        deterministic=True,
        logger=logger,
        callbacks=callbacks,
        precision=args.precision,
        log_every_n_steps=50,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
