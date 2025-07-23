# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import app
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pushworld.dm_env import PushWorldEnv
from pushworld.puzzle import NUM_ACTIONS


matplotlib.use('TkAgg')


def main(argv):
    # Choose puzzle
    path = "benchmark/puzzles/manual/simple.pwp"
    project_root = Path(__file__).resolve().parents[2]

    # Create gym environment
    env = PushWorldEnv(str(project_root / path))

    # Reset the environment and show observation
    timestep = env.reset()
    plt.figure(figsize=(5, 5))
    plt.imshow(timestep.observation)
    plt.ion()
    plt.show()

    # Randomly take 10 actions and show observation
    for _ in range(10):
        timestep = env.step(np.random.randint(NUM_ACTIONS))
        plt.imshow(timestep.observation)
        plt.draw()
        plt.pause(0.5)


if __name__ == '__main__':
    app.run(main)
