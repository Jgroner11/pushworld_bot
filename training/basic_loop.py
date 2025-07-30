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

from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pushworld.puzzle import NUM_ACTIONS
from pushworld.gym_env import PushWorldEnv

from chmm_actions import CHMM

from absl import app

matplotlib.use('TkAgg')


class IntEncoder:
    def __init__(self, max_size=np.inf):
        self._size = 0
        self._map = {}
        self.max_size = max_size

    def encode(self, x):
        key =  tuple(x.ravel().tolist())
        if key in self._map:
            return self._map[key]
        elif self._size == self.max_size:
            raise Exception("Encoder ran out of unique mappings")
        else:
            self._map[key] = self._size
            self._size += 1
            return self._map[key]
        
    def encode_array(self, arr):
        r = np.array(dtype=np.int64)
        for x in arr:
            r.append(self.encode(x))
        return r
    

def main(argv):
    seq_len = 10
    n_obs = 10
    # Choose puzzle
    path = "benchmark/puzzles/manual/actor_only.pwp"


    project_root = Path(__file__).resolve().parents[1]

    
    # Create gym environment
    env = PushWorldEnv(str(project_root / path), border_width=1, pixels_per_cell=5)
    
    # Reset the environment and show observation
    encoder = IntEncoder(max_size=n_obs)
    o = np.zeros((seq_len), dtype=np.int64)
    a = np.zeros(seq_len, dtype=np.int64)


    image, info = env.reset()
    
    # plt.figure(figsize=(5, 5))
    # plt.imshow(image)
    # plt.ion()
    # plt.show()


    # Randomly take 10 actions and show observation
    for i in range(seq_len):
        action = np.random.randint(NUM_ACTIONS)
        
        o[i] = encoder.encode(image)
        a[i] = action

        rets = env.step(action)

        image = rets[0]

        plt.imshow(image)
        plt.draw()
        plt.pause(0.01)

    n_clones = np.ones(n_obs, dtype=np.int64) * 1


    chmm = CHMM(n_clones=n_clones, pseudocount=2e-3, x=o, a=a, seed=42)  # Initialize the model
    progression = chmm.learn_em_T(o, a, n_iter=100)  # Training

    chmm.pseudocount = 0.0
    chmm.learn_viterbi_T(o, a, n_iter=100)

if __name__ == '__main__':
    app.run(main)
