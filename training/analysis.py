from experiment import ExperimentData
from cscg_helpers import Plotting, Reasoning

from pushworld.gym_env import PushWorldEnv
from pushworld.puzzle import Actions, NUM_ACTIONS

import numpy as np
import torch
from pathlib import Path
from queue import Queue, Empty

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

project_root = Path(__file__).resolve().parents[1]

class Analysis:
    @staticmethod
    def _place_figure(fig, x, y):
        # Works reliably with TkAgg on Windows
        mgr = fig.canvas.manager
        win = getattr(mgr, "window", fig.canvas.get_tk_widget().winfo_toplevel())

        fig.canvas.draw()             # ensure widgets exist
        win.update_idletasks()

        # Make sure it's a normal, deiconified window before moving
        try:
            if win.state() != "normal":
                win.state("normal")
        except Exception:
            pass

        # Withdraw -> set geometry -> deiconify tends to stick
        try:
            win.withdraw()
            win.geometry(f"+{x}+{y}")
            win.deiconify()
        except Exception:
            # Fallback: schedule after the current event loop turn
            win.after(0, lambda: win.geometry(f"+{x}+{y}"))

        # Bring to front and focus (optional)
        try:
            win.lift()
            win.focus_force()
        except Exception:
            pass

    @staticmethod
    def simulate_trained_model(experiment_name):
        e = ExperimentData.load(experiment_name=experiment_name)

        # Put a prior on transitions so that a never seen before sequence still has some probability
        e.chmm.pseudocount = 2e-3
        e.chmm.update_T()

        # Setup gym environment
        path = project_root / "benchmark/puzzles" / e.config.puzzle
        if not path.suffix:
            path = path.with_suffix(".pwp")
        puzzle_path = str(path)
        env = PushWorldEnv(puzzle_path, border_width=2, pixels_per_cell=20) # these are the defaults
        image, info = env.reset()

        # Give model a memory of the sequence, in the future this memory sequence could be a finite buffer
        input = [image]
        x = [e.encoder.classify(image)]
        a = []
        v = Reasoning.sum_product_decode(e.chmm, np.array(x, dtype=np.int64), np.array(a, dtype=np.int64)) # Identify initial state

        # Cache the unique states so you don't have to call decode every time you plot
        unique_states = np.unique(e.chmm.decode(e.x, e.a)[1])
            
        # Figs
        plt.ion()

        game_fig, game_ax = plt.subplots(figsize=(5, 5))
        game_fig.canvas.flush_events()
        Analysis._place_figure(game_fig, 2200, 0)
        game_image = game_ax.imshow(image)

        cscg_fig, cscg_ax = plt.subplots(figsize=(5, 5))
        cscg_ax.axis("off")
        cscg_fig.canvas.flush_events()
        Analysis._place_figure(cscg_fig, 1000, 0)

        img_path = project_root / "experiments" / experiment_name / "current_state.png"
        Plotting.plot_heat_map(e.chmm, e.x, e.a, v, output_file=img_path, unique_states=unique_states)
        cscg_image = cscg_ax.imshow(mpimg.imread(img_path), cmap="viridis")
        cbar = plt.colorbar(cscg_image, ax=cscg_ax, orientation='vertical')

        # Set up game loop
        loop = True
        act_q = Queue()

        # Assign keys to 
        RESET = 4
        act_map = {
            "up": Actions.UP,
            "right": Actions.RIGHT,
            "left": Actions.LEFT,
            "down": Actions.DOWN,
            "r": RESET,
        }

        def on_press(event):
            nonlocal loop, act_q 
            if event.key == 'q':
                loop = False
                plt.close('all')
            elif event.key == 'a':
                act_q.put(np.random.randint(NUM_ACTIONS))
            elif event.key in act_map.keys():
                act_q.put(act_map[event.key])

        game_fig.canvas.mpl_connect('key_press_event', on_press)

        while loop and plt.fignum_exists(game_fig.number) and plt.fignum_exists(cscg_fig.number):
            try:
                action = act_q.get(timeout=.01)
                print(e.encoder.classify(image))
                print(e.encoder.forward(torch.as_tensor(image)))


                if action == RESET:
                    # Reset env and reset memory
                    image, info = env.reset()
                    input = [image]
                    x = [e.encoder.classify(image)]
                    a = []
                else:
                    # step env and update memory
                    rets = env.step(action)
                    image = rets[0]
                    a.append(action)
                    input.append(image)
                    x.append(e.encoder.classify(image))

                game_image.set_data(image)
                v = Reasoning.sum_product_decode(e.chmm, np.array(x, dtype=np.int64), np.array(a, dtype=np.int64))
                Plotting.plot_heat_map(e.chmm, e.x, e.a, v, output_file=img_path, unique_states=unique_states)
                cscg_image.set_data(mpimg.imread(img_path))
            except Empty:
                pass
            plt.pause(.01)

