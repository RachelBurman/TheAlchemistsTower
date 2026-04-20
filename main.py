import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agent.train import train, plot_training

if __name__ == "__main__":
    train(
        n_episodes=1000,
        max_steps=400,
        print_every=25,
        save_dir="checkpoints",
    )
    plot_training(save_dir="checkpoints")
