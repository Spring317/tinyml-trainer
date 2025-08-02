import numpy as np
import matplotlib.pyplot as plt
import json


class MonteCarlosimulation:
    """A class to perform Monte Carlo simulations for the DL model to evaluate it actual performance.

    Args:
        model (object): The DL model to be evaluated.
        data (list): The dataset to be used for the simulation.
        num_simulations (int): The number of simulations to run.
    """

    def __init__(self, model, data, num_simulations=1000):
        self.model = model
        self.data = data
        self.num_simulations = num_simulations

    def run_simulation(self):
        """Run the Monte Carlo simulation."""
        results = []
        for _ in range(self.num_simulations):
            # Simulate a random sample from the data
            sample = self._get_random_sample()
            # Evaluate the model on the sample
            result = self.model.evaluate(sample)
            results.append(result)
        return results

    def _get_random_sample(self):
        """Get a random sample from the dataset."""
        import random

        return random.choice(self.data)

    def get_summary_statistics(self, results):
        """Get summary statistics from the simulation results."""
        import numpy as np

        mean = np.mean(results)
        std_dev = np.std(results)
        return {
            "mean": mean,
            "std_dev": std_dev,
            "min": np.min(results),
            "max": np.max(results),
        }

    def plot_results(self, results):
        """Plot the results of the simulation."""
        import matplotlib.pyplot as plt

        plt.hist(results, bins=30, alpha=0.7)
        plt.title("Monte Carlo Simulation Results")
        plt.xlabel("Evaluation Metric")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    def save_results(self, results, filename):
        """Save the results of the simulation to a file."""
        import json

        with open(filename, "w") as f:
            json.dump(results, f)
