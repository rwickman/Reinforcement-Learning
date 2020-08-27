"""The 10-armed Bandits Testbed.

Uses the example explained in section 2.3 in Reinforcement Learning: An Introduction.
"""

import numpy as np
from matplotlib import pyplot as plt
import argparse


class Bandits:
    def __init__(self, k=10):
        self.k = k
        self.reset_q_true()

    def reset_q_true(self):
        """Reset the true action-values."""
        self.q_true = np.random.normal(size=10)

    def take_action(self, a):
        """Take the action.

        Args:
            a: the action to take.

        Returns:
            the sampled reward.
        """
        return np.random.normal(loc=self.q_true[a])

    def run(self, epsilon, steps, num_runs):
        """Run the k-armed bandits problem."""
        rewards = np.zeros(steps)
        rng = np.random.default_rng()

        for cur_run in range(num_runs):
            if cur_run > 0:
                self.reset_q_true()

            # Action-values estimates
            q_vals = np.zeros(self.k)

            # Action counts
            a_counts = np.zeros(self.k)
 
            # Run the steps
            for i in range(steps):
                ep_prob = rng.random()
                if ep_prob <= epsilon:
                    # Select randomly from all actions
                    action = rng.integers(0, self.k)
                else:
                    # Select the greedy action and break ties randomly
                    q_max = np.max(q_vals)
                    action = np.random.choice(np.argwhere(q_vals == q_max).flatten())
               
                # Take the chosen action
                reward = self.take_action(action)

                # Save the reward for plotting later
                rewards[i] += (1/(cur_run + 1)) * (reward - rewards[i])

                # Update the q(a) value (action-value estimate)
                a_counts[action] += 1
                q_vals[action] +=  (1/a_counts[action]) * (reward - q_vals[action])

        # Return the averaged rewards
        return rewards
    

def main(args):
    bandits = Bandits()
    r = []
    for e in args.epsilon:
        print("Running epsilon", e)
        rewards = bandits.run(e, args.steps, args.num_runs)2
        plt.plot(rewards)

    plt.ylabel("Average reward")
    plt.xlabel("Steps")
    plt.legend([str(e) for e in args.epsilon])    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, nargs="*", default=[0, 0.01, 0.1],
                        help="Epsilon value to use for epsilon greedy action selection.")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Amount of steps to take per bandits run.")
    parser.add_argument("--num_runs", type=int, default=2000,
                        help="Amount of times to run the bandits.")
    main(parser.parse_args())
