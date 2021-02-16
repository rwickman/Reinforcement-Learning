"""
Value Iteration over Gambler Problem Ex. 4.9
"""

import seaborn as sns
import argparse
import numpy as np
from matplotlib import pyplot as plt


class ValueIteration:
    def __init__(self, args):
        # Gambler's capital
        self.states = np.arange(0, 101)
        self.gamma = args.gamma
        self.theta = args.theta
        #self.actions = np.arange(0, ) 

    def run(self, p_h):
        # State-value function
        V = np.zeros(self.states.shape[0])
        V[100] = 1
        policy = np.zeros(self.states.shape[0])
        
        # Iterate over all the states
        while True:
            delta = 0
            for s in self.states[1:-1]:
                # Find the max state-value, where actions are stakes
                max_v  = 0  
                for a in range(1, min(s, 100 - s)+1):
                    # If heads lose "a" amount of capital, else win "a" amount of captial
                    cur_v = p_h * (self.gamma  * V[s + a]) + (1-p_h) * (self.gamma  * V[s - a])
                    if cur_v > max_v:
                        policy[s] = a
                        max_v = cur_v
                    
                delta = max(delta, abs(V[s] - max_v))
                # Take max over actions
                V[s] = max_v
            print("delta", delta)
            if delta <= self.theta:
                break
        return V, policy



def main(args):
    value_iter = ValueIteration(args)
    V, policy = value_iter.run(args.p_h)
    
    # Plot value estimates
    sns.set(style="darkgrid", font_scale=1.5)
    print(V)
    sns.lineplot(x=np.arange(0, 101), y=V)
    plt.xlabel("Capital")
    plt.ylabel("Value esimates")
    plt.show()

    # Plot final policy
    print(policy)
    #sns.barplot(x=np.arange(0, 101), y=policy)
    sns.barplot(x=np.arange(0, 101), y=policy, color="black")
    plt.xlabel("Capital")
    plt.ylabel("Final policy (stake)")
    plt.xticks([1, 25, 50, 75, 99])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p_h", type=float, default=0.4,
                    help="Probablity of heads.")
    parser.add_argument("--gamma", type=float, default=1.0,
                    help="Gamme reward.")
    parser.add_argument("--theta", type=float, default=0.0,
                    help="Stopping threshold.")
    main(parser.parse_args())

