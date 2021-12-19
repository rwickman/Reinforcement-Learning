"""Dyna-Q Testbed.

Uses the algorithm explained in section 8.2 in Reinforcement Learning: An Introduction.
Tests on example 8.1.
"""

import numpy as np
from matplotlib import pyplot as plt
import argparse
import random
from tqdm import tqdm

NUM_ACTIONS = 4

class Maze:
    def __init__(self):
        self.maze_bounds = (6, 9)
        self.goal_pos = (0, 8)
        self.start_pos = (2, 0)
        self.walls = set([
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 5),
            (0, 7),
            (1, 7),
            (2, 7),
        ])
        self.cur_pos = list(self.start_pos)
    
    def reset(self):
        self.cur_pos = list(self.start_pos)

    def move(self, move_dir):
        # Move Up
        updated_pos = self.cur_pos.copy()
        if move_dir == 0:
            updated_pos[0] -= 1
        elif move_dir == 1:
            # Move right
            updated_pos[1] += 1
        elif move_dir == 2:
            # Move down
            updated_pos[0] += 1
        else:
            # Move left
            updated_pos[1] -= 1
        
        # Only update position if it is withing bounds and no blocking walls
        if self.can_move(updated_pos):
            self.cur_pos = updated_pos
        
        if tuple(self.cur_pos) == self.goal_pos:
            reward = 1
        else:
            reward = 0

        return self.cur_pos, reward
    
    def can_move(self, cur_pos):
        """Check if this a valid move position."""
        # Check if the position is within bounds of the maze
        is_in_maze = cur_pos[0] >= 0 and cur_pos[0] < self.maze_bounds[0] and cur_pos[1] >= 0 and cur_pos[1] < self.maze_bounds[1]

        # Also check if a wall is not blocking
        return is_in_maze and tuple(cur_pos) not in self.walls
        

class EnvModel:
    def __init__(self):
        self._model_dict = {}
    
    def update(self, state, action, reward, next_state):
        self._model_dict[(tuple(state), action)] = [reward, next_state]
    
    def sample_exp(self):
        pairs = list(self._model_dict.items())
        rand_idx = random.randint(0, len(pairs) - 1)
        state_action, reward_next_state = pairs[rand_idx]
        state, action = state_action[0], state_action[1]
        reward, next_state = reward_next_state[0], reward_next_state[1]
        return state, action, reward, next_state

    def __call__(self, state, action):
        return self._model_dict[(tuple(state), action)]


class DynaQ:
    def __init__(self, args):
        self.action_vals = np.zeros((6, 9, NUM_ACTIONS))
        # Model for getting next state and reward given current state and action
        self.model = EnvModel()
        self.epsilon = args.epsilon
        self.gamma = args.gamma
        self.step_size = args.step_size

    def update(self, state, action, reward, next_state, update_model=True):
        # Update the model
        if update_model:
            self.model.update(state, action, reward, next_state)

        # Update the q-values
        q_val_next = self.action_vals[next_state[0], next_state[1], self.get_action(next_state, argmax=True)]
        
        q_val = self.action_vals[state[0], state[1], action] 
        self.action_vals[state[0], state[1], action] = q_val + self.step_size * (reward + self.gamma * q_val_next - q_val)  


    def get_action(self, state, argmax = False):
        if not argmax and random.random() > self.epsilon:
            # Select the greedy action and break ties randomly
            q_vals = self.action_vals[state[0], state[1]]            
            q_max = np.max(q_vals)
            action = np.random.choice(np.argwhere(q_vals == q_max).flatten())

        else:
            action = random.randint(0, NUM_ACTIONS - 1)
        
        return action

    def sample_exp(self):
        return self.model.sample_exp()    
    
        
class Trainer:
    def __init__(self, n_planning_steps, args):
        self.maze = Maze()
        self.agent = DynaQ(args)
        self.n_planning_steps = n_planning_steps
        self.num_episodes = args.episodes

    def model_updates(self):
        """Update the q-values using the environment model."""
        for i in range(self.n_planning_steps): 
            self.agent.update(*self.agent.sample_exp(), update_model=False)

    def print_maze(self):
        maze_str = "The Maze"
        maze_str += "\n" + "--" * 11 + "\n"
        for i in range(-1, 7):
            if i > -1:
                maze_str += "\n"
            for j in range(-1, 11):
                can_move = self.maze.can_move([i, j])
                if tuple([i, j]) == self.maze.goal_pos:
                    maze_str += "G "
                elif tuple([i, j]) == self.maze.start_pos:
                    maze_str += "S "
                elif can_move:
                    maze_str += "o "
                else:
                    maze_str += "x "
        print(maze_str)

    def run(self):
        ep_steps = []
        for _ in range(self.num_episodes):
            reward = 0
            num_steps = 0
            while reward == 0:
                state = self.maze.cur_pos.copy()
                action = self.agent.get_action(state)
                
                # Perform the action in the maze
                next_state, reward = self.maze.move(action)

                # Update the agent
                self.agent.update(state, action, reward, next_state.copy())

                self.model_updates()
                num_steps += 1
            
            ep_steps.append(num_steps)
            self.maze.reset()
        
        return ep_steps



def main(args):
    # Train on environment for multiple runs
    trainer = Trainer(0, args)
    trainer.print_maze()
    ep_steps_list = [[] for _ in range(len(args.planning_steps))] 
    fig, ax = plt.subplots(1)
    for _ in tqdm(range(args.num_runs)):
        for i, planning_step in enumerate(args.planning_steps):
            trainer = Trainer(planning_step, args)
            ep_steps_list[i].append(trainer.run())

    # Plot average steps over multiple runs 
    for i, planning_step in enumerate(args.planning_steps): 
        cur_ep_steps = np.mean(ep_steps_list[i], axis=0)
        ax.plot(cur_ep_steps, label=f"{planning_step} Planning Steps")

    ax.set(ylabel="Steps per episode", xlabel="Episodes")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Epsilon value to use for epsilon greedy action selection.")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Reward discount.")
    parser.add_argument("--step_size", type=float, default=0.1,
                        help="Learning rate.")
    parser.add_argument("--num_runs", type=int, default=30,
                        help="Number of times to test each planning step.")
    parser.add_argument("--planning_steps", type=int, nargs="*", default=[0, 5, 50],
                        help="List of model training planning steps.")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of episodes to train each planning steps.")
    main(parser.parse_args())
