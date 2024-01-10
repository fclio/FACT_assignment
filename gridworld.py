# grid-world self collected dataset and use of lstm
# seaquest data from https://github.com/takuseno/d4rl-atari and pretrained transformer from https://huggingface.co/edbeeching/decision_transformer_atari
# halfcheetah data from https://github.com/Farama-Foundation/D4RL and pretrained transformer from https://github.com/jannerm/trajectory-transformer
# https://github.com/google-deepmind/mujoco



import random
import numpy as np

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.epsilon = epsilon
        
        self.q = np.zeros((n_states, self.n_actions))
        
    def select_action(self, state):   
        # If a random number is within the explore rate we explore by choosing a random action
        if np.random.uniform(0,1) <= self.epsilon:
            return np.random.choice(range(self.n_actions))
        
        # If we don't explore we choose the action with the highest reward 
        # If there are multiple actions with the same reward we choose the first one
        return np.argmax(self.q[state])

    def update(self, state, next_state, action, next_reward, alpha):
        df = 1 # Discount factor

        # Update q value
        self.q[state][action] += alpha*(next_reward+df*max(self.q[next_state])-self.q[state][action])
        
class Gridworld():
    def __init__(self, seed=None):
        self.rows = 7
        self.columns = 7
        self.s = np.array([["_", "_", "_", "_", "_", "_", "G"],
                           ["_", "W", "_", "W", "_", "_", "_"],
                           ["_", "_", "_", "_", "_", "_", "_"],
                           ["_", "W", "_", "W", "_", "_", "_"],
                           ["_", "_", "_", "L", "W", "_", "W"],
                           ["_", "_", "_", "_", "_", "_", "_"],
                           ["_", "_", "_", "_", "W", "_", "G"],
                        ])
        self.reset()
    
    def reset(self):
        self.x = random.randint(0, self.columns-1)
        self.y = random.randint(0, self.rows-1)
        while self.s[self.y][self.x] == "W":
            self.x = random.randint(0, self.columns-1)
            self.y = random.randint(0, self.rows-1)
        self.isdone = False
    
    def state(self):
        return self.y*self.columns + self.x
    
    def state_size(self):
        return self.columns*self.rows
    
    def action_size(self):
        return 4
    
    def done(self):
        return self.isdone
    
    def possible_actions(self):
        return [0, 1, 2, 3]
    
    def step(self, action):
        if self.isdone:
            raise ValueError('Environment has to be reset.')
        
        if not action in self.possible_actions():
            raise ValueError(f'Action ({action}) not in set of possible actions.')
        
        if action == 0:
            if self.y>0 and self.s[self.y-1, self.x] != 'W':
                self.y -= 1
        elif action == 1:
            if self.y<self.rows-1 and self.s[self.y+1, self.x] != 'W':
                self.y += 1
        elif action == 2:
            if self.x>0 and self.s[self.y, self.x-1] != 'W':
                self.x -= 1
        elif action == 3:
            if self.x<self.columns-1 and self.s[self.y, self.x+1] != 'W':
                self.x += 1
        
        if self.s[self.y, self.x]=='G': # Goal reached
            self.isdone = True
            return 1
        elif self.s[self.y, self.x] == 'L': # Lava
            self.isdone = True
            return -1
        return -0.1
    
    def render(self):
        s = self.s.copy()
        s[self.y, self.x] = 'p'
        print(s.tobytes().decode('utf-8'))


env = Gridworld()
agent = QLearningAgent
agent = agent(n_actions=env.action_size(),n_states=env.state_size(), epsilon=0.1)
for episode in range(1000):
    env.reset() # Reset the environment
    
    while not env.done():
        # Select and take a new action, observe reward
        current_state = env.state()
        action = agent.select_action(state=current_state)
        reward = env.step(action)
        next_state = env.state()
        
        # Update the Q-value
        agent.update(state=current_state, next_state=next_state, action=action, next_reward=reward, alpha=0.1)

q_values = agent.q
arrows = ["↑", "↓", "←", "→"]
for i in range(env.columns):
    for j in range(env.rows):
        if env.s[i][j] == "_":
            print(arrows[np.argmax(q_values[i*env.columns + j])], end=" ")
        else:
            print(env.s[i][j], end=" ")
    print()
