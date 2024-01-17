# grid-world self collected dataset and use of lstm
# seaquest data from https://github.com/takuseno/d4rl-atari and pretrained transformer from https://huggingface.co/edbeeching/decision_transformer_atari
# halfcheetah data from https://github.com/Farama-Foundation/D4RL and pretrained transformer from https://github.com/jannerm/trajectory-transformer
# https://github.com/google-deepmind/mujoco



import random
import numpy as np
              
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
        return self.state()
    
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
    

class DynaQAgent(object):
    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma # discount factor
        self.Q_sa = np.random.random((n_states, n_actions))

        self.n = np.zeros((n_states, n_actions, n_states))
        self.r = np.zeros((n_states, n_actions, n_states))
        
    def select_action(self, state, epsilon=0.1):
        # If a random number is within the explore rate we explore by choosing a random action
        if np.random.uniform(0,1) <= epsilon:
            return np.random.choice(range(self.n_actions))
        
        # If we don't explore we choose the action with the highest reward 
        # If there are multiple actions with the same reward we choose the first one
        return np.argmax(self.Q_sa[state])
        
    def update(self, state, action, reward, done, next_state, n_planning_updates):
        if done:
            return
        
        self.n[state][action][next_state] += 1 
        self.r[state][action][next_state] += reward
        
        self.Q_sa[state][action] += self.learning_rate * (reward + self.gamma * max(self.Q_sa[next_state])-self.Q_sa[state][action])

        for _ in range(n_planning_updates):
            # Choose random state with n>0
            state_sums = np.sum(self.n, axis=(1,2))
            state = np.random.choice(np.arange(state_sums.size)[state_sums>0])

            # Choose previously taken action for state s  
            action_sums = np.sum(self.n[state], axis=1)
            action = np.random.choice(np.arange(action_sums.size)[action_sums>0])
            
            # Choose next state
            p_hat = self.n[state][action]/np.sum(self.n[state][action])
            next_state = np.random.choice(np.arange(self.n_states), p=p_hat)

            reward = self.r[state][action][next_state]/self.n[state][action][next_state]
            
            # Update q value
            self.Q_sa[state][action] += self.learning_rate * (reward + self.gamma * max(self.Q_sa[next_state])-self.Q_sa[state][action])
    
    def train(self, env: Gridworld, episodes, n_planning_updates):
        s = env.reset()
        cumulative_r = 0
        for t in range(episodes):            
            # Select action, transition, update policy
            a = self.select_action(s)
            
            r = env.step(a)
            s_next = env.state()
            done = env.done()
            cumulative_r += r
            self.update(s, a, r, done, s_next,n_planning_updates=n_planning_updates)

            # Reset environment when terminated
            if done:
                s = env.reset()
            else:
                s = s_next
                
    def gen_traj(self, env: Gridworld, max_traj_len=15):
        traj = []
        traj_len = 0
        s = env.reset()
        while traj_len < max_traj_len:
            a = self.select_action(s)
            r = env.step(a)
            s_next = env.state()
            done = env.done()
            traj.append((s, a, r, s_next))
            if done:
                break
            s = s_next
        return traj, r

