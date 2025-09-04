import random as rd
import numpy as np

class QAgent():
    def __init__(self):
        self.q_table = np.zeros((5, 7, 4))
        self.epsilon = 0.9
        self.alpha   = 0.01
        self.gamma   = 0.001

    def select_action(self, s):
        x, y = s
        coin = rd.random()

        if coin < self.epsilon:
            action = rd.randint(0,3)
        else:
            action_val = self.q_table[x,y,:]
            action     = np.argmax(action_val)
        return action
    
    def update_table(self, history):
        cum_reward = 0
        for transition in history[::-1]:
            s, a, r, s_prime = transition
            x, y             = s

            self.q_table[x, y, a] = self.q_table[x, y, a] + self.alpha*(cum_reward - self.q_table[x, y, a])
            cum_reward            = self.gamma*cum_reward + r

    def anneal_epsilon(self):
        self.epsilon -= 0.03
        self.epsilon  = max(self.epsilon, 0.1)

    def show_table(self):
        q_lst = self.q_table.tolist()
        data  = np.zeros((5, 7))
        for row_idx in range(len(q_lst)):
            row = q_lst[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action
                
        print(data)