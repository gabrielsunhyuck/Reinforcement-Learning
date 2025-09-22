import numpy as np
import random as rd

class QAgent():
    def __init__(self):
        self.q_table = np.zeros((5, 7, 4))
        self.epsilon = 0.9
        self.alpha   = 0.1

    def select_action(self, s):
        x, y = s
        coin = rd.random()

        if coin < self.epsilon:
            action = rd.randint(0,3)
        else:
            action_val = self.q_table[x,y,:]
            action     = np.argmax(action_val)
        return action
    
    def update_table(self, transition):
        s, a, r, s_prime = transition
        x, y             = s
        next_x, next_y   = s_prime
        a_prime          = self.select_action(s_prime) # 한 차례 미래 상태에서 선택할 액션

        # SARSA 업데이트 식
        self.q_table[x,y,a] = self.q_table[x,y,a] + self.alpha * (r + np.amax(self.q_table[next_x, next_y, :]) - self.q_table[x, y, a])

    def anneal_epsilon(self):
        self.epsilon -= 0.01
        self.epsilon  = max(self.epsilon, 0.2)

    def show_table(self):
        q_lst = self.q_table.tolist()
        data  = np.zeros((5, 7))

        for row_idx in range(len(q_lst)):
            row = q_lst[row_idx]
            for col_idx in range(len(row)):
                col                    = row[col_idx]
                action                 = np.argmax(col)
                data[row_idx, col_idx] = action

        print(data)
