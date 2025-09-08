import random as rd
import numpy as np

class QAgent():
    def __init__(self):
        self.q_table = np.zeros((5, 7, 4))
        self.epsilon = 0.9   # 입실론 그리디 파라미터
        self.alpha   = 0.01  # Learning rate
        self.gamma   = 0.001 # 감쇠 비율 (현재 보상의 중요도)

    def select_action(self, s):
        x, y = s # 에이전트의 상태
        coin = rd.random()

        ''' [ 입실론-그리디 정책 ]
        (1) 무작위로 뽑은 coin이 epsilon의 값보다 작으면 에이전트는 탐험을 선택
        (2) 무작위로 뽑은 coin이 epsilon의 값보다 크면 에이전트는 max(상태-행동 가치) 선택
        '''
        if coin < self.epsilon:
            action = rd.randint(0,3)
        else:
            action_val = self.q_table[x,y,:]
            action     = np.argmax(action_val)
        return action
    
    def update_table(self, history):
        ''' [ 학습 단계 ]
        (1) 한 에피소드가 끝난 후, 기록된 history를 바탕으로 상태-행동 가치(Q-value) 업데이트
        (2) 목표 지점에서부터 거꾸로 보상을 전파 (for transition in history[::-1])
        '''
        cum_reward = 0 # 누적 보상 초기화
        for transition in history[::-1]:
            s, a, r, s_prime = transition
            x, y             = s

            # 업데이트 공식
            self.q_table[x, y, a] = self.q_table[x, y, a] + self.alpha*(cum_reward - self.q_table[x, y, a])
            cum_reward            = self.gamma*cum_reward + r

    def anneal_epsilon(self):
        ''' 에피소드가 끝날 때마다 epsilon의 값을 점진적으로(-0.03 씩) 줄이는 함수
        - epsilon의 값은 0.1 밑으로 떨어지지 않게 설정
        '''
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