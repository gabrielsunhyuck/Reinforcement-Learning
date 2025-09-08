import random as rd
import numpy as np
from grid import *
from agent import *

def main():
    env   = Grid()
    agent = QAgent()

    for n_epi in range(1000): # 1000번의 에피스드 반복 (n_epi : 에피소드 번호)
        done    = False # 도착 플래그
        history = []    # 에이전트의 상태, 행동, 보상, 다음 상태에 대한 정보 저장

        s       = env.reset() # 에피소드가 끝나고 초기 상태로 되돌림 (재시작)

        while not done: # 에피소드 한 바퀴
            # 에이전트가 행동을 선택 
            a = agent.select_action(s)
            # 행동에 따라 에이전트의 행동 실행
            s_prime, r, done = env.step(a)
            # 에이전트의 행동에 따른 상태, 행동, 보상, 다음 상태 저장
            history.append((s, a, r, s_prime))
            # 다음상태를 현재의 상태로 초기화
            s = s_prime

        agent.update_table(history)
        agent.anneal_epsilon()

    agent.show_table()