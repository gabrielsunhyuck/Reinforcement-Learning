from agent import *
from grid  import *

def main():
    for n_epi in range(1000): # 1000번의 에피스드 반복 (n_epi : 에피소드 번호)
        done = False # 도착 플래그
        s    = GridWorld.reset() # 에피소드가 끝나고 초기 상태로 되돌림 (재시작)
        
        while not done:  # 에피소드 한 바퀴
            # 에이전트가 행동을 선택 
            a = QAgent.select_action(s)
            # 행동에 따라 에이전트의 행동 실행
            s_prime, r, done = GridWorld.step(a)
            # 에이전트의 행동에 따른 상태, 행동, 보상, 다음 상태 저장
            QAgent.update_table((s, a, r, s_prime))
            # 다음상태를 현재의 상태로 초기화
            s = s_prime
        QAgent.anneal_epsilon() # epsilon 을 낮춤 (에이전트의 탐험률 저하)

    QAgent.show_table()