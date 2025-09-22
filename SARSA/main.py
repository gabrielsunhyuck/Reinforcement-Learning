from agent import *
from grid  import *

def main():
    # 1. QAgent와 GridWorld의 인스턴스(객체)를 생성
    agent = QAgent()
    grid = GridWorld()

    for n_epi in range(1000):
        done = False
        # 2. grid 인스턴스를 사용하여 환경을 리셋
        s = grid.reset() 

        while not done:
            # 3. agent 인스턴스를 사용하여 메서드를 호출
            a = agent.select_action(s)
            s_prime, r, done = grid.step(a)

            # 4. 현재 스텝의 경험을 transition 튜플로 만듦
            transition = (s, a, r, s_prime)
            
            # 5. 매 스텝마다 Q-테이블을 바로 업데이트 (SARSA 방식)
            agent.update_table(transition)
            
            s = s_prime
        
        # 에피소드가 끝난 후 엡실론 감소
        agent.anneal_epsilon()

    # 6. 학습이 끝난 후 agent 인스턴스의 테이블을 출력
    agent.show_table()

main()