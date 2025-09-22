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

        # 3. 에피소드 기록을 저장할 히스토리 리스트 초기화
        history = []
        while not done:
            # 4. agent 인스턴스를 사용하여 메서드를 호출
            a = agent.select_action(s)
            s_prime, r, done = grid.step(a)

            # 5. 현재 스텝의 경험을 히스토리에 추가
            history.append((s, a, r, s_prime))
            s = s_prime
        
        # 6. 에피소드가 끝난 후, 전체 히스토리를 이용해 Q-테이블 업데이트
        agent.update_table(history)
        agent.anneal_epsilon()

    # 7. 학습이 끝난 후 agent 인스턴스의 테이블을 출력합니다.
    agent.show_table()

main()