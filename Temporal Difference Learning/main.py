from grid import GridWorld
from agent import Agent

w = GridWorld()
ag = Agent()

data  = [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]

gamma = 1.0
alpha = 0.01 # 몬테카를로 법칙에 비해 큰 값을 사용
'''
(1) gamma : 감쇠율 (Discount factor)
--> 미래 보상을 현재 가치에 얼마나 반영할지를 결정

(2) alpha : 학습률 (Learning rate)
--> 에이전트가 새로운 정보를 얼마나 빠르게 받아들여 가치 함수를 업데이트할지 결정
'''

for k in range(50000):
    '''
    50,000번의 에피소드를 반복하는 루프
    '''
    done    = False # 에피소드의 종료 여부를 boolean 처리 (도착 지점 도착 시 True)

    while not done:
        # 1. 현재 에이전트의 위치를 저장 [t]
        x, y = w.get_state()
        # 2. 에이전트가 무작위로 행동 선택 (agent.py)
        action = ag.select_action()
        # 3. 에이전트가 선택한 행동을 환경에 반영 [t+1] (grid.py)
        (x_prime, y_prime), reward, done = w.step(action)
        # 4. 에이전트가 행동을 한 후의 상태 [t+1]
        x_prime, y_prime = w.get_state()

        # 5. Temporal Difference 학습을 통한 상태 업데이트
        data[x][y] = data[x][y] + alpha * (reward + gamma*data[x_prime][y_prime] - data[x][y])

    w.reset()

    # 가독성을 위해 일정 주기마다 결과 출력
    if (k + 1) % 5000 == 0:
        print(f"Iteration: {k + 1}")
        for row in data:
            print(row)
        print("-" * 20)