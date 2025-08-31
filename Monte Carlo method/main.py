from grid import GridWorld
from agent import Agent

w = GridWorld()
ag = Agent()

data  = [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]

gamma = 1.0
alpha = 0.001

for k in range(50000):
    '''
    50,000번의 에피소드를 반복하는 루프
    --> 각 에피소드는 에이전트가 시작 지점에서 목표 지점까지 이동하는 한 번의 전체 탐험
    '''
    done    = False # 에피소드의 종료 여부를 boolean 처리 (도착 지점 도착 시 True)
    history = []    # 하나의 에피소드동안 에이전트의 모든 행동과 결과를 저장
    
    # 각 에피소드 시작 시 환경 초기화
    x, y = w.reset()

    while not done:
        # 메서드 호출 (agent.py)
        action = ag.select_action()
        # 메서드 호출 (grid.py)
        (x, y), reward, done = w.step(action)
        history.append((x, y, reward))

    c_reward = 0 # 누적 보상 초기화 (하나의 에피소드)
    '''
    Monte Carlo 학습: history를 거꾸로 순회하며 가치 업데이트
    '''
    for transition in history[::-1]:
        x, y, reward = transition
        data[x][y] = data[x][y] + alpha * (c_reward - data[x][y])
        c_reward = reward + gamma * c_reward

    # 가독성을 위해 일정 주기마다 결과 출력
    if (k + 1) % 5000 == 0:
        print(f"Iteration: {k + 1}")
        for row in data:
            print(row)
        print("-" * 20)