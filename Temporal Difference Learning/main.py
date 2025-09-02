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

for k in range(50000):
    '''
    50,000번의 에피소드를 반복하는 루프
    --> 각 에피소드는 에이전트가 시작 지점에서 목표 지점까지 이동하는 한 번의 전체 탐험
    '''
    done    = False # 에피소드의 종료 여부를 boolean 처리 (도착 지점 도착 시 True)

    while not done:
        x, y = w.get_state()
        # 메서드 호출 (agent.py)
        action = ag.select_action()
        # 메서드 호출 (grid.py)
        (prev_x, prev_y), reward, done = w.step(action)
        # history.append((x, y, reward))
        prev_x, prev_y = w.get_state()

        data[x][y] = data[x][y] + alpha * (reward + gamma*data[prev_x][prev_y] - data[x][y])

    w.reset()

    # 가독성을 위해 일정 주기마다 결과 출력
    if (k + 1) % 5000 == 0:
        print(f"Iteration: {k + 1}")
        for row in data:
            print(row)
        print("-" * 20)