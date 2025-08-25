import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 그리드 맵 크기 설정
rows = 4
cols = 4

# 환경 변수 설정 (정책 평가)
reward = -1  # 한 칸 이동 시 보상
gamma  = 1.0  # 감쇠율
theta  = 1e-4 # 수렴을 위한 임계값


'''
최초 정책 (무작위 그리드 이동)
무작위 이동을 기반으로 한 정책을 따를 떄의 상태 가치 계산
'''
value_table_random = np.zeros((rows, cols), dtype=float)
prob_random = 0.25

while True:
    delta            = 0 # 상태 가치 수렴성 파악을 위한 초기값 (이전 상태 가치 - 현재 상태 가치)
    temp_value_table = value_table_random.copy() # 상태 가치 초기화 (모든 그리드의 상태 가치 0.0으로 세팅)
    for i in range(rows):
        for j in range(cols):
            if i == rows - 1 and j == cols - 1:
                continue # 4행 4열은 상태 가치 0으로 고정하기 때문에 패스
            
            # 이전 상태 가치 저장
            v_old          = value_table_random[i, j]
            # 상태 가치 업데이트
            expected_value = 0
            # 그리드 이동 방향 정의
            actions        = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            
            for dr, dc in actions:
                # 이동 방향에 따른 next 그리드 정의 (그리드 경계 내)
                next_i, next_j = i + dr, j + dc
                if 0 <= next_i < rows and 0 <= next_j < cols:
                    v_prime = value_table_random[next_i, next_j]
                else:
                    v_prime = value_table_random[i, j]
                '''벨만 방정식
                벨만 기대 방정식을 통해 상태 가치(new_value) 업데이트
                (1) reward는 한 칸 이동마다 -1씩 쌓임
                (2) v_prime은 다음 상태로의 전이 확률 (100%) * 다음 상태에서의 상태 가치
                (3) 최종 상태 가치(new_value)는 expected_value에 현재 상태에서 액션을 취할 확률을 곱해준다.
                '''
                expected_value     += (reward + gamma * v_prime)
            new_value              = prob_random * expected_value
            temp_value_table[i, j] = new_value
            delta                  = max(delta, abs(v_old - new_value))
    value_table_random = temp_value_table # 최초 정책을 통해 계산되어지는 상태 가치 (반복 정책 평가법 적용)
    if delta < theta: # 수렴성 파악
        break

'''
무작위 이동 정책을 기반으로 계산한 상태 가치를 고려하여 새로운 정책 생성
반복 정책 평가법을 --> 그리디 정책을 통한 최적의 행동 추출
'''
policy_table = {}  # (i, j) -> [ (dr, dc), ... ]
for i in range(rows):
    for j in range(cols):
        if i == rows - 1 and j == cols - 1:
            continue
        
        # 각 그리드에서 최적의 행동 저장
        best_actions  = []
        # 현재 상태 가치 저장
        current_value = value_table_random[i, j]
        # 가능한 행동 정의
        actions       = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in actions:
            next_i, next_j = i + dr, j + dc
            if 0 <= next_i < rows and 0 <= next_j < cols:
                # 행동을 통해 이동한 그리드의 상태 가치 계산
                neighbor_value = value_table_random[next_i, next_j]
                if neighbor_value > current_value:
                    # 이웃하는 그리드의 상태 가치 > 현재 그리드 상태 가치
                    # --> 최적의 경로로 판단
                    best_actions.append((dr, dc))
        if best_actions:
            '''policy_table은 최적의 행동을 저장한 딕셔너리 구조 [(i, j), [actions]]'''
            policy_table[(i, j)] = best_actions

'''
새로운 정책을 통한 상태 가치 계산 (반복 정책 계산법)
'''
value_table_new = np.zeros((rows, cols), dtype=float)

while True:
    delta = 0
    temp_value_table_new = value_table_new.copy()
    for i in range(rows):
        for j in range(cols):
            if i == rows - 1 and j == cols - 1:
                continue

            v_old          = value_table_new[i, j]
            expected_value = 0
            
            actions_to_consider = policy_table.get((i, j), [])
            # 각 그리드에서 가능한 최적의 경로 개수를 세아림
            # --> 이를 기반으로 행동 확률 계산 pi(a|s)
            num_actions = len(actions_to_consider)
            
            if num_actions == 0:
                # 최적의 경로가 없는 경우
                new_value = reward + gamma * value_table_new[i,j]
            else:
                # 최적의 경로가 있는 경우
                # 1/그리드에서 가능한 최적의 행동 개수 (최적 이동 방향 개수)은 상태 s에서 a 행동을 할 확률
                prob_new = 1.0 / num_actions
                
                for dr, dc in actions_to_consider:
                    next_i, next_j = i + dr, j + dc
                    
                    # 다음 상태에서의 가치
                    v_prime         = value_table_new[next_i, next_j]
                    expected_value += (reward + gamma * v_prime)
                
                # 현재 상태에서 a 행동을 할 확률 고려
                new_value = prob_new * expected_value

            temp_value_table_new[i, j] = new_value
            delta = max(delta, abs(v_old - new_value))

    value_table_new = temp_value_table_new
    if delta < theta:
        break

fig, ax = plt.subplots(figsize=(8, 8))#
ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

for i in range(rows):
    for j in range(cols):
        x = j + 0.5
        y = (rows - 1 - i) + 0.5
        
        color = 'black'
        facecolor = 'white'
        if i == 0 and j == 0:
            facecolor = 'green'
            color = 'white'
        elif i == rows - 1 and j == cols - 1:
            facecolor = 'red'
            color = 'white'
        
        ax.add_patch(plt.Rectangle((j, rows - 1 - i), 1, 1, facecolor=facecolor, alpha=0.5))
        
        value_text = f'{value_table_new[i, j]:.2f}'
        ax.text(x, y, value_text, ha='center', va='center', fontsize=14, color=color, fontweight='bold')

ax.set_xticks(np.arange(0, cols + 1, 1))
ax.set_yticks(np.arange(0, rows + 1, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlim(0, cols)
ax.set_ylim(0, rows)

ax.set_title('Policy Improvement [Greedy policy] & Policy Evaluation', fontsize=16)

start_patch = mpatches.Patch(color='green', alpha=0.5, label='Departure (1, 1)')
arrival_patch = mpatches.Patch(color='red', alpha=0.5, label='Arrival (4, 4)')
ax.legend(handles=[start_patch, arrival_patch], loc='lower right', fontsize=12)

plt.show()