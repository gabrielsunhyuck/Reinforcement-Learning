import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

'''
정책 평가 :: 본 정책에서 "반복적 정책 평가" 방법론을 통해 각 그리드의 상태 가치를 계산하는 것
정책 평가를 통해 각 상태의 밸류를 계산한다면, 정책 개선 단계로 넘어가게 된다.

정책 평가를 통해 각 그리드의 상태 가치를 판별하고,
최선의 정책을 사용할 수 있도록 그리디 정책을 활용하는 예시이다.
'''

# 그리드 맵 크기 설정
rows = 4
cols = 4

# 환경 변수 설정
reward = -1  # 한 칸 이동 시 보상
prob = 0.25  # 동,서,남,북 각 방향으로 이동할 확률
gamma = 1.0  # 감쇠율
theta = 1e-4 # 수렴을 위한 임계값

# 초기 상태 가치 함수
value_table = np.zeros((rows, cols), dtype=float)

# 상태 가치가 수렴할 때까지 반복
while True:
    delta = 0
    temp_value_table = value_table.copy()

    for i in range(rows):
        for j in range(cols):
            if i == rows - 1 and j == cols - 1:
                continue

            v_old = value_table[i, j]
            expected_value = 0
            
            actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            
            for dr, dc in actions:
                next_i, next_j = i + dr, j + dc
                
                if 0 <= next_i < rows and 0 <= next_j < cols:
                    v_prime = value_table[next_i, next_j]
                else:
                    v_prime = value_table[i, j]

                expected_value += (reward + gamma * v_prime)

            new_value = prob * expected_value
            temp_value_table[i, j] = new_value

            delta = max(delta, abs(v_old - new_value))

    value_table = temp_value_table
    if delta < theta:
        break

# ----------------- 상태 가치 개선 정책 추출 -----------------
policy_table = {}  # 각 칸의 최적 행동들을 저장할 딕셔너리

for i in range(rows):
    for j in range(cols):
        # 도착점에서는 이동하지 않음
        if i == rows - 1 and j == cols - 1:
            continue

        best_actions = []
        current_value = value_table[i, j]
        
        actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dr, dc in actions:
            next_i, next_j = i + dr, j + dc
            
            # 그리드 경계 밖으로 이동하는 경우 제외
            if 0 <= next_i < rows and 0 <= next_j < cols:
                neighbor_value = value_table[next_i, next_j]
                
                # 이웃 칸의 가치가 현재 칸의 가치보다 높으면 최적 경로에 추가
                if neighbor_value > current_value:
                    best_actions.append((dr, dc))
        
        if best_actions:
            policy_table[(i, j)] = best_actions

# ----------------- 최종 결과 시각화 및 최적 경로 표시 -----------------
fig, ax = plt.subplots(figsize=(8, 8))

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

        value_text = f'{value_table[i, j]:.2f}'
        ax.text(x, y + 0.2, value_text, ha='center', va='center', fontsize=12, color=color, fontweight='bold')
        
        if (i, j) in policy_table:
            best_actions = policy_table[(i, j)]
            for action in best_actions:
                dr, dc = action
                
                arrow_offsets = {
                    (0, 1): (0.1, -0.15),
                    (0, -1): (-0.1, -0.15),
                    (1, 0): (0, -0.15),
                    (-1, 0): (0, -0.15)
                }
                
                arrow_map = {
                    (0, 1): '→',
                    (0, -1): '←',
                    (1, 0): '↓',
                    (-1, 0): '↑'
                }
                
                offset_x, offset_y = arrow_offsets[action]
                arrow_char = arrow_map[action]
                ax.text(x + offset_x, y + offset_y, arrow_char, ha='center', va='center', fontsize=20, color='blue', fontweight='bold')

# 축 설정 및 레이블 제거
ax.set_xticks(np.arange(0, cols + 1, 1))
ax.set_yticks(np.arange(0, rows + 1, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlim(0, cols)
ax.set_ylim(0, rows)

ax.set_title('Best policy [based on Greedy policy]', fontsize=16)

start_patch = mpatches.Patch(color='green', alpha=0.5, label='Departure (1, 1)')
arrival_patch = mpatches.Patch(color='red', alpha=0.5, label='Arrival (4, 4)')
ax.legend(handles=[start_patch, arrival_patch], loc='lower right', fontsize=12)

plt.show()