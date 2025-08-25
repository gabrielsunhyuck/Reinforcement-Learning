import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

'''
본 코드에서 정책은
1) 한 칸 이동 시 보상 -1
2) 동, 서, 남, 북 이동 확률 0.25
3) 상태 전이 확률 100%
4) 감쇠 인자 1.0

정책 평가 :: 본 정책에서 "반복적 정책 평가" 방법론을 통해 각 그리드의 상태 가치를 계산하는 것
정책 평가를 통해 각 상태의 밸류를 계산한다면, 정책 개선 단계로 넘어가게 된다.
'''

# 그리드 맵 크기 설정
rows = 4
cols = 4

# 환경 변수 설정
reward = -1   # 한 칸 이동 시 보상
prob   = 0.25 # 동,서,남,북 각 방향으로 이동할 확률
gamma  = 1.0  # 감쇠율 :: 미래의 보상이 당장 받는 보상에 비해 얼마나 더 중요한가
theta  = 1e-4 # 수렴을 위한 임계값

# 초기 상태 가치 함수 (모든 상태의 가치를 0.0으로 초기화)
value_table = np.zeros((rows, cols), dtype=float)

# 상태 가치가 수렴할 때까지 반복
while True:
    delta = 0 # 상태 가치 변화율 (수렴성 파악)
    temp_value_table = value_table.copy() # 상태 가치를 업데이트 하기 전에 이전 상태 가치들을 저장

    for i in range(rows):
        for j in range(cols):
            if i == rows - 1 and j == cols - 1:
                # 도착점 (4행 4열)은 상태 가치 0으로 고정이기 때문에 업데이트 건너뜀
                continue
            
            # 이전 상태 가치 저장
            v_old          = value_table[i, j] 
            # 상태 가치 업데이트 (벨만 기대 방정식)
            expected_value = 0
            # 동서남북 이동 경로 정의
            actions        = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            
            for dr, dc in actions:
                next_i, next_j = i + dr, j + dc
                
                # 이동 후 위치가 그리드 경계 내에 있는지 점검
                if 0 <= next_i < rows and 0 <= next_j < cols:
                    v_prime = value_table[next_i, next_j]
                else:
                    v_prime = value_table[i, j]

            
                expected_value += (reward + gamma * v_prime)
            '''벨만 기대 방정식----------------------------------------'''
            new_value              = prob * expected_value
            temp_value_table[i, j] = new_value
            '''------------------------------------------------------'''
            
            delta = max(delta, abs(v_old - new_value))

    # 이전 상태 가치와 현재 상태 가치의 차이가 theta보다 작은 경우 수렴한다고 판단
    value_table = temp_value_table
    if delta < theta:
        break

print(value_table)
# 맵 시각화 설정
fig, ax = plt.subplots(figsize=(8, 8))

# 주요 그리드 선 그리기
ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

# 각 셀에 상태 가치와 좌표 추가 (시각화)
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
        ax.text(x, y, value_text, ha='center', va='center', fontsize=14, color=color, fontweight='bold')

ax.set_xticks(np.arange(0, cols + 1, 1))
ax.set_yticks(np.arange(0, rows + 1, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlim(0, cols)
ax.set_ylim(0, rows)

ax.set_title('4x4 Final State Values', fontsize=16)

start_patch = mpatches.Patch(color='green', alpha=0.5, label='Departure (1, 1)')
arrival_patch = mpatches.Patch(color='red', alpha=0.5, label='Arrival (4, 4)')
ax.legend(handles=[start_patch, arrival_patch], loc='lower right', fontsize=12)

plt.show()