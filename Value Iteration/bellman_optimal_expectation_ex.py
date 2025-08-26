import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

# 그리드 맵 크기 설정
rows = 4
cols = 4

# 환경 변수
reward = -1  # 한 칸 이동 시 보상
gamma = 1.0  # 감쇠율
theta = 1e-4 # 수렴을 위한 임계값

# 초기 상태 가치 함수
value_table = np.zeros((rows, cols), dtype=float)
frames = [] # 애니메이션 프레임을 저장할 리스트

# ----------------- 가치 반복 (벨만 최적 방정식) -----------------
while True:
    delta = 0
    temp_value_table = value_table.copy()

    # 현재 상태 테이블을 애니메이션 프레임으로 저장
    frames.append(temp_value_table.copy())
    
    for i in range(rows):
        for j in range(cols):
            if i == rows - 1 and j == cols - 1:
                continue

            v_old = value_table[i, j]
            
            # 벨만 최적 방정식: 모든 행동 중 최대 가치 찾기
            max_value_for_state = -np.inf
            
            actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            
            for dr, dc in actions:
                next_i, next_j = i + dr, j + dc
                
                if 0 <= next_i < rows and 0 <= next_j < cols:
                    v_prime = value_table[next_i, next_j]
                else:
                    v_prime = value_table[i, j]
                
                # Q-가치 계산 (상태에서 특정 행동을 취했을 때의 가치)
                q_value = reward + gamma * v_prime
                
                if q_value > max_value_for_state:
                    max_value_for_state = q_value

            # 상태 가치를 최대 Q-가치로 업데이트
            new_value = max_value_for_state
            temp_value_table[i, j] = new_value

            delta = max(delta, abs(v_old - new_value))

    value_table = temp_value_table
    if delta < theta:
        frames.append(value_table.copy()) # 최종 수렴된 프레임 추가
        break

# ----------------- 애니메이션 시각화 -----------------
fig, ax = plt.subplots(figsize=(8, 8))
current_frame_number = [0] # 프레임 번호를 저장할 리스트

def update(frame):
    ax.clear()
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_title(f'Bellman Optimality Equation-Finding optimal policy [Iteration : {current_frame_number[0]}]', fontsize=16)

    current_table = frames[frame]
    
    for i in range(rows):
        for j in range(cols):
            x = j + 0.5
            y = (rows - 1 - i) + 0.5
            
            facecolor = 'white'
            if i == 0 and j == 0:
                facecolor = 'green'
            elif i == rows - 1 and j == cols - 1:
                facecolor = 'red'
            
            ax.add_patch(plt.Rectangle((j, rows - 1 - i), 1, 1, facecolor=facecolor, alpha=0.5))

            text_color = 'white' if facecolor in ['green', 'red'] else 'black'
            value_text = f'{current_table[i, j]:.2f}'
            ax.text(x, y, value_text, ha='center', va='center', fontsize=14, color=text_color, fontweight='bold')

    ax.set_xticks(np.arange(0, cols + 1, 1))
    ax.set_yticks(np.arange(0, rows + 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    
    start_patch = mpatches.Patch(color='green', alpha=0.5, label='Departure (1, 1)')
    arrival_patch = mpatches.Patch(color='red', alpha=0.5, label='Arrival (4, 4)')
    ax.legend(handles=[start_patch, arrival_patch], loc='lower right', fontsize=12)
    
    current_frame_number[0] += 1
    
    return ax.patches + ax.texts

ani = FuncAnimation(fig, update, frames=len(frames), interval=500, repeat=False)
plt.show()