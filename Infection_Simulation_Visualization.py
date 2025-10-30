import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

# 기본 설정
N = 100
T = 20

# 감염 상태 컬러맵: 0=미감염, 1=감염, 2=회복, 3=사망
state_cmap = ListedColormap(['white', 'red', 'green', 'black'])
state_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], state_cmap.N)

# 도시 배경 컬러맵: 0=일반, 1=도로, 2=건물
terrain_cmap = ListedColormap(['yellow', 'blue', 'red'])

# 도시 마스크 로드
city_map = np.load("city_map.npy")

# 변화 기록용 리스트
time_vals = []
infected_vals = []
recovered_vals = []
dead_vals = []

# 시각화 레이아웃
fig, (ax_grid, ax_graph) = plt.subplots(1, 2, figsize=(13, 6))

# 그래프 설정
ax_graph.set_xlim(0, T)
ax_graph.set_ylim(0, N * N * 0.3)
ax_graph.set_xlabel("Time")
ax_graph.set_ylabel("Persons")
ax_graph.set_title("Trends of infection statue transition")
ax_graph.grid(True)
# ax_graph.legend(loc='upper right')  # 레전드 나중에 정의함

# 도시 마스크 범례
terrain_legend = [
    mpatches.Patch(color='yellow', label='general'),
    mpatches.Patch(color='blue', label='road'),
    mpatches.Patch(color='red', label='building'),
]

# 애니메이션 업데이트 함수
def update(t):
    ax_grid.clear()
    ax_graph.clear()

    data = np.load(f"frame_{t:02d}_road.npy")

    # 격자 시각화
    ax_grid.imshow(city_map, cmap=terrain_cmap, alpha=0.5, origin='upper')
    ax_grid.imshow(data, cmap=state_cmap, norm=state_norm, alpha=0.8, origin='upper')
    ax_grid.set_title(f"Time Step {t}")
    ax_grid.set_xlim(0, N)
    ax_grid.set_ylim(N, 0)  # y축 위쪽부터 시작
    ax_grid.set_aspect('equal')
    ax_grid.axis('off')
    ax_grid.legend(handles=terrain_legend, loc='upper right', fontsize='small')

    # --- 상태 변화 그래프 ---
    infected = np.sum(data == 1)
    recovered = np.sum(data == 2)
    dead = np.sum(data == 3)

    time_vals.append(t)
    infected_vals.append(infected)
    recovered_vals.append(recovered)
    dead_vals.append(dead)

    ax_graph.plot(time_vals, infected_vals, color='red', label='Infectors')
    ax_graph.plot(time_vals, recovered_vals, color='green', label='Recovers')
    ax_graph.plot(time_vals, dead_vals, color='black', label='Deads')

    ax_graph.set_xlim(0, T)
    ax_graph.set_ylim(0, max(1, max(infected_vals + recovered_vals + dead_vals)) * 1.1)
    ax_graph.set_xlabel("Time")
    ax_graph.set_ylabel("Persons")
    ax_graph.set_title("Trends of infection statue transition")
    ax_graph.grid(True)
    ax_graph.legend(loc='upper right')

    return []

# 애니메이션 생성
anim = animation.FuncAnimation(fig, update, frames=T, interval=500, blit=False)

# --- 저장 옵션 ---
anim.save("infection_visual6.mp4", writer="ffmpeg", dpi=150)
# anim.save("infection_visual.gif", writer="pillow", fps=2)

plt.tight_layout()
plt.show()
