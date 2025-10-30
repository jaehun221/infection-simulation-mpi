import numpy as np
import time

# [A] 파라미터 설정
N = 100                    # 도시 크기 (NxN)
T = 20                     # 시뮬레이션 반복 횟수 (시간)
infection_prob = 0.3       # 감염 확률 (0~1 사이)
recovery_time = 5          # 회복까지 걸리는 시간
death_prob = 0.1           # 사망 확률
num_initial_infected = 20  # 초기 감염자 수

# [B] 배열 초기화
grid = np.zeros((N, N), dtype=np.int32)               # 상태 격자
infection_duration = np.zeros((N, N), dtype=np.int32) # 감염 지속 시간 추적

# [C] 도시 지도 생성
city_map = np.zeros((N, N), dtype=np.int32)  # 0: 일반, 1: 도로, 2: 건물

# 수평 도로
city_map[20:25, :] = 1

# 수직 도로
city_map[:, 45:48] = 1

# 곡선 도로
for i in range(N):
    j = int(N/2 + (N/4) * np.sin(i / (N/15)))
    if 0 <= j < N:
        city_map[i, j] = 1
        if j + 1 < N:
            city_map[i, j + 1] = 1

# 건물
city_map[10:20, 10:20] = 2
city_map[60:75, 60:75] = 2

# 도시 지도 저장
np.save("city_map.npy", city_map)

# [D] 초기 감염자 무작위 배치
init_coords = [(np.random.randint(0, N), np.random.randint(0, N)) for _ in range(num_initial_infected)]
for i, j in init_coords:
    if city_map[i, j] != 2:  # 건물이 아니면 배치
        grid[i, j] = 1

# [E] 시뮬레이션 단계 함수
def step(grid, infection_duration, city_map):
    new = grid.copy()
    duration = infection_duration.copy()

    for i in range(N):
        for j in range(N):
            if grid[i, j] == 0:  # 건강한 사람
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < N and 0 <= nj < N:
                        if grid[ni, nj] == 1 and city_map[i, j] == 1:
                            if np.random.rand() < infection_prob:
                                new[i, j] = 1
                                duration[i, j] = 1
                                break
            elif grid[i, j] == 1:  # 감염자
                duration[i, j] += 1
                if duration[i, j] >= recovery_time:
                    new[i, j] = 3 if np.random.rand() < death_prob else 2
                    duration[i, j] = 0

    return new, duration

# [F] 시뮬레이션 실행
start_time = time.time()

for t in range(T):
    grid, infection_duration = step(grid, infection_duration, city_map)
    np.save(f"frame_{t:02d}_road.npy", grid.copy())

# [G] 감염/사망 집계
total_infected = np.sum(grid == 1)
total_dead = np.sum(grid == 3)
print(f"Total infected: {total_infected}")
print(f"Total dead: {total_dead}")
np.save("final_infection_map.npy", grid)

# [H] 성능 측정 루프 (심화)
start_time = time.time()
infection_duration = np.zeros((N, N), dtype=np.int32)

for t in range(T):
    grid, infection_duration = step(grid, infection_duration, city_map)

end_time = time.time()
print(f"총 처리 시간: {end_time - start_time:.4f}초")