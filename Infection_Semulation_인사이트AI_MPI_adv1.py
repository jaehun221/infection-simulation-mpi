import numpy as np
import time
from mpi4py import MPI

# ===== MPI 초기화 =====
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ===== 시뮬레이션 파라미터 =====
N = 100
T = 20
infection_prob = 0.3
recovery_time = 5
death_prob = 0.1
num_initial_infected = 20

# ===== 불균등 행 분할 계산 =====
rows_per_proc = [N // size + (1 if i < N % size else 0) for i in range(size)]
start_rows = [sum(rows_per_proc[:i]) for i in range(size)]
start_row = start_rows[rank]
end_row = start_row + rows_per_proc[rank]

# ===== 배열 초기화 =====
grid = np.zeros((rows_per_proc[rank] + 2, N), dtype=np.int32)  # 상태 격자 (halo 포함)
infection_duration = np.zeros_like(grid, dtype=np.int32)

# ===== 도시 지도 생성 =====
if rank == 0:
    city_map = np.zeros((N, N), dtype=np.int32)
    city_map[20:25, :] = 1
    city_map[:, 45:48] = 1
    for i in range(10, 80):
        city_map[i, int(20 + 10 * np.sin(i / 10))] = 1
    city_map[60:68, 10:18] = 2
    city_map[30:40, 70:80] = 2
    city_map[50:52, 50:70] = 2
    np.save("city_map.npy", city_map)
else:
    city_map = None

city_map = comm.bcast(city_map, root=0)

# ===== 도로 위 초기 감염자 배치 =====
def get_random_road_positions(city_map, count):
    road_positions = np.argwhere(city_map == 1)
    indices = np.random.choice(len(road_positions), count, replace=False)
    return [tuple(pos) for pos in road_positions[indices]]

if rank == 0:
    init_coords = get_random_road_positions(city_map, num_initial_infected)
else:
    init_coords = None

init_coords = comm.bcast(init_coords, root=0)

for i_global, j in init_coords:
    if start_row <= i_global < end_row:
        local_i = i_global - start_row + 1
        grid[local_i, j] = 1

# ===== 경계 교환 함수 =====
def exchange_borders(grid):
    top = grid[1, :].copy()
    bottom = grid[-2, :].copy()
    if rank > 0:
        comm.Sendrecv(top, dest=rank-1, sendtag=11,
                      recvbuf=grid[0, :], source=rank-1, recvtag=22)
    if rank < size - 1:
        comm.Sendrecv(bottom, dest=rank+1, sendtag=22,
                      recvbuf=grid[-1, :], source=rank+1, recvtag=11)

# ===== 시뮬레이션 단계 =====
def step(grid, infection_duration, city_map):
    new = grid.copy()
    duration = infection_duration.copy()

    for i in range(1, rows_per_proc[rank] + 1):
        for j in range(N):
            gi = start_row + i - 1
            if grid[i, j] == 0:
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + dx, j + dy
                    ngi = gi + dx
                    if 0 <= nj < N and 0 <= ni < rows_per_proc[rank] + 2 and 0 <= ngi < N:
                        if grid[ni, nj] == 1 and city_map[ngi, nj] == 1:
                            if np.random.rand() < infection_prob:
                                new[i, j] = 1
                                duration[i, j] = 1
                                break
            elif grid[i, j] == 1:
                duration[i, j] += 1
                if duration[i, j] >= recovery_time:
                    new[i, j] = 3 if np.random.rand() < death_prob else 2
                    duration[i, j] = 0

    return new, duration

# ===== 시뮬레이션 실행 =====
for t in range(T):
    exchange_borders(grid)
    grid, infection_duration = step(grid, infection_duration, city_map)

    final_grid = None
    if rank == 0:
        final_grid = np.zeros((N, N), dtype=np.int32)

    # GatherV 준비
    recvcounts = [r * N for r in rows_per_proc]
    displs = [sum(recvcounts[:i]) for i in range(size)]

    comm.Gatherv(grid[1:-1, :].flatten(), (final_grid, recvcounts, displs, MPI.INT), root=0)

    if rank == 0:
        np.save(f"frame_{t:02d}_road.npy", final_grid.copy())

# ===== 감염자/사망자 수 집계 =====
local_infected = np.sum(grid[1:-1, :] == 1)
local_dead = np.sum(grid[1:-1, :] == 3)

total_infected = comm.reduce(local_infected, op=MPI.SUM, root=0)
total_dead = comm.reduce(local_dead, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Total infected: {total_infected}")
    print(f"Total dead: {total_dead}")
    np.save("final_infection_map.npy", final_grid)

# ===== 성능 측정 =====
start_time = time.time()
infection_duration = np.zeros_like(grid)

for t in range(T):
    exchange_borders(grid)
    grid, infection_duration = step(grid, infection_duration, city_map)

end_time = time.time()
elapsed = end_time - start_time
print(f"[Rank {rank}] 처리 시간: {elapsed:.4f}초")

total_time = comm.reduce(elapsed, op=MPI.MAX, root=0)
if rank == 0:
    print(f"총 처리 시간 (최대 기준): {total_time:.4f}초")
