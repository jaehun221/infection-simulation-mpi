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

rows_per_proc = N // size
start_row = rank * rows_per_proc
end_row = start_row + rows_per_proc

grid = np.zeros((rows_per_proc + 2, N), dtype=np.int32)  # 상태 격자 (halo 포함)
infection_duration = np.zeros_like(grid, dtype=np.int32)

# 도시 지도 생성 (0: 일반, 1: 도로, 2: 건물)
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

# 초기 감염자 배치
if rank == 0:
    init_coords = [(np.random.randint(0, N), np.random.randint(0, N)) for _ in range(num_initial_infected)]
else:
    init_coords = None

init_coords = comm.bcast(init_coords, root=0)

for i_global, j in init_coords:
    if start_row <= i_global < end_row:
        local_i = i_global - start_row + 1
        if city_map[i_global, j] != 2:
            grid[local_i, j] = 1

# 경계 교환 함수
def exchange_borders(grid):
    top = grid[1, :].copy()
    bottom = grid[-2, :].copy()
    if rank > 0:
        comm.Sendrecv(top, dest=rank-1, sendtag=11,
                      recvbuf=grid[0, :], source=rank-1, recvtag=22)
    if rank < size - 1:
        comm.Sendrecv(bottom, dest=rank+1, sendtag=22,
                      recvbuf=grid[-1, :], source=rank+1, recvtag=11)

# 시뮬레이션 단계 함수
def step(grid, infection_duration, city_map):
    new = grid.copy()
    duration = infection_duration.copy()

    for i in range(1, rows_per_proc + 1):
        for j in range(N):
            gi = start_row + i - 1
            if grid[i, j] == 0:
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + dx, j + dy
                    ngi = gi + dx
                    if 0 <= nj < N and 0 <= ni < rows_per_proc + 2 and 0 <= ngi < N:
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
    # (1) 경계 교환
    exchange_borders(grid)
    
    # (2) 감염 상태 업데이트
    grid, infection_duration = step(grid, infection_duration, city_map)

    # (3) 결과 수집 (0번 프로세스만)
    final_grid = None
    if rank == 0:
        final_grid = np.zeros((N, N), dtype=np.int32)
    
    comm.Gather(grid[1:-1, :], final_grid, root=0)

    # (5) 저장
    if rank == 0:
        np.save(f"frame_{t:02d}_road.npy", final_grid.copy())
        np.save("city_map.npy", city_map)

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