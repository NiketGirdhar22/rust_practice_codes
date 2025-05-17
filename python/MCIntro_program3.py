import sys
import time

def simple_rand(seed):
    seed[0] = (seed[0] * 1664525 + 1013904223) & 0xFFFFFFFFFFFFFFFF
    return ((seed[0] >> 16) & 0x7FFF) / 32768.0

def montecarlo(num_particles, max_time, decay_prob, seed):
    result = [0] * (max_time + 1)
    remaining = num_particles
    result[0] = remaining
    for t in range(1, max_time):
        decayed = 0
        for _ in range(remaining):
            if simple_rand(seed) <= decay_prob:
                decayed += 1
        remaining = max(remaining - decayed, 0)
        result[t] = remaining
        if remaining == 0:
            break
    return result

if len(sys.argv) != 6:
    print(f"Usage: {sys.argv[0]} outfilename num_particles_init max_time num_cycles decay_prob")
    sys.exit(1)

filename = sys.argv[1]
num_particles = int(sys.argv[2])
max_time = int(sys.argv[3])
num_cycles = int(sys.argv[4])
decay_prob = float(sys.argv[5])

total = [0] * (max_time + 1)
base_seed = int(time.time())

for cycle in range(num_cycles):
    seed = [base_seed + cycle]
    result = montecarlo(num_particles, max_time, decay_prob, seed)
    for i, val in enumerate(result):
        total[i] += val

with open(filename, "w") as f:
    for val in total:
        avg = val / num_cycles
        f.write(f"{avg:.6e}\n")

print(f"Simulation complete. Results written to {filename}")