import sys
import time

def simple_rand(seed):
    seed[0] = (seed[0] * 1664525 + 1013904223) & 0xFFFFFFFFFFFFFFFF
    return ((seed[0] >> 16) & 0x7FFF) / 32768.0

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} filename N")
    sys.exit(1)

filename = sys.argv[1]
n = int(sys.argv[2])
nl = n

seed = [int(time.time())]

try:
    with open(filename, 'w') as f:
        for t in range(10 * n):
            r = int(simple_rand(seed) * n)
            if r < nl:
                nl -= 1
            else:
                nl += 1
            f.write(f"{t} {nl}\n")
    print(f"Simulation complete. Data written to {filename}")
except Exception as e:
    print(f"Failed to write to file: {e}")
    sys.exit(1)