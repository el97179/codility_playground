import time


def solution(N):
    # Convert N to binary and strip the '0b' prefix
    binary = bin(N)[2:]
    max_gap = 0
    current_gap = 0
    in_gap = False
    for digit in binary:
        if digit == '1':
            if in_gap:
                max_gap = max(max_gap, current_gap)
            in_gap = True
            current_gap = 0
        elif in_gap:
            current_gap += 1
    return max_gap


def solution2(N):
    # Implement your solution here
    N_bin = bin(N)[2:]
    pos1 = N_bin.find("1")
    max_gap = 0
    while True:
        pos2 = N_bin[pos1+1:].find("1") + pos1 + 1
        if pos2 == pos1:  # No more '1's found
            break
        gap = pos2 - pos1 - 1
        if gap > max_gap:
            max_gap = gap
        pos1 = pos2

    return max_gap


test_cases = [137, 9, 529, 20, 15, 32, 1025634, 56230, 1234567890]

start = time.time()
for N in test_cases:
    print(f"Binary gap for {N} is {solution(N)}")  # Expected output: 3, 2, 4, 1, 0, 0
end = time.time()
print(f"solution execution time: {end - start:.6f} seconds\n")

start = time.time()
for N in test_cases:
    print(f"Binary gap for {N} is {solution2(N)}")  # Expected output: 3, 2, 4, 1, 0, 0
end = time.time()
print(f"solution2 execution time: {end - start:.6f} seconds\n")
