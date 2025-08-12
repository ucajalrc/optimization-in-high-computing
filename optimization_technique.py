import time
import random

# Generate a dataset of random integers
data = [random.randint(0, 1000) for _ in range(100000)]

# -------------------------
# Naive O(n^2) implementation
# -------------------------
def freq_naive(data):
    """
    Naive frequency counter using list.count()
    Time complexity: O(n^2)
    Very inefficient for large datasets because it iterates over
    the entire list for each element.
    """
    result = {}
    for x in data:
        result[x] = data.count(x)
    return result

# -------------------------
# Optimized O(n) implementation
# -------------------------
def freq_fast(data):
    """
    Optimized frequency counter using a hash table (dict).
    Time complexity: O(n)
    Processes the dataset in a single pass.
    """
    result = {}
    for x in data:
        result[x] = result.get(x, 0) + 1
    return result

# Benchmark both methods
start = time.time()
freq_naive(data)
print(f"Naive method took: {time.time() - start:.4f} seconds")

start = time.time()
freq_fast(data)
print(f"Optimized method took: {time.time() - start:.4f} seconds")
