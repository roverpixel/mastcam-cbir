import timeit

setup_code = """
import os
from collections import namedtuple

# Mock PointStruct since we just want to measure loop overhead
class PointStruct:
    __slots__ = ['id', 'vector', 'payload']
    def __init__(self, id, vector, payload):
        self.id = id
        self.vector = vector
        self.payload = payload

# Mock data
batch_size = 64
vectors = [[0.1] * 512 for _ in range(batch_size)]
valid_paths = [f"/path/to/image_{k}.jpg" for k in range(batch_size)]
i = 1000
"""

baseline_code = """
points = []
for j, (vector, filepath) in enumerate(zip(vectors, valid_paths)):
    point_id = i + j
    points.append(
        PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "filepath": filepath,
                "filename": os.path.basename(filepath)
            }
        )
    )
"""

optimized_code = """
points = [
    PointStruct(
        id=i + j,
        vector=vector,
        payload={
            "filepath": filepath,
            "filename": os.path.basename(filepath)
        }
    )
    for j, (vector, filepath) in enumerate(zip(vectors, valid_paths))
]
"""

# Run benchmarks
n_iters = 10000

baseline_time = timeit.timeit(stmt=baseline_code, setup=setup_code, number=n_iters)
print(f"Baseline (append): {baseline_time:.4f} seconds for {n_iters} iterations")

optimized_time = timeit.timeit(stmt=optimized_code, setup=setup_code, number=n_iters)
print(f"Optimized (list comprehension): {optimized_time:.4f} seconds for {n_iters} iterations")

improvement = (baseline_time - optimized_time) / baseline_time * 100
print(f"Improvement: {improvement:.2f}%")
