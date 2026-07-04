import timeit

class Hit:
    def __init__(self, score, filename):
        self.score = score
        self.payload = {'filename': filename}

# Simulate a typical search result size (e.g., limit=10)
search_result = [Hit(0.95 + (i * 0.001), f'file_{i}.jpg') for i in range(10)]

def using_loop():
    results = []
    for hit in search_result:
        results.append({
            'score': round(hit.score, 4),
            'filename': hit.payload['filename']
        })
    return results

def using_list_comp():
    return [{
        'score': round(hit.score, 4),
        'filename': hit.payload['filename']
    } for hit in search_result]

loop_time = timeit.timeit(using_loop, number=100000)
list_comp_time = timeit.timeit(using_list_comp, number=100000)

print(f"Loop time: {loop_time:.6f} seconds")
print(f"List comp time: {list_comp_time:.6f} seconds")
print(f"Improvement: {(loop_time - list_comp_time) / loop_time * 100:.2f}%")
