from functools import wraps
import time

def timer(prefix: str):
    # ① prefix を覚える（クロージャ）
    def deco(func):
        # ② wrapsでメタ情報を保持（重要）
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                dt = (time.time() - t0) * 1000
                print(f"{prefix} {func.__name__}: {dt:.1f} ms")
        return wrapper
    return deco

@timer("[API]")
def work(n: int) -> int:
    s = 0
    for i in range(n):
        s += i*i
    return s

print(work(200000))
print(work.__name__)  # wrapsが効くと "work" のまま
