from concurrent.futures import ProcessPoolExecutor
import os
from typing import List, Tuple


def work(x: int) -> Tuple[int, int, int]:
    """与えられた整数 x の二乗を計算し、(入力, 結果, 実行PID) を返す。"""
    pid = os.getpid()
    y = x * x
    return x, y, pid


def run_parallel(values: List[int]) -> List[Tuple[int, int, int]]:
    """ProcessPoolExecutorで並列実行して結果をリストで返す。"""
    # max_workers は明示的に指定（使用するCPU論理コア数）
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(work, values))
    return results


def main() -> None:
    values = list(range(10))
    results = run_parallel(values)
    for x, y, pid in results:
        print(f"pid={pid} f({x})={y}")


if __name__ == "__main__":
    main()

