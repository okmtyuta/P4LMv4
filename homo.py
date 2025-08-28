from __future__ import annotations

import math
import multiprocessing as mp
import os
from typing import List

from tqdm import tqdm

from src.modules.protein.protein_list import ProteinList

# seed = 5900308802214385025 is better
# seed = 5911646470734835975 is more better
# seed = 10338668691691671231 is more and more better
protein_list = ProteinList.load_from_hdf5("outputs/plasma_lumos_1h/plasma_lumos_1h_data_esm2.h5").shuffle(
    seed=10338668691691671231
)

tlpl, elpl, vlpl = protein_list.split_by_ratio(ratios=[0.8, 0.1, 0.1])

tlpla = tlpl.map(lambda p: p.seq).to_list()
elpla = tlpl.map(lambda p: p.seq).to_list()


def _nw_identity(a: str, b: str) -> float:
    # Needleman–Wunsch global alignment: match=+1, mismatch=-1, gap=-1
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 0.0

    score = [[0] * (m + 1) for _ in range(n + 1)]
    ptr = [[0] * (m + 1) for _ in range(n + 1)]  # 1:diag, 2:up, 3:left

    for i in range(1, n + 1):
        score[i][0] = -i
        ptr[i][0] = 2
    for j in range(1, m + 1):
        score[0][j] = -j
        ptr[0][j] = 3

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            bj = b[j - 1]
            diag = score[i - 1][j - 1] + (1 if ai == bj else -1)
            up = score[i - 1][j] - 1
            left = score[i][j - 1] - 1
            best = diag
            p = 1
            if up > best:
                best, p = up, 2
            if left > best:
                best, p = left, 3
            score[i][j] = best
            ptr[i][j] = p

    i, j = n, m
    matches = 0
    columns = 0
    while i > 0 or j > 0:
        p = ptr[i][j]
        if p == 1:
            if a[i - 1] == b[j - 1]:
                matches += 1
            i -= 1
            j -= 1
            columns += 1
        elif p == 2:
            i -= 1
            columns += 1
        else:
            j -= 1
            columns += 1
    if columns == 0:
        return 0.0
    return matches / columns


_DST: List[str] = []


def _init_pool(dst: List[str]) -> None:
    global _DST
    _DST = dst


def _best_for_one(seq: str) -> float:
    # Uses global _DST set by pool initializer
    best = 0.0
    for ref in _DST:
        ident = _nw_identity(seq, ref)
        if ident > best:
            best = ident
    return best


def _workers() -> int:
    c = os.cpu_count()
    return 1 if c is None or c <= 1 else c


def _best_hit_average(src: List[str], dst: List[str]) -> float:
    if not src or not dst:
        return 0.0
    with mp.Pool(processes=_workers(), initializer=_init_pool, initargs=(dst,)) as pool:
        totals = 0.0
        for val in tqdm(
            pool.imap_unordered(_best_for_one, src, chunksize=1), total=len(src), desc="Best-hit averaging", unit="seq"
        ):
            totals += val
    return totals / float(len(src))


def _best_hit_identities(src: List[str], dst: List[str], desc: str) -> List[float]:
    if not src or not dst:
        return []
    with mp.Pool(processes=_workers(), initializer=_init_pool, initargs=(dst,)) as pool:
        out = list(tqdm(pool.imap_unordered(_best_for_one, src, chunksize=1), total=len(src), desc=desc, unit="seq"))
    return out


def _percentile(sorted_vals: List[float], p: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if p <= 0:
        return sorted_vals[0]
    if p >= 1:
        return sorted_vals[-1]
    k = (n - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def _print_histogram(values: List[float]) -> None:
    if not values:
        print("(no data)")
        return
    bins = [i / 20.0 for i in range(21)]  # 0.00 .. 1.00 step 0.05
    counts = [0] * 20
    for v in values:
        idx = min(int(v * 20), 19)
        counts[idx] += 1
    total = len(values)
    max_count = max(counts) if counts else 1
    bar_width = 40
    for i in range(20):
        lo = bins[i]
        hi = bins[i + 1]
        bar_len = 0 if max_count == 0 else math.ceil(counts[i] / max_count * bar_width)
        bar = "#" * bar_len
        pct = counts[i] / total * 100.0
        print(f"[{lo:>4.2f},{hi:>4.2f}) {counts[i]:5d} ({pct:5.1f}%) {bar}")


def main() -> None:
    a_list = tlpla
    b_list = tlpla

    ab = _best_hit_average(a_list, b_list)
    ba = _best_hit_average(b_list, a_list)
    sym = (ab + ba) / 2.0

    # Print results (as decimals between 0 and 1)
    print(f"A->B Best-Hit Average Identity: {ab:.6f}")
    print(f"B->A Best-Hit Average Identity: {ba:.6f}")
    print(f"Symmetric Best-Hit Homology   : {sym:.6f}")

    # Report: max / 95th percentile / histogram for both directions
    print()
    print("A->B Best-Hit Identity Report:")
    ab_vals = _best_hit_identities(a_list, b_list, desc="A->B best-hit")
    ab_sorted = sorted(ab_vals)
    ab_max = ab_sorted[-1] if ab_sorted else 0.0
    ab_p95 = _percentile(ab_sorted, 0.95)
    print(f"  max: {ab_max:.6f}")
    print(f"  p95: {ab_p95:.6f}")
    _print_histogram(ab_vals)

    print()
    print("B->A Best-Hit Identity Report:")
    ba_vals = _best_hit_identities(b_list, a_list, desc="B->A best-hit")
    ba_sorted = sorted(ba_vals)
    ba_max = ba_sorted[-1] if ba_sorted else 0.0
    ba_p95 = _percentile(ba_sorted, 0.95)
    print(f"  max: {ba_max:.6f}")
    print(f"  p95: {ba_p95:.6f}")
    _print_histogram(ba_vals)


if __name__ == "__main__":
    main()
