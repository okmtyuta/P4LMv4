"""
ProteinList を指定バッチサイズで処理し、言語モデルにかける抽出器。

- 並列（ProcessPoolExecutor）/逐次の双方に対応。
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from tqdm import tqdm

from src.modules.extract.language._language import _Language
from src.modules.protein.protein_list import ProteinList


def _process_batch_with_language(language: _Language, batch: ProteinList) -> ProteinList:
    """プロセスプールで 1 バッチを処理するための補助関数。"""
    language(batch)
    return batch


class Extractor:
    """ProteinList をバッチ処理で言語モデルにかける呼び出しラッパ。"""

    def __init__(self, language: _Language):
        """使用する言語モデルを受け取って初期化。"""
        self._language = language

    def __call__(
        self, protein_list: ProteinList, batch_size: int, parallel: bool = False, max_workers: Optional[int] = None
    ) -> ProteinList:
        """ProteinList をバッチに分割し、逐次/並列で処理する。

        Args:
            protein_list: 入力 ProteinList。
            batch_size: バッチサイズ。
            parallel: 並列処理を行うか。
            max_workers: 並列時のワーカー数（None なら自動）。

        Returns:
            処理後の ProteinList。
        """
        # バッチサイズに基づいてProteinListを手動で分割
        protein_lists: list[ProteinList] = []
        total_proteins = len(protein_list)
        for start_idx in range(0, total_proteins, batch_size):
            end_idx = min(start_idx + batch_size, total_proteins)
            batch = protein_list[start_idx:end_idx]
            if isinstance(batch, ProteinList):
                protein_lists.append(batch)

        if parallel and len(protein_lists) > 1:
            # 並列処理（ProcessPoolExecutor固定）
            return self._process_parallel(protein_lists, max_workers)
        else:
            # 逐次処理
            return self._process_sequential(protein_lists)

    def _process_sequential(self, protein_lists: list[ProteinList]) -> ProteinList:
        """逐次処理で全バッチを処理する。"""
        for batch_protein_list in tqdm(protein_lists, desc="Processing batches"):
            self._language(batch_protein_list)
        return ProteinList.join(protein_lists)

    def _process_parallel(self, protein_lists: list[ProteinList], max_workers: Optional[int]) -> ProteinList:
        """プロセスプールで全バッチを並列処理する。"""
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            # 各バッチを並列で処理
            future_to_batch = {
                executor.submit(_process_batch_with_language, self._language, batch): batch for batch in protein_lists
            }

            # 進行状況表示付きで結果を取得
            processed_batches = []
            with tqdm(total=len(protein_lists), desc="Processing batches (process)", mininterval=0) as pbar:
                for future in as_completed(future_to_batch):
                    processed_batch = future.result()
                    processed_batches.append(processed_batch)
                    pbar.update(1)
                    pbar.refresh()  # force flush

        return ProteinList.join(processed_batches)
