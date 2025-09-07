"""
学習過程の履歴・最良エポックを管理し、HDF5 へ保存するユーティリティ。
"""

from typing import Optional

import h5py

from src.modules.train.train_result import EpochSummary, TrainingHistory


class TrainRecorder:
    """学習履歴と最良エポックを管理する。

    - 1 エポックごとの結果は `EpochSummary`。
    - 履歴は `TrainingHistory` として保持。
    - 早期停止の判定は「最良エポックからの経過エポック数 < patience」で継続。
    """

    def __init__(self, patience: int = 50) -> None:
        """猶予エポック数を受け取って初期化する。"""
        self._current_epoch: int = 1
        self._patience: int = patience

        self._belle_epoch: Optional[int] = None
        self._belle_epoch_summary: Optional[EpochSummary] = None

        self._history: TrainingHistory = TrainingHistory([])

    # --- Properties ---
    @property
    def current_epoch(self) -> int:
        """現在のエポック番号（1 始まり）。"""
        return self._current_epoch

    @property
    def patience(self) -> int:
        """早期停止の猶予エポック数。"""
        return self._patience

    @property
    def belle_epoch_summary(self) -> Optional[EpochSummary]:
        """最良エポックの要約。未決定時は None。"""
        return self._belle_epoch_summary

    @property
    def history(self) -> TrainingHistory:
        """全エポックの履歴。"""
        return self._history

    # --- Core API ---
    def append_epoch_results(self, summary: EpochSummary) -> None:
        """履歴へ追加し、必要なら最良エポックを更新する。"""
        self._history.append(summary)
        if self._is_better(summary):
            self._belle_epoch = self._current_epoch
            self._belle_epoch_summary = summary

    def _is_better(self, summary: EpochSummary) -> bool:
        """精度（validate の accuracy）が最大であるかを判定。"""
        if self._belle_epoch_summary is None:
            return True
        return summary.validate.accuracy >= self._belle_epoch_summary.validate.accuracy

    def next_epoch(self) -> None:
        """エポック番号を 1 進める。"""
        if self._belle_epoch is not None:
            print(self._current_epoch, self._belle_epoch_summary.evaluate.accuracy)
        self._current_epoch += 1

    def to_continue(self) -> bool:
        """早期停止の継続判定。

        - 最良エポック未決定なら継続。
        - 決定済みなら「現在エポック − 最良エポック < patience」で継続。
        """
        if self._belle_epoch is None:
            return True
        return (self._current_epoch - self._belle_epoch) < self._patience

    # --- Persistence ---
    def finalize(self, group: h5py.Group) -> None:
        """結果をHDF5グループへ保存。

        - attrs: current_epoch, patience, belle_epoch
        - history: TrainingHistory を `history` サブグループへ
        - belle: 最良エポックの EpochSummary を `belle` サブグループへ（存在する場合）
          保存直前に evaluate の予測を ProteinList に書き戻します。
        """
        group.attrs["current_epoch"] = self._current_epoch
        group.attrs["patience"] = self._patience
        if self._belle_epoch is not None:
            group.attrs["belle_epoch"] = self._belle_epoch

        for summary in self._history:
            if self._belle_epoch is not None and summary.epoch != self._belle_epoch:
                summary.train.forget()
                summary.validate.forget()
                summary.evaluate.forget()

        # history
        hist_g = group.create_group("history")
        self._history.to_hdf5_group(hist_g)

        # belle
        if self._belle_epoch_summary is not None:
            # 保存直前に評価フェーズの予測を書き戻す（必要時のみ）
            eval_phase = self._belle_epoch_summary.evaluate
            if eval_phase.protein_list is not None:
                eval_phase.assign_predictions_to_proteins()
            belle_g = group.create_group("belle")
            self._belle_epoch_summary.to_hdf5_group(belle_g)

    def save(self, file_path: str) -> None:
        """HDF5 ファイルとして保存。"""
        with h5py.File(file_path, "w") as f:
            root = f.create_group("result")
            self.finalize(root)
