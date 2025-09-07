from __future__ import annotations

"""
学習結果（各フェーズの出力・評価）を保持・集約するデータ構造群。

- `EpochPhaseResult`: 1 フェーズの出力/正解/指標/ProteinList を束ねる。
- `EpochSummary`: 1 エポックの train/validate/evaluate をまとめる。
- `TrainingHistory`: 複数エポックの履歴（並べ替え等の補助機能あり）。
"""

from dataclasses import dataclass, field
from typing import Optional

import torch

from src.modules.container.serializable_container import SerializableContainer
from src.modules.container.serializable_container_list import SerializableContainerList
from src.modules.protein.protein_list import ProteinList
from src.modules.protein.protein_types import ProteinPropName
from src.modules.train.criterion import Criteria, Criterion


@dataclass
class EpochPhaseResult(SerializableContainer):
    """1 フェーズ分（train/validate/evaluate）の推論・評価結果。"""

    epoch: int
    output: torch.Tensor
    label: torch.Tensor
    input_props: list[ProteinPropName]
    output_props: list[ProteinPropName]
    protein_list: Optional[ProteinList] = None
    criteria: Optional[dict[ProteinPropName, Criteria]] = field(default=None, repr=False)

    @property
    def output_by_prop(self) -> dict[ProteinPropName, torch.Tensor]:
        """出力テンソルをプロパティ名で列アクセス可能にするビュー。"""
        return {p: self.output[:, i] for i, p in enumerate(self.output_props)}

    @property
    def label_by_prop(self) -> dict[ProteinPropName, torch.Tensor]:
        """教師テンソルをプロパティ名で列アクセス可能にするビュー。"""
        return {p: self.label[:, i] for i, p in enumerate(self.output_props)}

    @property
    def accuracy(self) -> float:
        """単純合算の精度指標（各プロパティの Pearson を和）。"""
        if self.criteria is None:
            raise RuntimeError("criteria is not computed. call `compute_criteria()` first.")
        return float(sum(v["pearsonr"] for v in self.criteria.values()))

    def compute_criteria(self) -> "EpochPhaseResult":
        """各プロパティごとの指標を計算し `criteria` に格納する。"""
        metrics: dict[ProteinPropName, Criteria] = {}
        for i, name in enumerate(self.output_props):
            metrics[name] = Criterion.call(output=self.output[:, i], label=self.label[:, i])
        self.criteria = metrics
        return self

    def assign_predictions_to_proteins(self) -> "EpochPhaseResult":
        """予測値を `protein_list` の各 Protein の `predicted` へ書き戻す。"""
        if self.protein_list is None:
            raise RuntimeError("proteins is None. set ProteinList before assigning predictions.")
        for i, protein in enumerate(self.protein_list):
            predicted = dict(protein.predicted or {})
            for j, prop in enumerate(self.output_props):
                predicted[prop] = float(self.output[i, j].item())
            protein.set_predicted(predicted)
        return self


@dataclass
class EpochSummary(SerializableContainer):
    """1 エポックの train/validate/evaluate を束ねた要約。"""

    epoch: int
    train: EpochPhaseResult
    validate: EpochPhaseResult
    evaluate: EpochPhaseResult


class TrainingHistory(SerializableContainerList[EpochSummary]):
    """複数エポックの履歴。HDF5 へまとめて保存可能。"""

    @property
    def epochs(self) -> list[int]:
        """履歴に含まれるエポック番号の一覧。"""
        return [item.epoch for item in self]

    def sort_by_epoch(self) -> "TrainingHistory":
        """エポック番号で昇順ソートした新しい履歴を返す。"""
        sorted_items = sorted(list(self), key=lambda x: x.epoch)
        return type(self)(sorted_items)
