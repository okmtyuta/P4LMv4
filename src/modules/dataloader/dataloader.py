from __future__ import annotations

"""
学習用データの生成を担う軽量データローダ。

- `ProteinList` と前処理パイプラインから、学習に使えるテンソルバッチを構築する。
- バッチは `DataBatch` としてカプセル化し、必要に応じてキャッシュする。
"""

from dataclasses import dataclass
from typing import Optional

import torch

from src.modules.data_process.data_process_list import DataProcessList
from src.modules.protein.protein_list import ProteinList


@dataclass
class DataloaderConfig:
    """Dataloader の構成情報。

    Attributes:
        protein_list: 入力となるタンパク質列。
        input_props: 入力に連結する数値プロパティ名。
        output_props: 教師信号となるプロパティ名。
        batch_size: バッチサイズ。
        cacheable: バッチ内テンソルをキャッシュするか。
        process_list: 前処理パイプライン。
    """

    protein_list: ProteinList
    input_props: list[str]
    output_props: list[str]
    batch_size: int
    cacheable: bool
    process_list: DataProcessList

    def with_protein_list(self, protein_list: ProteinList) -> "DataloaderConfig":
        """同設定のまま `protein_list` だけ差し替えた新しい設定を返す。"""
        return DataloaderConfig(
            protein_list=protein_list,
            input_props=self.input_props,
            output_props=self.output_props,
            batch_size=self.batch_size,
            cacheable=self.cacheable,
            process_list=self.process_list,
        )


UsableDataBatch = tuple[torch.Tensor, torch.Tensor, ProteinList]


class DataBatch:
    """1 バッチ分のテンソルと関連メタを遅延生成・キャッシュするラッパ。"""

    def __init__(self, config: DataloaderConfig) -> None:
        """設定を受け取り、未計算状態で初期化する。"""
        self._config = config
        self._cache: Optional[UsableDataBatch] = None

    def __len__(self) -> int:
        """バッチに含まれるサンプル数（=内部 ProteinList の長さ）。"""
        return len(self._config.protein_list)

    def use(self) -> UsableDataBatch:
        """学習に使用可能な `(input, label, ProteinList)` を返す。

        - キャッシュが有効かつ存在する場合は再計算しない。
        - `process_list` を逐次適用して `processed` を更新後、
          入力テンソルに `input_props` を連結する。
        """
        if self._config.cacheable and self._cache is not None:
            return self._cache

        inputs = []
        labels = []

        process_list = self._config.process_list(protein_list=self._config.protein_list)

        for protein in process_list:
            processed = protein.get_processed()
            input_props = torch.Tensor([protein.read_props(key) for key in self._config.input_props])
            input_tensor = torch.cat([processed, input_props], dim=0)
            inputs.append(input_tensor)

            label = [protein.read_props(key) for key in self._config.output_props]
            labels.append(label)

        usable: UsableDataBatch = (
            torch.stack(inputs).to(torch.float32),
            torch.Tensor(labels).to(torch.float32),
            self._config.protein_list,
        )

        if self._config.cacheable:
            self._cache = usable

        return usable


class Dataloader:
    """ProteinList をバッチ列へ分割し、`DataBatch` の列として提供する。"""

    def __init__(self, config: DataloaderConfig) -> None:
        """設定を保持する。バッチ生成は遅延評価。"""
        self._config = config
        self._batches: Optional[list[DataBatch]] = None

    def __len__(self) -> int:
        """生成済みバッチ数。必要なら内部で生成される。"""
        return len(self.batches)

    @property
    def input_props(self):
        """入力に連結されるプロパティ名の一覧。"""
        return self._config.input_props

    @property
    def output_props(self):
        """教師信号として用いるプロパティ名の一覧。"""
        return self._config.output_props

    @property
    def batches(self):
        """`DataBatch` の列を返す（未生成なら生成する）。"""
        if self._batches is None:
            self._batches = self._generate_batches()
        return self._batches

    def __iter__(self):
        """`DataBatch` のイテレータ。"""
        return iter(self.batches)

    def __getitem__(self, key):
        """スライス時はバッチ列を部分列に差し替えたビューを返す。"""
        if isinstance(key, slice):
            new = object.__new__(type(self))
            new.__dict__ = self.__dict__.copy()
            # 実体化したバッチをスライス（必要に応じて生成）
            new._batches = self.batches[key]
            return new
        return self.batches[key]

    def _generate_batches(self) -> list[DataBatch]:
        """内部 ProteinList を `batch_size` ごとに分割して DataBatch を生成。"""
        protein_lists: list[ProteinList] = self._config.protein_list.split_by_size(self._config.batch_size)
        batches: list[DataBatch] = []

        for protein_list in protein_lists:
            config = self._config.with_protein_list(protein_list=protein_list)
            batches.append(DataBatch(config=config))

        return batches

    def split_by_ratio(self, ratios: list[float]) -> list["Dataloader"]:
        """内部 ProteinList を比率で分割し、各部分に対する Dataloader を返す。"""
        if any(r <= 0 for r in ratios):
            raise ValueError("All ratios must be positive")

        parts: list[ProteinList] = self._config.protein_list.split_by_ratio(ratios)
        loaders: list[Dataloader] = []
        for plist in parts:
            cfg = self._config.with_protein_list(plist)
            loaders.append(Dataloader(cfg))
        return loaders

    def output_dim(self, input_dim: int) -> int:
        """前処理後の特徴次元 + 入力プロパティ次元を返す。

        Args:
            input_dim: 元の系列表現の特徴次元。

        Returns:
            モデル入力となる次元数。
        """
        processed_dim = self._config.process_list.output_dim(input_dim=input_dim)
        input_props_dim = len(self._config.input_props)
        return processed_dim + input_props_dim
