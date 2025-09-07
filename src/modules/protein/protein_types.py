"""
Protein 関連の型定義。
"""

from typing import Literal, Optional, TypedDict

import torch

ProteinLanguageName = Literal["esm2", "esm1b"]
protein_language_names: list[ProteinLanguageName] = ["esm2", "esm1b"]


ProteinProps = dict[str, str | int | float]
ProteinPropName = str


class ProteinSource(TypedDict):
    """Protein のソース情報（シリアライズ用の最小構成）。"""

    key: str
    seq: str
    props: ProteinProps
    representations: Optional[torch.Tensor]
