from typing import Optional, Union

import torch

from src.modules.container.serializable_container import SerializableContainer
from src.modules.protein.protein_types import ProteinProps


class Protein(SerializableContainer):
    """配列・属性・表現テンソルを扱う Protein の単一要素。"""

    def __init__(
        self,
        key: str,
        props: ProteinProps,
        representations: Optional[torch.Tensor] = None,
        processed: Optional[torch.Tensor] = None,
        predicted: Optional[ProteinProps] = None,
    ) -> None:
        """キー・属性・任意のテンソルを受け取り初期化する。"""
        self.key = key
        self.props = props
        self.representations = representations
        self.processed = processed
        self.predicted = predicted if predicted is not None else {}

    # Note: `representations` はパブリック属性として保持（テスト仕様に合わせる）

    @property
    def seq(self) -> str:
        """文字列の一次配列（props['seq']）を返す。"""
        value = self.props.get("seq")
        if not isinstance(value, str):
            raise TypeError("Protein.props['seq'] must be a str")
        return value

    def read_props(self, name: str) -> Union[str, int, float]:
        """指定名の属性値を読み出す（存在しない/None は例外）。"""
        if name not in self.props:
            raise RuntimeError(f"Prop {name} is not readable")

        prop = self.props[name]
        if prop is None:
            raise RuntimeError(f"Prop {name} is not readable")

        return prop

    def set_props(self, props: ProteinProps) -> "Protein":
        """属性辞書を置き換え、self を返す。"""
        self.props = props
        return self

    def set_representations(self, representations: torch.Tensor) -> "Protein":
        """表現テンソルを設定し、self を返す。"""
        self.representations = representations
        return self

    def set_processed(self, processed: torch.Tensor) -> "Protein":
        """処理済テンソルを設定し、self を返す。"""
        self.processed = processed
        return self

    def set_predicted(self, predicted: ProteinProps) -> "Protein":
        """予測値辞書を設定し、self を返す。"""
        self.predicted = predicted
        return self

    def get_representations(self) -> torch.Tensor:
        """表現テンソルを返す（未設定なら例外）。"""
        if self.representations is None:
            raise RuntimeError("Protein representations unavailable")

        return self.representations

    def get_processed(self) -> torch.Tensor:
        """処理済テンソルを返す（未設定なら例外）。"""
        if self.processed is None:
            raise RuntimeError("Protein processed unavailable")

        return self.processed
