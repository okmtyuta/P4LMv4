"""
ESM2 言語モデルの薄いラッパ。
"""

from src.modules.extract.language.esm._esm import _ESMLanguage


class ESM2Language(_ESMLanguage):
    """ESM2 を用いて表現を付与する言語クラス。"""

    name = "esm2"

    def __init__(self):
        super().__init__("esm2")
