"""
ESM1b 言語モデルの薄いラッパ。
"""

from src.modules.extract.language.esm._esm import _ESMLanguage


class ESM1bLanguage(_ESMLanguage):
    """ESM1b を用いて表現を付与する言語クラス。"""

    name = "esm1b"

    def __init__(self):
        super().__init__("esm1b")
