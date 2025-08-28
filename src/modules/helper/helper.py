from pathlib import Path
from typing import ClassVar


class Helper:
    """プロジェクト共通のヘルパー。

    - `ROOT`: リポジトリのルートディレクトリを指すパス。
    """

    ROOT: ClassVar[Path] = Path(__file__).resolve().parents[3]
