from pathlib import Path
from typing import ClassVar


class Dir:
    """プロジェクトのディレクトリパスを管理するクラス"""

    # プロジェクトのルートディレクトリ（static変数）
    ROOT: ClassVar[Path] = Path(__file__).parent.parent.parent.parent
