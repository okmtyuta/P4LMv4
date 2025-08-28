from abc import ABC, abstractmethod
from typing import Any


class RunnerConfig(ABC):
    """Runner設定の抽象基底クラス"""

    pass


class Runner(ABC):
    """Runnerの抽象基底クラス"""

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config

    @abstractmethod
    def run(self) -> Any:
        """実行メソッド"""
        pass
