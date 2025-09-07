import logging
from typing import Any, Dict, List, Type

from src.main.extraction import ExtractionRunner, ExtractionRunnerConfig
from src.main.training import TrainingRunner, TrainingRunnerConfig
from src.main.utils.runner import Runner, RunnerConfig

logger = logging.getLogger(__name__)


class Manager:
    """複数のRunnerConfigを順次実行するマネージャークラス"""

    def __init__(self, configs: List[RunnerConfig]):
        """Managerインスタンスを初期化

        Args:
            configs: 実行するRunnerConfigのリスト
        """
        self.configs = configs
        self.results: List[Any] = []
        self.errors: List[Exception] = []

        # デフォルトのマッピングを設定
        self._runner_registry: Dict[Type[RunnerConfig], Type[Runner]] = {
            ExtractionRunnerConfig: ExtractionRunner,
            TrainingRunnerConfig: TrainingRunner,
        }

    def _create_runner(self, config: RunnerConfig) -> Runner:
        """ConfigからRunnerを生成する

        Args:
            config: Runner設定

        Returns:
            生成されたRunnerインスタンス

        Raises:
            ValueError: 対応するRunnerクラスが見つからない場合
        """
        config_class = type(config)
        if config_class not in self._runner_registry:
            raise ValueError(f"No runner registered for config type: {config_class.__name__}")

        runner_class = self._runner_registry[config_class]
        return runner_class(config)

    def run_all(self) -> List[Any]:
        """すべてのConfigに基づいてRunnerを順次実行

        一つのConfigで例外が発生した場合、次のConfigに遷移します。

        Returns:
            成功したRunnerの実行結果のリスト
        """
        logger.info(f"Starting execution of {len(self.configs)} configs")

        self.results.clear()
        self.errors.clear()

        for i, config in enumerate(self.configs):
            try:
                logger.info(f"Executing config {i + 1}/{len(self.configs)}: {type(config).__name__}")

                # ConfigからRunnerを生成
                runner = self._create_runner(config)

                # Runnerを実行
                result = runner.run()
                self.results.append(result)

                logger.info(f"Config {i + 1} completed successfully")

            except Exception as e:
                logger.error(f"Config {i + 1} failed with error: {e}")
                self.errors.append(e)
                # エラーが発生しても次のConfigに遷移
                continue

        logger.info(f"Execution completed. {len(self.results)} successful, {len(self.errors)} failed")
        return self.results

    def get_results(self) -> List[Any]:
        """成功した実行結果のリストを取得

        Returns:
            成功したRunnerの実行結果のリスト
        """
        return self.results.copy()

    def get_errors(self) -> List[Exception]:
        """発生したエラーのリストを取得

        Returns:
            発生した例外のリスト
        """
        return self.errors.copy()

    def has_errors(self) -> bool:
        """エラーが発生したかどうかを確認

        Returns:
            エラーが発生した場合True
        """
        return len(self.errors) > 0
