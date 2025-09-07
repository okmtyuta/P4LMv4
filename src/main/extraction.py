#!/usr/bin/env python3
"""
Extractorによる特徴抽出処理を行うメインモジュール

このモジュールは、設定可能な形でProteinListに対して言語モデルによる
特徴抽出を実行する機能を提供します。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.main.utils.runner import Runner, RunnerConfig
from src.modules.extract.extractor.extractor import Extractor
from src.modules.extract.language.esm.esm1b import ESM1bLanguage
from src.modules.extract.language.esm.esm2 import ESM2Language
from src.modules.protein.protein_list import ProteinList
from src.modules.protein.protein_types import ProteinLanguageName
from src.modules.services import SlackService
from src.modules.services.platform import PlatformService


@dataclass
class ExtractionRunnerConfig(RunnerConfig):
    """抽出処理の設定を管理するデータクラス

    Attributes:
        csv_path: 入力CSVファイルのパス
        output_path: 出力HDF5ファイルのパス
        dataset_name: データセット名（ログ用）
        language_model: 使用する言語モデル名（"esm2" or "esm1b"）
        batch_size: バッチサイズ
        parallel: 並列処理を使用するかどうか
        max_workers: 並列処理時の最大ワーカー数（Noneで自動）
    """

    csv_path: str | Path
    output_path: str | Path
    dataset_name: str
    language_model: ProteinLanguageName = "esm2"
    batch_size: int = 32
    parallel: bool = False
    max_workers: Optional[int] = None

    def __post_init__(self):
        """パスをPathオブジェクトに変換し、検証を行う"""
        self.csv_path = Path(self.csv_path)
        self.output_path = Path(self.output_path)

        # 入力ファイルの存在確認
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        # 出力ディレクトリの作成
        self.output_path.parent.mkdir(parents=True, exist_ok=True)


class ExtractionRunner(Runner):
    """特徴抽出処理を実行するクラス"""

    def __init__(self, config: ExtractionRunnerConfig):
        """ExtractionRunnerインスタンスを初期化

        Args:
            config: 抽出処理の設定
        """
        super().__init__(config)
        self.config: ExtractionRunnerConfig = config

    @staticmethod
    def _create_language_model(language_name: ProteinLanguageName):
        """言語モデル名から対応する言語モデルインスタンスを作成

        Args:
            language_name: 言語モデル名

        Returns:
            対応する言語モデルインスタンス

        Raises:
            ValueError: サポートされていない言語モデル名の場合
        """
        if language_name == "esm2":
            return ESM2Language()
        elif language_name == "esm1b":
            return ESM1bLanguage()
        else:
            raise ValueError(f"Unsupported language model: {language_name}")

    def run(self) -> ProteinList:
        """設定に基づいて特徴抽出を実行

        Returns:
            特徴抽出済みのProteinList
        """
        print(f"Starting extraction for dataset: {self.config.dataset_name}")
        print(f"Configuration: {self.config}")

        # Slack 通知（開始）
        try:
            server = PlatformService.server_name()
            text = "\n".join(
                [
                    "[EXTRACTION START]",
                    f"server={server}",
                    f"dataset_name={self.config.dataset_name}",
                    f"file={self.config.csv_path}",
                ]
            )
            SlackService().send(text=text)
        except Exception:
            pass

        # 1. CSVからProteinListを読み込み
        print(f"Loading protein data from: {self.config.csv_path}")
        protein_list = ProteinList.from_csv(str(self.config.csv_path))
        print(f"Loaded {len(protein_list)} proteins")

        # 2. 言語モデルの作成
        print(f"Initializing language model: {self.config.language_model}")
        language_model = self._create_language_model(self.config.language_model)

        # 3. Extractorの作成
        extractor = Extractor(language_model)

        # 4. 特徴抽出の実行
        print(f"Running extraction with batch_size={self.config.batch_size}, parallel={self.config.parallel}")
        if self.config.parallel:
            print(f"Using parallel processing with max_workers={self.config.max_workers}")

        extracted_protein_list = extractor(
            protein_list=protein_list,
            batch_size=self.config.batch_size,
            parallel=self.config.parallel,
            max_workers=self.config.max_workers,
        )

        # 5. 結果の保存
        print(f"Saving results to: {self.config.output_path}")
        extracted_protein_list.save_as_hdf5(str(self.config.output_path))
        print(f"ExtractionRunner completed successfully for {len(extracted_protein_list)} proteins")

        # Slack 通知（終了）
        try:
            server = PlatformService.server_name()
            text = "\n".join(
                [
                    "[EXTRACTION END]",
                    f"server={server}",
                    f"dataset_name={self.config.dataset_name}",
                    f"file={self.config.csv_path}",
                ]
            )
            SlackService().send(text=text)
        except Exception:
            pass

        return extracted_protein_list
