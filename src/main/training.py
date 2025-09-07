#!/usr/bin/env python3
"""
Trainer による学習処理を行うメインモジュール

このモジュールは、`src/main/extraction.py` と同様に Config + Runner 形式で
学習処理を構成します。Manager からは `TrainingRunnerConfig` を渡すだけで
トレーニングが実行できます。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from schedulefree import RAdamScheduleFree  # type: ignore[import-untyped]

from src.main.utils.runner import Runner, RunnerConfig
from src.modules.data_process.data_process_list import DataProcessList
from src.modules.dataloader.dataloader import Dataloader, DataloaderConfig
from src.modules.model.model import Model
from src.modules.protein.protein_list import ProteinList
from src.modules.services import SlackService
from src.modules.services.platform import PlatformService
from src.modules.train.recorder import TrainRecorder
from src.modules.train.trainer import Trainer


@dataclass
class TrainingRunnerConfig(RunnerConfig):
    """学習処理の設定を管理するデータクラス

    Notes:
        - 可変長・オプショナル引数は使用しません。必要な値は全て明示指定します。
    """

    input_hdf5_path: str | Path
    output_hdf5_path: str | Path
    dataset_name: str

    # Dataloader 関連（data_process も外部から指定）
    input_props: list[str]
    output_props: list[str]
    batch_size: int
    cacheable: bool
    process_list: DataProcessList

    # モデル/オプティマイザ（外部注入）
    model: Model
    optimizer: RAdamScheduleFree

    # 学習制御（デフォルトを許容）
    patience: int = 100
    shuffle_seed: int = 100

    def __post_init__(self) -> None:
        # Path 化と入出力の検証
        self.input_hdf5_path = Path(self.input_hdf5_path)
        self.output_hdf5_path = Path(self.output_hdf5_path)

        if not self.input_hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.input_hdf5_path}")

        self.output_hdf5_path.parent.mkdir(parents=True, exist_ok=True)


class TrainingRunner(Runner):
    """Training を実行する Runner。"""

    def __init__(self, config: TrainingRunnerConfig) -> None:
        super().__init__(config)
        self.config: TrainingRunnerConfig = config

    def run(self) -> TrainRecorder:
        """設定に基づいて学習を実行し、Recorder を返す。

        実行の開始と終了時に Slack 通知を送信する。通知には
        実行サーバー名と対象の設定（データセット名／入力ファイル名）を含める。
        """
        cfg = self.config
        print(f"Starting training for dataset: {cfg.dataset_name}")
        print(f"Configuration: {cfg}")

        # Slack 通知（開始）
        try:
            server = PlatformService.server_name()
            fname = Path(cfg.input_hdf5_path).name
            SlackService().send(text=f"[TRAIN START] server={server} config={cfg.dataset_name} file={fname}")
        except Exception:
            # 通知失敗は学習に影響させない
            pass

        # 1. データ読み込み（HDF5）+ シャッフル
        print(f"Loading proteins from: {cfg.input_hdf5_path}")
        protein_list = ProteinList.load_from_hdf5(str(cfg.input_hdf5_path)).shuffle(seed=cfg.shuffle_seed)
        print(f"Loaded {len(protein_list)} proteins")

        # 2. Dataloader 準備
        dl_conf = DataloaderConfig(
            protein_list=protein_list,
            input_props=cfg.input_props,
            output_props=cfg.output_props,
            batch_size=cfg.batch_size,
            cacheable=cfg.cacheable,
            process_list=cfg.process_list,
        )
        dataloader = Dataloader(config=dl_conf)
        # 3. モデル・オプティマイザ（外部注入済み）
        model = cfg.model
        optimizer = cfg.optimizer

        # 4. Trainer 準備
        trainer = Trainer(model=model, dataloader=dataloader, optimizer=optimizer)
        # 既存 Trainer は patience を受け取らないため、Recorder を差し替え
        trainer._recorder = TrainRecorder(patience=cfg.patience)

        # 5. 学習
        print(f"Training started: batch_size={cfg.batch_size}, patience={cfg.patience}")
        trainer.train()

        # 6. 保存（最良エポック含む履歴を HDF5 へ）
        print(f"Saving training results to: {cfg.output_hdf5_path}")
        trainer._recorder.save(str(cfg.output_hdf5_path))
        print("Training completed successfully")

        # Slack 通知（終了）
        try:
            server = PlatformService.server_name()
            fname = Path(cfg.input_hdf5_path).name
            SlackService().send(text=f"[TRAIN END] server={server} config={cfg.dataset_name} file={fname}")
        except Exception:
            pass

        return trainer._recorder
