#!/usr/bin/env python3
"""
hela_lumos_2hデータセットの特徴抽出設定
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.dir.dir import Dir

# 設定: ESM2モデル、バッチサイズ32、並列処理あり
hela_lumos_2h_config = ExtractionRunnerConfig(
    csv_path=Dir.ROOT / "data" / "hela_lumos_2h" / "data.csv",
    output_path=Dir.ROOT / "outputs" / "hela_lumos_2h" / "hela_lumos_2h_data_esm2.h5",
    dataset_name="hela_lumos_2h_data_esm2",
    language_model="esm2",
    batch_size=32,
    parallel=True,
)
