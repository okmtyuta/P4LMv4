#!/usr/bin/env python3
"""
plasma_lumos_2hデータセットの特徴抽出設定（ESM1b）
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.helper.helper import Helper

# 設定: ESM1bモデル、バッチサイズ32、並列処理あり
plasma_lumos_2h_config = ExtractionRunnerConfig(
    csv_path=Helper.ROOT / "data" / "plasma_lumos_2h" / "data.csv",
    output_path=Helper.ROOT / "outputs" / "plasma_lumos_2h" / "plasma_lumos_2h_data_esm1b.h5",
    dataset_name="plasma_lumos_2h_data_esm1b",
    language_model="esm1b",
    batch_size=32,
    parallel=False,
)
