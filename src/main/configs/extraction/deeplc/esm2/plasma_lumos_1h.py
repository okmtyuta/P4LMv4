#!/usr/bin/env python3
"""
plasma_lumos_1hデータセットの特徴抽出設定
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.helper.helper import Helper

# 設定: ESM2モデル、バッチサイズ32、並列処理あり
plasma_lumos_1h_config = ExtractionRunnerConfig(
    csv_path=Helper.ROOT / "data" / "plasma_lumos_1h" / "data.csv",
    output_path=Helper.ROOT / "outputs" / "plasma_lumos_1h" / "plasma_lumos_1h_data_esm2.h5",
    dataset_name="plasma_lumos_1h_data_esm2",
    language_model="esm2",
    batch_size=32,
    parallel=False,
)
