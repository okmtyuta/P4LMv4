#!/usr/bin/env python3
"""
hela_lumos_1hデータセットの特徴抽出設定
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.helper.helper import Helper

# 設定: ESM2モデル、バッチサイズ32、並列処理あり
hela_lumos_1h_config = ExtractionRunnerConfig(
    csv_path=Helper.ROOT / "data" / "hela_lumos_1h" / "data.csv",
    output_path=Helper.ROOT / "outputs" / "hela_lumos_1h" / "hela_lumos_1h_data_esm2.h5",
    dataset_name="hela_lumos_1h_data_esm2",
    language_model="esm2",
    batch_size=32,
    parallel=True,
    max_workers=4,
)
