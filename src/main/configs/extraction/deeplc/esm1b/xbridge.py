#!/usr/bin/env python3
"""
xbridgeデータセットの特徴抽出設定（ESM1b）
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.helper.helper import Helper

# 設定: ESM1bモデル、バッチサイズ32、並列処理あり
xbridge_config = ExtractionRunnerConfig(
    csv_path=Helper.ROOT / "data" / "xbridge" / "data.csv",
    output_path=Helper.ROOT / "outputs" / "xbridge" / "xbridge_data_esm1b.h5",
    dataset_name="xbridge_data_esm1b",
    language_model="esm1b",
    batch_size=32,
    parallel=False,
)
