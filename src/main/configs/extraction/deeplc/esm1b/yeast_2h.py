#!/usr/bin/env python3
"""
yeast_2hデータセットの特徴抽出設定（ESM1b）
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.helper.helper import Helper

# 設定: ESM1bモデル、バッチサイズ32、並列処理あり
yeast_2h_config = ExtractionRunnerConfig(
    csv_path=Helper.ROOT / "data" / "yeast_2h" / "data.csv",
    output_path=Helper.ROOT / "outputs" / "yeast_2h" / "yeast_2h_data_esm1b.h5",
    dataset_name="yeast_2h_data_esm1b",
    language_model="esm1b",
    batch_size=32,
    parallel=False,
)
