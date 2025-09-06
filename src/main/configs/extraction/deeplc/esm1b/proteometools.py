#!/usr/bin/env python3
"""
proteometoolsデータセットの特徴抽出設定（ESM1b）
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.helper.helper import Helper

# 設定: ESM1bモデル、バッチサイズ32、並列処理あり
proteometools_config = ExtractionRunnerConfig(
    csv_path=Helper.ROOT / "data" / "proteometools" / "data.csv",
    output_path=Helper.ROOT / "outputs" / "proteometools" / "proteometools_data_esm1b.h5",
    dataset_name="proteometools_data_esm1b",
    language_model="esm1b",
    batch_size=32,
    parallel=True,
    max_workers=4,
)
