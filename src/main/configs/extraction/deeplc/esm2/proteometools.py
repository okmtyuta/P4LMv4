#!/usr/bin/env python3
"""
proteometoolsデータセットの特徴抽出設定
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.helper.helper import Helper

# 設定: ESM2モデル、バッチサイズ32、並列処理あり
proteometools_config = ExtractionRunnerConfig(
    csv_path=Helper.ROOT / "data" / "proteometools" / "data.csv",
    output_path=Helper.ROOT / "outputs" / "proteometools" / "proteometools_data_esm2.h5",
    dataset_name="proteometools_data_esm2",
    language_model="esm2",
    batch_size=32,
    parallel=False,
)
