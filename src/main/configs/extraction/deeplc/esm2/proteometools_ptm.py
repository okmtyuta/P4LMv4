#!/usr/bin/env python3
"""
proteometools_ptmデータセットの特徴抽出設定
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.helper.helper import Helper

# 設定: ESM2モデル、バッチサイズ32、並列処理あり
proteometools_ptm_config = ExtractionRunnerConfig(
    csv_path=Helper.ROOT / "data" / "proteometools_ptm" / "data.csv",
    output_path=Helper.ROOT / "outputs" / "proteometools_ptm" / "proteometools_ptm_data_esm2.h5",
    dataset_name="proteometools_ptm_data_esm2",
    language_model="esm2",
    batch_size=32,
    parallel=True,
    max_workers=4,
)
