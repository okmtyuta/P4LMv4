#!/usr/bin/env python3
"""
dia_hfデータセットの特徴抽出設定（ESM1b）
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.helper.helper import Helper

# 設定: ESM1bモデル、バッチサイズ32、並列処理あり
dia_hf_config = ExtractionRunnerConfig(
    csv_path=Helper.ROOT / "data" / "dia_hf" / "data.csv",
    output_path=Helper.ROOT / "outputs" / "dia_hf" / "dia_hf_data_esm1b.h5",
    dataset_name="dia_hf_data_esm1b",
    language_model="esm1b",
    batch_size=32,
    parallel=False,
)
