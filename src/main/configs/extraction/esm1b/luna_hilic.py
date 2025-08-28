#!/usr/bin/env python3
"""
luna_hilicデータセットの特徴抽出設定（ESM1b）
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.helper.helper import Helper

# 設定: ESM1bモデル、バッチサイズ32、並列処理あり
luna_hilic_config = ExtractionRunnerConfig(
    csv_path=Helper.ROOT / "data" / "luna_hilic" / "data.csv",
    output_path=Helper.ROOT / "outputs" / "luna_hilic" / "luna_hilic_data_esm1b.h5",
    dataset_name="luna_hilic_data_esm1b",
    language_model="esm1b",
    batch_size=32,
    parallel=True,
)
