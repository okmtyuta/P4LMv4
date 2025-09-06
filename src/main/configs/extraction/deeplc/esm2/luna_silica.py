#!/usr/bin/env python3
"""
luna_silicaデータセットの特徴抽出設定
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.helper.helper import Helper

# 設定: ESM2モデル、バッチサイズ32、並列処理あり
luna_silica_config = ExtractionRunnerConfig(
    csv_path=Helper.ROOT / "data" / "luna_silica" / "data.csv",
    output_path=Helper.ROOT / "outputs" / "luna_silica" / "luna_silica_data_esm2.h5",
    dataset_name="luna_silica_data_esm2",
    language_model="esm2",
    batch_size=32,
    parallel=True,
)
