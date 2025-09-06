#!/usr/bin/env python3
"""
swath_libraryデータセットの特徴抽出設定
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.helper.helper import Helper

# 設定: ESM2モデル、バッチサイズ32、並列処理あり
swath_library_config = ExtractionRunnerConfig(
    csv_path=Helper.ROOT / "data" / "swath_library" / "data.csv",
    output_path=Helper.ROOT / "outputs" / "swath_library" / "swath_library_data_esm2.h5",
    dataset_name="swath_library_data_esm2",
    language_model="esm2",
    batch_size=32,
    parallel=True,
)
