#!/usr/bin/env python3
"""
swath_libraryデータセットの特徴抽出設定（ESM1b）
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.helper.helper import Helper

# 設定: ESM1bモデル、バッチサイズ32、並列処理あり
swath_library_config = ExtractionRunnerConfig(
    csv_path=Helper.ROOT / "data" / "swath_library" / "data.csv",
    output_path=Helper.ROOT / "outputs" / "swath_library" / "swath_library_data_esm1b.h5",
    dataset_name="swath_library_data_esm1b",
    language_model="esm1b",
    batch_size=32,
    parallel=True,
)
