#!/usr/bin/env python3
"""
swath_libraryデータセットの特徴抽出設定（ESM1b）
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.dir.dir import Dir

# 設定: ESM1bモデル、バッチサイズ32、並列処理あり
swath_library_config = ExtractionRunnerConfig(
    csv_path=Dir.ROOT / "data" / "swath_library" / "data.csv",
    output_path=Dir.ROOT / "outputs" / "swath_library" / "swath_library_data_esm1b.h5",
    dataset_name="swath_library_data_esm1b",
    language_model="esm1b",
    batch_size=32,
    parallel=True,
)
