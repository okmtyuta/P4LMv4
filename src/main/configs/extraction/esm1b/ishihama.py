#!/usr/bin/env python3
"""
ishihamaデータセットの特徴抽出設定（ESM1b）
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.dir.dir import Dir

# 設定: ESM1bモデル、バッチサイズ32、並列処理あり
ishihama_config = ExtractionRunnerConfig(
    csv_path=Dir.ROOT / "data" / "ishihama" / "data.csv",
    output_path=Dir.ROOT / "outputs" / "ishihama" / "ishihama_data_esm1b.h5",
    dataset_name="ishihama_data_esm1b",
    language_model="esm1b",
    batch_size=32,
    parallel=True,
)
