#!/usr/bin/env python3
"""
yeast_1hデータセットの特徴抽出設定（ESM1b）
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.dir.dir import Dir

# 設定: ESM1bモデル、バッチサイズ32、並列処理あり
yeast_1h_config = ExtractionRunnerConfig(
    csv_path=Dir.ROOT / "data" / "yeast_1h" / "data.csv",
    output_path=Dir.ROOT / "outputs" / "yeast_1h" / "yeast_1h_data_esm1b.h5",
    dataset_name="yeast_1h_data_esm1b",
    language_model="esm1b",
    batch_size=32,
    parallel=True,
)
