#!/usr/bin/env python3
"""
yeast_deeprtデータセットの特徴抽出設定
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.dir.dir import Dir

# 設定: ESM2モデル、バッチサイズ32、並列処理あり
yeast_deeprt_config = ExtractionRunnerConfig(
    csv_path=Dir.ROOT / "data" / "yeast_deeprt" / "data.csv",
    output_path=Dir.ROOT / "outputs" / "yeast_deeprt" / "yeast_deeprt_data_esm2.h5",
    dataset_name="yeast_deeprt_data_esm2",
    language_model="esm2",
    batch_size=32,
    parallel=True,
)
