#!/usr/bin/env python3
"""
atlantis_silicaデータセットの特徴抽出設定
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.dir.dir import Dir

# 設定: ESM2モデル、バッチサイズ32、並列処理あり
atlantis_silica_config = ExtractionRunnerConfig(
    csv_path=Dir.ROOT / "data" / "atlantis_silica" / "data.csv",
    output_path=Dir.ROOT / "outputs" / "atlantis_silica" / "atlantis_silica_data_esm2.h5",
    dataset_name="atlantis_silica_data_esm2",
    language_model="esm2",
    batch_size=32,
    parallel=True,
)
