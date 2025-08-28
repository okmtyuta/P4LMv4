#!/usr/bin/env python3
"""
luna_silicaデータセットの特徴抽出設定（ESM1b）
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.dir.dir import Dir

# 設定: ESM1bモデル、バッチサイズ32、並列処理あり
luna_silica_config = ExtractionRunnerConfig(
    csv_path=Dir.ROOT / "data" / "luna_silica" / "data.csv",
    output_path=Dir.ROOT / "outputs" / "luna_silica" / "luna_silica_data_esm1b.h5",
    dataset_name="luna_silica_data_esm1b",
    language_model="esm1b",
    batch_size=32,
    parallel=True,
)
