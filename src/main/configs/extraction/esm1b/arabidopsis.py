#!/usr/bin/env python3
"""
arabidopsisデータセットの特徴抽出設定（ESM1b）
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.dir.dir import Dir

# 設定: ESM1bモデル、バッチサイズ32、並列処理あり
arabidopsis_config = ExtractionRunnerConfig(
    csv_path=Dir.ROOT / "data" / "arabidopsis" / "data.csv",
    output_path=Dir.ROOT / "outputs" / "arabidopsis" / "arabidopsis_data_esm1b.h5",
    dataset_name="arabidopsis_data_esm1b",
    language_model="esm1b",
    batch_size=32,
    parallel=True,
)
