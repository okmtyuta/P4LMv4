#!/usr/bin/env python3
"""
arabidopsisデータセットの特徴抽出設定
"""

from src.main.extraction import ExtractionRunnerConfig
from src.modules.helper.helper import Helper

test_config = ExtractionRunnerConfig(
    csv_path=Helper.ROOT / "data" / "test" / "data.csv",
    output_path=Helper.ROOT / "outputs" / "test" / "test_data_esm2.h5",
    dataset_name="test",
    language_model="esm2",
    batch_size=32,
    parallel=True,
    max_workers=3,
)
