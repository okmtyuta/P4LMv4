#!/usr/bin/env python3
"""
esm1b用の全Extraction configをManagerで順次実行するスクリプト
（動的探索は用いず、明示importで収集）
"""

from typing import List

from src.main.configs.extraction.deeplc.esm1b import (
    arabidopsis_config,
    atlantis_silica_config,
    dia_hf_config,
    hela_deeprt_config,
    hela_hf_config,
    hela_lumos_1h_config,
    hela_lumos_2h_config,
    ishihama_config,
    luna_hilic_config,
    luna_silica_config,
    pancreas_config,
    plasma_lumos_1h_config,
    plasma_lumos_2h_config,
    proteometools_config,
    proteometools_ptm_config,
    scx_config,
    swath_library_config,
    xbridge_config,
    yeast_1h_config,
    yeast_2h_config,
    yeast_deeprt_config,
)
from src.main.utils.manager import Manager
from src.main.utils.runner import RunnerConfig

ALL_CONFIGS: List[RunnerConfig] = [
    arabidopsis_config,
    atlantis_silica_config,
    dia_hf_config,
    hela_deeprt_config,
    hela_hf_config,
    hela_lumos_1h_config,
    hela_lumos_2h_config,
    ishihama_config,
    luna_hilic_config,
    luna_silica_config,
    pancreas_config,
    plasma_lumos_1h_config,
    plasma_lumos_2h_config,
    proteometools_config,
    proteometools_ptm_config,
    scx_config,
    swath_library_config,
    xbridge_config,
    yeast_1h_config,
    yeast_2h_config,
    yeast_deeprt_config,
]


def main() -> None:
    configs = ALL_CONFIGS
    print(f"Found {len(configs)} configs under deeplc.esm1b")
    manager = Manager(configs)
    manager.run_all()
    if manager.has_errors():
        print(f"{len(manager.get_errors())} runs failed.")
    else:
        print("All runs succeeded.")


if __name__ == "__main__":
    main()
