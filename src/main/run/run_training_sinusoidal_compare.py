#!/usr/bin/env python3
"""
Additive/Multiplicative サイン波PEの比較実験を一括実行するスクリプト。
"""

from src.main.configs.training.sinusoidal_additive import additive_config
from src.main.configs.training.sinusoidal_multiplicative import multiplicative_config
from src.main.utils.manager import Manager
from src.main.utils.runner import RunnerConfig


def main() -> None:
    """2つの設定を順次実行する。"""
    configs: list[RunnerConfig] = [additive_config, multiplicative_config]
    manager = Manager(configs)
    manager.run_all()

    if manager.has_errors():
        print(f"{len(manager.get_errors())} runs failed.")
    else:
        print("All sinusoidal comparison runs succeeded.")


if __name__ == "__main__":
    main()
