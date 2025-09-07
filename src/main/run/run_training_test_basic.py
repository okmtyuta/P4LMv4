#!/usr/bin/env python3
"""
シンプルな training config を Manager で実行するスクリプト。
"""

from src.main.configs.training.test_basic import test_training_config
from src.main.utils.manager import Manager
from src.main.utils.runner import RunnerConfig


def main() -> None:
    configs: list[RunnerConfig] = [test_training_config]
    manager = Manager(configs)
    manager.run_all()

    if manager.has_errors():
        print(f"{len(manager.get_errors())} runs failed.")
    else:
        print("All training runs succeeded.")


if __name__ == "__main__":
    main()
