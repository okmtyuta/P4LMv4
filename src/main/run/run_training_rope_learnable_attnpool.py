#!/usr/bin/env python3
"""
Learnable RoPE + RoPE整合注意プーリング学習の実行スクリプト。
"""

from src.main.configs.training.rope_learnable_attnpool import rope_learnable_attnpool_config
from src.main.utils.manager import Manager
from src.main.utils.runner import RunnerConfig


def main() -> None:
    configs: list[RunnerConfig] = [rope_learnable_attnpool_config]
    manager = Manager(configs)
    manager.run_all()

    if manager.has_errors():
        print(f"{len(manager.get_errors())} runs failed.")
    else:
        print("RoPE learnable + attention pooling run succeeded.")


if __name__ == "__main__":
    main()
