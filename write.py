#!/usr/bin/env python3
"""
Visualizer を用いて学習曲線と散布図を出力するスクリプト。

データ: outputs/experiments/plasma_lumos_1h/sinusoidal/additive/train_result.h5
出力: 同ディレクトリに PNG を保存（learning_curve.png, belle_scatter.png）
"""

from __future__ import annotations

from pathlib import Path


from src.modules.train.loaded_train_result import LoadedTrainResult
from src.modules.helper.helper import Helper
from src.modules.visualize.visualizer import Visualizer


def main() -> None:
    """学習結果を読み込み、学習曲線と散布図を保存する。"""
    result_path = (
        Helper.ROOT
        / "outputs"
        / "experiments"
        / "plasma_lumos_1h"
        / "sinusoidal"
        / "multiplicative"
        / "train_result.h5"
    )

    loaded = LoadedTrainResult.load(str(result_path))
    viz = Visualizer(loaded)

    out_dir: Path = result_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    curve_path = out_dir / "learning_curve2.png"
    scatter_path = out_dir / "belle_scatter2.png"

    # 本プロジェクトでは出力プロパティ名は "rt" を使用
    viz.save_learning_result(path=str(curve_path), prop_name="rt")
    viz.save_belle_epoch_scatter(path=str(scatter_path), prop_name="rt")

    print(f"Saved: {curve_path}")
    print(f"Saved: {scatter_path}")


if __name__ == "__main__":
    main()

