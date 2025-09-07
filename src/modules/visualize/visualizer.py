"""
学習結果の可視化ユーティリティ。

- `read.py` の `LoadedTrainResult` を受け取って学習曲線・散布図を保存する。
- 余計な依存を持たず、必要最小限の API のみ提供する。
"""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes

from src.modules.protein.protein_types import ProteinPropName
from src.modules.train.loaded_train_result import LoadedTrainResult
from src.modules.train.train_result import EpochPhaseResult, EpochSummary


class Visualizer:
    """`LoadedTrainResult` を用いた簡易ビジュアライザ。"""

    def __init__(self, loaded: LoadedTrainResult) -> None:
        """読み込み済みの学習結果を受け取って初期化する。"""
        self._res = loaded
        # 使い回す色（train/validate/evaluate で色相を固定）
        self._color_train = "tab:blue"
        self._color_validate = "tab:orange"
        self._color_evaluate = "tab:green"
        self._marker_belle = "tab:red"

    def _epochs(self) -> list[int]:
        """history のエポック番号列を返す。"""
        return [s.epoch for s in self._res.history]

    def _curve_from(
        self,
        accessor: Callable[[EpochSummary], EpochPhaseResult],
        prop_name: ProteinPropName,
        metric: str,
    ) -> list[float]:
        """指定アクセサで取り出したフェーズの指標系列を返す（型安全）。

        - criteria が未格納で output/label が存在する場合は、その場で計算して利用する。
        - どちらも無い場合は `nan` を格納する。
        """
        values: list[float] = []
        for s in self._res.history:
            phase_obj = accessor(s)
            if phase_obj.criteria is None:
                if phase_obj.output is not None and phase_obj.label is not None:
                    phase_obj.compute_criteria()
                else:
                    values.append(float("nan"))
                    continue
            crit = phase_obj.criteria.get(prop_name)
            if crit is None:
                values.append(float("nan"))
            else:
                values.append(float(crit.get(metric, float("nan"))))
        return values

    def _curve_train(self, prop_name: ProteinPropName, metric: str) -> list[float]:
        """train フェーズの指標系列を返す。"""
        return self._curve_from(lambda s: s.train, prop_name, metric)

    def _curve_validate(self, prop_name: ProteinPropName, metric: str) -> list[float]:
        """validate フェーズの指標系列を返す。"""
        return self._curve_from(lambda s: s.validate, prop_name, metric)

    def _curve_evaluate(self, prop_name: ProteinPropName, metric: str) -> list[float]:
        """evaluate フェーズの指標系列を返す。"""
        return self._curve_from(lambda s: s.evaluate, prop_name, metric)

    def save_learning_result(self, path: str, prop_name: ProteinPropName) -> None:
        """学習曲線（RMSE と Pearson）を同図に描画して保存する。"""
        epochs = self._epochs()
        fig = plt.figure(dpi=100, figsize=(14, 8))
        fig.subplots_adjust(left=0.1, right=0.75, bottom=0.2, top=0.85)

        ax_left = fig.add_subplot(1, 1, 1)
        ax_right = ax_left.twinx()

        # RMSE 曲線（左軸）
        rmse_train = self._curve_train(prop_name, "root_mean_squared_error")
        rmse_val = self._curve_validate(prop_name, "root_mean_squared_error")
        rmse_eval = self._curve_evaluate(prop_name, "root_mean_squared_error")
        ax_left.plot(epochs, rmse_train, color=self._color_train, label=f"Train {prop_name} RMSE", linestyle="-")
        ax_left.plot(epochs, rmse_val, color=self._color_validate, label=f"Validate {prop_name} RMSE", linestyle="-")
        ax_left.plot(epochs, rmse_eval, color=self._color_evaluate, label=f"Evaluate {prop_name} RMSE", linestyle="-")

        # Pearson 曲線（右軸）
        pr_train = self._curve_train(prop_name, "pearsonr")
        pr_val = self._curve_validate(prop_name, "pearsonr")
        pr_eval = self._curve_evaluate(prop_name, "pearsonr")
        ax_right.plot(epochs, pr_train, color=self._color_train, label=f"Train {prop_name} r", linestyle="--")
        ax_right.plot(epochs, pr_val, color=self._color_validate, label=f"Validate {prop_name} r", linestyle="--")
        ax_right.plot(epochs, pr_eval, color=self._color_evaluate, label=f"Evaluate {prop_name} r", linestyle="--")

        # Belle の補助線
        belle_epoch = int(self._res.belle_epoch)
        ax_left.axvline(x=belle_epoch, color=self._marker_belle, linestyle=":", linewidth=2)
        try:
            belle_acc = float(self._res.belle.validate.accuracy)
            ax_right.axhline(y=belle_acc, color=self._marker_belle, linestyle=":", linewidth=2)
        except Exception:
            pass

        # 凡例（左右の軸の凡例を結合）
        h1, l1 = ax_left.get_legend_handles_labels()
        h2, l2 = ax_right.get_legend_handles_labels()
        # 判例をさらに右へオフセットして縦軸の数値と重ならないようにする
        ax_left.legend(
            h1 + h2,
            l1 + l2,
            bbox_to_anchor=(1.18, 1.0),  # 以前より右へオフセット
            loc="upper left",
            borderaxespad=0,
        )

        ax_left.set_xlabel("Epoch")
        ax_left.set_ylabel("RMSE")
        ax_right.set_ylabel("Pearson r")

        # 判例を右側へずらしたため、はみ出しを防ぐ
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def save_belle_epoch_scatter(self, path: str, prop_name: ProteinPropName) -> None:
        """Belle エポックの評価フェーズにおける予測 vs 観測の散布図を保存する。"""
        phase = self._res.belle.evaluate
        if phase.output is None or phase.label is None:
            raise RuntimeError("Belle evaluate phase tensors are unavailable (forgotten)")

        y = phase.output_by_prop[prop_name]
        x = phase.label_by_prop[prop_name]

        x = x.detach().to(torch.float32)
        y = y.detach().to(torch.float32)

        xy_min = torch.stack([x, y]).min().item()
        xy_max = torch.stack([x, y]).max().item()
        pad = 0.05 * (xy_max - xy_min if xy_max > xy_min else 1.0)

        fig = plt.figure(dpi=100, figsize=(8, 8))
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
        ax: Axes = fig.add_subplot(1, 1, 1)

        ax.scatter(x, y, color=self._marker_belle, s=4, alpha=0.7)
        ax.plot([xy_min - pad, xy_max + pad], [xy_min - pad, xy_max + pad], color="black", linewidth=1)

        ax.set_xlabel(f"Observed {prop_name}")
        ax.set_ylabel(f"Predicted {prop_name}")

        plt.savefig(path)
        plt.close(fig)
