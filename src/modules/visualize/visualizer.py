import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes

from src.modules.color.ColorPallet import ColorPallet
from src.modules.protein.types import ProteinPropName
from src.modules.train.train_recorder import TrainRecordedResult


class Visualizer:
    def __init__(self, train_recorded_result: TrainRecordedResult):
        self._train_recorded_result = train_recorded_result
        self._pallet = ColorPallet()

    def _filter_near_by(self, target: list[float], center: float, by: float):
        return [value for value in target if abs(value - center) > by]

    def _filter_near_of(self, target: list[float], center: float, scale: float):
        by = (max(target) - min(target)) / scale
        return self._filter_near_by(target=target, center=center, by=by)

    def save_learning_result(self, path: str, prop_name: ProteinPropName):
        figure = plt.figure(dpi=100, figsize=(14, 8))
        figure.subplots_adjust(left=0.1, right=0.75, bottom=0.2, top=0.85)

        left_axes = figure.add_subplot(1, 1, 1)
        self._render_rmse_curve(axes=left_axes, prop_name=prop_name, linestyle="-")

        right_axes = left_axes.twinx()
        self._render_pearsonr_curve(axes=right_axes, prop_name=prop_name, linestyle="--")

        left_axes_handles, left_axes_labels = left_axes.get_legend_handles_labels()
        right_axes_handles, right_axes_labels = right_axes.get_legend_handles_labels()
        left_axes.legend(
            left_axes_handles + right_axes_handles,
            left_axes_labels + right_axes_labels,
            bbox_to_anchor=(1.1, 1),
            loc="upper left",
            borderaxespad=0,
        )

        m_color = self._pallet.hex_universal_color["yellow"]

        max_accuracy = self._train_recorded_result._belle_epoch_results.validate_epoch_result.criteria["rt"]["pearsonr"]
        right_axes.axhline(y=max_accuracy, color=m_color, linestyle="--", linewidth=2)
        right_yticks = right_axes.get_yticks().tolist()
        right_yticks = self._filter_near_of(target=right_yticks, center=max_accuracy, scale=10)
        right_yticks.append(max_accuracy)
        right_axes.set_yticks(right_yticks)

        max_accuracy_epoch = self._train_recorded_result._belle_epoch
        left_axes.axvline(x=max_accuracy_epoch, color=m_color, linestyle="--", linewidth=2)
        xticks = left_axes.get_xticks().tolist()
        xticks = [tick for tick in xticks if tick >= -10]
        xticks = self._filter_near_by(target=xticks, center=max_accuracy_epoch, by=50)
        xticks.append(max_accuracy_epoch)
        left_axes.set_xticks(xticks)

        left_axes.set_xlabel("Epoch", fontsize=24, labelpad=16)
        left_axes.tick_params(axis="both", labelsize=20)
        right_axes.tick_params(axis="both", labelsize=20)

        plt.savefig(path)
        plt.close()

    def _render_rmse_curve(self, axes: Axes, prop_name: ProteinPropName, linestyle: str):
        epochs = self._train_recorded_result._epoch_results_list.epochs

        root_mean_squared_errors = self._train_recorded_result._epoch_results_list.get_train_criteria_curve(
            prop_name=prop_name, criteria_name="root_mean_squared_error"
        )
        color = self._pallet.consume_current_color()
        axes.plot(epochs, root_mean_squared_errors, color=color, label=f"Train {prop_name} RMSE", linestyle=linestyle)

        root_mean_squared_errors = self._train_recorded_result._epoch_results_list.get_validate_criteria_curve(
            prop_name=prop_name, criteria_name="root_mean_squared_error"
        )
        color = self._pallet.consume_current_color()
        axes.plot(
            epochs, root_mean_squared_errors, color=color, label=f"Validate {prop_name} RMSE", linestyle=linestyle
        )

        root_mean_squared_errors = self._train_recorded_result._epoch_results_list.get_evaluate_criteria_curve(
            prop_name=prop_name, criteria_name="root_mean_squared_error"
        )
        color = self._pallet.consume_current_color()
        axes.plot(
            epochs, root_mean_squared_errors, color=color, label=f"Evaluate {prop_name} RMSE", linestyle=linestyle
        )

    def _render_pearsonr_curve(self, axes: Axes, prop_name: ProteinPropName, linestyle: str):
        epochs = self._train_recorded_result._epoch_results_list.epochs

        root_mean_squared_errors = self._train_recorded_result._epoch_results_list.get_train_criteria_curve(
            prop_name=prop_name, criteria_name="pearsonr"
        )
        color = self._pallet.consume_current_color()
        axes.plot(
            epochs, root_mean_squared_errors, color=color, label=f"Train {prop_name} Pearsonr", linestyle=linestyle
        )

        root_mean_squared_errors = self._train_recorded_result._epoch_results_list.get_validate_criteria_curve(
            prop_name=prop_name, criteria_name="pearsonr"
        )
        color = self._pallet.consume_current_color()
        axes.plot(
            epochs, root_mean_squared_errors, color=color, label=f"Validate {prop_name} Pearsonr", linestyle=linestyle
        )

        root_mean_squared_errors = self._train_recorded_result._epoch_results_list.get_evaluate_criteria_curve(
            prop_name=prop_name, criteria_name="pearsonr"
        )
        color = self._pallet.consume_current_color()
        axes.plot(
            epochs, root_mean_squared_errors, color=color, label=f"Evaluate {prop_name} Pearsonr", linestyle=linestyle
        )

    def save_belle_epoch_scatter(self, path: str, prop_name: ProteinPropName):
        figure = plt.figure(dpi=100, figsize=(14, 10))
        figure.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)

        axes = figure.add_subplot(1, 1, 1)

        label = self._train_recorded_result._belle_epoch_results.evaluate_epoch_result.label_by_prop["rt"]
        output = self._train_recorded_result._belle_epoch_results.evaluate_epoch_result.output_by_prop["rt"]

        xy_min = torch.stack([label, output]).min().item() - 0.05
        xy_max = torch.stack([label, output]).max().item() + 0.05

        axes.scatter(
            label,
            output,
            color=self._pallet.hex_universal_color["red"],
            s=2,
        )

        axes.plot([xy_min, xy_max], [xy_min, xy_max], color="black")

        axes.set_xlabel(f"Observed {prop_name} value", fontsize=24, labelpad=16)
        axes.set_ylabel(f"Predicted {prop_name} value", fontsize=24, labelpad=16)
        axes.tick_params(axis="both", labelsize=20)

        plt.savefig(path)
