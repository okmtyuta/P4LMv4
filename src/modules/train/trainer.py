"""
学習ループ本体（Trainer）。

- DataLoader を train/validate/evaluate に 8:1:1 比率で分割し、各フェーズを順に実行。
- RAdamScheduleFree を最適化器として想定（`optimizer` が同等 API であれば他でも可）。
"""

import torch
from schedulefree import RAdamScheduleFree

from src.modules.dataloader.dataloader import DataBatch, Dataloader
from src.modules.model.model import Model
from src.modules.protein.protein_list import ProteinList
from src.modules.train.criterion import Criterion
from src.modules.train.recorder import TrainRecorder
from src.modules.train.train_result import EpochPhaseResult, EpochSummary


class Trainer:
    """モデル・データローダ・最適化器を束ねて学習を進める。"""

    def __init__(self, model: Model, dataloader: Dataloader, optimizer: RAdamScheduleFree):
        """コンストラクタ。学習/検証/評価のデータ分割も行う。"""
        self._model = model

        self._dataloader = dataloader
        self._train_loader, self._evaluate_loader, self._validate_loader = self._dataloader.split_by_ratio(
            ratios=[0.8, 0.1, 0.1]
        )

        self._model.train()
        self._optimizer = optimizer
        self._optimizer.train()

        self._recorder = TrainRecorder()

    def _batch_predict(self, batch: DataBatch, backward: bool = False):
        """1 バッチの順伝播（必要に応じて逆伝播）。"""
        self._optimizer.zero_grad()
        input, label, protein_list = batch.use()

        output = self._model(input=input)
        if backward:
            loss = Criterion.mean_squared_error(output, label)
            loss.backward()
            self._optimizer.step()

        return label, output, protein_list

    def _epoch_predict(self, dataloader: Dataloader, backward: bool = False):
        """指定ローダーに対して 1 エポック分の推論を実行。"""
        batch_labels: list[torch.Tensor] = []
        batch_outputs: list[torch.Tensor] = []
        batch_protein_lists: list[ProteinList] = []

        for i, batch in enumerate(dataloader.batches):
            print(f"{i} of {len(dataloader.batches)}")
            _label, _output, _protein_list = self._batch_predict(batch=batch, backward=backward)
            batch_labels.append(_label)
            batch_outputs.append(_output)
            batch_protein_lists.append(_protein_list)

        label = torch.cat(batch_labels)
        output = torch.cat(batch_outputs)

        protein_list = ProteinList.join(batch_protein_lists)
        input_props = self._dataloader.input_props
        output_props = self._dataloader.output_props

        epoch_result = EpochPhaseResult(
            epoch=self._recorder.current_epoch,
            output=output,
            label=label,
            input_props=input_props,
            output_props=output_props,
            protein_list=protein_list,
        )
        epoch_result.compute_criteria()
        return epoch_result

    def train(self) -> None:
        """早期停止条件を満たすまでエポックを進める。"""
        # self.recorder.timer.start()
        while self._recorder.to_continue():
            train_epoch_result = self._epoch_predict(dataloader=self._train_loader, backward=True)
            validate_epoch_result = self._epoch_predict(dataloader=self._validate_loader)
            evaluate_epoch_result = self._epoch_predict(dataloader=self._evaluate_loader)

            epoch_summary = EpochSummary(
                epoch=self._recorder.current_epoch,
                train=train_epoch_result,
                validate=validate_epoch_result,
                evaluate=evaluate_epoch_result,
            )
            self._recorder.append_epoch_results(summary=epoch_summary)

            #     self._recorder.log()

            self._recorder.next_epoch()

        # self.recorder.timer.stop()

    # def save(self, path: str):
    #     with h5py.File(path, mode="w") as f:
    #         result_group = f.create_group("result")
    #         self._recorder.finalize(group=result_group)

    # def as_result(self):
    #     train_result: TrainResult = {
    #         "duration": self.recorder.timer.duration,
    #         "epochs": self._recorder.current_epoch,
    #         "input_props": self._dataloader.state.input_props,
    #         "output_props": self._dataloader.state.output_props,
    #         "max_accuracy_epoch": self._recorder.max_accuracy_epoch,
    #         "max_accuracy_result": self._recorder.max_accuracy_result,
    #         "train_result": {
    #             "train": self._recorder.train_result,
    #             "validate": self._recorder.validate_result,
    #             "evaluate": self._recorder.evaluate_result,
    #         },
    #     }
    #     return train_result
