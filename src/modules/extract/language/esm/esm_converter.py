"""
fair-esm の事前学習モデルを介して配列表現を取得するシンプルな変換器。

- 入力: 文字列配列のリスト。
- 出力: 各配列に対応する (L, D) の表現テンソル（CLS/SEP 除去済み）。
"""

import esm
import torch

from src.modules.extract.language.esm.esm_types import (
    ESMModelName,
    ESMModelResult,
)


class ESMConverter:
    """ESM モデルをラップし、配列→表現テンソルへ変換する。"""

    def __init__(self, model_name: ESMModelName):
        """モデル名を受け取り、モデルとアルファベットを用意する。"""
        super().__init__()
        self._model_name = model_name
        self._model, self._alphabet = self._get_model_and_alphabet()
        self._batch_converter = self._alphabet.get_batch_converter()

        # CPU で実行（必要なら外部で .to(device) する）
        self._model = self._model
        self._model.eval()

    def __call__(self, seqs: list[str]) -> list[torch.Tensor]:
        """配列リストを受け取り、各要素の (L, D) 表現を返す。"""
        batch_tokens = self._batch_converter([(seq, seq) for seq in seqs])[2]

        # CPUで実行するように設定
        batch_tokens = batch_tokens
        batch_lens = (batch_tokens != self._alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results: ESMModelResult = self._model(
                batch_tokens,
                repr_layers=[33],
                return_contacts=False,
            )
        token_representations: torch.Tensor = results["representations"][33]

        sequence_representations: list[torch.Tensor] = []
        for i, tokens_len in enumerate(batch_lens):
            representation = token_representations[i, 1 : tokens_len - 1]
            sequence_representations.append(representation)  # noqa: E203
        return sequence_representations

    def _get_model_and_alphabet(self):
        """モデルとアルファベット（Tokenizer）を取得。"""
        return self._get_model_alphabet()

    def _get_model_alphabet(self):
        """モデル名に応じて適切な事前学習モデルを返す。"""
        if self._model_name == "esm2":
            return esm.pretrained.esm2_t33_650M_UR50D()
        if self._model_name == "esm1b":
            return esm.pretrained.esm1b_t33_650M_UR50S()
        else:
            raise Exception()
