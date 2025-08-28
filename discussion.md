# 位置エンコーディングと集約（Aggregator）の設計メモ

本ドキュメントは、タンパク質 RT（保持時間）予測を主眼に、DataProcess 形式での位置エンコーディング（PE）と集約（Aggregator）の設計・選択肢・実装・運用指針をまとめたものです。将来の実験再現や拡張のための備忘です。

## パイプライン設計
- 構成: `Initializer → Position Encoding → Aggregator → Model`
- データフロー: 各 `DataProcess` は `protein.get_processed()` を読み、変換後を `set_processed()` で書き戻す逐次合成。複数の PE を重ねる場合も順に合成されます。
- 次元整合: 一部の PE/集約は出力次元を変更します（例: 双方向系で 2D、区間連結で K×D）。後段モデルの `input_dim` を合わせること。

## 位置エンコーディング（PE）の選択肢

### 固定 PE（非学習）
- Sinusoidal（標準/逆順/双方向）
  - 特徴: 周波数を固定。長さに対する滑らかで安定な符号化。
  - 用途: 強い先験を入れたいベースラインに向く。
- RoPE（Rotary Positional Embedding）
  - 特徴: 偶数ペアごとに2D回転を適用。相対的な幾何を暗黙に表現。
  - 用途: 自己注意など後段で相互作用を使う構成と相性が良い。

### 学習可能 PE（本レポ内の代表）
- LearnableAbsolutePositionalScaler（乗算ゲート・スカラー）
  - 概要: 位置ごとのスカラー `g[p]=1+s[p]` を学習し `x[p]*g[p]`。
  - 利点: 平均集約前に位置の寄与度を直接調整。初期は恒等（安定）。
  - 注意: `max_length` 超は末尾の係数を再利用。
- LearnableAbsolutePositionalAdder（加算バイアス・スカラー）
  - 概要: `x[p] + b[p]` を学習。回帰のベース補正に有効な場合。
  - 注意: スケール感に敏感。正規化と併用検討。
- LearnableFourierPositionalEncoder（LFPE, スカラーゲート）
  - 概要: 学習可能周波数と位相の Fourier 基底 Φ(p) を作りスカラーゲート `1 + s·Φ(p)·v` を適用。
  - 利点: 少パラメータで滑らかな位置依存性、長さ外挿に強め。
- LearnableRoPEPositionalEncoder（周波数学習）
  - 概要: RoPE の周波数 `inv_freq` を学習（対数空間で保持し正値制約）。
  - 用途: 相対的パターンを保ちつつデータに合わせて周波数を調整。

### その他の拡張案（必要時に実装）
- 次元別（位置×次元）版／低ランク分解版（表現力↑、過学習・メモリ↑）
- 位置 MLP 生成（正規化位置→MLP→ゲート/バイアス）
- RBF/スプライン基底展開（局所性の明示、基底数の選定が必要）

## RT 予測での推奨順位（経験則）
1) LearnableAbsolutePositionalScaler（乗算スカラー）
   - 位置寄与の直接学習が効く。末端効果の強調に向く。
2) LearnableFourierPositionalEncoder（LFPE）
   - 平滑な重み付け・外挿性に優れる。データ長分布が広いと有利。
3) LearnableAbsolutePositionalAdder／次元別/低ランク版
   - 強力だがパラメータ増。データ量と正則化設計が鍵。
4) LearnableRoPE
   - 自己注意など相互作用層と組むと真価。平均主体の集約では寄与が限定的なことも。

## 集約（Aggregator）の選択肢

### 学習不要（非学習）
- Mean（既存）: `mean`。安定・基準線。
- EndsSegmentMeanAggregator（実装済）
  - 先頭 N・中央・末尾 M の平均を連結して `(3D,)` を出力。
  - 末端効果を明示的に保持。短鎖にも安全に切り分け（空区間は0）。
- SegmentMean（K等分・連結）: 大域形状を保持。要実装。
- LogSumExp（τ固定）: 平均〜最大の連続内挿。要実装。
- Moment（mean+std 連結）: 分散情報を保持。要実装。

### 学習あり（本レポ内の実装）
- WeightedMeanAggregator（実装済）
  - 重み `w = softmax((X@v)/τ + b[pos])` で `(L,D)`→`(D)` に集約。
  - 初期は平均に近い挙動（v≈0, b=0, τ≈10）。
- AttentionPooling（学習クエリ）: Kクエリで多様な要約（出力 `K×D`）。要実装。

## 実装済みモジュール一覧（抜粋）
- 位置エンコーディング
  - `SinusoidalPositionalEncoder`（固定, 正/逆/双方向）
  - `RoPEPositionalEncoder`（固定, 正/逆/双方向）
  - `LearnableAbsolutePositionalScaler`（学習, 乗算スカラー）
  - `LearnableAbsolutePositionalAdder`（学習, 加算スカラー）
  - `LearnableFourierPositionalEncoder`（学習, スカラーゲート）
  - `LearnableRoPEPositionalEncoder`（学習, 周波数学習, 正/逆/双方向）
- 集約
  - `Aggregator("mean")`（既存）
  - `WeightedMeanAggregator`（学習）
  - `EndsSegmentMeanAggregator`（非学習, 3分割連結）

## 使い方パターン（サンプル）
```python
from src.modules.data_process.initializer import Initializer
from src.modules.data_process.positional_encoder import (
    LearnableFourierPositionalEncoder,
    LearnableRoPEPositionalEncoder,
)
from src.modules.data_process.aggregator import (
    WeightedMeanAggregator,
    EndsSegmentMeanAggregator,
)
from src.modules.data_process.data_process_list import DataProcessList

# 例1: LFPE + WeightedMean（出力次元 D）
pe = LearnableFourierPositionalEncoder(num_bases=64, min_period=10.0, max_period=1000.0, projection_scale=0.1)
agg = WeightedMeanAggregator(dim=1280, max_length=64)
process_list = DataProcessList([Initializer(), pe, agg])

# 例2: LearnableRoPE（正） + 端部/中央/末尾平均（出力 3D）
pe2 = LearnableRoPEPositionalEncoder(dim=1280, theta_base=10000.0)
agg2 = EndsSegmentMeanAggregator(head_len=8, tail_len=8)
process_list2 = DataProcessList([Initializer(), pe2, agg2])
```

オプティマイザ例（PE/集約の学習率を分離）:
```python
optimizer = RAdamScheduleFree([
  {"params": model.parameters(), "lr": 1e-3},
  {"params": pe.parameters(),     "lr": 5e-4},
  {"params": agg.parameters(),    "lr": 5e-4},
])
```

## ハイパーパラメータ指針
- `max_length`: 配列長の上限以上に。超過は末尾再利用（テーブル型）。外挿性が必要なら LFPE/MLP/RBF 系が向く。
- LFPE: `num_bases=32〜64`, `min_period≈5〜10`, `max_period≈200〜1000`, `projection_scale=0.05〜0.2`。
- LearnableRoPE: `theta_base=10000` 初期化が安定。双方向は出力 2D→モデル次元調整。
- WeightedMean: 温度 `τ` は初期大きめ（≈10）。`v` と位置バイアス `b` は 0 近傍初期化。
- EndsSegmentMean: `head_len`/`tail_len` は 4〜16 程度から。実験で感度分析。
- 学習率: モデル本体より PE/集約は低め。`2e-4〜5e-4` のレンジを基準に調整。
- 正則化: ゲート系は小さめ `weight_decay`（1e-6〜1e-5）。過学習兆候があれば強める。

## 評価設計
- 指標: MAE / RMSE / Spearman / Pearson。
- 分割: 長さ分布で層化。シードを変えて 3 反復平均。
- アブレーション: no-PE, 固定 Sinusoidal, 各学習PE, 各集約（mean/LSE/Ends/Weighted）。
- 外挿テスト: 訓練=短鎖中心／評価=長鎖多め の条件で LFPE/Weighted の優位性を確認。

## よくある落とし穴と回避策
- 次元不一致: 双方向 PE や区間連結 Aggregator は出力次元が変わる。モデル `input_dim` を必ず更新。
- 長さ外挿: テーブル型 PE は `max_length` 超で性能劣化しやすい。LFPE/MLP/RBF を検討。
- スケール感: 加算系は分布の平均をずらす。標準化/LayerNorm/ゲート系の併用で安定化。
- 勾配の疎さ: `max` 系は非滑らか。代替として LSE（温度）や soft-topk 近似を検討。
- 末端/中央のデータ依存性: EndsSegment の `head_len`/`tail_len` はデータ特性に依存。感度分析を行う。

## 今後の拡張アイデア
- AttentionPooling（1〜K学習クエリ）/ Set Transformer PMA 互換の集約。
- SegmentMean（K等分）/ MomentAggregator（mean+std）/ LogSumExpAggregator（τ固定 or 学習）。
- 低ランク/次元別の Absolute PE、位置 RBF 基底による外挿性の改善。
- FFT/DCT 低周波プーリングで時系列の大域形状を特徴量化。

---
実装済みの主なクラスとファイル:
- `src/modules/data_process/learnable_absolute_positional_encoder.py`
- `src/modules/data_process/learnable_fourier_positional_encoder.py`
- `src/modules/data_process/learnable_rope_positional_encoder.py`
- `src/modules/data_process/weighted_mean_aggregator.py`
- `src/modules/data_process/ends_segment_mean_aggregator.py`
