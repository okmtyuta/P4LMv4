# Learnable RoPE Positional Encoder 概要

本ドキュメントは、`src/modules/data_process/positional_encoder/learnable_rope_positional_encoder.py` に実装された Learnable Rotary Positional Embedding (以下 Learnable RoPE; LRoPE) の数式定義・性質・実装要点をまとめたものです。標準の RoPE（固定周波数）との違いも併記します。

## 記号と形状
- 入力表現: 行列 `X ∈ R^{L×D}`（長さ `L`、次元 `D`）。
- 出力表現: `Y ∈ R^{L×D}`（双方向版は `Y ∈ R^{L×2D}`）。
- ペア数: `H = ⌊D/2⌋`（偶数・奇数次元での扱いは後述）。
- 位置インデックス: `p_l`（1 始まり）。
- 学習パラメータ（周波数の逆数ベクトルの対数）: `β ∈ R^H`。
- 周波数（正値制約付き）: `ω = exp(β) ∈ R_{>0}^H`。
- 初期化: `ω_k^{(0)} = θ_base^{−2k/D}`（`k = 0,1,...,H−1`）。

実装では `β = log(ω)` を `nn.Parameter` として保持し、正値制約と数値安定性を確保します。

## 位置インデックス `p_l`
- 通常（forward）: `p_l = l`（`l = 1..L`）。
- 逆順（reversed）: `p_l = L − l + 1`（`l = 1..L`）。

Bidirectional 版では forward と reversed をそれぞれ適用し、最後の次元で連結します（`dim_factor = 2`）。

## 角度と回転の定義（偶数次元ペアごと）
各ペア `k`（`k = 0..H−1`）に対し、角度を

- `φ_{l,k} = p_l · ω_k` で定義します。

ペア `(2k, 2k+1)` に対応する 2 次元ベクトルに、回転行列

- `R(φ) = [[cos φ, −sin φ], [sin φ, cos φ]]`

を適用します。すなわち、`x_{l,2k}, x_{l,2k+1}` を入力として、

- `y_{l,2k}   =  x_{l,2k} · cos φ_{l,k} − x_{l,2k+1} · sin φ_{l,k}`
- `y_{l,2k+1} =  x_{l,2k} · sin φ_{l,k} + x_{l,2k+1} · cos φ_{l,k}`

を計算します。`D` が奇数のとき、末尾チャネル `D−1` は未回転でそのまま保持します。

## 標準 RoPE との違い
- 標準 RoPE（固定周波数）: `ω_k = θ_base^{−2k/D}` を固定で用います。
- Learnable RoPE: `ω_k` を学習可能にし、`β_k = log ω_k` をパラメータ化します。

よって LRoPE はデータ分布に適応して各周波数スケールを自動調整でき、モデル性能や収束特性の改善が期待できます。`θ_base` は初期化にのみ使用されます。

## 勾配の流れ（概要）
`ω_k = exp(β_k)` より `∂ω_k/∂β_k = ω_k`。また `φ_{l,k} = p_l ω_k` なので `∂φ_{l,k}/∂β_k = p_l · ω_k`。例えば `y_{l,2k}` について、

- `∂y_{l,2k}/∂φ_{l,k} = −x_{l,2k} · sin φ_{l,k} − x_{l,2k+1} · cos φ_{l,k}`
- よって `∂y_{l,2k}/∂β_k = (∂y_{l,2k}/∂φ_{l,k}) · (p_l · ω_k)`

同様に `y_{l,2k+1}` も導出できます。これにより位置 `p_l` が大きいほど当該ペア角度への感度が増し、勾配が適切に流れます。

## 性質と実装上の注意
- ノルム保存: 各ペアの回転は直交変換であり、`∥(x_{l,2k}, x_{l,2k+1})∥₂` を保存します（奇数末尾は不変）。
- 計算量: `O(L·D)`、追加メモリは `cos/sin ∈ R^{L×H}` 程度。
- 安定性: `β` でのパラメータ化により `ω > 0` を保証しつつ指数的スケールを扱えます。
- 角度の再計算: `ω` が学習で更新されるため、`cos/sin` は毎ステップ再計算されます（標準 RoPE 実装は固定周波数のためキャッシュが可能）。
- 位置は 1 始まり: 実装では `p_l` を 1..L（逆順は L..1）で定義しています。

## 変種
- `LearnableRoPEPositionalEncoder`: forward のみ適用。
- `ReversedLearnableRoPEPositionalEncoder`: reversed のみ適用。
- `BidirectionalLearnableRoPEPositionalEncoder`: forward と reversed を適用し、`[Y_fwd, Y_rev]` を最後の次元で連結（出力次元は `2D`）。

## 実装対応（抜粋）
- ファイル: `src/modules/data_process/positional_encoder/learnable_rope_positional_encoder.py`
- 主要パラメータ: `self._log_inv_freq (β)`、初期値 `log(θ_base^{−2m/D})`。
- 角度計算: `angles = p[:,None] * exp(β)[None,:]`、`cos = cos(angles)`, `sin = sin(angles)`。
- 回転適用: 偶数・奇数インデックスをストライドで分割し、上記の 2×2 回転をブロードキャストで適用。
- 形状: 入出力は `(L, D)`（双方向は `(L, 2D)`）。奇数末尾チャネルは保持。

## 参考: 標準 RoPE 実装との差分箇所
- 標準: `src/modules/data_process/positional_encoder/rope_positional_encoder.py`
  - 固定 `ω` をキャッシュ可能。
- 学習版: `learnable_rope_positional_encoder.py`
  - `ω` が学習対象のため、毎回 `cos/sin` を再計算。

---

本ドキュメントは上記 2 ファイルの実装（2025-09-06 時点）を基に要約しています。
