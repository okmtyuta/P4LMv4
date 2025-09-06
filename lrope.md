# Learnable RoPE Positional Encoder 概要

本ドキュメントは，`src/modules/data_process/positional_encoder/learnable_rope_positional_encoder.py` に実装された Learnable Rotary Positional Embedding（以下 Learnable RoPE; LRoPE）の数式定義・性質・実装要点をまとめたものです。標準の RoPE（固定周波数）との差分も記します。

## 記号と形状

- 入力表現: 行列 $X \in \mathbb{R}^{L \times D}$（長さ $L$，次元 $D$）。
- 出力表現: $Y \in \mathbb{R}^{L \times D}$（双方向版は $Y \in \mathbb{R}^{L \times 2D}$）。
- ペア数: $H = \left\lfloor D/2 \right\rfloor$（偶数・奇数次元の扱いは後述）。
- 位置インデックス: $p_l$（1 始まり）。
- 学習パラメータ（周波数ベクトルの対数）: $\boldsymbol{\beta} \in \mathbb{R}^{H}$ 。
- 周波数（正値制約）: $\boldsymbol{\omega} = \exp\left(\boldsymbol{\beta}\right) \in \mathbb{R}_{>0}^{H}$ 。
- 初期化: $\omega_k^{(0)} = \theta_{\text{base}}^{\,-2k/D}$（$k=0,1,\dots,H-1$）。

実装では $\boldsymbol{\beta} = \log \boldsymbol{\omega}$ を `nn.Parameter` として保持し， $\omega_k>0$ を保証します。

## 位置インデックスの定義

- 通常（forward）: $p_l = l\;\;(l=1,\dots,L)$ 。
- 逆順（reversed）: $p_l = L - l + 1\;\;(l=1,\dots,L)$ 。

双方向版では forward と reversed をそれぞれ適用し，最後の次元で連結します（`dim_factor = 2`）。

## 角度と回転（偶数次元ペアごと）

各ペア $k\in\{0,\dots,H-1\}$ に対し角度を

$$
  \phi_{l,k} = p_l\, \omega_k
$$

と定義します。ペア $(2k,\,2k+1)$ に対応する 2 次元ベクトルに回転行列

$$
  R\left(\phi\right) =
  \begin{bmatrix}
    \cos\left(\phi\right) & -\sin\left(\phi\right) \\
    \sin\left(\phi\right) & \cos\left(\phi\right)
  \end{bmatrix}
$$

を適用します。成分表示では， $x_{l,j}$ を入力， $y_{l,j}$ を出力として

$$
  y_{l,2k} = x_{l,2k} \cos\left(\phi_{l,k}\right) - x_{l,2k+1} \sin\left(\phi_{l,k}\right),\quad
  y_{l,2k+1} = x_{l,2k} \sin\left(\phi_{l,k}\right) + x_{l,2k+1} \cos\left(\phi_{l,k}\right).
$$

$D$ が奇数のときは末尾チャネル $D-1$ をそのまま保持します: $\;y_{l,D-1}=x_{l,D-1}$ 。

## 標準 RoPE との違い

標準 RoPE では周波数が固定です: 

$$
  \omega_k = \theta_{\text{base}}^{-2k/D}\quad\left(\text{固定}\right).
$$

Learnable RoPE では $\omega_k$ を学習し， $\beta_k = \log \omega_k$ を直接最適化します。これによりデータ分布に応じた周波数スケールへの適応が可能で，表現力や収束特性の改善が期待できます（$\theta_{\text{base}}$ は初期化にのみ使用）。

## 前向き計算のまとめ

各 $l\in\{1,\dots,L\}$，各 $k\in\{0,\dots,H-1\}$ について

$$
  \phi_{l,k} = p_l e^{\beta_k}.
$$

$$
  \begin{bmatrix}
    y_{l,2k} \\
    y_{l,2k+1}
  \end{bmatrix}
  =
  \begin{bmatrix}
    \cos\left(\phi_{l,k}\right) & -\sin\left(\phi_{l,k}\right) \\
    \sin\left(\phi_{l,k}\right) & \cos\left(\phi_{l,k}\right)
  \end{bmatrix}
  \begin{bmatrix}
    x_{l,2k} \\
    x_{l,2k+1}
  \end{bmatrix}.
$$

かつ $D$ が奇数なら $y_{l,D-1}=x_{l,D-1}$ 。双方向版は $p_l=l$ と $p_l=L-l+1$ の両方で同式を計算し，最終次元で連結して $Y\in\mathbb{R}^{L\times 2D}$ を得ます。

## 勾配の流れ（概要）

$\omega_k = e^{\beta_k}$ より

$$
  \frac{\partial \omega_k}{\partial \beta_k} = \omega_k,\quad
  \frac{\partial \phi_{l,k}}{\partial \beta_k} = p_l\,\omega_k.
$$

また

$$
  \frac{\partial y_{l,2k}}{\partial \phi_{l,k}} = -x_{l,2k}\,\sin\left(\phi_{l,k}\right) - x_{l,2k+1}\,\cos\left(\phi_{l,k}\right),
$$

$$
  \frac{\partial y_{l,2k+1}}{\partial \phi_{l,k}} = x_{l,2k}\,\cos\left(\phi_{l,k}\right) - x_{l,2k+1}\,\sin\left(\phi_{l,k}\right),
$$

であるため，連鎖律により

$$
  \frac{\partial y_{l,j}}{\partial \beta_k}
  = \frac{\partial y_{l,j}}{\partial \phi_{l,k}}\,\frac{\partial \phi_{l,k}}{\partial \beta_k}
  = p_l\,\omega_k\,\frac{\partial y_{l,j}}{\partial \phi_{l,k}}\quad \left(j\in\left\{2k,2k+1\right\}\right)\,.
$$

位置 $p_l$ が大きいほど角度変化への感度が増し，学習信号が遠位置まで届きます。

## 性質と計算量

- ノルム保存: 各ペアの回転は直交変換であり，
  $$
    \left\|\begin{bmatrix}y_{l,2k}\\y_{l,2k+1}\end{bmatrix}\right\|_2
    =
    \left\|\begin{bmatrix}x_{l,2k}\\x_{l,2k+1}\end{bmatrix}\right\|_2
  $$
  が成り立ちます（奇数末尾は恒等）。
- 計算量: $\mathcal{O}(L\,D)$，追加メモリは $\cos/\sin\in\mathbb{R}^{L\times H}$ 程度。
- 数値安定性: $\beta$ によるパラメータ化で $\omega>0$ を保証し，指数スケールを安定に扱います。
- キャッシュ方針: Learnable RoPE は $\omega$ が更新されるため角度は毎回再計算（固定周波数の標準 RoPE はキャッシュ可能）。
- 位置は 1 始まり: 実装では $1,\dots,L$（逆順は $L,\dots,1$）。

## 実装対応（抜粋）

- ファイル: `src/modules/data_process/positional_encoder/learnable_rope_positional_encoder.py`
- パラメータ: `self._log_inv_freq \equiv \boldsymbol{\beta}`，初期値 $\log\left(\theta_{\text{base}}^{\,-2m/D}\right)$ 。
- 角度計算: $\texttt{angles} = p\,\texttt{[:,None]}\;\cdot\; \exp(\boldsymbol{\beta})\,\texttt{[None,:]}$， $\cos/\sin$ をブロードキャスト計算。
- 回転適用: 偶数・奇数インデックスをストライド分割し，上記の $2\times2$ 回転を適用。
- 形状: 入出力 $(L,D)$（双方向は $(L,2D)$）。奇数末尾は保持。

## 参考: 標準 RoPE 実装との差分

- 標準: `src/modules/data_process/positional_encoder/rope_positional_encoder.py`（固定 $\omega$，角度キャッシュあり）。
- 学習版: `learnable_rope_positional_encoder.py`（$\omega$ を学習，角度は毎回再計算）。

---

本ドキュメントは上記 2 ファイルの実装（2025-09-06 時点）に基づく要約です。
