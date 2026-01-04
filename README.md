# TimeCopilot 比較検証プロジェクト

時系列予測フレームワーク「TimeCopilot」の実用性を、Kaggleベンチマークデータセットを用いて検証するプロジェクトです。

## 概要

TimeCopilotは複数の基盤時系列モデル（TSFM）をLLM経由で統一的に扱うフレームワークです。本プロジェクトでは以下の比較検証を実施します：

- **TimeCopilot**: 18種類の予測モデルを自動選択・実行
- **ベースライン**: LightGBM, Prophet, Seq2Seq, Chronos-2

### 評価データセット

| データセット | 系列数 | 予測期間 | 評価指標 |
|--------------|--------|----------|----------|
| Store Item Demand | 500 | 90日 | SMAPE |
| Rossmann Store Sales | 1,115 | 42日 | RMSPE |
| M5 Forecasting (Level 6) | 9 | 28日 | WRMSSE |

## プロジェクト構成

```
.
├── configs/           # 実験設定（Hydra YAML）
├── dataset/           # Kaggleデータセット
├── notebooks/         # 可視化ノートブック
├── outputs/           # 実験結果出力
├── project_module/    # プロジェクト固有ロジック
├── scripts/           # 実行スクリプト
└── docs/              # ドキュメント
```

## 検証フロー

### 1. 環境構築

```bash
# 依存パッケージのインストール
uv sync
```

### 2. データセット取得

```bash
# Kaggle認証（事前に ~/.kaggle/access_token を設定）
uv run python scripts/download_kaggle_datasets.py
```

### 3. 実験実行

```bash
# Store Item実験
uv run python scripts/run_experiment.py experiment=store_item

# Rossmann実験
uv run python scripts/run_experiment.py experiment=rossmann

# M5実験
uv run python scripts/run_experiment.py experiment=m5
```

### 4. 結果確認

- `outputs/{experiment}/{timestamp}/metrics/summary.csv`: 評価結果
- `outputs/{experiment}/{timestamp}/predictions/`: モデル別予測
- `notebooks/visualization_*.ipynb`: 可視化ノートブック

## Kaggle認証

### 推奨: アクセストークン方式

```bash
kaggle auth login
# トークンを ~/.kaggle/access_token に保存
```

### 代替: 環境変数

```bash
export KAGGLE_API_TOKEN="your_token"
```
