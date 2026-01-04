# project_module

TimeCopilot比較検証のためのプロジェクト固有ロジックを実装したPythonパッケージです。

## 構成

```
src/project_module/
├── data_download/      # Kaggleデータ取得
├── data_preprocessing/ # 前処理（長形式変換、holdout分割）
├── models/             # ベースラインモデル（LightGBM, Prophet, Seq2Seq, Chronos-2）
├── timecopilot/        # TimeCopilot連携（モデル検出、実行、ログ）
├── evaluating/         # 評価指標（SMAPE, RMSPE, WRMSSE）
├── experiment/         # 実験ランナー
└── utils/              # ロギング、GPU管理
```

## インストール

```bash
# ルートディレクトリから
uv sync
```

## テスト

```bash
uv run pytest project_module/tests/
```
