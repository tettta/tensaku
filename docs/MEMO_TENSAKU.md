# Tensaku 復元メモ（自動生成）

*最終更新*: 2025-10-29 14:29
*生成元*: `scripts/gen_memo_from_modules.py`（各モジュール先頭docstringの @tags を収集）

## 方針
- **「1レス1スクリプト」** ルールを継続
- メンテは **各モジュール先頭docstringの @tags を編集**するだけ（本メモは自動生成）
- 優先キー: `@module, @role, @inputs, @outputs, @cli, @notes`（任意: `@deps`）

## 主要パスと環境（固定）
- ROOT: `/home/esakit25/work/tensaku`
- DATA: `/home/esakit25/work/tensaku/data_sas`（labeled.jsonl / dev.jsonl / test.jsonl / pool.jsonl）
- OUT : `/home/esakit25/work/tensaku/outputs`
- CKPT: `/home/esakit25/work/tensaku/outputs/checkpoints_min/{best.pt,last.pt}`
- 環境: VS Code + Jupyter / venv (Python 3.12) / PyTorch / Transformers≥4.44


## モジュール一覧

### split  (_split.py_)
- **役割**: "all.jsonl"（score有/無 混在可）から学習/検証/テスト/プールを作るデータ分割ユーティリティ
- **入力**: {run.data_dir}/{data.files.*} または data.input_all（JSONL: id, mecab|text, score?）
- **出力**: {data_dir}/labeled.jsonl, {data_dir}/dev.jsonl, {data_dir}/test.jsonl, {data_dir}/pool.jsonl
- **CLI**: tensaku split
- **補足**: freeze_dev_test=True のとき dev/test を固定維持。未採点は pool に落とす。
- **API**: run

### train  (_train.py_)
- **役割**: 日本語BERTで総得点の分類モデルを学習し、dev QWK ベースで best.pt を保存
- **入力**: {data_dir}/labeled.jsonl（id, mecab|text, score）, model.name, model/optim 設定
- **出力**: {out_dir}/checkpoints_min/best.pt, {out_dir}/checkpoints_min/last.pt, 学習ログ
- **CLI**: tensaku train
- **補足**: すべてのテンソルを同一 device に移送。AMP は CUDA 時のみ自動で有効化。
- **API**: EssayDS, run

### infer_pool  (_infer_pool.py_)
- **役割**: 未採点プールに対して一括推論を行い、予測ラベルとMSP確信度を保存
- **入力**: {data_dir}/pool.jsonl, {out_dir}/checkpoints_min/best.pt, model.name, data.max_len
- **出力**: {out_dir}/pool_preds.csv（id,y_pred,conf_msp）
- **CLI**: tensaku infer-pool
- **補足**: クラス数は ckpt→head形状→データ最大ラベル+1→cfg→fallback(6) の順で推定。
- **API**: EssayDS, run

### cli  (_cli.py_)
- **役割**: tensaku コマンドのディスパッチャ（split/train/infer-pool/confidence/gate/...）
- **入力**: YAML config（-c/--config, --set KEY=VAL）, サブコマンド引数
- **出力**: 各サブコマンドの成果物（このモジュール自体は副作用なし）
- **CLI**: tensaku {split,train,infer-pool,confidence,gate,al-sample,al-label-import,al-cycle,viz,eval}
- **補足**: 未実装コマンドは警告のみで終了。confidence.py 不在でも ImportError を回避。
- **API**: build_parser, main

### config  (_config.py_)
- **API**: load_config

### tensaku.calibration  (_calibration.py_)
- **役割**: Temperature scaling (T) and calibration metrics (NLL/ECE) with reliability bins.
- **CLI**: 直接のCLIは持たない。tensaku gate / tensaku infer-pool 等から内部利用。
- **API**: TemperatureScaler, nll_from_logits, ece, reliability_bins

### tensaku.confidence  (_confidence.py_)
- **役割**: Confidence estimators (MSP / entropy / energy / margin / MC-Dropout) with a lightweight registry hook.
- **CLI**: （直接のCLIは持たない。tensaku gate / tensaku infer-pool から内部利用）
- **API**: ConfidenceEstimator, MSP, OneMinusEntropy, Energy, Margin, MCDropoutConfig, MCDropout, create_estimator

### tensaku.gate  (_gate.py_)
- **役割**: 薄いオーケストレータ。Confidence / Calibration /（任意で Trust）を呼び出し、HITLゲートを決定する。
- **API**: fit_temperature_on_dev, compute_confidences, find_tau_for_constraint, decide_mask, save_gate_csv, score_trust

### tensaku.model_io  (_model_io.py_)
- **役割**: Model/Tokenizer I/O と推論ユーティリティ（薄い共通層）
- **API**: select_device, move_batch_to_device, load_tokenizer, load_cls_model, build_text_loader, predict_logits, predict_logits_from_texts, save_ckpt, load_ckpt_if_exists

### tensaku.registry  (_registry.py_)
- **役割**: Tensaku 内の軽量レジストリ（関数/クラス/ファクトリの名前→オブジェクト解決）
- **API**: register, list_names, get, create

### tensaku.trustscore  (_trustscore.py_)
- **役割**: kNN-based Trust Score（分類用）— v1（従来）と v2（ロバスト拡張）の両対応
- **API**: TrustScorer, trustscore


## 標準ワークフロー（AL × HITL）
1. 分割/プール準備: `tensaku split` または `scripts/make_pool.py`
2. 学習: `tensaku train` → best.pt
3. プール推論: `tensaku infer-pool`
4. ゲート: `tensaku gate`（温度T・τ・coverageの記録 / 擬似ラベル判定）
5. 取り込み: `python scripts/apply_gate.py --eps 0.05 --import_mode pseudo`
   - `label_import.jsonl` を `labeled.jsonl` に追記
   - `pool_hold.jsonl` を `pool.jsonl` に置換
6. 次サイクルへ
