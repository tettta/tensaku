# /home/esakit25/work/tensaku/src/tensaku/experiments/layout.py
# -*- coding: utf-8 -*-
"""
tensaku.experiments.layout

layout は「住所録」と「成果物メタ情報(kind等)」を定義するだけ。
保存処理・台帳処理（I/O）は fs_core.Layout が担当する。

このモジュールが知ってよいこと:
- 実験フォルダ内で「どこに何を置くか」（相対パス）
- kind などのメタ情報（定義）

知らなくてよいこと:
- 具体的な保存方法（JSONL/np.save/torch.save 等）
- 台帳フォーマットの詳細
"""

from __future__ import annotations

from tensaku.fs_core import ArtifactDir, ArtifactFile, Layout, define_file, define_family, define_dir_family


class QidLayout(Layout):
    """QID-level layout under outputs/{qid}.

    This layout provides stable, cross-experiment aggregation paths.
    In particular, it owns the per-qid experiment index.

    Boundary:
      - Knows: only QID root.
      - Does NOT know: per-experiment folder structure details.
    """

    experiments_index = define_file("_index/experiments.jsonl", kind="index", record=False)

class CheckpointDir(ArtifactDir):
    @property
    def best(self) -> ArtifactFile:
        return self / "model_best.pt"
    
    @property
    def last(self) -> ArtifactFile:
        return self / "model_last.pt"

class ExperimentLayout(Layout):
    # --- Catalog / Ledger ---
    artifact_index = define_file("artifacts/index.jsonl", kind="ledger", record=False)

    # --- Config / Meta ---
    run_meta = define_file("config/run_meta.json", kind="meta", record=True)
    exp_config_dump = define_file("config/exp_config.yaml", kind="meta", record=True)

    # --- Logs ---
    pipeline_log = define_file("logs/pipeline.log", kind="log", record=False)
    log_round = define_family("logs/round_{round:03d}.log", kind="log", record=False)

    # --- Metrics ---
    al_history = define_file("metrics/al_history.csv", kind="metric", record=True)
    al_learning_curve = define_file("metrics/al_learning_curve.csv", kind="metric", record=True)
    hitl_summary_final = define_file("metrics/hitl_summary_final.csv", kind="metric", record=True)
    hitl_summary_rounds = define_file("metrics/hitl_summary_rounds.csv", kind="metric", record=True)

    # --- Selection ---
    selection_all_samples = define_file("selection/al_samples_all.csv", kind="selection", record=True)
    selection_round_samples = define_family("selection/rounds/round_{round:03d}_al_sample.csv", kind="selection", record=True)
    selection_round_sample_ids = define_family("selection/rounds/round_{round:03d}_al_sample_ids.txt", kind="selection", record=True)

    # --- Predictions ---
    preds_detail_final = define_file("predictions/final/preds_detail.csv", kind="pred", record=True)
    gate_assign_final = define_file("predictions/final/gate_assign.csv", kind="pred", record=True)
    predictions_round_detail = define_family("predictions/rounds/round_{round:03d}_preds_detail.csv", kind="pred", record=True)
    predictions_round_gate_assign = define_family("predictions/rounds/round_{round:03d}_gate_assign.csv", kind="pred", record=True)

    # --- Arrays (long-lived optional) ---
    arrays_round_probs = define_family("arrays/rounds/round_{round:03d}_probs.npy", kind="prob", record=True)
    arrays_round_logits = define_family("arrays/rounds/round_{round:03d}_logits.npy", kind="logit", record=True)
    arrays_round_emb_cls = define_family("arrays/rounds/round_{round:03d}_emb_cls.npy", kind="emb", record=True)
    arrays_round_file = define_family("arrays/rounds/round_{round:03d}_{name}", kind="array", record=True)

    # --- Round Internals (short-lived by default) ---
    round_ckpt_dir = define_dir_family("rounds/round_{round:03d}/train/checkpoints", node_type=CheckpointDir, record=True, meta={"description": "Model checkpoints"})

    round_pool_embs = define_family("rounds/round_{round:03d}/infer/pool_embs.npy", kind="emb", record=True)
    round_pool_preds = define_family("rounds/round_{round:03d}/infer/pool_preds.csv", kind="pred", record=True)
    round_infer_dir = define_dir_family("rounds/round_{round:03d}/infer")

    # kind schema (optional validation)
    ALLOWED_KINDS = {
        "ledger",
        "meta",
        "log",
        "metric",
        "selection",
        "pred",
        "ckpt",
        "emb",
        "prob",
        "logit",
        "array",
    }

if __name__ == "__main__":
    from pathlib import Path
    out = Path("./out_experiment_layout_demo")
    layout = ExperimentLayout(root=out)
    print("\n" * 2)
    print("===================================================================")
    print("--- Layout Tree (Defined Paths) ---")
    print("===================================================================")
    # fs_core.py で追加した tree() メソッドを呼び出す
    print(layout.tree())