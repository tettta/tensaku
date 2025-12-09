# /home/esakit25/work/tensaku/scripts/run_al_grid.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@script: scripts/run_al_grid.py
@role  : 複数の AL 実験を (qid, experiment, seed) のグリッドで回すラッパー。
         - tensaku al-run を qid→experiment→seed の順に実行
         - 成功した run について tensaku viz single を実行（オプション）
         - 実行結果を CSV (al_run_status.csv) に保存
         - 成功 / 失敗 / スキップ件数を標準出力にサマリ表示

● 実験条件の指定
    - QIDS                 : 対象となる qid のリスト
    - SEEDS                : 回したい seed のリスト
    - EXPERIMENT_CONFIGS   : 各 experiment の構成
        - name      : run.experiment に入る名前（exp_trust, exp_random など）
        - overrides : al.sampler, al.uncertainty_key 等、CLI から --set したい項目
        - run_name_pattern : run.name のパターン（既定: "seed{seed}"）

    → 実行順序は「qid ごとに → experiment ごとに → seed ごとに」となる。

● AL 成功判定
    - al-run の exit_code == 0 のあと、
      exp_dir (= outputs/QID/EXP_NAME/RUN_NAME) に以下が揃っているか確認:
        - metrics/al_history.csv
        - metrics/hitl_summary_final.csv
        - predictions/final/preds_detail.csv
        - predictions/final/gate_assign.csv
      いずれか欠けていれば error 扱いにする（status="error"）。

● rounds / budget
    - AL_ROUNDS / AL_BUDGET を設定すると al.rounds / al.budget を --set で上書き。
    - YAML で管理したい場合は None にしておけば CLI では触らない。

使い方（例）:
    python scripts/run_al_grid.py \
        --config configs/exp_al_hitl.yaml \
        --root /home/esakit25/work/tensaku

    # viz は別でまとめて実行したいとき:
    python scripts/run_al_grid.py -c configs/exp_al_hitl.yaml --no-viz

    # 実際には実行せず、コマンドだけ確認したいとき:
    python scripts/run_al_grid.py -c configs/exp_al_hitl.yaml --dry-run
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 実験グリッド定義（ここを編集して実験条件を変える）
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """
    1つの「実験構成」を表す。
    - name           : run.experiment として使われる論理名（exp_trust など）
    - overrides      : al.sampler や al.uncertainty_key など、CLI から --set したい追加設定
    - run_name_pattern: run.name を決めるパターン
    - data_dir_pattern: data_dir のパターン（root からの相対パス）
    """
    name: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    run_name_pattern: str = "seed{seed}"
    data_dir_pattern: str = "data_sas/splits/{qid}"


# QID ごとにまとめて分析したいので、最上位ループは QIDS。
QIDS: List[str] = [
    "Y14_1-2_1_3",
    # 必要なら他の qid を追加
]

# 各 experiment 内で回す seed のリスト。
SEEDS: List[int] = [42]

# 「比較したい設定」は experiment 名に埋め込み、
# 必要な override はここで指定する。
EXPERIMENT_CONFIGS = [
    # trust を使う uncertainty sampling
    ExperimentConfig(
        name="exp_trust",
        overrides={
            "al.sampler.name": "uncertainty",
            "al.uncertainty_key": "trust",
        },
    ),
    # MSP を使う uncertainty sampling
    ExperimentConfig(
        name="exp_msp",
        overrides={
            "al.sampler.name": "uncertainty",
            "al.uncertainty_key": "msp",
        },
    ),
    # 純ランダム sampling
    ExperimentConfig(
        name="exp_random",
        overrides={
            "al.sampler.name": "random",
        },
    ),
]



# al.rounds / al.budget を CLI から上書きしたい場合に設定する。
# YAML 側で管理したい場合は None にしておけば CLI では触らない。
AL_ROUNDS: Optional[int] = 5
AL_BUDGET: Optional[int] = 50


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def now_iso() -> str:
    """現在時刻を ISO 8601 文字列（秒精度）で返す。"""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def build_al_run_command(
    root: Path,
    config_path: Path,
    qid: str,
    seed: int,
    ec: ExperimentConfig,
) -> List[str]:
    """
    単一 run に対する `tensaku al-run` コマンドを組み立てる。
    """
    run_name = ec.run_name_pattern.format(seed=seed)
    data_dir = root / ec.data_dir_pattern.format(qid=qid)

    cmd: List[str] = [
        "tensaku",
        "al-run",
        "-c",
        str(config_path),
        "--set",
        f"data.qid={qid}",
        "--set",
        f"run.experiment={ec.name}",
        "--set",
        f"run.name={run_name}",
        "--set",
        f"run.seed={seed}",
        "--set",
        f"run.data_dir={data_dir}",
    ]

    if AL_ROUNDS is not None:
        cmd.extend(["--set", f"al.rounds={AL_ROUNDS}"])
    if AL_BUDGET is not None:
        cmd.extend(["--set", f"al.budget={AL_BUDGET}"])

    # overrides はそのまま --set <key>=<value> として渡す
    # 例: {"al.sampler": "uncertainty"} → --set al.sampler=uncertainty
    for key, value in ec.overrides.items():
        cmd.extend(["--set", f"{key}={value}"])

    return cmd


def build_viz_single_command(run_root: Path) -> List[str]:
    """
    単一 run 用の可視化コマンド。
    viz 側は `tensaku viz single --exp-dir <run_root>` を受け取る前提。
    """
    return [
        "tensaku",
        "viz",
        "single",
        "--exp-dir",
        str(run_root),
    ]


def check_run_outputs(exp_dir: Path) -> bool:
    required_paths = [
        exp_dir / "metrics" / "al_history.csv",
        exp_dir / "predictions" / "final" / "preds_detail.csv",
        exp_dir / "predictions" / "final" / "gate_assign.csv",
    ]
    missing = [p for p in required_paths if not p.exists()]
    if missing:
        print(f"[run_al_grid] WARN: Required outputs missing for exp_dir={exp_dir}:")
        for p in missing:
            print(f"    - MISSING: {p}")
        return False
    return True



@dataclass
class RunStatus:
    """
    1 本の run の実行結果を表すサマリ行。
    """
    qid: str
    experiment: str
    run_name: str
    seed: int
    overrides: str  # "al.sampler=...,al.uncertainty_key=..." のような簡易表現
    rounds: int
    budget: int
    exit_code: int
    status: str  # "success" / "error" / "skipped"
    started_at: str
    ended_at: str
    exp_dir: str


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple AL experiments over (qid, experiment, seed) grid."
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Base config YAML for tensaku al-run (e.g., configs/exp_al_hitl.yaml)",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Do NOT run 'tensaku viz single' even if al-run succeeds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned commands without executing them.",
    )
    parser.add_argument(
        "--status-csv",
        default="outputs/al_run_status.csv",
        help=(
            "Path to output CSV summary of all runs "
            "(relative to --root if not absolute)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    root = Path(args.root).resolve()
    config_path = (
        (root / args.config).resolve()
        if not Path(args.config).is_absolute()
        else Path(args.config)
    )
    if not config_path.exists():
        print(f"[run_al_grid] ERROR: Config not found: {config_path}")
        return 1

    if not root.exists():
        print(f"[run_al_grid] ERROR: Root not found: {root}")
        return 1

    # status CSV のパスを決定
    status_csv_path = Path(args.status_csv)
    if not status_csv_path.is_absolute():
        status_csv_path = root / status_csv_path
    status_csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[run_al_grid] Project root    : {root}")
    print(f"[run_al_grid] Base config     : {config_path}")
    print(f"[run_al_grid] Status CSV      : {status_csv_path}")
    print(f"[run_al_grid] Dry-run         : {args.dry_run}")
    print(f"[run_al_grid] Run viz single  : {not args.no_viz}")
    print(f"[run_al_grid] QIDS            : {QIDS}")
    print(f"[run_al_grid] SEEDS           : {SEEDS}")
    print(f"[run_al_grid] EXPERIMENTS     : {[ec.name for ec in EXPERIMENT_CONFIGS]}")

    statuses: List[RunStatus] = []
    n_success = 0
    n_error = 0
    n_skipped = 0

    # 実行順序: qid ごとに → experiment ごとに → seed ごとに
    for qid in QIDS:
        print(f"\n[run_al_grid] === QID: {qid} ===")
        for ec in EXPERIMENT_CONFIGS:
            print(f"[run_al_grid]  - Experiment: {ec.name}")
            for seed in SEEDS:
                run_name = ec.run_name_pattern.format(seed=seed)
                exp_dir = root / "outputs" / qid / ec.name / run_name

                print(
                    f"[run_al_grid]    > Run: seed={seed}, "
                    f"run_name={run_name}, exp_dir={exp_dir}"
                )

                al_cmd = build_al_run_command(root, config_path, qid, seed, ec)
                print(f"[run_al_grid]      al-run cmd: {' '.join(al_cmd)}")

                started_at = now_iso()

                if args.dry_run:
                    exit_code = 2
                    ended_at = now_iso()
                    print("[run_al_grid]      DRY-RUN: skipped actual execution.")
                else:
                    try:
                        proc = subprocess.run(
                            al_cmd,
                            cwd=str(root),
                            check=False,
                        )
                        exit_code = proc.returncode
                    except Exception as e:
                        print(
                            f"[run_al_grid]      ERROR: Exception while running al-run: {e}"
                        )
                        exit_code = 1
                    ended_at = now_iso()

                    # 成功コードでも成果物欠如なら error 扱い
                    if exit_code == 0:
                        if not check_run_outputs(exp_dir):
                            print(
                                "[run_al_grid]      WARN: al-run returned 0 but required "
                                "outputs are missing. Marking this run as error."
                            )
                            exit_code = 1

                overrides_str = ",".join(
                    f"{k}={v}" for k, v in ec.overrides.items()
                )
                rounds_val = AL_ROUNDS if AL_ROUNDS is not None else -1
                budget_val = AL_BUDGET if AL_BUDGET is not None else -1

                if exit_code == 0:
                    status_str = "success"
                elif exit_code == 2:
                    status_str = "skipped"
                else:
                    status_str = "error"

                statuses.append(
                    RunStatus(
                        qid=qid,
                        experiment=ec.name,
                        run_name=run_name,
                        seed=seed,
                        overrides=overrides_str,
                        rounds=rounds_val,
                        budget=budget_val,
                        exit_code=exit_code,
                        status=status_str,
                        started_at=started_at,
                        ended_at=ended_at,
                        exp_dir=str(exp_dir),
                    )
                )

                if status_str == "success":
                    n_success += 1
                elif status_str == "error":
                    n_error += 1
                else:
                    n_skipped += 1

                # 成功した run に対してのみ viz を実行
                if (not args.dry_run) and (not args.no_viz) and exit_code == 0:
                    viz_cmd = build_viz_single_command(exp_dir)
                    print(f"[run_al_grid]      viz cmd : {' '.join(viz_cmd)}")
                    try:
                        _ = subprocess.run(
                            viz_cmd,
                            cwd=str(root),
                            check=False,
                        )
                    except Exception as e:
                        print(
                            f"[run_al_grid]      WARN: Exception while running viz: {e}"
                        )

    # 全 run 終了後、ステータスを CSV に保存
    print(f"\n[run_al_grid] Writing status CSV: {status_csv_path}")

    if statuses:
        fieldnames = list(asdict(statuses[0]).keys())
        with status_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in statuses:
                writer.writerow(asdict(s))

    # 件数サマリを標準出力に表示
    print(
        "\n[run_al_grid] Summary: "
        f"success={n_success}, error={n_error}, skipped={n_skipped}"
    )

    # 1本でも error があれば全体としては 1 を返す
    return 1 if n_error > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
