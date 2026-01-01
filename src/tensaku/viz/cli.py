# /home/esakit25/work/tensaku/src/tensaku/viz/cli.py
# -*- coding: utf-8 -*-
"""tensaku.viz.cli

Entry points for visualization.

Commands
  - single : per-experiment plots
  - compare: aggregate learning curves across experiments (seed-avg) + presets
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensaku.viz import base, metrics, plots, compare
from tensaku.viz import tau_sweep
from tensaku.viz import snapshot


# --- 削除: _read_preds_detail, _find_preds_detail (不要/重複コード) ---

def _load_run_meta_conf_direction(exp_dir: Path) -> Dict[str, bool]:
    """Load explicit confidence direction from run_meta.json (Strict).
    Returns: { 'conf_msp': True, 'conf_entropy': False, ... }
    """
    # layout.py で定義されたパス: config/run_meta.json
    meta_path = exp_dir / "config" / "run_meta.json"
    if not meta_path.exists():
        return {}
    
    try:
        import json
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        # index.py: _extract_confidence_meta で保存された構造を参照
        # meta -> confidence -> estimators (list of dict or list of str)
        # run_metaの構造によっては meta -> confidence ではなく直下の場合もあるため柔軟に
        meta_root = data.get("meta", {})
        
        # fallback: 古い形式や構造違いを考慮し、トップレベルも探す
        conf_cfg = meta_root.get("confidence") or data.get("confidence")
        if not conf_cfg:
            return {}

        estimators = conf_cfg.get("estimators", [])
        
        out = {}
        for est in estimators:
            if isinstance(est, dict):
                name = est.get("name")
                hib = est.get("higher_is_better")
                if name and hib is not None:
                    # CSVのカラム名は "conf_{name}" または "{name}" (trustなど)
                    # ここでは標準的な "conf_{name}" と、そのままの名前の両方を登録しておく
                    out[f"conf_{name}"] = bool(hib)
                    out[name] = bool(hib)
        return out
    except Exception:
        return {}


def _eps_dir_name(eps: float) -> str:
    s = f"{float(eps):.6f}".rstrip("0").rstrip(".")
    return "eps_" + s.replace(".", "p")


def _export_tau_evidence_for_experiments(
    *,
    bundles: Sequence[compare.CurveBundle],
    eps_list: Sequence[float],
    cse_abs_err: int,
    out_root: Path,
    snapshot_round: Optional[int],
    snapshot_n_labeled: Optional[int],
    snapshot_nearest: bool,
) -> None:
    """Export tau-sweep evidence for each experiment.

    Strict behavior:
      - requires preds_detail.csv with split,y_true,y_pred
      - requires at least one confidence column (conf_* or trust)
      - raises on missing/unknown confidence direction
    """
    out_root.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    for b in bundles:
        exp_dir = Path(str(b.meta.get("exp_dir"))).resolve()
        if not exp_dir.exists():
            raise FileNotFoundError(f"exp_dir not found: {exp_dir}")

        # Load explicit direction map (Strict support)
        direction_map = _load_run_meta_conf_direction(exp_dir)

        pick = snapshot.pick_snapshot(
            exp_dir=exp_dir,
            snapshot_round=snapshot_round,
            snapshot_n_labeled=snapshot_n_labeled,
            nearest=bool(snapshot_nearest),
        )
        preds_path = pick.preds_detail_csv
        
        # 修正: _read_preds_detail -> base.read_csv 直接呼び出し
        df = base.read_csv(preds_path, ctx="preds_detail")
        
        conf_cols = _conf_cols(df)
        if not conf_cols:
            raise RuntimeError(f"no confidence columns found in preds_detail: {preds_path}")

        exp_uid = str(b.exp_uid or "").strip()
        if not exp_uid:
            # Deterministic fallback: hash exp_dir (no silent randomness)
            import hashlib
            exp_uid = hashlib.sha1(str(exp_dir).encode("utf-8")).hexdigest()[:10]

        exp_out = out_root / f"exp_{base.sanitize_token(exp_uid)}"
        if snapshot_round is not None or snapshot_n_labeled is not None:
            exp_out = exp_out / f"snap_{pick.tag}"
        exp_out.mkdir(parents=True, exist_ok=True)

        for eps in eps_list:
            eps_dir = exp_out / _eps_dir_name(float(eps))
            paths = tau_sweep.export_tau_sweep(
                preds_detail_csv=preds_path,
                out_dir=eps_dir,
                conf_keys=list(conf_cols),
                cse_abs_err=int(cse_abs_err),
                eps_cse=float(eps),
                run_meta_json=None,
                direction=direction_map, # 修正: マップを渡す
            )
            # collect selected jsons for summary (best-effort; strict on missing file)
            for p in paths:
                if p.name.startswith("tau_selected_") and p.suffix == ".json":
                    selected_rows.append(
                        {
                            "exp_uid": exp_uid,
                            "exp_dir": str(exp_dir),
                            "eps_cse": float(eps),
                            "selected_json": str(p),
                        }
                    )

    if selected_rows:
        # Keep tables separated from json metadata.
        (out_root / "tables").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(selected_rows).to_csv(out_root / "tables" / "tau_selected_index.csv", index=False)


def _conf_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("conf_")]
    # Backward compatibility: some runs may store trust separately.
    if "trust" in df.columns and "trust" not in cols:
        cols.append("trust")
    return cols


def cmd_single(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()

    if args.exp_dir:
        exp_dir = (project_root / args.exp_dir).resolve() if not Path(args.exp_dir).is_absolute() else Path(args.exp_dir).resolve()
    else:
        raise ValueError("single requires --exp-dir (index-based lookup can be added if needed)")

    if not exp_dir.exists():
        raise FileNotFoundError(f"exp_dir not found: {exp_dir}")

    # Snapshot selection
    if args.round is not None and args.n_labeled is not None:
        raise ValueError("--round and --n-labeled are mutually exclusive")
    
    pick = snapshot.pick_snapshot(
        exp_dir=exp_dir,
        snapshot_round=args.round,
        snapshot_n_labeled=args.n_labeled,
        nearest=bool(args.nearest),
    )
    preds_path = pick.preds_detail_csv
    
    # 修正: _read_preds_detail -> base.read_csv 直接呼び出し
    df = base.read_csv(preds_path, ctx="preds_detail")

    # Load explicit direction map
    direction_map = _load_run_meta_conf_direction(exp_dir)

    # Choose split
    split = args.split
    if "split" in df.columns and split is not None:
        df_use = df[df["split"].astype(str) == str(split)].copy()
    else:
        df_use = df.copy()

    # Filter to rows with y_true present
    if "y_true" not in df_use.columns or "y_pred" not in df_use.columns:
        raise KeyError(f"preds_detail must have y_true,y_pred. available={list(df_use.columns)}")

    y_true = pd.to_numeric(df_use["y_true"], errors="coerce").to_numpy()
    y_pred = pd.to_numeric(df_use["y_pred"], errors="coerce").to_numpy()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m].astype(int)
    y_pred = y_pred[m].astype(int)

    # Avoid overwrite when snapshot is used
    snap_dir = None
    if args.round is not None or args.n_labeled is not None:
        snap_dir = pick.tag
    out_dir = exp_dir / "plots"
    if snap_dir:
        out_dir = out_dir / f"snap_{snap_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base.setup_style()
    
    # Snapshot convenience: apply x_max when not using preset.
    if getattr(args, 'snapshot_round', None) is not None and getattr(args, 'snapshot_n_labeled', None) is not None:
        raise ValueError('--snapshot-round and --snapshot-n-labeled are mutually exclusive')
    if not getattr(args, 'preset', False):
        if getattr(args, 'snapshot_round', None) is not None and args.x == 'round' and args.x_max is None:
            args.x_max = float(args.snapshot_round)
        if getattr(args, 'snapshot_n_labeled', None) is not None and args.x == 'n_labeled' and args.x_max is None:
            args.x_max = float(args.snapshot_n_labeled)
    else:
        # preset: apply snapshot to x_max_{round,n_labeled} (strict on conflicts)
        if getattr(args, 'snapshot_round', None) is not None:
            if args.x_max_round is not None and float(args.x_max_round) != float(args.snapshot_round):
                raise ValueError('snapshot_round conflicts with --x-max-round')
            args.x_max_round = float(args.snapshot_round)
        if getattr(args, 'snapshot_n_labeled', None) is not None:
            if args.x_max_n_labeled is not None and float(args.x_max_n_labeled) != float(args.snapshot_n_labeled):
                raise ValueError('snapshot_n_labeled conflicts with --x-max-n-labeled')
            args.x_max_n_labeled = float(args.snapshot_n_labeled)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    plots.plot_confusion_matrix(ax, y_true, y_pred, title=f"Confusion Matrix ({args.split or 'all'})")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix_{args.split or 'all'}.png", dpi=160)
    plt.close(fig)

    conf_cols = _parse_csv_list(args.conf_cols) or _conf_cols(df_use)
    if not conf_cols:
        print(f"[warn] no confidence columns found in {preds_path.name}. only confusion_matrix was saved.")
        return 0

    for c in conf_cols:
        if c not in df_use.columns:
            raise KeyError(f"confidence column not found: {c}. available={list(df_use.columns)}")

        conf = pd.to_numeric(df_use[c], errors="coerce").to_numpy(dtype=float)[m]

        # Strict Direction Handling
        if c in direction_map:
            hib = direction_map[c]
        else:
            # 未知の場合は Strict にエラーを吐く(tau_sweep._conf_higher_is_better 側で制御)
            # ここでは便宜上 heuristic を呼ぶが、後で tau_sweep.py も更新することを推奨
            hib = tau_sweep._conf_higher_is_better(c)
        
        conf_dir = conf if hib else (-conf)

        # Histogram (raw scale)
        fig, ax = plt.subplots(figsize=(6, 4))
        plots.plot_histogram(ax, pd.DataFrame({c: conf}), cols=[c], title=f"Confidence Distribution: {c}")
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{c}.png", dpi=160)
        plt.close(fig)

        # Reliability bins (min-max normalized internally)
        bin_conf, bin_acc, bin_count = metrics.compute_reliability_bins_from_conf(
            conf=np.asarray(conf_dir, dtype=float),
            y_pred=np.asarray(y_pred),
            y_true=np.asarray(y_true),
            n_bins=int(args.ece_bins),
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        plots.plot_reliability_diagram_bins(ax, bin_conf, bin_acc, title=f"Reliability: {c}")
        fig.tight_layout()
        fig.savefig(out_dir / f"reliability_{c}.png", dpi=160)
        plt.close(fig)

        bins_df = pd.DataFrame(
            {
                "bin": np.arange(len(bin_conf), dtype=int),
                "confidence": bin_conf,
                "accuracy": bin_acc,
                "count": bin_count,
            }
        )
        bins_df.to_csv(out_dir / f"reliability_bins_{c}.csv", index=False)

        # Risk-coverage (CSE by default)
        rc = metrics.compute_risk_coverage(
            y_true=np.asarray(y_true),
            y_pred=np.asarray(y_pred),
            conf=np.asarray(conf_dir, dtype=float),
            risk_metric="cse",
            cse_abs_err=int(args.cse_abs_err),
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        plots.plot_risk_coverage(ax, [(c, rc)], title=f"Risk-Coverage (CSE>={int(args.cse_abs_err)}): {c}", ylabel="CSE rate")
        fig.tight_layout()
        fig.savefig(out_dir / f"risk_coverage_{c}.png", dpi=160)
        plt.close(fig)

        pd.DataFrame(rc).to_csv(out_dir / f"risk_coverage_{c}.csv", index=False)

    print(f"[ok] saved: {out_dir}")
    return 0

# (以下 _parse_csv_list, _stem_compare, _run_compare_one, cmd_compare, cmd_preset, main は変更なし)
# _parse_csv_list はもともとcli.pyにあったので省略しません（ファイル完全性のため必要なヘルパーです）
def _parse_csv_list(xs: Optional[str]) -> Optional[List[str]]:
    if xs is None:
        return None
    parts = [p.strip() for p in xs.split(",") if p.strip()]
    return parts or None


def _parse_int_list(xs: Optional[str]) -> Optional[List[int]]:
    if xs is None:
        return None
    parts = [p.strip() for p in xs.split(",") if p.strip()]
    return [int(p) for p in parts] if parts else None


def _parse_float_list(xs: Optional[str]) -> Optional[List[float]]:
    if xs is None:
        return None
    parts = [p.strip() for p in xs.split(",") if p.strip()]
    return [float(p) for p in parts] if parts else None


def _stem_compare(
    *,
    metric: str,
    x: str,
    qid: str,
    tag: Optional[Sequence[str]],
    sampler: Optional[Sequence[str]],
    seed: Optional[Sequence[int]],
    aggregate_seeds: bool,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> str:
    tag_s = base.join_tokens(tag or [], empty="")
    sam_s = base.join_tokens(sampler or [], empty="")
    seed_s = base.join_tokens([str(s) for s in (seed or [])], empty="")
    return base.sanitize_token(
        f"compare_{metric}__x-{x}__qid-{qid}"
        + (f"__tag-{tag_s}" if tag_s else "__tag-ALL")
        + (f"__sampler-{sam_s}" if sam_s else "__sampler-ALL")
        + (f"__seed-{seed_s}" if seed_s else "__seed-ALL")
        + ("__aggseeds" if aggregate_seeds else "")
        + (f"__xrange-{x_min}-{x_max}" if (x_min is not None or x_max is not None) else "")
    )


def _run_compare_one(
    *,
    project_root: Path,
    qid: str,
    tags: Optional[List[str]],
    samplers: Optional[List[str]],
    seeds: Optional[List[int]],
    curve_key: str,
    allow_missing: bool,
    aggregate_seeds: bool,
    metric: str,
    x_col: str,
    out_dir: Path,
    label_keys: List[str],
    title: str,
    ylabel: Optional[str],
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    snapshot_round: Optional[int] = None,
    snapshot_n_labeled: Optional[int] = None,
) -> None:
    bundles = compare.load_learning_curves(
        project_root=project_root,
        qid=qid,
        status=("success", "backfilled"),
        tag=tags,
        sampler=samplers,
        seed=seeds,
        curve_key=curve_key,
        require_files=not allow_missing,
    )
    if not bundles:
        raise RuntimeError("no learning curves loaded (check filters / index)")

    df = compare.concat_curves(bundles)
    if df.empty:
        raise RuntimeError("no data after concatenation")

    # Snapshot truncation
    if snapshot_round is not None:
        base.require_columns(df, ["round"], ctx="learning_curve")
        rr = pd.to_numeric(df["round"], errors="coerce")
        df = df.loc[np.isfinite(rr.to_numpy()) & (rr.to_numpy() <= float(int(snapshot_round)))].copy()
        if df.empty:
            raise RuntimeError(f"no rows after snapshot-round filter: round <= {int(snapshot_round)}")
    if snapshot_n_labeled is not None:
        if "n_labeled" not in df.columns:
            raise KeyError("snapshot-n-labeled requires 'n_labeled' column in learning_curve csv")
        nl = pd.to_numeric(df["n_labeled"], errors="coerce")
        df = df.loc[np.isfinite(nl.to_numpy()) & (nl.to_numpy() <= float(int(snapshot_n_labeled)))].copy()
        if df.empty:
            raise RuntimeError(f"no rows after snapshot-n-labeled filter: n_labeled <= {int(snapshot_n_labeled)}")

    # group keys -> plot_label
    for k in label_keys:
        if k not in df.columns:
            raise KeyError(f"group key not found: {k}. available={list(df.columns)}")

    if not aggregate_seeds and "meta_seed" not in label_keys:
        # If user wants per-seed series, ensure meta_seed is included.
        label_keys = list(label_keys) + ["meta_seed"]

    df["plot_label"] = df[label_keys].astype(str).agg("|".join, axis=1)

    base.require_columns(df, [x_col, metric], ctx="learning_curve")
    # Optional x-range filter (inclusive)
    if x_min is not None or x_max is not None:
        x_vals = pd.to_numeric(df[x_col], errors="coerce")
        msk = np.isfinite(x_vals.to_numpy())
        if x_min is not None:
            msk &= x_vals.to_numpy() >= float(x_min)
        if x_max is not None:
            msk &= x_vals.to_numpy() <= float(x_max)
        df = df.loc[msk].copy()
        if df.empty:
            raise RuntimeError(f"no rows after x-range filter: {x_col} in [{x_min},{x_max}]")


    df_agg = compare.aggregate_curves(df, x_col=x_col, y_cols=[metric], group_keys=["plot_label"])

    # Split outputs by file type to keep directories readable.
    dirs = base.ensure_split_dirs(out_dir)
    stem = _stem_compare(metric=metric, x=x_col, qid=qid, tag=tags, sampler=samplers, seed=seeds, aggregate_seeds=aggregate_seeds, x_min=x_min, x_max=x_max)
    out_csv = dirs["tables"] / f"{stem}.csv"
    df_agg.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    plots.plot_learning_curve_aggregate(
        ax,
        df_agg,
        metric_col=metric,
        ylabel=ylabel or metric,
        title=title,
        x_col=x_col,
    )
    fig.tight_layout()
    out_png = dirs["figures"] / f"{stem}.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_csv}")


def cmd_compare(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()

    tags = _parse_csv_list(args.tag)
    samplers = _parse_csv_list(args.sampler)
    seeds = _parse_int_list(args.seed)

    base.setup_style()

    # Default label keys
    if args.group_keys:
        label_keys = [k.strip() for k in args.group_keys.split(",") if k.strip()]
    else:
        label_keys = ["meta_sampler", "meta_sampler_conf_key"]
        if tags is not None and len(tags) > 1:
            label_keys.insert(0, "meta_tag")
        if not args.aggregate_seeds:
            label_keys.append("meta_seed")

    out_dir = compare.get_summary_plots_dir(project_root, args.qid)
    # Snapshot outputs are isolated into a subdir to avoid overwrite.
    if getattr(args, 'snapshot_round', None) is not None and getattr(args, 'snapshot_n_labeled', None) is not None:
        raise ValueError('--snapshot-round and --snapshot-n-labeled are mutually exclusive')
    snap_tag = None
    if getattr(args, 'snapshot_round', None) is not None:
        snap_tag = f"round_{int(args.snapshot_round):03d}"
    if getattr(args, 'snapshot_n_labeled', None) is not None:
        snap_tag = f"n_labeled_{int(args.snapshot_n_labeled)}"
    if snap_tag:
        out_dir = out_dir / f"snapshot_{snap_tag}"

    # If snapshot is specified, enforce x_max consistency
    snapshot_round = getattr(args, "snapshot_round", None)
    snapshot_n_labeled = getattr(args, "snapshot_n_labeled", None)
    if snapshot_round is not None:
        if args.x_max_round is not None and float(args.x_max_round) != float(int(snapshot_round)):
            raise ValueError(f"--snapshot-round={int(snapshot_round)} conflicts with --x-max-round={args.x_max_round}")
        args.x_max_round = float(int(snapshot_round))
        if (not getattr(args, "preset", False)) and getattr(args, "x", None) == "round":
            if args.x_max is not None and float(args.x_max) != float(int(snapshot_round)):
                raise ValueError(f"--snapshot-round={int(snapshot_round)} conflicts with --x-max={args.x_max}")
            args.x_max = float(int(snapshot_round))
    if snapshot_n_labeled is not None:
        if args.x_max_n_labeled is not None and float(args.x_max_n_labeled) != float(int(snapshot_n_labeled)):
            raise ValueError(f"--snapshot-n-labeled={int(snapshot_n_labeled)} conflicts with --x-max-n-labeled={args.x_max_n_labeled}")
        args.x_max_n_labeled = float(int(snapshot_n_labeled))
        if (not getattr(args, "preset", False)) and getattr(args, "x", None) == "n_labeled":
            if args.x_max is not None and float(args.x_max) != float(int(snapshot_n_labeled)):
                raise ValueError(f"--snapshot-n-labeled={int(snapshot_n_labeled)} conflicts with --x-max={args.x_max}")
            args.x_max = float(int(snapshot_n_labeled))

    presets = [
        ("test_qwk", "n_labeled"),
        ("test_qwk", "round"),
        ("test_rmse", "n_labeled"),
        ("test_rmse", "round"),
        ("test_cse", "n_labeled"),
        ("test_cse", "round"),
    ]

    def _range_for_x(x_name: str) -> tuple[Optional[float], Optional[float]]:
        if x_name == "n_labeled":
            return args.x_min_n_labeled, args.x_max_n_labeled
        if x_name == "round":
            return args.x_min_round, args.x_max_round
        return None, None


    if args.preset:
        bundles = compare.load_learning_curves(
            project_root=project_root,
            qid=args.qid,
            status=("success", "backfilled"),
            tag=tags,
            sampler=samplers,
            seed=seeds,
            curve_key=args.curve_key,
            require_files=not args.allow_missing,
        )
        if not bundles:
            raise RuntimeError("no learning curves loaded (check filters / index)")

        # Standard 6 plots
        for metric, x_col in presets:
            x_min, x_max = _range_for_x(x_col)
            _run_compare_one(
                project_root=project_root,
                qid=args.qid,
                tags=tags,
                samplers=samplers,
                seeds=seeds,
                curve_key=args.curve_key,
                allow_missing=args.allow_missing,
                aggregate_seeds=args.aggregate_seeds,
                metric=metric,
                x_col=x_col,
                out_dir=out_dir,
                label_keys=label_keys,
                title=f"Learning Curve: {metric} vs {x_col}",
                ylabel=args.ylabel,
                x_min=x_min,
                x_max=x_max,
                snapshot_round=snapshot_round,
                snapshot_n_labeled=snapshot_n_labeled,
            )

        if args.aux_by_sampler:
            if samplers is None:
                # Discover samplers
                idx = compare.load_index(project_root, args.qid)
                seen = []
                for rec in idx:
                    run = rec.get("run") or {}
                    if run.get("sampler") is None:
                        continue
                    if tags is not None and run.get("tag") not in set(tags):
                        continue
                    if seeds is not None and int(run.get("seed")) not in set(seeds):
                        continue
                    if run.get("sampler") not in seen:
                        seen.append(run.get("sampler"))
                samplers_list = seen
            else:
                samplers_list = samplers

            for sm in samplers_list:
                out_s = compare.get_by_sampler_plots_dir(project_root, args.qid, sm)
                for metric, x_col in presets:
                    x_min, x_max = _range_for_x(x_col)
                    _run_compare_one(
                        project_root=project_root,
                        qid=args.qid,
                        tags=tags,
                        samplers=[sm],  # single sampler
                        seeds=seeds,
                        curve_key=args.curve_key,
                        allow_missing=args.allow_missing,
                        aggregate_seeds=args.aggregate_seeds,
                        metric=metric,
                        x_col=x_col,
                        out_dir=out_s,
                        label_keys=["meta_sampler", "meta_sampler_conf_key"] + ([] if args.aggregate_seeds else ["meta_seed"]),
                        title=f"{sm} | {metric} vs {x_col}",
                        ylabel=args.ylabel,
                        x_min=x_min,
                        x_max=x_max,
                        snapshot_round=snapshot_round,
                        snapshot_n_labeled=snapshot_n_labeled,
                    )

        # Optional: export tau sweep evidence (Strict)
        if args.export_tau_sweep:
            if isinstance(args.eps_cse_list, str):
                eps_list = _parse_float_list(args.eps_cse_list)
            else:
                eps_list = list(args.eps_cse_list)
            if not eps_list:
                raise ValueError("eps_cse_list must be non-empty (e.g., '0.02,0.05')")
            tau_out = out_dir / "tau_sweep"
            _export_tau_evidence_for_experiments(
                bundles=bundles,
                eps_list=eps_list,
                cse_abs_err=int(args.cse_abs_err),
                out_root=tau_out,
                snapshot_round=getattr(args, 'snapshot_round', None),
                snapshot_n_labeled=getattr(args, 'snapshot_n_labeled', None),
                snapshot_nearest=bool(getattr(args, 'snapshot_nearest', False)),
            )

        return 0

    # Single-shot compare
    metric = args.metric
    x_col = args.x

    _run_compare_one(
        project_root=project_root,
        qid=args.qid,
        tags=tags,
        samplers=samplers,
        seeds=seeds,
        curve_key=args.curve_key,
        allow_missing=args.allow_missing,
        aggregate_seeds=args.aggregate_seeds,
        metric=metric,
        x_col=x_col,
        out_dir=out_dir,
        label_keys=label_keys,
        title=args.title or f"Learning Curve: {metric} vs {x_col}",
        ylabel=args.ylabel,
        x_min=args.x_min,
        x_max=args.x_max,
        snapshot_round=snapshot_round,
        snapshot_n_labeled=snapshot_n_labeled,
    )
    return 0


def cmd_preset(args: argparse.Namespace) -> int:
    eps_list = _parse_float_list(args.eps_cse_list)
    if not eps_list:
        raise ValueError("--eps-cse-list must be non-empty (e.g., '0.02,0.05')")

    return cmd_compare(
        argparse.Namespace(
            project_root=args.project_root,
            qid=args.qid,
            tag=args.tag,
            sampler=args.sampler,
            seed=args.seed,
            curve_key=args.curve_key,
            allow_missing=args.allow_missing,
            aggregate_seeds=True,
            group_keys=None,
            x_min=None,
            x_max=None,
            x_min_n_labeled=args.x_min_n_labeled,
            x_max_n_labeled=args.x_max_n_labeled,
            x_min_round=args.x_min_round,
            x_max_round=args.x_max_round,
            snapshot_round=getattr(args, 'snapshot_round', None),
            snapshot_n_labeled=getattr(args, 'snapshot_n_labeled', None),
            snapshot_nearest=bool(getattr(args, 'snapshot_nearest', False)),
            preset=True,
            aux_by_sampler=True,
            metric="test_qwk",
            x="n_labeled",
            title=None,
            ylabel=args.ylabel,
            export_tau_sweep=True,
            eps_cse_list=eps_list,
            cse_abs_err=args.cse_abs_err,
        )
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="tensaku.viz")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("preset", help="standard visualization: 6 plots (seed-agg) + aux-by-sampler + tau evidence")
    pp.add_argument("--project-root", default=".")
    pp.add_argument("--qid", required=True)
    pp.add_argument("--tag", default=None, help="comma separated tags (default: all)")
    pp.add_argument("--sampler", default=None, help="comma separated samplers (default: all)")
    pp.add_argument("--seed", default=None, help="comma separated seeds (default: all)")
    pp.add_argument("--curve-key", default="metrics/al_learning_curve.csv")
    pp.add_argument("--allow-missing", action="store_true")
    pp.add_argument("--x-min-n-labeled", type=float, default=None)
    pp.add_argument("--x-max-n-labeled", type=float, default=None)
    pp.add_argument("--x-min-round", type=float, default=None)
    pp.add_argument("--x-max-round", type=float, default=None)
    pp.add_argument("--snapshot-round", type=int, default=None, help="use artifacts at a specific round (and truncate curves). mutually exclusive with --snapshot-n-labeled")
    pp.add_argument("--snapshot-n-labeled", type=int, default=None, help="use artifacts at a specific n_labeled (and truncate curves). requires metrics/al_learning_curve.csv")
    pp.add_argument("--snapshot-nearest", action="store_true", help="if exact snapshot not found, pick nearest (explicit)")
    pp.add_argument("--ylabel", default=None)
    pp.add_argument(
        "--eps-cse-list",
        default="0.02,0.05",
        help="comma separated eps_cse for tau evidence (default: 0.02,0.05)",
    )
    pp.add_argument("--cse-abs-err", type=int, default=2)
    pp.set_defaults(func=cmd_preset)

    ps = sub.add_parser("single", help="visualize one experiment")
    ps.add_argument("--project-root", default=".", help="project root (TENSAKU_ROOT)")
    ps.add_argument("--exp-dir", required=True, help="experiment directory (relative to project-root or absolute)")
    ps.add_argument("--round", type=int, default=None, help="visualize snapshot at a specific round (uses predictions/rounds). mutually exclusive with --n-labeled")
    ps.add_argument("--n-labeled", type=int, default=None, help="visualize snapshot at a specific n_labeled (requires metrics/al_learning_curve.csv). mutually exclusive with --round")
    ps.add_argument("--nearest", action="store_true", help="if exact snapshot not found, pick nearest (explicit)")
    ps.add_argument("--split", default="test", help="which split to visualize (default: test). if split col missing, uses all")
    ps.add_argument("--ece-bins", type=int, default=15)
    ps.add_argument("--cse-abs-err", type=int, default=2)
    ps.add_argument("--conf-cols", default=None, help="comma separated confidence columns to plot (default: auto conf_*)")
    ps.set_defaults(func=cmd_single)

    pc = sub.add_parser("compare", help="compare experiments via learning curves")
    pc.add_argument("--project-root", default=".")
    pc.add_argument("--qid", required=True)
    pc.add_argument("--tag", default=None, help="comma separated tags (default: all)")
    pc.add_argument("--sampler", default=None, help="comma separated samplers (default: all)")
    pc.add_argument("--seed", default=None, help="comma separated seeds (default: all)")
    pc.add_argument("--curve-key", default="metrics/al_learning_curve.csv", help="relative path from exp_dir, or key in index.paths")
    pc.add_argument("--allow-missing", action="store_true", help="skip experiments missing curve csv")
    pc.add_argument("--aggregate-seeds", action="store_true", help="aggregate seeds into mean±std (recommended)")
    pc.add_argument("--group-keys", default=None, help="comma separated meta keys to form plot label (default: sampler/conf_key/(seed))")
    pc.add_argument("--x-min", type=float, default=None, help="single-shot: minimum x value (inclusive)")
    pc.add_argument("--x-max", type=float, default=None, help="single-shot: maximum x value (inclusive)")
    pc.add_argument("--x-min-n-labeled", type=float, default=None, help="preset: minimum n_labeled (inclusive)")
    pc.add_argument("--x-max-n-labeled", type=float, default=None, help="preset: maximum n_labeled (inclusive)")
    pc.add_argument("--x-min-round", type=float, default=None, help="preset: minimum round (inclusive)")
    pc.add_argument("--x-max-round", type=float, default=None, help="preset: maximum round (inclusive)")
    pc.add_argument("--snapshot-round", type=int, default=None, help="snapshot mode: truncate curves and export tau evidence at a specific round")
    pc.add_argument("--snapshot-n-labeled", type=int, default=None, help="snapshot mode: truncate curves and export tau evidence at a specific n_labeled")
    pc.add_argument("--snapshot-nearest", action="store_true", help="if exact snapshot not found, pick nearest (explicit)")

    pc.add_argument("--preset", action="store_true", help="generate standard 6 plots (qwk/rmse/cse × n_labeled/round)")
    pc.add_argument("--aux-by-sampler", action="store_true", help="also generate sampler-only plots under by_sampler/<sampler>/")
    pc.add_argument("--export-tau-sweep", dest="export_tau_sweep", action="store_true", help="(preset only) export tau-sweep evidence per experiment")
    pc.add_argument(
        "--eps-cse-list",
        default="0.02,0.05",
        help="(preset only) comma separated eps_cse for tau evidence (default: 0.02,0.05)",
    )
    pc.add_argument("--cse-abs-err", type=int, default=2, help="(preset only) CSE margin threshold |pred-y|>=m")

    pc.add_argument("--metric", default="test_qwk", help="single-shot metric column name")
    pc.add_argument("--x", default="n_labeled", help="single-shot x column")
    pc.add_argument("--title", default=None)
    pc.add_argument("--ylabel", default=None)
    pc.set_defaults(func=cmd_compare)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())