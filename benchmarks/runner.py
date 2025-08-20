# benchmarks/runner.py
"""
Benchmark runner for "GoT for Bias Mitigation"

Usage:
    # Synthetic quick run (offline-safe)
    python -m benchmarks.runner --dataset synthetic --samples 50

    # CrowS-Pairs (requires internet + `pip install datasets`)
    python -m benchmarks.runner --dataset crows --samples 200

Outputs:
    - JSONL with per-sample results
    - CSV summary
    - Markdown report with aggregate metrics
    - Debiased artifacts produced by got.py are NOT modified here; we only evaluate.
"""
import argparse
import csv
import os
import time
from datetime import datetime

from got import GraphOfThought
from benchmarks import datasets as ds
from benchmarks import baselines as bl
from benchmarks import metrics as mt

def debias_with_got(got: GraphOfThought, text: str, max_depth: int, bias_threshold: float):
    root = got.create_root_node(text)
    best = got.traverse_graph(root, max_depth=max_depth, bias_threshold=bias_threshold)
    return best

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def run(args):
    ensure_dir(args.outdir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, f"{args.dataset}_{stamp}")
    ensure_dir(run_dir)

    # Load data
    items = ds.load_dataset(args.dataset, split="test", n=args.samples)

    # Initialize GoT once to reuse caches and be deterministic
    got = GraphOfThought()

    # Prepare outputs
    jsonl_path = os.path.join(run_dir, "results.jsonl")
    csv_path = os.path.join(run_dir, "summary.csv")
    md_path = os.path.join(run_dir, "report.md")

    rows_for_csv = []
    agg_accum = []

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        writer = None  # write lines manually

        for idx, ex in enumerate(items):
            text = ex["text"]
            ex_id = ex.get("id", f"{args.dataset}-{idx:06d}")
            meta = ex.get("metadata", {})

            # --- GoT ---
            t0 = time.time()
            best = debias_with_got(got, text, args.max_depth, args.bias_threshold)
            t1 = time.time()
            got_time = t1 - t0

            got_cbs, got_crs, got_break = mt.got_scores(got, text, best.text_content)
            got_sim = mt.semantic_similarity(text, best.text_content)
            got_repl = mt.token_replacement_rate(best.text_content)

            # --- Baselines ---
            baseline_results = {}
            for name, fn in bl.BASELINES.items() if hasattr(bl, "BASELINES") else bl.BASELINES.items():
                pass  # guard for typo; below use correct dict

            baseline_results = {}
            for name, fn in bl.BASELINES.items():
                btext, binfo = fn(text)
                b_cbs, b_crs, b_break = mt.got_scores(got, text, btext)
                b_sim = mt.semantic_similarity(text, btext)
                b_repl = mt.token_replacement_rate(btext)
                baseline_results[name] = {
                    "text": btext,
                    "meta": binfo,
                    "cbs": b_cbs,
                    "crs": b_crs,
                    "sim": b_sim,
                    "replace_rate": b_repl,
                    "time_sec": None,  # trivial baselines; negligible
                    "breakdown": b_break,
                }

            record = {
                "id": ex_id,
                "dataset": args.dataset,
                "original": {
                    "text": text,
                    "meta": meta,
                },
                "got": {
                    "node_id": best.node_id,
                    "text": best.text_content,
                    "cbs": got_cbs,
                    "crs": got_crs,
                    "sim": got_sim,
                    "replace_rate": got_repl,
                    "time_sec": got_time,
                    "breakdown": got_break,
                    "transformations": best.transformation_history,
                    "manifest_len": len(best.masked_manifest),
                },
                "baselines": baseline_results,
            }

            jf.write((str(record) + "\n"))  # human-readable; not strict JSON (python dict repr)
            # also store machine-friendly row summaries
            rows_for_csv.append({
                "id": ex_id,
                "variant": "GoT",
                "cbs": got_cbs,
                "crs": got_crs,
                "sim": got_sim,
                "replace_rate": got_repl,
                "time_sec": got_time,
            })
            agg_accum.append({
                "got_cbs": got_cbs,
                "got_crs": got_crs,
                "got_sim": got_sim,
                "got_replace_rate": got_repl,
            })
            for name, res in baseline_results.items():
                rows_for_csv.append({
                    "id": ex_id,
                    "variant": name,
                    "cbs": res["cbs"],
                    "crs": res["crs"],
                    "sim": res["sim"],
                    "replace_rate": res["replace_rate"],
                    "time_sec": res["time_sec"] if res["time_sec"] is not None else 0.0,
                })
                agg_accum.append({
                    f"{name}_cbs": res["cbs"],
                    f"{name}_crs": res["crs"],
                    f"{name}_sim": res["sim"],
                    f"{name}_replace_rate": res["replace_rate"],
                })

    # write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=["id", "variant", "cbs", "crs", "sim", "replace_rate", "time_sec"])
        writer.writeheader()
        writer.writerows(rows_for_csv)

    # aggregate (simple means over pooled rows)
    # We compute means for each shared metric key
    # For clarity in the report we compute grouped means variant-wise:
    def group_mean(rows, variant):
        xs = [r for r in rows if r["variant"] == variant]
        if not xs:
            return {}
        return {
            "mean_cbs": sum(r["cbs"] for r in xs)/len(xs),
            "mean_crs": sum(r["crs"] for r in xs)/len(xs),
            "mean_sim": sum(r["sim"] for r in xs)/len(xs),
            "mean_replace_rate": sum(r["replace_rate"] for r in xs)/len(xs),
            "mean_time_sec": sum(r["time_sec"] for r in xs)/len(xs),
            "n": len(xs),
        }

    variants = sorted({r["variant"] for r in rows_for_csv})
    grouped = {v: group_mean(rows_for_csv, v) for v in variants}

    # write Markdown report
    with open(md_path, "w", encoding="utf-8") as mf:
        mf.write(f"# GoT for Bias Mitigation — Benchmark Report\n\n")
        mf.write(f"- Date: {stamp}\n")
        mf.write(f"- Dataset: `{args.dataset}`\n")
        mf.write(f"- Samples: {len(items)}\n")
        mf.write(f"- GoT params: max_depth={args.max_depth}, bias_threshold={args.bias_threshold}\n\n")

        mf.write("## Aggregate Metrics (means)\n\n")
        mf.write("| Variant | mean_cbs ↓ | mean_crs | mean_sim | mean_replace_rate | mean_time_sec | n |\n")
        mf.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for v in variants:
            gm = grouped[v]
            mf.write(f"| {v} | {gm.get('mean_cbs', float('nan')):.4f} | {gm.get('mean_crs', float('nan')):.4f} | "
                     f"{gm.get('mean_sim', float('nan')):.4f} | {gm.get('mean_replace_rate', float('nan')):.4f} | "
                     f"{gm.get('mean_time_sec', float('nan')):.4f} | {gm.get('n', 0)} |\n")

        mf.write("\nNotes:\n")
        mf.write("- `cbs` (Composite Bias Score) is lower-better, computed via got.py weights.\n")
        mf.write("- `crs` (Content Retention Score) ~ SequenceMatcher ratio vs original (0–1).\n")
        mf.write("- `sim` is SBERT cosine if available, else SequenceMatcher.\n")
        mf.write("- `replace_rate` is fraction of `[MASK:...]` tokens among words (approx.).\n")
        mf.write("- Expect GoT to achieve **lower mean_cbs** than fixed-order baselines while keeping **crs/sim** high.\n")

    print(f"\nWrote:\n- {jsonl_path}\n- {csv_path}\n- {md_path}\n- Run dir: {run_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="synthetic", help="synthetic|crows")
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--max_depth", type=int, default=5)
    ap.add_argument("--bias_threshold", type=float, default=0.15)
    ap.add_argument("--outdir", type=str, default=os.path.join("benchmarks", "results"))
    args = ap.parse_args()
    run(args)
