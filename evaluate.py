import json
import re
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from llm import LLM
from knowno.embedding import EmbeddingSelector
from knowno.classify import AmbiguityClassifier
from knowno.pipeline import EntityExtractor, TaskHandler


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def parse_plan_steps(plan_text: str) -> list[str]:
    """Parse numbered plan text ('1. Step...\n2. Step...') into a list."""
    if not isinstance(plan_text, str):
        return []
    steps = []
    for line in plan_text.strip().split('\n'):
        line = line.strip()
        if re.match(r'^\d+\.', line):
            step = re.sub(r'^\d+\.\s*', '', line).strip()
            if step:
                steps.append(step)
    return steps


def parse_csv_list(value: str) -> list[str]:
    """Parse comma-separated string into a list of lowercase stripped strings."""
    if not isinstance(value, str) or not value.strip():
        return []
    return [item.strip().lower() for item in value.split(',') if item.strip()]


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_step(handler: TaskHandler, task: str, environment: list[str],
             act: str, prefix: list[str]) -> dict:
    """
    Run the full pipeline (extract → embed → classify) on a single step.
    Injects task and environment directly instead of going through find_environment.
    """
    handler.current_task = task
    handler.environment = environment

    history = [{"user_message": s, "robot_message": "Done."} for s in prefix]
    return handler.handle_step(act, history=history if history else None)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def fuzzy_match(a: str, b: str) -> bool:
    """True if a is a substring of b or b is a substring of a (case-insensitive)."""
    a, b = a.lower().strip(), b.lower().strip()
    return a in b or b in a


def compute_icr(viable_objects: list[str], user_intent_str: str) -> float:
    """
    ICR = (# user_intent items found in viable_objects) / (# total intent items).
    Matching is done via substring (fuzzy) match.
    Returns -1 if user_intent is empty.
    """
    intents = parse_csv_list(user_intent_str)
    if not intents:
        return -1

    viable_lower = [v.lower().strip() for v in (viable_objects or [])]
    found = sum(
        1 for intent in intents
        if any(fuzzy_match(intent, v) for v in viable_lower)
    )
    return found / len(intents)


def compute_ssc(viable_objects: list[str], amb_shortlist_str: str) -> float:
    """
    SSC = |CS ∩ PS| / |CS ∪ PS| using fuzzy matching between items.
    CS = amb_shortlist, PS = viable_objects.
    Returns -1 if either set is empty.
    """
    cs_items = parse_csv_list(amb_shortlist_str)
    ps_items = [v.lower().strip() for v in (viable_objects or [])]

    if not cs_items or not ps_items:
        return -1

    # Build matched intersection: CS items that have a fuzzy match in PS
    cs_matched = set()
    ps_matched = set()
    for ci, c in enumerate(cs_items):
        for pi, p in enumerate(ps_items):
            if fuzzy_match(c, p):
                cs_matched.add(ci)
                ps_matched.add(pi)

    intersection = len(cs_matched)
    # Union = total unique items in CS + PS minus matched pairs (counted once)
    union = len(cs_items) + len(ps_items) - intersection
    if union == 0:
        return -1
    return intersection / union


def compute_ambdiff(status_amb: str, status_clear: str) -> float:
    """
    AmbDiff score tại step end_of_ambiguity:
      - Cả hai đúng (amb=Ambiguous, clr=Unambiguous) → 1.0
      - Chỉ một trong hai đúng                       → 0.5
      - Cả hai sai                                   → 0.0
    """
    amb_ok = (status_amb or "").strip().lower() == "ambiguous"
    clr_ok = (status_clear or "").strip().lower() == "unambiguous"
    return amb_ok * 0.5 + clr_ok * 0.5


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def safe_mean(values: list[float]) -> float:
    filtered = [v for v in values if v >= 0]
    return float(np.mean(filtered)) if filtered else -1.0


def build_summary_text(results: list[dict]) -> str:
    df = pd.DataFrame(results)
    labels = ['preferences', 'common_sense_knowledge', 'safety']

    lines = []
    lines.append("=" * 60)
    lines.append(f"EVALUATION RESULTS ({len(results)} rows)")
    lines.append("=" * 60)

    # AmbDiff — overall
    amdbdiff_vals = [r['AmbDiff'] for r in results if r['AmbDiff'] >= 0]
    lines.append("")
    lines.append(f"AmbDiff (all labels): {safe_mean(amdbdiff_vals):.4f}  (n={len(amdbdiff_vals)})")

    # ICR — per label
    lines.append("")
    lines.append("ICR per label:")
    lines.append(f"  {'Label':<30} {'ICR':>8}  {'n':>5}")
    lines.append(f"  {'-'*45}")
    for label in labels:
        sub = df[df['ambiguity_type'] == label]
        vals = [v for v in sub['ICR'] if v >= 0]
        lines.append(f"  {label:<30} {safe_mean(vals):>8.4f}  {len(vals):>5}")

    # SSC — preferences only
    pref = df[df['ambiguity_type'] == 'preferences']
    ssc_vals = [v for v in pref['SSC'] if v >= 0]
    lines.append("")
    lines.append(f"SSC (preferences only): {safe_mean(ssc_vals):.4f}  (n={len(ssc_vals)})")
    lines.append("=" * 60)
    return "\n".join(lines)


def aggregate_and_print(results: list[dict]) -> str:
    summary = build_summary_text(results)
    print("\n" + summary)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline on AmbiK dataset.")
    parser.add_argument("--n_rows", type=int, default=300, help="Number of rows to evaluate.")
    parser.add_argument("--top_k", type=int, default=3, help="Top-K for embedding selector.")
    parser.add_argument("--output", type=str, default="eval_results.jsonl",
                        help="Path to save per-row results (JSONL).")
    parser.add_argument("--metric_output", type=str, default="metric.txt",
                        help="Path to save aggregated metric summary (TXT).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file (skip already-processed rows).")
    args = parser.parse_args()

    dataset_path = _root / "ambik_dataset" / "ambik_test_400.csv"
    output_path = _root / args.output

    df = pd.read_csv(dataset_path).head(args.n_rows)

    # Load already-processed row IDs if resuming
    done_ids: set[int] = set()
    existing_results: list[dict] = []
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done_ids.add(row['id'])
                    existing_results.append(row)
                except Exception:
                    pass
        print(f"[Resume] Loaded {len(done_ids)} already-processed rows.")

    # Setup pipeline (LLMs + components)
    llm = LLM("ollama:qwen2.5:32b", {"max_new_tokens": 150, "temperature": 0.7})
    embed_model = LLM("ollama:mxbai-embed-large:latest", {})

    extractor = EntityExtractor(llm)
    embedding_selector = EmbeddingSelector(embed_model)
    classifier = AmbiguityClassifier(llm)
    # Pass env_matcher=None; we inject environment directly in run_step()
    handler = TaskHandler(extractor, None, embedding_selector, classifier)

    new_results: list[dict] = []

    with open(output_path, 'a') as f_out:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            row_id = int(row.get('id', -1))
            if row_id in done_ids:
                continue

            try:
                # ── Parse inputs ─────────────────────────────────────────────
                environment = [e.strip() for e in
                               row.get('environment_short', '').split(',') if e.strip()]

                amb_steps   = parse_plan_steps(row.get('plan_for_amb_task', ''))
                clear_steps = parse_plan_steps(row.get('plan_for_clear_task', ''))

                step_idx    = int(row.get('end_of_ambiguity', 0))
                amb_type    = str(row.get('ambiguity_type', ''))
                user_intent = str(row.get('user_intent', ''))
                amb_short   = str(row.get('amb_shortlist', ''))
                amb_task    = str(row.get('ambiguous_task', ''))
                clear_task  = str(row.get('unambiguous_direct', ''))

                def safe_step(steps, idx):
                    if idx < len(steps):
                        return steps[idx], steps[:idx]
                    return steps[-1] if steps else '', steps[:-1]

                amb_act,   amb_prefix   = safe_step(amb_steps,   step_idx)
                clear_act, clear_prefix = safe_step(clear_steps, step_idx)

                # ── Run pipeline on AMBIGUOUS step ────────────────────────────
                res_amb = run_step(handler, amb_task, environment, amb_act, amb_prefix)
                cls_amb = res_amb.get('classification') or {}
                viable_amb  = cls_amb.get('viable_objects', [])
                set_size_amb = len(viable_amb)

                # ── Run pipeline on CLEAR step ────────────────────────────────
                res_clr = run_step(handler, clear_task, environment, clear_act, clear_prefix)
                cls_clr = res_clr.get('classification') or {}
                viable_clr  = cls_clr.get('viable_objects', [])
                set_size_clr = len(viable_clr)

                # ── Compute metrics ───────────────────────────────────────────
                status_amb = cls_amb.get('status', '')
                status_clr = cls_clr.get('status', '')
                amdbdiff = compute_ambdiff(status_amb, status_clr)
                icr      = compute_icr(viable_amb, user_intent)
                ssc      = (compute_ssc(viable_amb, amb_short)
                            if 'preference' in amb_type.lower() else -1)

                record = {
                    'id':            row_id,
                    'ambiguity_type': amb_type,
                    'amb_act':        amb_act,
                    'clear_act':      clear_act,
                    'viable_amb':     viable_amb,
                    'viable_clr':     viable_clr,
                    'set_size_amb':   set_size_amb,
                    'set_size_clr':   set_size_clr,
                    'AmbDiff':        amdbdiff,
                    'ICR':            icr,
                    'SSC':            ssc,
                    'status_amb':     status_amb,
                    'status_clr':     status_clr,
                    'label_amb':      cls_amb.get('label', ''),
                }

                new_results.append(record)
                f_out.write(json.dumps(record) + '\n')
                f_out.flush()

            except Exception as e:
                tqdm.write(f"[Error] Row {row_id}: {e}")

    all_results = existing_results + new_results
    if all_results:
        summary_text = aggregate_and_print(all_results)
        metric_path = _root / args.metric_output
        with open(metric_path, "w", encoding="utf-8") as f_metric:
            f_metric.write(summary_text + "\n")
        print(f"[Saved] Metric summary written to: {metric_path}")
    else:
        print("No results to aggregate.")


if __name__ == '__main__':
    main()
