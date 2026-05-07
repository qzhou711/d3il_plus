"""Aggregate all CD eval JSONL files into a single Markdown table."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_FILES = {
    "iter1_pureCD_200": "/tmp/cd_iter1_eval.jsonl",
    "iter2_long_500":   "/tmp/cd_iter2_eval.jsonl",
    "iter3_anchor01":   "/tmp/iter3_eval.jsonl",
    "iter4_huber02_sigW": "/tmp/iter4_eval.jsonl",
    "iter6_anchor02":   "/tmp/iter6_eval.jsonl",
    "iter7_lr5e5_300":  "/tmp/iter7_eval.jsonl",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None,
                    help="experiment=path pairs (default: hard-coded set)")
    args = ap.parse_args()

    files = DEFAULT_FILES.copy()
    if args.files:
        for kv in args.files:
            k, v = kv.split("=", 1)
            files[k] = v

    by_exp = defaultdict(list)
    for name, path in files.items():
        p = Path(path)
        if not p.exists():
            print(f"[skip] {name}: {path} missing")
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_exp[name].append(row)

    print(f"\n{'experiment':<22} {'K':>3} {'epoch':>10} {'success':>8} {'entropy':>8}")
    print("-" * 56)
    for name in files:
        rows = by_exp.get(name, [])
        if not rows:
            continue

        def keyf(r):
            e = r.get("epoch", "")
            ord_e = -1 if e == "eval_best" else (10**9 if e == "last" else int(e) if isinstance(e, int) or (isinstance(e, str) and e.isdigit()) else 0)
            return (r.get("num_inference_steps", 0), ord_e)

        for r in sorted(rows, key=keyf):
            print(f"{name:<22} {r.get('num_inference_steps','?'):>3} {str(r.get('epoch','?')):>10} "
                  f"{r.get('success_rate', 0):>8.3f} {r.get('entropy', 0):>8.3f}")
        print()

    # Best per experiment
    print("\n=== BEST PER EXPERIMENT (by success_rate) ===")
    print(f"{'experiment':<22} {'K':>3} {'epoch':>10} {'success':>8} {'entropy':>8}")
    print("-" * 56)
    for name in files:
        rows = by_exp.get(name, [])
        if not rows:
            continue
        best = max(rows, key=lambda r: r.get("success_rate", 0))
        print(f"{name:<22} {best.get('num_inference_steps','?'):>3} {str(best.get('epoch','?')):>10} "
              f"{best.get('success_rate', 0):>8.3f} {best.get('entropy', 0):>8.3f}")


if __name__ == "__main__":
    main()
