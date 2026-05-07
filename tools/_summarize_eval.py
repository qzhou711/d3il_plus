"""Pretty-print a CD eval JSONL produced by tools/eval_avoiding_checkpoints.py."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    rows = []
    for p in args.paths:
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["_src"] = p
            rows.append(r)

    def order(e):
        if e == "eval_best":
            return -1
        if e == "last":
            return 10**9
        try:
            return int(e)
        except (ValueError, TypeError):
            return 0

    print(f'{"src":<40} {"K":>3} {"epoch":>10} {"success":>8} {"entropy":>8}')
    last_src = None
    for r in sorted(rows, key=lambda x: (x["_src"], x.get("num_inference_steps", 0), order(x.get("epoch", "0")))):
        src = Path(r["_src"]).name
        if src != last_src:
            print()
            last_src = src
        print(
            f'{src:<40} '
            f'{r.get("num_inference_steps","?"):>3} '
            f'{str(r.get("epoch","?")):>10} '
            f'{r.get("success_rate", 0):>8.3f} '
            f'{r.get("entropy", 0):>8.3f}'
        )


if __name__ == "__main__":
    main()
