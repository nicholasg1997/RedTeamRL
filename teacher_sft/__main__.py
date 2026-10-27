"""Offline commands; never creates Modal resources or executes harvested commands."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--harvest", required=True)
    prepare.add_argument("--output", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--dataset", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--tuned", required=True)
    compare.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        from teacher_sft.data import prepare
        result = prepare(args.harvest, args.output)
    elif args.command == "inspect":
        from collections import Counter
        from teacher_sft.author import load_examples
        meta, rows = load_examples(args.dataset)
        result = {**meta, "by_split_kind": dict(Counter(f"{r['split']}/{r['kind']}" for r in rows))}
    else:
        from teacher_sft.evaluate import compare
        result = compare(args.baseline, args.tuned, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
