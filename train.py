"""Unified training entry for NORMST, ProNORMST, AE-NORMST, and Visium HD.

Examples::

    python train.py --task visium --manifest manifests/split.json \
      --output-dir save/multislice/run2027 --seed 2027

    python train.py --task visium --model pro-normst \
      --manifest pre-train/manifests/random_pair_8_2_2_seed2027.json \
      --output-dir save/pro_normst/round-001/pilot2027 --seed 2027 \
      --round-id round-001 --round-reason "frozen baseline candidate"

    python train.py --task visium_hd \
      --lr-dir /data/square_016um --hr-dir /data/square_008um \
      --output-dir save/visium_hd/seed2027 --seed 2027

    python train.py --task ae_visium \
      --manifest pre-train/manifests/random_pair_8_2_2_seed2027.json \
      --ae-checkpoint pre-train/runs/example/best.pt \
      --output-dir save/ae_normst/seed2027 --seed 2027

Use ``python train.py --task <task> --help`` for task-specific options.
"""

from __future__ import annotations

import argparse
import sys


TASKS = ("visium", "ae_visium", "visium_hd")


def _task_parser(add_help: bool = False):
    parser = argparse.ArgumentParser(
        description=__doc__,
        add_help=add_help,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=TASKS,
        required=True,
        help=(
            "visium: gene-space multi-slice reconstruction; "
            "ae_visium: frozen-AE composition reconstruction; "
            "visium_hd: paired 16-to-8 training"
        ),
    )
    parser.add_argument(
        "--model",
        choices=("legacy", "pro-normst"),
        default="legacy",
        help=(
            "model implementation for --task visium; legacy preserves the "
            "existing VisiumNORMST path, pro-normst selects direct-512 ProNORMST"
        ),
    )
    return parser


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or (
        any(value in {"-h", "--help"} for value in arguments)
        and not any(
            value == "--task" or value.startswith("--task=")
            for value in arguments
        )
    ):
        _task_parser(add_help=True).print_help()
        return 0

    task_args, remaining = _task_parser().parse_known_args(arguments)
    if task_args.model == "pro-normst" and task_args.task != "visium":
        _task_parser(add_help=True).error("--model pro-normst requires --task visium")
    if task_args.task == "visium" and task_args.model == "pro-normst":
        from training.pro_normst import main as task_main
    elif task_args.task == "visium":
        from training.visium import main as task_main
    elif task_args.task == "visium_hd":
        from training.visium_hd import main as task_main
    else:
        from training.ae_visium import main as task_main
    return task_main(remaining)


if __name__ == "__main__":
    main()
