"""Unified training entry for NORMST Visium and paired Visium HD tasks.

Examples::

    python train.py --task visium --visium-root /data/visium \
      --output-dir save/multislice/seed2027 --seed 2027

    python train.py --task visium_hd \
      --lr-dir /data/square_016um --hr-dir /data/square_008um \
      --output-dir save/visium_hd/seed2027 --seed 2027

Use ``python train.py --task <task> --help`` for task-specific options.
"""

from __future__ import annotations

import argparse
import sys


TASKS = ("visium", "visium_hd")


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
        help="visium: multi-slice reconstruction; visium_hd: paired 16-to-8 training",
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
    if task_args.task == "visium":
        from training.visium import main as task_main
    else:
        from training.visium_hd import main as task_main
    return task_main(remaining)


if __name__ == "__main__":
    main()
