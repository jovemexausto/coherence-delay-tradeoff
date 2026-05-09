from __future__ import annotations

import importlib
import sys
from pathlib import Path


COMMANDS = {
    "particle": "experiments.cli.run_particle",
    "gaussian": "experiments.cli.run_gaussian",
    "cuberoot_adwin": "experiments.cli.run_cuberoot_adwin",
    "rajput": "experiments.cli.run_rajput",
    "bikes": "experiments.cli.run_bikes",
    "elec2": "experiments.cli.run_elec2",
    "airlines": "experiments.cli.run_airlines",
    "kuairand": "experiments.cli.run_kuairand",
    "kuairand_followup": "experiments.cli.run_kuairand_followup",
    "all": "experiments.cli.run_all",
}


def _print_help() -> None:
    print("Usage: python run.py <command> [args]\n")
    print("Commands:")
    for name in COMMANDS:
        print(f"  {name}")


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        _print_help()
        return

    command = sys.argv[1]
    module_name = COMMANDS.get(command)
    if module_name is None:
        raise SystemExit(f"Unknown command: {command}")

    module = importlib.import_module(module_name)
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    module.main()


if __name__ == "__main__":
    main()
