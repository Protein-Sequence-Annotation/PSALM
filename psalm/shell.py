from __future__ import annotations

import atexit
import argparse
import shlex
import sys
from pathlib import Path

try:
    import readline
except ImportError:  # pragma: no cover - platform-dependent
    readline = None

from psalm import __version__ as PSALM_VERSION
from psalm.cli import (
    DEFAULT_MODEL_NAME,
    build_scan_parser,
    load_model_with_startup,
    run_scan_from_args,
    validate_scan_args,
)
from psalm.fast_scan import FastWorkerPoolManager


HISTORY_LIMIT = 100
HISTORY_PATH = Path.home() / ".psalm_history"


def _build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive PSALM shell with one-time model loading.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", "--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("-d", "--device", default="auto")
    parser.add_argument(
        "-c",
        "--cpu-workers",
        type=int,
        default=None,
        help=(
            "Pre-create and warm this many persistent fast decode helper processes for the shell. "
            "Default behavior is equivalent to 0; no persistent worker pool is created until a later fast scan requests one."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {PSALM_VERSION}",
    )
    return parser


def _print_shell_help() -> None:
    print("Commands:")
    print("  scan [args]       Run one scan (use scan --help for flags)")
    print("  help              Show shell command help")
    print("  quit              Leave shell (recommended)")
    print("  exit              Leave shell (alias)")


def _configure_history() -> None:
    if readline is None:
        return
    readline.set_history_length(HISTORY_LIMIT)
    try:
        readline.read_history_file(str(HISTORY_PATH))
    except FileNotFoundError:
        pass

    def _save_history() -> None:
        readline.set_history_length(HISTORY_LIMIT)
        try:
            readline.write_history_file(str(HISTORY_PATH))
        except OSError:
            pass

    atexit.register(_save_history)


def _remember_command(raw: str) -> None:
    if readline is None:
        return
    if raw.strip() == "":
        return
    readline.add_history(raw)
    while readline.get_current_history_length() > HISTORY_LIMIT:
        readline.remove_history_item(0)


def _shell_status_callback(msg: str) -> None:
    print(f"   {msg}", flush=True)


def main() -> None:
    parser = _build_main_parser()
    if len(sys.argv) > 1 and sys.argv[1] == "help":
        parser.print_help()
        return
    args = parser.parse_args()
    _configure_history()
    if args.cpu_workers is not None and args.cpu_workers < 0:
        print("startup error: --cpu-workers must be >= 0")
        return

    pool_manager = None

    def _startup_fast_worker_setup(model, printer) -> None:
        nonlocal pool_manager
        if args.cpu_workers is None or args.cpu_workers <= 0:
            return
        pool_manager = FastWorkerPoolManager(model)
        pool_manager.ensure_size(args.cpu_workers, status_callback=printer)

    try:
        model = load_model_with_startup(
            model_name=args.model_name,
            device=args.device,
            extra_setup_callback=_startup_fast_worker_setup,
        )
    except KeyboardInterrupt:
        print("\nStartup cancelled.")
        return
    except Exception as exc:
        print(f"startup error: {exc}")
        return

    model._fast_shell_mode = True
    model._fast_worker_status_callback = _shell_status_callback
    if pool_manager is not None:
        model._fast_worker_pool_manager = pool_manager

    scan_parser = build_scan_parser(prog="scan", add_help=True, exit_on_error=False)
    try:
        while True:
            try:
                raw = input(">>PSALM>> ")
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue

            raw = raw.strip()
            if not raw:
                continue
            _remember_command(raw)
            if raw in {"quit", "exit"}:
                break
            if raw == "help":
                _print_shell_help()
                continue
            if not raw.startswith("scan"):
                print("Unknown command. Use: scan/help/quit")
                continue

            try:
                tokens = shlex.split(raw)
            except ValueError as exc:
                print(f"scan error: {exc}")
                continue
            if any(token in {"-h", "--help"} for token in tokens[1:]):
                scan_parser.print_help()
                continue

            try:
                scan_args = scan_parser.parse_args(tokens[1:])
            except argparse.ArgumentError as exc:
                print(f"scan error: {exc}")
                continue
            except SystemExit:
                print("scan error: invalid arguments")
                continue
            try:
                validate_scan_args(scan_args)
            except ValueError as exc:
                print(f"scan error: {exc}")
                continue

            try:
                run_scan_from_args(model, scan_args)
            except KeyboardInterrupt:
                manager = getattr(model, "_fast_worker_pool_manager", None)
                if manager is not None:
                    desired_workers = manager.target_size
                    manager.close()
                    if desired_workers > 0:
                        replacement = FastWorkerPoolManager(model)
                        replacement.target_size = desired_workers
                        model._fast_worker_pool_manager = replacement
                    else:
                        model._fast_worker_pool_manager = None
                print("\nscan cancelled")
            except Exception as exc:
                print(f"scan error: {exc}")
    finally:
        manager = getattr(model, "_fast_worker_pool_manager", None)
        if manager is not None:
            manager.close()


if __name__ == "__main__":
    main()
