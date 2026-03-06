from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import List

from psalm import __version__ as PSALM_VERSION
from psalm.cli import DEFAULT_MODEL_NAME, _format_model_source, _print_startup_banner
from psalm.psalm_model import PSALM


def _build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive PSALM session with one-time model loading.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", "--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("-d", "--device", default="auto")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {PSALM_VERSION}",
    )
    return parser


def _build_scan_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scan",
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-s", "--sequence", default=None)
    parser.add_argument("-f", "--fasta", default=None)
    parser.add_argument("-t", "--score-thresh", type=float, default=0.0)
    parser.add_argument("-b", "--beam-size", type=int, default=64)
    parser.add_argument("--to-tsv", default=None)
    parser.add_argument("--to-txt", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    return parser


def _print_shell_help() -> None:
    print("Commands:")
    print("  scan [args]       Run one scan (use scan --help for flags)")
    print("  help              Show command help")
    print("  quit              Leave shell (recommended)")
    print("  exit              Leave shell (alias)")


def main() -> None:
    args = _build_main_parser().parse_args()

    _print_startup_banner()
    print(f"[startup] Requested device: {args.device}")
    print(f"[startup] Model ID/path: {args.model_name}")
    print("[startup] Checking model location and loading...")
    startup_events: List[str] = []

    def _status(msg: str) -> None:
        startup_events.append(msg)

    model = PSALM(
        model_name=args.model_name,
        device=args.device,
        status_callback=_status,
    )
    resolved_device = getattr(model, "resolved_device", None)
    if resolved_device is not None:
        if str(args.device).strip().lower() == "auto":
            print(f"[startup] Auto resolved device: {resolved_device}")
        else:
            print(f"[startup] Resolved device: {resolved_device}")
    model_source = _format_model_source(getattr(model, "model_source", None))
    if model_source not in {"None", ""}:
        print(f"[startup] Model source: {model_source}")
    for event in startup_events:
        print(f"[startup] {event}")
    if not getattr(model, "warmup_executed", False):
        print("[startup] Warmup: skipped")
    print("─" * 80)
    print("PSALM shell ready. Type 'help' for commands.")

    scan_parser = _build_scan_parser()
    while True:
        try:
            raw = input("psalm> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        if raw in {"quit", "exit"}:
            break
        if raw == "help":
            _print_shell_help()
            continue
        if not raw.startswith("scan"):
            print("Unknown command. Use: scan/help/quit")
            continue

        tokens = shlex.split(raw)
        scan_args = scan_parser.parse_args(tokens[1:])
        if scan_args.help:
            scan_parser.print_help()
            continue
        if (scan_args.sequence is None) == (scan_args.fasta is None):
            print("scan error: provide exactly one of --sequence or --fasta")
            continue

        result = model.scan(
            sequence=scan_args.sequence,
            fasta=scan_args.fasta,
            score_thresh=scan_args.score_thresh,
            beam_size=scan_args.beam_size,
            refine_extended=(not scan_args.no_refine),
            verbose=scan_args.verbose,
            to_tsv=scan_args.to_tsv,
            to_txt=scan_args.to_txt,
            _print_output=(not scan_args.quiet),
        )
        if scan_args.json_out:
            out_path = Path(scan_args.json_out)
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
