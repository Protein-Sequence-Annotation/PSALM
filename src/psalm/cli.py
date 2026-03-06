from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from psalm import __version__ as PSALM_VERSION
from psalm.psalm_model import PSALM

DEFAULT_MODEL_NAME = "ProteinSequenceAnnotation/PSALM-2-InterPro30-104.0"


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def _print_startup_banner() -> None:
    width = 80
    inner = width - 2
    top = "┌" + "─" * inner + "┐"
    bottom = "└" + "─" * inner + "┘"
    art = [
        "██████╗ ███████╗ █████╗ ██╗     ███╗   ███╗",
        "██╔══██╗██╔════╝██╔══██╗██║     ████╗ ████║",
        "██████╔╝███████╗███████║██║     ██╔████╔██║",
        "██╔═══╝ ╚════██║██╔══██║██║     ██║╚██╔╝██║",
        "██║     ███████║██║  ██║███████╗██║ ╚═╝ ██║",
        "╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝",
    ]
    print(top)
    print(f"│{'':{inner}}│")
    print(f"│{'':{inner}}│")
    for line in art:
        print(f"│{line:^{inner}}│")
    print(f"│{'Protein Sequence Annotation using a Language Model':^{inner}}│")
    print(f"│{'':{inner}}│")
    print(bottom)
    print(f"{('PSALM CLI (v' + PSALM_VERSION + ')'):^{width}}")


def _format_model_source(source: str | None) -> str:
    mapping = {
        "local_path": "local path",
        "repo_models_dir": "repo models directory",
        "hf_cache": "Hugging Face cache",
        "hf_download": "Hugging Face download",
    }
    return mapping.get(str(source), str(source))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run PSALM domain scanning for one sequence or a FASTA file.\n"
            "Use psalm-shell for a persistent session (load once, scan many times)."
        ),
        epilog=(
            "Examples:\n"
            "  psalm-scan -f Q09870.fa\n"
            "  psalm-scan -v -f Q09870.fa --to-tsv out.tsv\n"
            "  psalm-scan -q --no-refine -f Q09870.fa\n"
            "  psalm-shell -d auto"
        ),
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model bundle path or Hugging Face repo id.",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="auto",
        help="Device to use: auto, cpu, mps, cuda, cuda:<index>.",
    )
    parser.add_argument(
        "-s",
        "--sequence",
        default=None,
        help="Raw amino-acid sequence string.",
    )
    parser.add_argument(
        "-f",
        "--fasta",
        default=None,
        help="Path to FASTA file.",
    )
    parser.add_argument(
        "-t",
        "--score-thresh",
        type=float,
        default=0.0,
        help="CBM score threshold for filtering predictions.",
    )
    parser.add_argument(
        "-b",
        "--beam-size",
        type=int,
        default=64,
        help="Beam size for decoding.",
    )
    parser.add_argument(
        "--to-tsv",
        default=None,
        help="Write tabular predictions to TSV path.",
    )
    parser.add_argument(
        "--to-txt",
        default=None,
        help="Write console-style report to TXT path.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON output path for full structured results.",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable extended refinement pass.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed model tables and alignment output.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress scan result output in terminal (startup banner/status still shown).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {PSALM_VERSION}",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.sequence is None and args.fasta is None:
        parser.error("No input provided. Try `psalm-scan -h` for usage and examples.")
    if args.sequence is not None and args.fasta is not None:
        parser.error(
            "Provide exactly one of --sequence or --fasta (not both). "
            "Try `psalm-scan -h` for usage and examples."
        )

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

    result = model.scan(
        sequence=args.sequence,
        fasta=args.fasta,
        score_thresh=args.score_thresh,
        beam_size=args.beam_size,
        refine_extended=(not args.no_refine),
        verbose=args.verbose,
        to_tsv=args.to_tsv,
        to_txt=args.to_txt,
        _print_output=(not args.quiet),
    )

    if args.json_out:
        out_path = Path(args.json_out)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
