#!/usr/bin/env python3
import argparse
import pickle


def _strip_version(rfam_id: str) -> str:
    return rfam_id.split(".", 1)[0]


def build_label_mapping(ids):
    label_mapping = {"None": 0}
    next_value = 1
    seen = set()
    for raw_id in ids:
        rfam_id = _strip_version(raw_id)
        if not rfam_id or rfam_id in seen:
            continue
        seen.add(rfam_id)
        label_mapping[rfam_id] = {
            "start": next_value,
            "middle": next_value + 1,
            "stop": next_value + 2,
        }
        next_value += 3
    return label_mapping


def read_ids(path):
    ids = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            ids.append(line.split()[0])
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Create a label mapping from an RFAM ID list."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to .txt file with one RFAM ID per line.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output pickle file for label mapping.",
    )
    args = parser.parse_args()

    ids = read_ids(args.input)
    label_mapping = build_label_mapping(ids)

    with open(args.output, "wb") as f:
        pickle.dump(label_mapping, f)

    print(f"Wrote {len(label_mapping) - 1} RFAM labels to {args.output}")


if __name__ == "__main__":
    main()
