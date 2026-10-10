import argparse
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESCR_COLUMN = "description:token_seq"


def clean(text):
    # tabs or newlines inside a field would break RecBole's tab-separated parser, and pandas
    # treats double quotes as CSV quoting, so swap them for single quotes
    return " ".join(text.split()).replace('"', "'")


def load_descriptions(path):
    descriptions = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                raise ValueError(f"{path}:{line_no}: expected 4 tab-separated columns, got {len(parts)}")
            descriptions[parts[0]] = clean("\t".join(parts[3:]))
    return descriptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--descr", default=os.path.join(root, "dataset", "ml-100k-unformatted", "items_descr.txt"))
    parser.add_argument("--item", default=os.path.join(root, "dataset", "ml-100k", "ml-100k.item"))
    args = parser.parse_args()

    descriptions = load_descriptions(args.descr)

    with open(args.item, encoding="utf-8") as f:
        lines = f.read().splitlines()

    header = lines[0].split("\t")
    if DESCR_COLUMN in header:
        descr_idx = header.index(DESCR_COLUMN)
    else:
        descr_idx = len(header)
        header.append(DESCR_COLUMN)

    out = ["\t".join(header)]
    missing = []
    for line in lines[1:]:
        if not line:
            continue
        row = line.split("\t")
        row += [""] * (len(header) - len(row))  # pad so descr_idx is always valid
        item_id = row[0]
        if item_id not in descriptions:
            missing.append(item_id)
        row[descr_idx] = descriptions.get(item_id, "")
        out.append("\t".join(row))

    with open(args.item, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

    print(f"Wrote {len(out) - 1} items to {args.item}")
    if missing:
        print(f"Warning: {len(missing)} items have no description: {missing[:20]}")


if __name__ == "__main__":
    main()
