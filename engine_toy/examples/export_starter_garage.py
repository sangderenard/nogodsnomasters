"""Export the complete starter-garage authoring and production documents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter_garage import garage_document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="starter")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "hardware_data" / "starter_garage.json")
    args = parser.parse_args()
    document = garage_document(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"wrote {args.output}: {document['inventory_entry_count']} entries, "
          f"{document['constituent_record_count']} constituent records, "
          f"{document['machine_count']} real Machine graphs")


if __name__ == "__main__":
    main()
