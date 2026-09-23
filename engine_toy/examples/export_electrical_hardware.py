"""Export the initial construction catalogue or real Machine graph documents.

Run from the normal engine-toy/Turing workspace for graph exports. Construction
JSON is dependency-free and can also be generated in an isolated checkout.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hardware_construction import document, unknown_facts
from hardware_catalogue import SOURCES, cable_catalogue, connector_catalogue


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--construction-only",action="store_true",
                        help="Export evidence-bearing definitions, without importing the game runtime")
    args=parser.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    cables=cable_catalogue()
    connectors=connector_catalogue()
    manifest={"schema":"engine-toy-hardware-catalogue-v1",
              "scope":"integrated game catalogue; not a certified component database",
              "base_commit":"d2a66def49fc8f75ff2dcbdea56db0d7b6e5cddf",
              "sources":SOURCES,
              "cables":[document(x) for x in cables.values()],
              "connectors":[document(x) for x in connectors.values()],
              "unresolved":{key:unknown_facts(value) for key,value in {**cables,**connectors}.items()}}
    (out/"initial_catalogue.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Wrote {len(cables)} cable and {len(connectors)} connector declarations")
    if args.construction_only:
        return 0
    from electrical_hardware import (build_cable,build_connector,build_duplex,
        build_breaker_panel,build_patch_panel,build_cord,BreakerPosition)
    try:
        objects=[*(build_cable(s) for s in cables.values()),
                 *(build_connector(s) for s in connectors.values()),build_duplex(),
                 build_breaker_panel(neutral_to_frame_bond=True),
                 build_breaker_panel(identity="military.panel",sector="military",
                    phases=("line-1","line-2","line-3"),main_rating_a=100,
                    breakers=(BreakerPosition("CB1","SHELTER FEED",(1,3,5),60),)),
                 build_breaker_panel(identity="industrial.panel",sector="industrial",
                    phases=("line-1","line-2","line-3"),main_rating_a=100,
                    breakers=(BreakerPosition("CB1","MACHINE FEED",(2,4,6),60),)),
                 build_patch_panel(),
                 build_cord(cables["military.flex.6-5"],connectors["military.class-l32-12.plug"],
                            connectors["military.class-l32-12.receptacle"],identity="military.service-cord")]
    except ModuleNotFoundError as exc:
        print(f"Construction JSON was exported. Graph exports require the actual engine-toy/Turing workspace: {exc}",file=sys.stderr)
        return 2
    for machine in objects:
        path=out/(machine.identity.replace("/","__")+".graph.json")
        path.write_text(json.dumps(machine.build_graph(),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Wrote {len(objects)} real Machine graph documents")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
