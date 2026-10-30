"""Command line for the decision agents.

    python -m app.cli train    XAU XAG BTC NIFTY
    python -m app.cli decide   XAU XAG BTC NIFTY CRUDE
    python -m app.cli evaluate XAU
    python -m app.cli history  --instrument XAU

Add --source synthetic to run offline on random test data.
"""

from __future__ import annotations

import argparse
import dataclasses
import json

from app.config import settings
from app.data.history import INSTRUMENTS
from app.decision.service import DecisionService


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train", "decide", "evaluate", "history"])
    ap.add_argument("instruments", nargs="*", default=["XAU", "XAG", "BTC", "NIFTY"])
    ap.add_argument("--source", choices=["yahoo", "csv", "synthetic"], help="override DATA_SOURCE")
    ap.add_argument("--instrument", help="history filter")
    ap.add_argument("--json", action="store_true", help="print full JSON")
    args = ap.parse_args()

    for i in args.instruments:
        if i not in INSTRUMENTS:
            ap.error(f"unknown instrument {i}; choose from {sorted(INSTRUMENTS)}")
    cfg = dataclasses.replace(settings, data_source=args.source) if args.source else settings
    svc = DecisionService(cfg)
    if cfg.data_source == "synthetic":
        print("NOTE: synthetic random data - results say nothing about real markets.\n")

    if args.command == "train":
        for inst in args.instruments:
            m = svc.train(inst).metrics
            edge = "EDGE" if m.has_edge else "no edge (ML agent will abstain)"
            print(f"{inst:6} hold-out acc {m.accuracy:.1%} vs baseline {m.majority_baseline:.1%} | "
                  f"log-loss {m.log_loss} vs {m.prior_log_loss} | {edge}")
    elif args.command == "decide":
        for d in svc.decide_many(args.instruments):
            if args.json:
                print(json.dumps(d, indent=2))
            elif "error" in d:
                print(f"{d['instrument']:6} ERROR {d['error']}")
            else:
                r = d["risk"]
                levels = f" | stop {r['stop_loss']} target {r['target']}" if d["final_action"] != "HOLD" else ""
                print(f"{d['instrument']:6} {d['final_action']:4} (as of {d['as_of']}, close {d['last_close']}){levels}")
                print(f"       {d['explanation']}")
                for v in d["votes"]:
                    tag = "  -  " if v["abstained"] else f"{v['action']:4} {v['confidence']:.2f}"
                    print(f"         {v['agent']:15} {tag:10} {v['reasons'][0]}")
                print()
    elif args.command == "evaluate":
        for inst in args.instruments:
            print(json.dumps(svc.evaluate(inst), indent=2))
    else:
        for row in svc.store.history(args.instrument):
            print(row)


if __name__ == "__main__":
    main()
