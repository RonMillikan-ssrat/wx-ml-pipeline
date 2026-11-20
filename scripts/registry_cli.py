"""Inspect and operate the model registry by hand.

    python scripts/registry_cli.py status
    python scripts/registry_cli.py promote 7          # make version 7 production
    python scripts/registry_cli.py rollback           # production -> previous

Inside the stack: docker compose exec airflow-scheduler python /opt/wx/scripts/registry_cli.py status
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wxml import config, registry  # noqa: E402


def status():
    c = registry.client()
    prod, prev = registry.alias_version(config.PROD_ALIAS), registry.alias_version(config.PREV_ALIAS)
    print(f"{config.MODEL_NAME}   @production={prod}   @previous={prev}")
    versions = sorted(c.search_model_versions(f"name='{config.MODEL_NAME}'"), key=lambda v: int(v.version))
    for v in versions:
        t = v.tags
        print(f"  v{v.version:<4} {t.get('status', ''):<12} skill={t.get('skill', ''):<8} "
              f"train_end={t.get('train_end', ''):<11} {t.get('gate_reason', t.get('rollback_reason', ''))}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("promote")
    p.add_argument("version")
    sub.add_parser("rollback")
    a = ap.parse_args()
    if a.cmd == "status":
        status()
    elif a.cmd == "promote":
        print(registry.promote(a.version, reason="manual promote"))
    else:
        current = registry.alias_version(config.PROD_ALIAS)
        print(registry.rollback(current, reason="manual rollback"))


if __name__ == "__main__":
    main()
