"""Command line interface: `python -m ingest <command>`."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from ingest.config import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ingest", description="sh-rag document ingestion")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("serve", help="run the HTTP API and folder watcher")

    p_scan = sub.add_parser("scan", help="sync DOCS_DIR into the vector index once")
    p_scan.add_argument("--force", action="store_true", help="re-embed every file even if unchanged")
    p_scan.add_argument("--recreate", action="store_true", help="drop and recreate the collection first")
    p_scan.add_argument("--no-prune", action="store_true", help="keep index entries for deleted files")

    p_add = sub.add_parser("add", help="copy files into DOCS_DIR/uploads and index them")
    p_add.add_argument("files", nargs="+", type=Path)

    sub.add_parser("list", help="list indexed documents")

    p_emb = sub.add_parser("embedding", help="switch the embedding model (re-embeds into its own index)")
    p_emb.add_argument("model")

    p_rm = sub.add_parser("remove", help="remove a document by doc_id or source path")
    p_rm.add_argument("key")
    p_rm.add_argument("--keep-file", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = Settings.from_env()

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("ingest.api:app", host=settings.host, port=settings.port)
        return 0

    from ingest.pipeline import Ingestor

    ingestor = Ingestor(settings)

    if args.cmd == "scan":
        ingestor.setup(recreate=args.recreate)
        results = ingestor.scan(force=args.force or args.recreate, prune=not args.no_prune)
        for r in results:
            if r.status != "unchanged":
                print(f"{r.status:<11} {r.source}" + (f"  ({r.chunks} chunks)" if r.chunks else "")
                      + (f"  error: {r.error}" if r.error else ""))
        print(json.dumps(dict(Counter(r.status for r in results))))
        return 1 if any(r.status == "error" for r in results) else 0

    if args.cmd == "add":
        failed = False
        for path in args.files:
            r = ingestor.save_upload(path.name, path.read_bytes())
            failed |= r.status in {"error", "unsupported"}
            print(f"{r.status:<11} {r.source}" + (f"  ({r.chunks} chunks)" if r.chunks else ""))
        return 1 if failed else 0

    if args.cmd == "list":
        for d in ingestor.list_docs():
            print(f"{d.get('doc_id')}  {d.get('chunks', '?'):>5} chunks  {d.get('source')}")
        return 0

    if args.cmd == "embedding":
        def report(msg, frac=None):
            print(f"  {msg}" + (f" ({frac:.0%})" if frac is not None else ""), flush=True)

        results = ingestor.switch_embedding(args.model, on_status=report)
        print(json.dumps(dict(Counter(r.status for r in results))))
        print(f"active embedding model: {ingestor.index.embed_model} (collection {ingestor.index.collection})")
        return 0

    if args.cmd == "remove":
        r = ingestor.remove(args.key, delete_file=not args.keep_file)
        if r is None:
            print(f"not found: {args.key}")
            return 1
        print(f"removed {r.source}")
        return 0

    return 2
