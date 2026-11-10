import argparse
import json
from pathlib import Path
import sys
import time

from . import ASREnhancer, __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="thinkingears")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="Show the bundled lexicon and license.")
    enhance = commands.add_parser("enhance", help="Enhance ASR text.")
    enhance.add_argument("text", nargs="?", help="Read stdin when omitted.")
    enhance.add_argument("--hotwords", type=Path)
    enhance.add_argument("--corrections", type=Path)
    enhance.add_argument("--lexicon", type=Path)
    enhance.add_argument("--final", action="store_true")
    listen = commands.add_parser("listen", help="Recognize a microphone recording.")
    listen.add_argument("--model-dir", type=Path, required=True)
    listen.add_argument("--hotwords", type=Path)
    listen.add_argument("--seconds", type=int, default=10, choices=range(1, 31))
    listen.add_argument("--threshold", type=int, default=5, choices=range(1, 11))
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        if args.command == "info":
            enhancer = ASREnhancer()
            print(json.dumps({
                "version": __version__,
                "lexicon_entries": enhancer.lexicon_count,
                "license": "ThinkingEars-ASR Noncommercial License v1.0",
            }, ensure_ascii=False))
        elif args.command == "enhance":
            corrections = (
                json.loads(args.corrections.read_text(encoding="utf-8-sig"))
                if args.corrections else None
            )
            enhancer = ASREnhancer(
                hotwords=args.hotwords, corrections=corrections,
                lexicon_path=args.lexicon,
            )
            text = args.text if args.text is not None else sys.stdin.read().rstrip("\r\n")
            print(enhancer.enhance(text, final=args.final))
        else:
            enhancer = ASREnhancer(
                model_dir=args.model_dir, hotwords=args.hotwords,
            ).initialize()
            enhancer.set_threshold_level(args.threshold)
            ok, message = enhancer.start(args.seconds)
            if not ok:
                raise RuntimeError(message)
            try:
                while enhancer.worker is not None:
                    time.sleep(0.05)
            except KeyboardInterrupt:
                enhancer.stop()
                worker = enhancer.worker
                if worker is not None:
                    worker.join()
            result = enhancer.snapshot()
            print(json.dumps(result, ensure_ascii=False))
            return 1 if result["error"] else 0
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0
