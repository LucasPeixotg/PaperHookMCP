"""Console entry point (`paperhook-mcp`).

Kept apart from `server.py` because importing the server builds the clients and loads
the cross-encoder, so a broken model or network would crash before `--doctor` was
even parsed. Both paths import what they need only once the flag is known.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="paperhook-mcp",
        description="PaperHook research assistant, an MCP server over stdio.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="run startup diagnostics in this terminal and exit instead of serving",
    )
    args = parser.parse_args()

    if args.doctor:
        from .doctor import run

        sys.exit(run())

    from .server import main as serve

    serve()


if __name__ == "__main__":
    main()
