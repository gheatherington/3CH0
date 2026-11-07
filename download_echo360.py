#!/usr/bin/env python3
"""Compatibility entry point for the original ``download_echo360.py`` script.

The implementation now lives in the ``echo360_dl`` package (``echo360-dl`` command).
Old flags are translated as follows:

    download_echo360.py URL                  -> echo360-dl download URL --streams all
    --output-dir DIR                         -> --output DIR
    --request-file FILE                      -> --auth request-file --request-file FILE
    --browser-profile DIR                    -> --profile-dir DIR
    --timeout N                              -> --timeout N
    --list-formats                           -> echo360-dl inspect URL
    --fresh                                  -> (default: the browser session is always used
                                                 unless --request-file is given)

Behaviour change: the old hard-coded ``~/Downloads/work/echo-request.txt`` default is no
longer read implicitly; pass ``--request-file`` explicitly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def translate(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(prog="download_echo360.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", help="Echo360.ca lesson classroom URL")
    parser.add_argument("--request-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--browser-profile", type=Path)
    parser.add_argument("--fresh", action="store_true", help="accepted for compatibility")
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--list-formats", action="store_true")
    opts = parser.parse_args(argv)

    args = ["inspect", opts.url] if opts.list_formats else ["download", opts.url,
                                                            "--streams", "all"]
    if opts.request_file and not opts.fresh:
        args += ["--auth", "request-file", "--request-file", str(opts.request_file)]
    if opts.browser_profile:
        args += ["--profile-dir", str(opts.browser_profile)]
    if opts.timeout:
        args += ["--timeout", str(opts.timeout)]
    if opts.output_dir and not opts.list_formats:
        args += ["--output", str(opts.output_dir)]
    return args


def main(argv: list[str] | None = None) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from echo360_dl.cli.app import main as cli_main

    sys.argv = ["echo360-dl", *translate(sys.argv[1:] if argv is None else argv)]
    cli_main()


if __name__ == "__main__":
    main()
