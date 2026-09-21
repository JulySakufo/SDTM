"""Compatibility entry point selecting Qwen SDTM_testidm2 by default."""

from __future__ import annotations

from typing import Iterable

from sample import build_parser, run, validate_args


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    parser.set_defaults(tore_type="SDTM_testidm2")
    args = parser.parse_args(argv)
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
