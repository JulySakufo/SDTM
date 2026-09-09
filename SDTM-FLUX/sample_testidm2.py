"""Compatibility entry point selecting ``SDTM_testidm2`` by default."""

from __future__ import annotations

import sys

from sample import main


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--tore-type" not in arguments:
        arguments = ["--tore-type", "SDTM_testidm2", *arguments]
    main(arguments)
