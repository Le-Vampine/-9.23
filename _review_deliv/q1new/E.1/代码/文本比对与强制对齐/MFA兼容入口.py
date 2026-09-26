"""E.1-local MFA launcher for OpenFST builds whose --help exits with code 1.

MFA 3.4.2's startup canary treats a successful OpenFST help page (exit code 1)
as a missing binary. The executable is present in the configured MFA env, so
this launcher skips only that help-based canary; it does not replace any MFA
alignment or feature-extraction operation. Missing executables will still fail
when MFA invokes them.
"""
from __future__ import annotations

import sys
import faulthandler
import os


def main() -> int:
    trace_path = os.environ.get("MFA_E1_TRACE_FILE")
    trace_stream = None
    if trace_path:
        trace_stream = open(trace_path, "a", encoding="utf-8", buffering=1)
        faulthandler.enable(file=trace_stream, all_threads=True)
        faulthandler.dump_traceback_later(15, repeat=True, file=trace_stream)

    from montreal_forced_aligner.command_line import utils as cli_utils
    from montreal_forced_aligner.command_line.mfa import mfa_cli

    cli_utils.check_third_party = lambda: None
    mfa_cli.main(args=sys.argv[1:], prog_name="mfa", standalone_mode=True)
    faulthandler.cancel_dump_traceback_later()
    if trace_stream is not None:
        faulthandler.disable()
        trace_stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
