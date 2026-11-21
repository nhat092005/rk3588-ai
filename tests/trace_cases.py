"""Paths of the submit-trace fixtures (also used by the generator)."""

from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
# A slice of the legacy-final trace_c012.log and the outputs of the
# legacy summarize.py and diff_cores.py on it.
TRACE_LOG = FIXTURES / "submit_trace_slice.log"
TRACE_SUMMARY = FIXTURES / "submit_trace_slice_summary.txt"
TRACE_DIFF = FIXTURES / "submit_trace_slice_diff.txt"
