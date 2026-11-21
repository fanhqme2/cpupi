"""Run the five release benchmarks and compare their complete decimal SHA-256."""
import argparse
import csv
from pathlib import Path
import re
import subprocess
import sys
import tempfile

REFERENCE = {
    1: "f5b642520190fea7f15589a6dc2d364aa7dac4c70f8b02685ed75d3c4b64e03e",
    2: "b788ff73a39b9fb1a78510c372028bf52d92b4e56a2195849df676e5ad5dbe44",
    3: "8e83eefc243bcde9ae233d2b3c1849a671cc62086bb6d3c40c124f65d2c569f8",
    4: "50298c68d5c3b9b6ef20830b63e21f4e9603cdc95651458031ad3f8c182aaca9",
    5: "4da7d7659e4eb850697f385c7f391ed6d2fa40011619a629e1f266fef3bb2ba5",
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--threads", type=int, default=32)
parser.add_argument("--repeats", type=int, default=3)
args = parser.parse_args()
if args.repeats < 1:
    parser.error("--repeats must be positive")
binary = Path(__file__).resolve().parents[1] / "pi_cpu"
writer = csv.writer(sys.stdout, lineterminator="\n")
writer.writerow(["digits", "run", "threads", "compute_s", "peak_rss_KiB", "sha256"])
with tempfile.TemporaryDirectory() as work:
    rss = Path(work) / "rss"
    for run in range(1, args.repeats + 1):
        for size in (REFERENCE if run % 2 else reversed(REFERENCE)):
            digits = size * 10**9
            result = subprocess.check_output(
                ["/usr/bin/time", "-f", "%M", "-o", rss, binary, str(digits),
                 "--benchmark", "--threads", str(args.threads)], text=True)
            digest = re.search(r"^SHA-256: ([0-9a-f]{64})$", result, re.M)
            seconds = re.search(r"^Computation time: ([\d.]+) s$", result, re.M)
            if not digest or digest[1] != REFERENCE[size] or not seconds:
                raise RuntimeError(f"Benchmark verification failed at {digits} digits: {result}")
            writer.writerow([digits, run, args.threads, seconds[1], int(rss.read_text()), digest[1]])
            sys.stdout.flush()
