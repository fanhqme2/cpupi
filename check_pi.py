"""Check the release CLI, decimal boundaries and the benchmark digest."""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile

PI = Path(__file__).with_name("pi_cpu").resolve()
REFERENCE = ("3.1415926535897932384626433832795028841971693993751058209749445923078164"
             "062862089986280348253421170679821480865132823066470938446095505822317253")


def run(*args):
    return subprocess.run([PI, *map(str, args)], capture_output=True, check=True).stdout


def check_benchmark(digits, text):
    lines = run(digits, "--benchmark", "--threads", 3).decode().splitlines()
    tail = text[2:][-32:] if digits else b"3"
    assert lines[:2] == [f"Last digits: {tail.decode()}", f"SHA-256: {hashlib.sha256(text).hexdigest()}"]
    assert len(lines) == 3 and re.fullmatch(r"Computation time: \d+\.\d+ s", lines[2])


for n in (0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 140):
    text = REFERENCE[:n + 2 if n else 1].encode()
    assert run(n, "--stdout", "1") == text + b"\n"
    check_benchmark(n, text)

allowed = os.sched_getaffinity(0)
try:
    os.sched_setaffinity(0, sorted(allowed)[:3])
    assert run(100) == REFERENCE[:102].encode() + b"\n"
finally:
    os.sched_setaffinity(0, allowed)

for n in (8191, 32769, 100003):
    one = run(n, "-t", 1)
    assert one == run(n, "--output", "-", "--threads", 7) and len(one) == n + 3

with tempfile.TemporaryDirectory() as work:
    target = Path(work) / "pi digits.txt"
    assert run(1000003, "-o", target, "-t", 7) == b""
    digits = target.read_bytes()
    assert digits.endswith(b"\n") and len(digits) == 1000006
    assert hashlib.sha256(digits[:-1]).hexdigest() == (
        "ce05d7f44541622379683128dbc81d9a7d5b996145f50874d872c925433e7d73")
    check_benchmark(1000003, digits[:-1])

for args in ([], ["-1"], ["1.5"], ["90", "-t", "0"], ["90", "257"], ["90", "--output"],
             ["90", "--threads"], ["90", "--stdout", "-b"], ["90", "2", "-t", "2"],
             ["90", "--unknown"], ["18446744073709551615", "-b"]):
    assert subprocess.run([PI, *args], capture_output=True).returncode != 0, args
print("Decimal output, SHA-256 and CLI checks passed.")
