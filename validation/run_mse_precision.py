import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

BASE = "a724e78b166c828a81d858b029daf4e4f4011a27"
HEAD = "b18ee6cebb744b5201429931622f9d6e3928a147"
SOURCE = "torchao/quantization/observer.py"
TESTS = "test/quantization/test_observer.py"
ROOT = Path.cwd()
subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", BASE, HEAD], check=True)
for path in (SOURCE, TESTS):
    expected = subprocess.check_output(["git", "show", f"{HEAD}:{path}"])
    assert (ROOT / path).read_bytes() == expected, path
print("Production source and tests match", HEAD, flush=True)
environment = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="2")
import torch
print("Native torch:", torch.__version__, torch.__file__, flush=True)

def run(label, *extra):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", *extra,
         f"--junitxml={label}.xml"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment,
    )
    Path(f"{label}.log").write_text(result.stdout)
    print(result.stdout, flush=True)
    suites = ElementTree.parse(f"{label}.xml").getroot()
    totals = {k: sum(int(s.get(k, 0)) for s in suites.iter("testsuite"))
              for k in ("tests", "failures", "errors", "skipped")}
    totals["returncode"] = result.returncode
    return totals

fixed = (ROOT / SOURCE).read_bytes()
try:
    (ROOT / SOURCE).write_bytes(subprocess.check_output(["git", "show", f"{BASE}:{SOURCE}"]))
    before = run("before", "-k", "TestMSEObserverPrecision")
    metrics_before = subprocess.check_output([sys.executable, "validation/probe_mse_precision.py"], env=environment, text=True)
    Path("metrics_before.jsonl").write_text(metrics_before)
    print(metrics_before, flush=True)
finally:
    (ROOT / SOURCE).write_bytes(fixed)
assert before == {"tests": 16, "failures": 12, "errors": 0, "skipped": 0, "returncode": 1}, before
after = run("after")
metrics_after = subprocess.check_output([sys.executable, "validation/probe_mse_precision.py"], env=environment, text=True)
Path("metrics_after.jsonl").write_text(metrics_after)
print(metrics_after, flush=True)
assert after["returncode"] == 0 and after["failures"] == after["errors"] == 0, after
assert after["tests"] == 33 and after["skipped"] == 0, after
for path in (SOURCE, TESTS):
    assert (ROOT / path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"])
result = {"production_head": HEAD, "base": BASE, "torch": torch.__version__,
          "before": before, "after": after}
Path("validation_result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
