import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

BASE = "a724e78b166c828a81d858b029daf4e4f4011a27"
HEAD = "49d23b3410c6870bab011b5f6d9e4b3c1e3d927b"
SOURCE = "torchao/optim/adam.py"
TEST = "test/test_low_bit_optim_state_dict.py"
FILES = [SOURCE, TEST]
subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", BASE, HEAD], check=True)
for path in FILES:
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
import torch
print("Native PyTorch", torch.__version__, torch.__file__, flush=True)

def run(label, path, expression):
    result = subprocess.run([sys.executable, "-m", "pytest", path, "-q", "-k", expression, f"--junitxml={label}.xml"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    Path(f"{label}.log").write_text(result.stdout)
    print(result.stdout, flush=True)
    suites = ElementTree.parse(f"{label}.xml").getroot()
    totals = {k: sum(int(s.get(k, 0)) for s in suites.iter("testsuite")) for k in ("tests", "failures", "errors", "skipped")}
    totals["returncode"] = result.returncode
    print(label, totals, flush=True)
    return totals

fixed = Path(SOURCE).read_bytes()
try:
    Path(SOURCE).write_bytes(subprocess.check_output(["git", "show", f"{BASE}:{SOURCE}"]))
    before = run("before", TEST, "restore_")
finally:
    Path(SOURCE).write_bytes(fixed)
assert before == {"tests": 36, "failures": 30, "errors": 0, "skipped": 0, "returncode": 1}, before
after = run("after", TEST, "restore_")
assert after == {"tests": 36, "failures": 0, "errors": 0, "skipped": 0, "returncode": 0}, after
existing = run("existing", "test/test_low_bit_optim.py", "TestQuantize or test_subclass_slice or test_subclass_appearance_dtype or test_optim_smoke")
assert existing["returncode"] == existing["failures"] == existing["errors"] == 0, existing
for path in FILES:
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
result = {"head": HEAD, "base": BASE, "torch": torch.__version__, "before": before, "after": after, "existing": existing}
Path("validation_result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
