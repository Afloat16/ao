import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

BASE = "a724e78b166c828a81d858b029daf4e4f4011a27"
HEAD = "9e0beae00fd32dfb34fe173cacd97f22bbfc356a"
FILES = ["torchao/optim/subclass_4bit.py","torchao/optim/subclass_8bit.py","torchao/optim/subclass_fp8.py","test/test_low_bit_optim.py"]
SOURCES = FILES[:-1]
TESTS = FILES[-1]
subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", BASE, HEAD], check=True)
for path in FILES:
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
import torch
print("Native PyTorch", torch.__version__, torch.__file__, flush=True)

def run(label, selection):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "-k", selection, f"--junitxml={label}.xml"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    Path(f"{label}.log").write_text(result.stdout)
    print(result.stdout, flush=True)
    tree = ElementTree.parse(f"{label}.xml").getroot()
    totals = {k: sum(int(s.get(k, 0)) for s in tree.iter("testsuite"))
              for k in ("tests", "failures", "errors", "skipped")}
    totals["returncode"] = result.returncode
    print(label, totals, flush=True)
    return totals

fixed = {path: Path(path).read_bytes() for path in SOURCES}
try:
    for path in SOURCES:
        Path(path).write_bytes(subprocess.check_output(["git", "show", f"{BASE}:{path}"]))
    before = run("before", "TestOptimStateMaximum or TestOptimAMSGrad")
finally:
    for path, content in fixed.items():
        Path(path).write_bytes(content)
assert before["tests"] == 30 and before["errors"] == before["skipped"] == 0, before
assert before["returncode"] == 1 and before["failures"] == 30, before
after = run("after", "TestOptimStateMaximum or TestOptimAMSGrad or TestQuantize or test_subclass_slice or test_subclass_appearance_dtype or test_optim_smoke")
assert after["returncode"] == after["failures"] == after["errors"] == 0, after
for path in FILES:
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
result = {"head": HEAD, "base": BASE, "torch": torch.__version__, "before": before, "after": after}
Path("validation_result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
