import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

BASE = "a724e78b166c828a81d858b029daf4e4f4011a27"
HEAD = "e4da8a230aaf9ca316b67dc1cd1cb5cea2f6893c"
FILES = ["torchao/optim/subclass_4bit.py","torchao/optim/subclass_8bit.py","torchao/optim/subclass_fp8.py","test/test_low_bit_optim_clone.py"]
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
    before = run("before", "test_quantized_state_clone or test_quantized_optimizer_snapshot_resume")
finally:
    for path, content in fixed.items():
        Path(path).write_bytes(content)
assert before["tests"] == 72 and before["errors"] == before["skipped"] == 0, before
assert before["returncode"] == 1 and before["failures"] == 72, before
after = run("after", "test_quantized_state_clone or test_quantized_optimizer_snapshot_resume")
assert after["tests"] == 72 and after["skipped"] == 0, after
assert after["returncode"] == after["failures"] == after["errors"] == 0, after
TESTS = "test/test_low_bit_optim.py"
existing = run("existing", "TestQuantize or test_subclass_slice or test_subclass_appearance_dtype or test_optim_smoke")
assert existing["returncode"] == existing["failures"] == existing["errors"] == 0, existing
for path in FILES:
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
result = {"head": HEAD, "base": BASE, "torch": torch.__version__, "before": before, "after": after, "existing": existing}
Path("validation_result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
