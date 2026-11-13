import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

BASE = "a724e78b166c828a81d858b029daf4e4f4011a27"
HEAD = "2cb2497e32703910b9998c90886bcf44eb0e4322"
FILES = ["torchao/quantization/quant_api.py", "test/quantization/test_conv_linear_swap.py"]
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
    before = run("before", "pointwise_conv")
finally:
    for path, content in fixed.items():
        Path(path).write_bytes(content)
assert before["tests"] == 28 and before["errors"] == before["skipped"] == 0, before
assert before["returncode"] == 1 and before["failures"] > 0, before
after = run("after", "pointwise_conv")
assert after["returncode"] == after["failures"] == after["errors"] == 0, after
TESTS = "test/quantization/test_quant_api.py"
existing = run("existing", "test_quantize_does_not_leak_fp32_precision or test_quantize_reapply_is_noop or test_dynamic_quant_gpu_singleline or test_fqn_to_config_regex_skip")
assert existing["returncode"] == existing["failures"] == existing["errors"] == 0, existing
for path in FILES:
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
result = {"head": HEAD, "base": BASE, "torch": torch.__version__, "before": before, "after": after, "existing": existing}
Path("validation_result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
