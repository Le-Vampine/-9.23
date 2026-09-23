# -*- coding: utf-8 -*-
"""环境自检：把结果写入 _env_check.txt（避免终端输出丢失）。"""
import os
import sys
import traceback

out = []


def log(s):
    out.append(str(s))
    print(s)


log("python: %s" % sys.version.replace("\n", " "))
log("executable: %s" % sys.executable)
for mod in ["numpy", "scipy", "sklearn", "pandas", "torch", "matplotlib", "openpyxl"]:
    try:
        m = __import__(mod)
        log("OK   %-12s %s" % (mod, getattr(m, "__version__", "?")))
    except Exception as e:
        log("FAIL %-12s %s" % (mod, e))

try:
    import torch
    log("torch cuda available: %s" % torch.cuda.is_available())
except Exception:
    log("torch import failed:\n%s" % traceback.format_exc())

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_env_check.txt"),
          "w", encoding="utf-8") as f:
    f.write("\n".join(out))
