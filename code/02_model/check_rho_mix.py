"""快速自检：确认 --rho_mix 的分段混合采样确实产生目标分布。"""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(errors="replace")
from missing_sim import sample_rho     # noqa: E402

MIX = [(0.02, 0.22, 0.75), (0.22, 0.50, 0.25)]
rng = np.random.default_rng(0)
a = np.array([sample_rho(0.04, 0.25, rng, mix=MIX) for _ in range(40000)])
b = np.array([sample_rho(0.04, 0.25, rng, mix=None) for _ in range(40000)])

print("目标（附件3 实测）: mean 0.183 / p50 0.150 / p90 0.333 / max 0.478")
for tag, x in (("mix ", a), ("旧课程", b)):
    print("%s : mean %.3f / p50 %.3f / p90 %.3f / max %.3f / P(>0.25)=%.3f"
          % (tag, x.mean(), np.percentile(x, 50), np.percentile(x, 90),
             x.max(), (x > 0.25).mean()))
print("\n结论：mix 的均值/上界应贴近附件3；旧课程 P(>0.25)=0（这正是被修的分布外问题）")
