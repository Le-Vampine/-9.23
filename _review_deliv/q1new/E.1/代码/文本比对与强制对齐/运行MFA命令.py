"""Run the locally installed MFA CLI with all writable state inside E.1."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MFA_ENV = Path(r"C:\Users\Administrator\miniforge3\envs\mfa")
MFA_EXE = MFA_ENV / "Scripts" / "mfa.exe"
MFA_PYTHON = MFA_ENV / "python.exe"
MFA_COMPAT_LAUNCHER = ROOT / "代码" / "文本比对与强制对齐" / "MFA兼容入口.py"
MFA_DIR = ROOT / "MFA"
CORPUS = MFA_DIR / "corpus" / "audio"
WHISPER_CORPUS = MFA_DIR / "corpus_whisper" / "audio"
DICTIONARY = MFA_DIR / "models" / "dictionary" / "english_us_arpa.dict"
ACOUSTIC_MODEL = MFA_DIR / "models" / "acoustic" / "english_us_arpa.zip"
VALIDATION_OUTPUT = MFA_DIR / "validation_parallel"
ALIGNMENT_OUTPUT = MFA_DIR / "aligned_json"
WHISPER_ALIGNMENT_OUTPUT = MFA_DIR / "aligned_whisper_json"
LOG_DIR = ROOT / "日志" / "第4步文本一致性检查"
NUM_JOBS = 1


def environment(alias_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["MFA_ROOT_DIR"] = str(alias_root / "MFA" / "work")
    env["NUMBA_CACHE_DIR"] = str(alias_root / "MFA" / "numba_cache")
    env["CONDA_PREFIX"] = str(MFA_ENV)
    env["CONDA_DEFAULT_ENV"] = "mfa"
    env["CONDA_SHLVL"] = "1"
    extra_path = [MFA_ENV / "Library" / "bin", MFA_ENV / "Scripts", MFA_ENV]
    env["PATH"] = os.pathsep.join([str(path) for path in extra_path] + [env.get("PATH", "")])
    return env


def validate_command(alias_root: Path) -> list[str]:
    tmp = alias_root / "MFA" / "work_validate_parallel"
    return [
        str(MFA_PYTHON), str(MFA_COMPAT_LAUNCHER), "validate", str(alias_root / "MFA" / "corpus" / "audio"),
        str(alias_root / "MFA" / "models" / "dictionary" / "english_us_arpa.dict"),
        "--acoustic_model_path", str(alias_root / "MFA" / "models" / "acoustic" / "english_us_arpa.zip"),
        "--output_directory", str(alias_root / "MFA" / "validation"),
        "--temporary_directory", str(tmp),
        "--num_jobs", str(NUM_JOBS),
        "--single_speaker",
    ]


def align_command(alias_root: Path) -> list[str]:
    tmp = alias_root / "MFA" / "work_align"
    return [
        str(MFA_PYTHON), str(MFA_COMPAT_LAUNCHER), "align", str(alias_root / "MFA" / "corpus" / "audio"),
        str(alias_root / "MFA" / "models" / "dictionary" / "english_us_arpa.dict"),
        str(alias_root / "MFA" / "models" / "acoustic" / "english_us_arpa.zip"),
        str(alias_root / "MFA" / "aligned_json"),
        "--output_format", "json",
        "--temporary_directory", str(tmp),
        "--num_jobs", str(NUM_JOBS),
        "--single_speaker",
        "--no_tokenization",
        "--no_use_mp",
        "--verbose",
    ]


def align_whisper_command(alias_root: Path) -> list[str]:
    tmp = alias_root / "MFA" / "work_align_whisper"
    return [
        str(MFA_PYTHON), str(MFA_COMPAT_LAUNCHER), "align",
        str(alias_root / "MFA" / "corpus_whisper" / "audio"),
        str(alias_root / "MFA" / "models" / "dictionary" / "english_us_arpa.dict"),
        str(alias_root / "MFA" / "models" / "acoustic" / "english_us_arpa.zip"),
        str(alias_root / "MFA" / "aligned_whisper_json"),
        "--output_format", "json",
        "--temporary_directory", str(tmp),
        "--num_jobs", "1",
        "--single_speaker",
        "--no_tokenization",
        "--no_use_mp",
        "--verbose",
    ]


def create_ascii_alias() -> tuple[Path, str]:
    """Map an unused drive letter to E.1 so OpenFST avoids Unicode paths."""
    for letter in "ZYXWVUTSRQP":
        drive = f"{letter}:"
        if os.path.exists(drive + "\\"):
            continue
        result = subprocess.run(["subst", drive, str(ROOT)], capture_output=True, text=True, errors="replace")
        if result.returncode == 0 and os.path.exists(drive + "\\MFA"):
            return Path(drive + "\\"), drive
    raise RuntimeError("找不到可用于临时映射 E.1 的空闲盘符")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "align", "align_whisper", "help"))
    args = parser.parse_args()
    required = (MFA_EXE, MFA_PYTHON, MFA_COMPAT_LAUNCHER)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MFA运行文件缺失：{missing}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (MFA_DIR / "numba_cache").mkdir(parents=True, exist_ok=True)
    alias_root, drive = create_ascii_alias()
    try:
        (alias_root / "MFA" / "work").mkdir(parents=True, exist_ok=True)
        if args.action == "validate":
            command = validate_command(alias_root)
            log_path = LOG_DIR / "MFA语料验证控制台.json"
        elif args.action == "align":
            command = align_command(alias_root)
            log_path = LOG_DIR / "MFA强制对齐控制台.json"
        elif args.action == "align_whisper":
            command = align_whisper_command(alias_root)
            log_path = LOG_DIR / "MFA全量Whisper文本强制对齐控制台.json"
        else:
            command = [str(MFA_EXE), "--help"]
            log_path = LOG_DIR / "MFA本机命令环境检查.json"
        proc = subprocess.run(
            command,
            cwd=str(alias_root),
            env=environment(alias_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    finally:
        subprocess.run(["subst", drive, "/d"], capture_output=True, text=True, errors="replace")
    log = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "mfa_executable": str(MFA_EXE),
        "mfa_environment": str(MFA_ENV),
        "mfa_root_dir": str(alias_root / "MFA" / "work"),
        "return_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"MFA action: {args.action}\nreturn code: {proc.returncode}\nlog: {log_path}")
    output = (proc.stdout + "\n" + proc.stderr).strip()
    if output:
        safe_output = "\n".join(output.splitlines()[-80:])
        sys.stdout.write(safe_output.encode("utf-8", "backslashreplace").decode("utf-8") + "\n")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
