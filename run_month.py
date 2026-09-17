#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""單月批次：CTI 錄音 → whisper 逐字稿 → 地端 Qwen 結構化分析

把 CTI/<診所>/<YYYYMM*>/ 的錄音跑完整條鍊，可無人值守整夜執行、可中斷續跑。

    python run_month.py --month 202509 --clinic A院區    # 單月單院區（建議）
    python run_month.py --month 202509                 # 全月全院區
    python run_month.py --month 202509 --clinic A院區 --limit 30   # 小量驗證
    python run_month.py --month 202509 --clinic A院區 --stage asr  # 只跑辨識

流程
  1. 建立 <WORK>/00_raw 的月份鏡像（硬連結，不複製 —— 省磁碟也省時間）
  2. cti_whisper_batch.py：正規化 + 辨識，轉完即刪正規化檔
  3. cti_llm_pipeline.py：逐字稿 → 結構化分析
兩支子腳本各自有帳本／快取，重跑只會做沒做過的。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
CTI = BASE / "CTI"
PY = sys.executable

# Windows 主控台預設 cp950，子行程輸出裡的替代字元會讓 print 直接炸掉。
# 整條 pipeline 全是中文，統一走 UTF-8。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def out(line: str) -> None:
    """安全輸出：無論主控台編碼為何都不會拋例外。"""
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(line.encode(enc, "replace").decode(enc, "replace"), flush=True)


def log(msg: str, fh=None) -> None:
    line = "[%s] %s" % (datetime.now().strftime("%m-%d %H:%M:%S"), msg)
    out(line)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def mirror_month(month: str, dst_root: Path, limit: int, only: str = "") -> int:
    """把該月錄音以硬連結鏡像到 00_raw/<診所>/<日期>/。

    用硬連結而非複製：同一顆磁碟上不佔額外空間、瞬間完成，
    且下游把 00_raw 當唯讀來源，不會動到 CTI 原始檔。
    only 非空時只鏡像該院區。
    """
    clinics = sorted(p for p in CTI.iterdir() if p.is_dir())
    if only:
        names = [c.name for c in clinics]
        if only not in names:
            sys.exit("[fatal] 找不到院區「%s」。可用院區：%s" % (only, "、".join(names)))
        clinics = [c for c in clinics if c.name == only]

    n = 0
    for clinic in clinics:
        for day in sorted(p for p in clinic.iterdir() if p.is_dir()):
            if not day.name.startswith(month):
                continue
            dst = dst_root / clinic.name / day.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in sorted(day.iterdir()):
                if not f.is_file():
                    continue
                target = dst / f.name
                if not target.exists():
                    try:
                        os.link(f, target)
                    except OSError:
                        target.write_bytes(f.read_bytes())
                n += 1
                if limit and n >= limit:
                    return n
    return n


def run(cmd: list[str], fh) -> int:
    log("$ " + " ".join(str(c) for c in cmd), fh)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", cwd=str(BASE),
                         env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    for line in p.stdout:
        line = line.rstrip()
        if line:
            out("   " + line)
            fh.write("   " + line + "\n")
            fh.flush()
    return p.wait()


def main() -> None:
    ap = argparse.ArgumentParser(description="單月 CTI 批次")
    ap.add_argument("--month", required=True, help="YYYYMM，如 202509")
    ap.add_argument("--clinic", default="",
                    help="只處理指定院區（如 A院區）。留空＝全部院區")
    ap.add_argument("--work", default="",
                    help="工作目錄，預設 _run_<month> 或 _run_<month>_<clinic>")
    ap.add_argument("--model", default="ggml-large-v3.bin", help="whisper 模型")
    ap.add_argument("--llm", default="qwen3.8", help="Ollama 模型")
    ap.add_argument("--limit", type=int, default=0, help="只處理前 N 檔（驗證用）")
    ap.add_argument("--stage", choices=["all", "asr", "llm"], default="all")
    args = ap.parse_args()

    # 每個院區獨立工作目錄與帳本：一個院區約 4 小時就能看到結果，
    # 不必等 20 小時才知道有沒有問題；出錯也只影響該院區。
    tag = args.month + ("_" + args.clinic if args.clinic else "")
    work = Path(args.work) if args.work else BASE / ("_run_" + tag)
    raw = work / "00_raw"
    analysis = work / "05_analysis"
    work.mkdir(parents=True, exist_ok=True)
    logf = work / ("run_%s.log" % datetime.now().strftime("%Y%m%d_%H%M%S"))

    with logf.open("a", encoding="utf-8") as fh:
        t0 = time.time()
        log("=" * 66, fh)
        log("單月批次 %s ｜ 工作目錄 %s" % (tag, work), fh)
        log("院區=%s ｜ whisper=%s ｜ LLM=%s ｜ limit=%s ｜ stage=%s"
            % (args.clinic or "全部", args.model, args.llm,
               args.limit or "全部", args.stage), fh)
        log("=" * 66, fh)

        n = mirror_month(args.month, raw, args.limit, args.clinic)
        if not n:
            sys.exit("[fatal] CTI 下找不到 %s 的資料" % tag)
        log("鏡像完成：%d 檔 → %s" % (n, raw), fh)

        env = dict(os.environ, CTI_SRC=str(raw), CTI_OUT=str(work),
                   CTI_MODEL=args.model, PYTHONIOENCODING="utf-8")

        if args.stage in ("all", "asr"):
            log("── 階段 1／2：whisper 辨識 ──", fh)
            t = time.time()
            p = subprocess.Popen(
                [PY, str(BASE / "cti_whisper_batch.py"), "run"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", cwd=str(BASE), env=env)
            for line in p.stdout:
                line = line.rstrip()
                if line:
                    out("   " + line)
                    fh.write("   " + line + "\n")
                    fh.flush()
            rc = p.wait()
            log("階段 1 結束 rc=%d，耗時 %.2f 小時" % (rc, (time.time() - t) / 3600), fh)
            if rc != 0:
                log("[warn] whisper 回傳非 0，仍繼續進入分析階段（已完成的檔可用）", fh)

        if args.stage in ("all", "llm"):
            log("── 階段 2／2：地端 Qwen 分析 ──", fh)
            t = time.time()
            rc = run([PY, str(BASE / "cti_llm_pipeline.py"),
                      "--src", str(work / "04_transcript"),
                      "--out", str(analysis),
                      "--model", args.llm], fh)
            log("階段 2 結束 rc=%d，耗時 %.2f 小時" % (rc, (time.time() - t) / 3600), fh)

        res = analysis / "results.json"
        if res.exists():
            calls = json.loads(res.read_text(encoding="utf-8"))["calls"]
            valid = sum(1 for c in calls if c.get("valid"))
            log("分析完成：%d 通（有效 %d）" % (len(calls), valid), fh)
        log("全部結束，總耗時 %.2f 小時 ｜ log: %s"
            % ((time.time() - t0) / 3600, logf), fh)


if __name__ == "__main__":
    main()
