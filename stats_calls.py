#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTI 全量通話量統計 —— 只讀檔名與 WAV 檔頭，不辨識、不上 GPU

用途：算出 CTI/<院區>/<日期>/*.WAV 每個診所每天的通話次數、通話總時數、
內外線分類（來電／撥出／內線／外轉外），並依既有實測速度估算跑完
whisper + Qwen 兩階段大概要多久，供排期用。純 I/O，數十萬檔也是分鐘等級，
完全不動 GPU，可以跟現有轉檔批次同時跑。

方向判定直接呼叫 cti_whisper_batch.py 的 parse_meta()，跟正式轉檔管線
共用同一套規則，不兩邊各寫一份、之後改一邊忘了改另一邊。

通話時長讀 WAV 檔頭的 fmt chunk（byte_rate）+ data chunk 大小算出來，
不叫 ffprobe（幾十萬檔會慢很多）。抓到標準 chunk 時是精確值（byte_rate
是檔頭自己宣告的播放位元率，對 gsm_ms 這種固定位元率格式一樣準）；
抓不到才退回「檔案大小 / 13000 bit/s（gsm_ms 實測值，見參考數據.md）」
估算，並在輸出中標記筆數。

用法：
    python stats_calls.py                          # 掃 CTI/，輸出到目前資料夾
    python stats_calls.py --root CTI --workers 24
    python stats_calls.py --out-prefix call_stats_0902
"""

from __future__ import annotations

import argparse
import csv
import re
import struct
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from cti_whisper_batch import parse_meta, AUDIO_EXTS  # noqa: E402  跟正式管線共用同一套方向判定

DAY_RE = re.compile(r"^\d{8}$")

# 常數來源：參考數據.md §3「效能基準」
#   whisper：A院區 2025-09 實測 75.1 小時音訊 → 2.25 小時完成（large-v3，含正規化），
#            速度係數 = 75.1 / 2.25 ≈ 33.4x
#   Qwen：26 欄位版（目前用的版本）無競爭時 7.4–8.0 秒/通，取上界保守估
WHISPER_SPEEDUP = 33.4
QWEN_SEC_PER_CALL = 8.0
WORKDAY_HOURS = 12  # 使用者實測：一個診所一個月量約跑 12 小時，換算成「工作天」數用


def log(msg: str) -> None:
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def wav_duration_sec(path: Path) -> tuple[float, bool]:
    """回傳 (秒數, 是否為估計值)。只讀檔頭前 4KB，不讀整檔內容。"""
    try:
        with path.open("rb") as f:
            head = f.read(4096)
    except OSError:
        return 0.0, True

    if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        try:
            size = path.stat().st_size
        except OSError:
            return 0.0, True
        return max(0.0, (size - 44) * 8 / 13000), True

    pos, byte_rate, data_size = 12, None, None
    while pos + 8 <= len(head):
        cid = head[pos:pos + 4]
        (csize,) = struct.unpack_from("<I", head, pos + 4)
        body = pos + 8
        if cid == b"fmt " and body + 16 <= len(head):
            _, _, _, byte_rate, _, _ = struct.unpack_from("<HHIIHH", head, body)
        elif cid == b"data":
            data_size = csize
            break  # data 通常是最後一個且體積最大的 chunk，抓到就不必往後找
        pos = body + csize + (csize & 1)  # chunk 依 RIFF 規格 word-align

    if byte_rate and data_size is not None:
        return data_size / byte_rate, False

    try:
        size = path.stat().st_size
    except OSError:
        return 0.0, True
    return max(0.0, (size - 44) * 8 / 13000), True


def scan_clinic(clinic_dir: Path, workers: int) -> list[tuple[str, str, float, bool]]:
    """回傳該院區底下每個音檔的 (日期, 方向, 秒數, 是否估計)。"""
    clinic = clinic_dir.name
    tasks = []
    for day_dir in sorted(p for p in clinic_dir.iterdir() if p.is_dir() and DAY_RE.match(p.name)):
        day = day_dir.name
        for f in day_dir.iterdir():
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                tasks.append((day, f))
    if not tasks:
        return []

    def work(item):
        day, f = item
        rel = "%s/%s/%s" % (clinic, day, f.name)
        direction = parse_meta(rel)["direction"]
        sec, est = wav_duration_sec(f)
        return day, direction, sec, est

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(work, tasks))


def main() -> None:
    ap = argparse.ArgumentParser(description="CTI 通話量統計（純檔名/檔頭，不辨識、不用 GPU）")
    ap.add_argument("--root", default=str(BASE / "CTI"), help="CTI 根目錄")
    ap.add_argument("--workers", type=int, default=24, help="每個院區的檔頭讀取執行緒數")
    ap.add_argument("--out-prefix", default="call_stats", help="輸出 CSV 檔名前綴")
    args = ap.parse_args()

    root = Path(args.root)
    clinics = sorted(p for p in root.iterdir() if p.is_dir())
    log("院區數：%d" % len(clinics))

    # daily[(診所, 日期, 方向)] = [通數, 總秒數, 估計值筆數]
    daily: dict[tuple[str, str, str], list] = defaultdict(lambda: [0, 0.0, 0])
    t0 = time.time()
    for i, clinic_dir in enumerate(clinics, 1):
        t1 = time.time()
        rows = scan_clinic(clinic_dir, args.workers)
        for day, direction, sec, est in rows:
            k = (clinic_dir.name, day, direction)
            row = daily[k]
            row[0] += 1
            row[1] += sec
            row[2] += 1 if est else 0
        log("[%d/%d] %-8s：%6d 檔，%.1f 秒" % (i, len(clinics), clinic_dir.name, len(rows), time.time() - t1))
    log("掃描完成，共耗時 %.1f 秒" % (time.time() - t0))

    daily_path = BASE / (args.out_prefix + "_daily.csv")
    with daily_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["診所", "日期", "方向", "通數", "通話時數", "估計值筆數"])
        for (clinic, day, direction), (n, sec, est) in sorted(daily.items()):
            w.writerow([clinic, day, direction, n, round(sec / 3600, 3), est])
    log("已寫出每日明細：%s" % daily_path)

    monthly: dict[tuple[str, str], list] = defaultdict(lambda: [0, 0.0])
    for (clinic, day, _direction), (n, sec, _est) in daily.items():
        row = monthly[(clinic, day[:6])]
        row[0] += n
        row[1] += sec

    monthly_path = BASE / (args.out_prefix + "_monthly.csv")
    with monthly_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["診所", "月份", "通數", "通話時數", "預估管線耗時_小時"])
        for (clinic, ym), (n, sec) in sorted(monthly.items()):
            est_hr = sec / 3600 / WHISPER_SPEEDUP + n * QWEN_SEC_PER_CALL / 3600
            w.writerow([clinic, ym, n, round(sec / 3600, 2), round(est_hr, 2)])
    log("已寫出月彙整：%s" % monthly_path)

    by_clinic: dict[str, list] = defaultdict(lambda: [0, 0.0, set()])
    for (clinic, ym), (n, sec) in monthly.items():
        row = by_clinic[clinic]
        row[0] += n
        row[1] += sec
        row[2].add(ym)

    log("=" * 84)
    log("%-10s %6s %10s %10s %12s %14s" % ("診所", "月數", "通數", "時數", "平均通數/月", "預估總耗時(h)"))
    grand_n = grand_sec = grand_hr = grand_months = 0
    for clinic, (n, sec, months) in sorted(by_clinic.items()):
        m = len(months)
        est_hr = sec / 3600 / WHISPER_SPEEDUP + n * QWEN_SEC_PER_CALL / 3600
        log("%-10s %6d %10d %10.1f %12.0f %14.1f" % (clinic, m, n, sec / 3600, n / m if m else 0, est_hr))
        grand_n += n
        grand_sec += sec
        grand_hr += est_hr
        grand_months += m
    log("-" * 84)
    log("合計：%d 通｜%.1f 小時音訊｜%d 個診所-月｜全部跑完預估 %.1f 小時（以每天跑 %d 小時計，約 %.0f 個工作天）"
        % (grand_n, grand_sec / 3600, grand_months, grand_hr, WORKDAY_HOURS, grand_hr / WORKDAY_HOURS))
    log("（此為「全部歷史資料」的參考上限，不代表實際排程——上線月份另議）")


if __name__ == "__main__":
    main()
