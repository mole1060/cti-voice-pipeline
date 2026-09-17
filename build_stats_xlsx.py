#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 call_stats_daily.csv / call_stats_monthly.csv 整理成一份 Excel，
外加一份 33 診所簡易版（總時數／本地執行天數／可分辨語者 API 價格區間概算）。

簡易版全部用公式算，改假設區的數字（匯率、API 低/高價、每日可執行小時數）
表格會自動重算，不是寫死的數字。
"""
import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

BASE = Path(__file__).resolve().parent
FONT = "Arial"

wb = Workbook()

# ── 1. 簡易估算_33院所（放第一個分頁）──────────────────────────────
ws1 = wb.active
ws1.title = "簡易估算_33院所"

bold = Font(name=FONT, bold=True)
normal = Font(name=FONT)
title_font = Font(name=FONT, bold=True, size=14)
assump_fill = PatternFill("solid", fgColor="FFFF00")
header_fill = PatternFill("solid", fgColor="D9E1F2")

ws1["A1"] = "CTI 全量 33 診所 — 簡易時程與 API 成本估算"
ws1["A1"].font = title_font

ws1["A3"] = "假設（改這裡，下面表格會自動重算）"
ws1["A3"].font = bold

assumptions = [
    ("匯率 USD→TWD", 31.69, "來源：參考數據.md §4，2026-09 報價（簽約前需以官方報價為準）"),
    ("API 低價（US$/小時，可分辨語者）", 0.230, "AssemblyAI Universal-2 Pro + 語者分離，參考數據.md §4"),
    ("API 高價（US$/小時，可分辨語者）", 0.312, "Deepgram Nova-3 多語，參考數據.md §4"),
    ("本地每日可執行小時數", 12, "使用者實測節奏（2026-09-02 對話：一個診所一個月量約跑 12 小時）"),
]
for i, (label, value, note) in enumerate(assumptions, start=4):
    ws1.cell(row=i, column=1, value=label).font = normal
    c = ws1.cell(row=i, column=2, value=value)
    c.font = Font(name=FONT, color="0000FF")
    c.fill = assump_fill
    ws1.cell(row=i, column=3, value=note).font = Font(name=FONT, italic=True, size=9, color="808080")

FX_CELL = "$B$4"
LOW_CELL = "$B$5"
HIGH_CELL = "$B$6"
WORKDAY_CELL = "$B$7"

header_row = 9
headers = ["診所", "總時數（小時）", "本地執行天數", "API 價格區間_低（NT$）", "API 價格區間_高（NT$）"]
for col, h in enumerate(headers, start=1):
    c = ws1.cell(row=header_row, column=col, value=h)
    c.font = bold
    c.fill = header_fill
    c.alignment = Alignment(horizontal="center")

# ── 讀月彙整 CSV，抓出不重複診所清單（依總時數由大到小排序，跟前面回報一致）──
monthly_rows = list(csv.DictReader((BASE / "call_stats_monthly.csv").open(encoding="utf-8-sig")))
clinic_hours = {}
for r in monthly_rows:
    clinic_hours[r["診所"]] = clinic_hours.get(r["診所"], 0.0) + float(r["通話時數"])
clinics = sorted(clinic_hours, key=lambda k: -clinic_hours[k])

first_data_row = header_row + 1
for i, clinic in enumerate(clinics):
    row = first_data_row + i
    ws1.cell(row=row, column=1, value=clinic).font = normal
    # 總時數：從「月彙整」分頁用 SUMIF 加總，不寫死數字
    ws1.cell(row=row, column=2,
              value='=SUMIF(月彙整!$A:$A,A%d,月彙整!$D:$D)' % row).font = normal
    # 本地執行天數 = 該診所總管線工時 / 每日可執行小時數
    ws1.cell(row=row, column=3,
              value='=SUMIF(月彙整!$A:$A,A%d,月彙整!$E:$E)/%s' % (row, WORKDAY_CELL)).font = normal
    ws1.cell(row=row, column=4, value='=B%d*%s*%s' % (row, LOW_CELL, FX_CELL)).font = normal
    ws1.cell(row=row, column=5, value='=B%d*%s*%s' % (row, HIGH_CELL, FX_CELL)).font = normal

total_row = first_data_row + len(clinics)
ws1.cell(row=total_row, column=1, value="合計").font = bold
for col, letter in ((2, "B"), (3, "C"), (4, "D"), (5, "E")):
    rng = "%s%d:%s%d" % (letter, first_data_row, letter, total_row - 1)
    c = ws1.cell(row=total_row, column=col, value="=SUM(%s)" % rng)
    c.font = bold

for row in range(first_data_row, total_row + 1):
    ws1.cell(row=row, column=2).number_format = "#,##0.0"
    ws1.cell(row=row, column=3).number_format = "#,##0.0"
    ws1.cell(row=row, column=4).number_format = '"NT$"#,##0'
    ws1.cell(row=row, column=5).number_format = '"NT$"#,##0'

ws1.cell(row=total_row + 2, column=1,
          value="註：本表為「全部歷史資料」的參考上限，不代表實際排程；上線月份另議（見 決策記錄.md）。").font = \
    Font(name=FONT, italic=True, size=9, color="808080")

widths1 = [14, 16, 14, 20, 20]
for i, w in enumerate(widths1, start=1):
    ws1.column_dimensions[get_column_letter(i)].width = w
ws1.column_dimensions["C"].width = 45  # 容納假設區備註（第3欄同時是備註欄）

ws1.freeze_panes = "A%d" % first_data_row


def dump_csv_to_sheet(csv_path: Path, ws) -> None:
    rows = list(csv.reader(csv_path.open(encoding="utf-8-sig")))
    for r_i, row in enumerate(rows, start=1):
        for c_i, val in enumerate(row, start=1):
            cell = ws.cell(row=r_i, column=c_i, value=val)
            if r_i == 1:
                cell.font = bold
                cell.fill = header_fill
            else:
                cell.font = normal
                # 數字欄位轉成真的數字，Excel 才能排序/加總
                try:
                    if "." in val:
                        cell.value = float(val)
                    else:
                        cell.value = int(val)
                except ValueError:
                    pass
    ws.freeze_panes = "A2"
    for i in range(1, len(rows[0]) + 1):
        ws.column_dimensions[get_column_letter(i)].width = 16


# ── 2. 月彙整 ──────────────────────────────────────────────
ws2 = wb.create_sheet("月彙整")
dump_csv_to_sheet(BASE / "call_stats_monthly.csv", ws2)

# ── 3. 每日明細 ────────────────────────────────────────────
ws3 = wb.create_sheet("每日明細")
dump_csv_to_sheet(BASE / "call_stats_daily.csv", ws3)

out = BASE / "call_stats.xlsx"
wb.save(out)
print("已寫出：%s" % out)
