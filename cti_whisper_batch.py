#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTI 錄音批次轉逐字稿 — 本地 Docker whisper.cpp (CUDA)

來源結構：  SRC_ROOT / 診所 / 日期 / 檔案.wav
產出結構：  OUT_ROOT / 02_normalized / 診所 / 日期 / 檔案.wav      (16k WAV，供辨識)
           OUT_ROOT / 04_transcript / 診所 / 日期 / 檔案.txt      (帶時間戳的對話純文字)
           OUT_ROOT / 04_transcript / 診所 / 日期 / 檔案.json     (含時間軸原始結果)
           OUT_ROOT / state/ledger.sqlite                        (斷點續跑帳本)

設計重點
  1. 全程 idempotent：任何時候 Ctrl-C，重跑只做沒做過的。一年份檔案必須能分多天跑完。
  2. 模型只載入一次 / 每批：whisper-cli 支援一次吃多個檔，避免每檔重啟容器重載數 GB 模型。
  3. 語者標記三種模式（見 SPEAKER_MODE）。單軌來源只能用 "none"。

用法
    python cti_whisper_batch.py scan          # 只掃描建帳本
    python cti_whisper_batch.py normalize     # ffmpeg 正規化
    python cti_whisper_batch.py transcribe    # 送 whisper
    python cti_whisper_batch.py run           # 以上三步一次做完（常用）
    python cti_whisper_batch.py status        # 看進度
    python cti_whisper_batch.py retry         # 把 failed 退回重做
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# ════════════════════════════════════════════════════════════════════
# 設定區 —— 只改這裡
# ════════════════════════════════════════════════════════════════════

BASE = Path(__file__).resolve().parent

# 預設全部落在本腳本所在資料夾（= 目前授權資料夾），可用環境變數覆寫，
# 不必改程式。正式跑一年份時建議把 CTI_SRC / CTI_OUT 指到專用大容量磁碟。
#   set CTI_SRC=E:\cti\00_raw
#   set CTI_OUT=E:\cti
SRC_ROOT = Path(os.environ.get("CTI_SRC") or BASE / "00_raw")    # 診所/日期/檔案 的根目錄
OUT_ROOT = Path(os.environ.get("CTI_OUT") or BASE)               # 產出根目錄
MODEL_DIR = Path(os.environ.get("CTI_MODELS") or BASE / "models")  # 放 ggml-*.bin 的目錄

MODEL_FILE = os.environ.get("CTI_MODEL") or "ggml-large-v3.bin"   # 或 ggml-large-v3-turbo.bin
VAD_MODEL_FILE = "ggml-silero-v6.2.0.bin"         # 設為 "" 表示不啟用 VAD

DOCKER_IMAGE = "ghcr.io/ggml-org/whisper.cpp:main-cuda"
USE_GPU = True

AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".gsm", ".au", ".m4a", ".alaw", ".ulaw"}

# 語者標記模式
#   "none"    — 不分語者。每個 segment 一行、附時間戳，由下游 LLM 依脈絡判斷身分。
#               ★ 實測 CTI/A院區 全部 50,581 檔皆為 gsm_ms / 8000 Hz / 單軌，
#                 沒有聲道可分，所以這是唯一可用的模式。
#   "split"   — 左右聲道拆開各辨識一次再依時間軸合併。需雙軌來源。
#   "diarize" — whisper.cpp 內建 -di（比較左右聲道能量）。需雙軌來源；且精細度受限於
#               segment 切分，須搭配 -ml 強制切短（見 whisper_cmd()）。
SPEAKER_MODE = os.environ.get("CTI_SPEAKER_MODE") or "none"

# 聲道對應（務必先用真實樣本確認左右哪邊是誰！）
SPEAKER_LABELS = {"0": "客服", "1": "客戶", "?": "未知"}

WITH_TIMESTAMP = True            # 輸出行首是否帶 [00:00:12.340]
WRITE_META_HEADER = True         # .txt 第一行寫入通話中介資料（方向/分機/對方號碼）
PROMPT_ECHO_FILTER = True        # 若日後重新啟用 INITIAL_PROMPT，擋掉吐回 prompt 的段落
HALLUCINATION_FILTER = True      # 擋掉 whisper 訓練語料殘留的固定幻覺句（見 KNOWN_HALLUCINATIONS）

LANGUAGE = os.environ.get("CTI_LANG") or "zh"

# ★ 預設不下 initial prompt。實測（large-v3、真實 CTI 檔）下 prompt 會造成 prompt 洩漏，
#   而且不是只多一行雜訊 —— 語音內容偏少的通話會「只吐出 prompt 文字」，整通逐字稿全毀：
#     preset-20250905_085936-…（15 秒）有 prompt → 只有「以繁體中文轉寫。」
#                               無 prompt → 完整 10 行對話
#   代價：不下 prompt 時輸出為簡體且少標點，靠 opencc 轉繁（見 CONVERT_TO_TRADITIONAL）。
#   資料完整性優先於標點，所以預設關閉。要開的話務必先用 PROMPT_ECHO_FILTER 擋。
INITIAL_PROMPT = os.environ.get("CTI_PROMPT", "")
BEAM_SIZE = 5
THREADS = 8
DIARIZE_MAX_LEN = 30             # 僅 diarize 模式用：單一 segment 最大字元數

BATCH_SIZE = 48                  # 每個 docker run 餵幾個檔（模型只載一次）
NORMALIZE_WORKERS = 6            # ffmpeg 併發數

# gsm_ms 8k → 16k mono PCM 會膨脹 16 倍（實測 3 MB → 48 MB）。
# 一年份 5.19 GB 原始檔正規化後約 83 GB。轉完就刪，尖峰用量壓到單批大小；
# 原始檔不可變，隨時可重新產生，所以刪掉不會失去任何東西。
KEEP_NORMALIZED = False

CONVERT_TO_TRADITIONAL = True    # 需要 pip install opencc-python-reimplemented

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

# ════════════════════════════════════════════════════════════════════

NORM_ROOT = OUT_ROOT / "02_normalized"
TRANS_ROOT = OUT_ROOT / "04_transcript"
STATE_DIR = OUT_ROOT / "state"
LOG_DIR = OUT_ROOT / "logs"
DB_PATH = STATE_DIR / "ledger.sqlite"

_cc = None
if CONVERT_TO_TRADITIONAL:
    try:
        from opencc import OpenCC
        _cc = OpenCC("s2twp")
    except Exception:
        print("[warn] 未安裝 opencc，略過簡轉繁：pip install opencc-python-reimplemented")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ─────────────────────────── 帳本 ───────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    rel_path    TEXT PRIMARY KEY,
    clinic      TEXT NOT NULL,
    day         TEXT NOT NULL,
    src_size    INTEGER,
    src_mtime   REAL,
    duration    REAL,
    channels    INTEGER,
    sample_rate INTEGER,
    status      TEXT NOT NULL,
    stage       TEXT,
    err         TEXT,
    attempts    INTEGER DEFAULT 0,
    updated_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_status ON files(status);
CREATE INDEX IF NOT EXISTS idx_clinic_day ON files(clinic, day);
"""


def db_connect() -> sqlite3.Connection:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def set_status(conn, rel, status, *, stage=None, err=None, bump=False, **cols):
    sets = ["status=?", "stage=?", "err=?", "updated_at=?"]
    vals = [status, stage, (err or "")[:2000], datetime.now(timezone.utc).isoformat()]
    for k, v in cols.items():
        sets.append(k + "=?")
        vals.append(v)
    if bump:
        sets.append("attempts=attempts+1")
    vals.append(rel)
    conn.execute("UPDATE files SET " + ", ".join(sets) + " WHERE rel_path=?", vals)
    conn.commit()


# ─────────────────────────── 1. 掃描 ───────────────────────────

def cmd_scan(conn) -> None:
    if not SRC_ROOT.is_dir():
        sys.exit("[fatal] 來源目錄不存在：" + str(SRC_ROOT))

    known = {r[0] for r in conn.execute("SELECT rel_path FROM files")}
    new = 0
    now = datetime.now(timezone.utc).isoformat()

    for path in SRC_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        rel = path.relative_to(SRC_ROOT)
        parts = rel.parts
        if len(parts) < 3:
            log("[skip] 深度不符 診所/日期/檔案：" + str(rel))
            continue
        key = str(rel)
        if key in known:
            continue
        st = path.stat()
        conn.execute(
            "INSERT INTO files(rel_path,clinic,day,src_size,src_mtime,status,updated_at) "
            "VALUES(?,?,?,?,?,'discovered',?)",
            (key, parts[0], parts[1], st.st_size, st.st_mtime, now),
        )
        new += 1
        if new % 2000 == 0:
            conn.commit()
            log("  掃描中… 新增 " + str(new))
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    log(f"掃描完成：新增 {new} 筆，帳本共 {total} 筆")


# ─────────────────────────── 2. 正規化 ───────────────────────────

# ★ 每一個捕獲文字輸出的 subprocess 都必須帶這個。
#
# 2026-09-09 查到的坑：`text=True` 沒指定 encoding 時，Python 用 locale 預設編碼
# （這台機器是 cp950）去解子行程的輸出。ffprobe／ffmpeg 失敗時會把**含中文路徑**的
# 錯誤訊息以 UTF-8 寫進 stderr，cp950 解不動 →
# `UnicodeDecodeError: 'cp950' codec can't decode byte 0xe5`（0xe5 是「垂」的首位元組）
# 在 subprocess 的讀取執行緒裡拋出 → **該執行緒死掉，`out.stderr` 變成 `None`** →
# 後面的 `.strip()` 拋 AttributeError，把真正的原因整個蓋掉。
#
# 實際後果：某院區 8/24–8/26 的 75 筆失敗，訊息全是
# `probe: 'NoneType' object has no attribute 'strip'`，
# 真正的原因（`Invalid data found when processing input`，來源檔是 FTP 零填充空殼）
# 一直到用 python 手動重現才查出來。只有「檔案真的壞掉」時才會寫 stderr，
# 所以健康檔案永遠不觸發，看起來像個別現象，其實是全域的。
TEXT_ENC = {"encoding": "utf-8", "errors": "replace"}


def _err(r) -> str:
    """從 CompletedProcess 取錯誤訊息，永不回傳 None。

    有了 TEXT_ENC 之後 stderr 理論上不會是 None，這層是防禦 ——
    上一次就是「只有一層、而那一層失效」才把原因弄丟的。
    """
    msg = ((r.stderr or "") + (r.stdout or "")).strip()
    return msg[:300] or "（無 stderr／stdout，rc=%s）" % r.returncode


def probe(path: Path) -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=channels,sample_rate:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=120, **TEXT_ENC,
    )
    if out.returncode != 0:
        raise RuntimeError("ffprobe: " + _err(out))
    info = json.loads(out.stdout or "{}")
    streams = info.get("streams") or []
    if not streams:
        raise RuntimeError("無音訊軌")
    s = streams[0]
    return {
        "channels": int(s.get("channels") or 0),
        "sample_rate": int(s.get("sample_rate") or 0),
        "duration": float((info.get("format") or {}).get("duration") or 0),
    }


def normalize_one(rel: str):
    """回傳 (rel, status, cols, err)。多執行緒執行，不碰 DB。"""
    src = SRC_ROOT / rel
    try:
        meta = probe(src)
    except Exception as e:
        return rel, "failed", {}, "probe: " + str(e)

    if meta["duration"] <= 0.5:
        return rel, "skipped", meta, "長度過短（未接通）"

    dst = (NORM_ROOT / rel).with_suffix(".wav")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")

    stereo = meta["channels"] >= 2 and SPEAKER_MODE in ("diarize", "split")
    r = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-ac", "2" if stereo else "1", "-ar", "16000",
         "-c:a", "pcm_s16le", "-f", "wav", str(tmp)],
        capture_output=True, text=True, timeout=900, **TEXT_ENC)
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return rel, "failed", meta, "ffmpeg: " + _err(r)

    if SPEAKER_MODE == "split" and stereo:
        for idx, tag in ((0, "_spk0"), (1, "_spk1")):
            side = dst.with_name(dst.stem + tag + ".wav")
            rr = subprocess.run(
                [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(tmp),
                 "-filter_complex", "[0:a]pan=mono|c0=c%d[a]" % idx, "-map", "[a]",
                 "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", str(side)],
                capture_output=True, text=True, timeout=900, **TEXT_ENC)
            if rr.returncode != 0:
                tmp.unlink(missing_ok=True)
                return rel, "failed", meta, "ffmpeg split: " + _err(rr)

    tmp.replace(dst)
    return rel, "normalized", meta, ""


def cmd_normalize(conn) -> None:
    rows = [r[0] for r in conn.execute(
        "SELECT rel_path FROM files WHERE status='discovered' ORDER BY clinic, day")]
    if not rows:
        log("正規化：無待處理檔案")
        return
    log(f"正規化：{len(rows)} 檔，{NORMALIZE_WORKERS} 併發")
    done = 0
    with ThreadPoolExecutor(max_workers=NORMALIZE_WORKERS) as ex:
        futs = [ex.submit(normalize_one, r) for r in rows]
        for f in as_completed(futs):
            rel, status, cols, err = f.result()
            set_status(conn, rel, status,
                       stage="normalize" if status == "failed" else None,
                       err=err, bump=(status == "failed"), **cols)
            done += 1
            if done % 100 == 0 or done == len(rows):
                log(f"  正規化 {done}/{len(rows)}")


# ─────────────────────────── 3. 辨識 ───────────────────────────

def norm_targets(rel: str):
    """回傳 [(語者鍵, 正規化後檔案路徑)]。split 模式回兩個。"""
    base = (NORM_ROOT / rel).with_suffix(".wav")
    if SPEAKER_MODE == "split":
        s0 = base.with_name(base.stem + "_spk0.wav")
        s1 = base.with_name(base.stem + "_spk1.wav")
        if s0.exists() and s1.exists():
            return [("0", s0), ("1", s1)]
    return [("", base)]


def whisper_cmd(container_paths):
    cmd = ["docker", "run", "--rm"]
    if USE_GPU:
        cmd += ["--gpus", "all"]
    cmd += [
        "-v", str(NORM_ROOT.resolve()) + ":/norm",
        "-v", str(MODEL_DIR.resolve()) + ":/models:ro",
        "--entrypoint", "/app/build/bin/whisper-cli",
        DOCKER_IMAGE,
        "-m", "/models/" + MODEL_FILE,
        "-l", LANGUAGE,
        "-t", str(THREADS),
        "-bs", str(BEAM_SIZE),
        "-oj", "-otxt",
        "-np", "-pp",
        "--suppress-nst",
    ]
    if INITIAL_PROMPT:
        cmd += ["--prompt", INITIAL_PROMPT]
    if SPEAKER_MODE == "diarize":
        # -ml 強制切短 segment，否則能量比對的解析度會粗到把一整輪對話標成同一人
        cmd += ["-di", "-ml", str(DIARIZE_MAX_LEN), "-sow"]
    if VAD_MODEL_FILE:
        cmd += ["--vad", "-vm", "/models/" + VAD_MODEL_FILE]
    return cmd + list(container_paths)


def to_container(p: Path) -> str:
    return "/norm/" + p.resolve().relative_to(NORM_ROOT.resolve()).as_posix()


def fmt_ts(ms: int) -> str:
    s, ms = divmod(max(int(ms), 0), 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d.%03d" % (h, m, s, ms)


def _is_cjk(ch: str) -> bool:
    """中日韓文字與全角標點：ord >= 0x2E80。用來決定合併時要不要補空白。"""
    return bool(ch) and ord(ch) >= 0x2E80


def zh(text: str) -> str:
    return _cc.convert(text) if _cc else text


def load_segments(wav: Path, forced_speaker: str = ""):
    """讀 whisper.cpp 產出的 <wav>.json；抓不到就退回 <wav>.txt。"""
    jf = Path(str(wav) + ".json")
    segs = []
    if jf.exists():
        data = json.loads(jf.read_text(encoding="utf-8", errors="replace"))
        for s in data.get("transcription", []):
            off = s.get("offsets") or {}
            segs.append({
                "start": int(off.get("from", 0) or 0),
                "end": int(off.get("to", 0) or 0),
                "speaker": forced_speaker or str(s.get("speaker", "?")),
                "text": (s.get("text") or "").strip(),
            })
        return segs

    tf = Path(str(wav) + ".txt")
    if tf.exists():
        pat = re.compile(r"^\((?:speaker\s*)?(\d|\?)\)\s*(.*)$")
        for line in tf.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            m = pat.match(line)
            spk, txt = (m.group(1), m.group(2)) if m else (forced_speaker or "?", line)
            segs.append({"start": 0, "end": 0,
                         "speaker": forced_speaker or spk, "text": txt.strip()})
    return segs


_PUNCT_RE = re.compile(r"[\s，。、！？,.!?；;：:「」『』（）()]+")


def _is_prompt_echo(txt: str) -> bool:
    """擋 prompt 洩漏：辨識結果只是把 initial prompt 吐回來。

    只在 INITIAL_PROMPT 非空時生效。比對前先去標點與空白，因為模型吐回來的
    版本標點常有出入（實測「以繁體中文轉寫。」「請以繁體中文轉寫。」都出現過）。
    """
    if not PROMPT_ECHO_FILTER or not INITIAL_PROMPT:
        return False
    a = _PUNCT_RE.sub("", txt)
    b = _PUNCT_RE.sub("", INITIAL_PROMPT)
    return len(a) >= 4 and a in b


# 已知的 whisper 訓練語料殘留幻覺句：安靜／低語音段落時模型會吐出這類固定罐頭句
# （源自訓練資料中某 YouTube 頻道的結尾推廣詞），與通話內容無關。
# 實測樣本（A院區 2025-09）：整通僅 13~38 秒，內容 100% 是這句，
# 已被下游 LLM 正確判為 invalid／未接通，不會誤刪真實對話 —— 直接在轉檔階段丟棄即可，
# 省下一次 LLM 呼叫，也不會污染 unresolved_terms（曾被誤猜成「明鏡週報」之類的媒體名稱）。
# 繁簡與「點讚/點贊」「打賞/打赏」用字皆有出現，用不受用字影響的關鍵字「點點欄目」比對。
KNOWN_HALLUCINATIONS = (
    "點點欄目",
)


def _is_known_hallucination(txt: str) -> bool:
    return any(h in txt for h in KNOWN_HALLUCINATIONS)


FNAME_RE = re.compile(
    r"^preset-(?P<d>\d{8})_(?P<t>\d{6})-(?P<a>[^-]+)-(?P<b>[^-]+)$", re.I)


def _is_external(party: str) -> bool:
    """外線判定：非純數字（anonymous）或位數 >= 7 視為外部號碼，2-3 位視為內線分機。"""
    if not party.isdigit():
        return True
    return len(party) >= 7


def parse_meta(rel: str) -> dict:
    """從路徑與檔名抽出通話中介資料，供下游 LLM 判斷身分用。

    檔名規則：preset-<YYYYMMDD>_<HHMMSS>-<主叫>-<被叫>.WAV
    """
    p = Path(rel)
    parts = p.parts
    meta = {"clinic": parts[0] if parts else "", "day_dir": parts[1] if len(parts) > 1 else "",
            "file": p.name, "call_time": "", "caller": "", "callee": "",
            "direction": "未知", "extension": "", "counterparty": ""}
    m = FNAME_RE.match(p.stem)
    if not m:
        return meta
    d, t = m.group("d"), m.group("t")
    meta["call_time"] = "%s-%s-%s %s:%s:%s" % (d[:4], d[4:6], d[6:8], t[:2], t[2:4], t[4:6])
    a, b = m.group("a"), m.group("b")
    meta["caller"], meta["callee"] = a, b
    ext_a, ext_b = _is_external(a), _is_external(b)
    if ext_a and not ext_b:
        meta.update(direction="來電", extension=b, counterparty=a)
    elif not ext_a and ext_b:
        meta.update(direction="撥出", extension=a, counterparty=b)
    elif not ext_a and not ext_b:
        meta.update(direction="內線", extension=a, counterparty=b)
    else:
        meta.update(direction="外轉外", extension="", counterparty="%s→%s" % (a, b))
    return meta


def meta_header(meta: dict, duration: float | None) -> str:
    bits = ["診所: " + meta["clinic"]]
    if meta["call_time"]:
        bits.append("通話時間: " + meta["call_time"])
    bits.append("方向: " + meta["direction"])
    if meta["extension"]:
        bits.append("分機: " + meta["extension"])
    if meta["counterparty"]:
        bits.append("對方: " + meta["counterparty"])
    if duration:
        bits.append("長度: %.0f 秒" % duration)
    return "# " + " | ".join(bits)


def write_transcript(rel: str, segs, duration: float | None = None) -> None:
    out_txt = (TRANS_ROOT / rel).with_suffix(".txt")
    out_json = (TRANS_ROOT / rel).with_suffix(".json")
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    merged = []
    dropped = 0
    for s in sorted(segs, key=lambda x: x["start"]):
        txt = zh(s["text"])
        if not txt:
            continue
        if _is_prompt_echo(txt):
            dropped += 1
            continue
        if HALLUCINATION_FILTER and _is_known_hallucination(txt):
            dropped += 1
            continue
        # 同一語者連續段落合併成一句，讀起來像對話。
        # none 模式不能合併：所有 segment 的 speaker 都相同，合併會把整通通話併成一行，
        # 下游 LLM 就失去判斷輪替（誰換誰講）的唯一線索。
        if SPEAKER_MODE != "none" and merged and merged[-1]["speaker"] == s["speaker"]:
            prev = merged[-1]["text"]
            # 中日韓文字不補空白；兩邊都是拉丁文字時要補，否則會黏成 "youcando"
            sep = "" if (not prev or prev.endswith(" ")
                         or _is_cjk(prev[-1]) or _is_cjk(txt[0])) else " "
            merged[-1]["text"] = prev + sep + txt
            merged[-1]["end"] = s["end"]
        else:
            merged.append({**s, "text": txt})

    meta = parse_meta(rel)
    lines = [meta_header(meta, duration)] if WRITE_META_HEADER else []
    for s in merged:
        who = SPEAKER_LABELS.get(s["speaker"], s["speaker"])
        if SPEAKER_MODE == "none":
            # 單軌無法分語者：保留每個 segment 一行 + 時間戳，
            # 停頓長短是下游 LLM 判斷「換人講話」的主要線索，不可合併掉。
            lines.append("[%s] %s" % (fmt_ts(s["start"]), s["text"])
                         if WITH_TIMESTAMP else s["text"])
        elif WITH_TIMESTAMP:
            lines.append("[%s] %s: %s" % (fmt_ts(s["start"]), who, s["text"]))
        else:
            lines.append("%s: %s" % (who, s["text"]))

    if dropped:
        log("  [filtered] %s 丟棄 %d 段（prompt 洩漏／已知幻覺句）" % (rel, dropped))

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(
        json.dumps({"source": rel, "speaker_mode": SPEAKER_MODE,
                    "duration_sec": duration, "call": meta, "segments": merged},
                   ensure_ascii=False, indent=1), encoding="utf-8")


def preflight_docker() -> None:
    """開跑前確認 Docker daemon 活著。

    沒有這道檢查的話，daemon 沒開會讓每一批 docker run 都非 0 退出，
    程式會把整批 48 檔標成 failed —— 一輪就能汙染幾千筆狀態，
    而真正的原因只是 Docker Desktop 沒啟動。寧可一開始就停。
    """
    try:
        r = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                           capture_output=True, text=True, timeout=30, **TEXT_ENC)
    except (OSError, subprocess.TimeoutExpired) as e:
        sys.exit("[fatal] 無法執行 docker：%s" % e)
    if r.returncode != 0:
        sys.exit("[fatal] Docker daemon 沒有回應，請先啟動 Docker Desktop。\n        "
                 + _err(r))


def cleanup_normalized(tgts) -> None:
    """逐字稿寫出後刪掉正規化檔與 whisper 的 sidecar，避免累積 ~83 GB。"""
    for _, wav in tgts:
        for p in (wav, Path(str(wav) + ".txt"), Path(str(wav) + ".json")):
            try:
                p.unlink(missing_ok=True)
            except OSError as e:
                log("  [warn] 刪除失敗 %s：%s" % (p.name, e))


def cmd_transcribe(conn) -> None:
    rows = [(r[0], r[1]) for r in conn.execute(
        "SELECT rel_path, duration FROM files WHERE status='normalized' "
        "ORDER BY clinic, day")]
    durations = dict(rows)
    rows = [r[0] for r in rows]
    if not rows:
        log("辨識：無待處理檔案")
        return

    if not (MODEL_DIR / MODEL_FILE).exists():
        sys.exit("[fatal] 找不到模型：" + str(MODEL_DIR / MODEL_FILE))
    if VAD_MODEL_FILE and not (MODEL_DIR / VAD_MODEL_FILE).exists():
        sys.exit("[fatal] 找不到 VAD 模型：" + str(MODEL_DIR / VAD_MODEL_FILE)
                 + "（或把 VAD_MODEL_FILE 設為空字串）")

    preflight_docker()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    runlog = LOG_DIR / ("whisper-%s.log" % datetime.now().strftime("%Y%m%d"))
    nbatch = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
    log(f"辨識：{len(rows)} 檔 / {nbatch} 批，模式 {SPEAKER_MODE}")

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        wavs = []
        mapping = {}
        for rel in batch:
            tgts = norm_targets(rel)
            missing = [p for _, p in tgts if not p.exists()]
            if missing:
                set_status(conn, rel, "failed", stage="transcribe",
                           err="缺正規化檔：" + missing[0].name, bump=True)
                continue
            mapping[rel] = tgts
            wavs += [p for _, p in tgts]
        if not wavs:
            continue

        t0 = time.time()
        cmd = whisper_cmd([to_container(p) for p in wavs])
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        with runlog.open("a", encoding="utf-8") as fh:
            fh.write("\n===== batch %d rc=%s %s =====\n"
                     % (i // BATCH_SIZE + 1, r.returncode, datetime.now().isoformat()))
            fh.write(" ".join(cmd) + "\n")
            fh.write((r.stdout or "")[-4000:])
            fh.write((r.stderr or "")[-8000:])

        if r.returncode != 0:
            log("  [批次失敗] rc=%s，詳見 %s" % (r.returncode, runlog))
            print((r.stderr or "")[-1500:])
            for rel in mapping:
                set_status(conn, rel, "failed", stage="transcribe",
                           err="whisper rc=%s" % r.returncode, bump=True)
            continue

        ok = 0
        for rel, tgts in mapping.items():
            try:
                segs = []
                for spk, wav in tgts:
                    segs += load_segments(wav, forced_speaker=spk)
                if not segs:
                    set_status(conn, rel, "skipped", err="無辨識結果（可能全靜音）")
                    if not KEEP_NORMALIZED:
                        cleanup_normalized(tgts)
                    continue
                write_transcript(rel, segs, durations.get(rel))
                set_status(conn, rel, "done")
                if not KEEP_NORMALIZED:
                    cleanup_normalized(tgts)
                ok += 1
            except Exception as e:
                set_status(conn, rel, "failed", stage="postprocess",
                           err=str(e), bump=True)

        el = time.time() - t0
        log("  批次 %d/%d：%d/%d 成功，耗時 %.0fs（%.1fs/檔）"
            % (i // BATCH_SIZE + 1, nbatch, ok, len(mapping), el, el / max(len(wavs), 1)))


# ─────────────────────────── 其他指令 ───────────────────────────

def cmd_status(conn) -> None:
    print("\n狀態統計")
    for status, n in conn.execute(
            "SELECT status, COUNT(*) FROM files GROUP BY status ORDER BY 2 DESC"):
        print("  %-12s %8d" % (status, n))
    tot = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    dur = conn.execute("SELECT COALESCE(SUM(duration),0) FROM files").fetchone()[0]
    print("  %-12s %8d   音訊總長 %.1f 小時" % ("總計", tot, dur / 3600))

    rows = list(conn.execute(
        "SELECT rel_path, stage, err FROM files WHERE status='failed' LIMIT 10"))
    if rows:
        print("\n失敗樣本（最多 10 筆）")
        for rel, stage, err in rows:
            print("  [%s] %s\n      %s" % (stage, rel, err))


def cmd_retry(conn) -> None:
    if KEEP_NORMALIZED:
        sql = ("UPDATE files SET status = CASE WHEN stage='normalize' THEN 'discovered' "
               "ELSE 'normalized' END, err='' WHERE status='failed'")
    else:
        # 正規化檔已被刪除，一律從原始檔重做
        sql = "UPDATE files SET status='discovered', err='' WHERE status='failed'"
    n = conn.execute(sql).rowcount
    conn.commit()
    log("已將 %d 筆失敗退回重做" % n)


def main() -> None:
    ap = argparse.ArgumentParser(description="CTI 錄音批次轉逐字稿")
    ap.add_argument("command",
                    choices=["scan", "normalize", "transcribe", "run", "status", "retry"])
    args = ap.parse_args()

    for d in (NORM_ROOT, TRANS_ROOT, STATE_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if not shutil.which(FFMPEG):
        sys.exit("[fatal] 找不到 ffmpeg，請確認已在 PATH")

    conn = db_connect()
    try:
        if args.command == "scan":
            cmd_scan(conn)
        elif args.command == "normalize":
            cmd_normalize(conn)
        elif args.command == "transcribe":
            cmd_transcribe(conn)
        elif args.command == "run":
            cmd_scan(conn)
            cmd_normalize(conn)
            cmd_transcribe(conn)
            cmd_status(conn)
        elif args.command == "status":
            cmd_status(conn)
        elif args.command == "retry":
            cmd_retry(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
