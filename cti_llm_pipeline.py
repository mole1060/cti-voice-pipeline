#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTI 逐字稿 → 結構化分析（地端 Ollama）  v3

輸入：cti_whisper_batch.py 產出的 04_transcript/<診所>/<日期>/*.txt
輸出：<OUT>/calls/<診所>/<日期>/<檔名>.json   每通一份（斷點續跑的單位）
      <OUT>/results.json                      彙整

設計重點
  1. 全程地端：只打 127.0.0.1 的 Ollama，不經任何雲端 API。
  2. 斷點續跑：每通一個 json 檔，存在就跳過。四千通必須能中斷後接著跑。
  3. 結構化輸出：schema 由 labels.json 產生，enum 直接當 grammar，
     模型只能吐出清單內的值 —— 產出可以直接聚合，不必事後清洗。
  4. 判斷類欄位一律必填：實測選填欄位會被模型直接省略（18 通有 17 通回空）。
  5. 欄位順序＝生成順序：先產出整理後的對話，再基於它下標籤。

用法
    python cti_llm_pipeline.py --src <逐字稿目錄> --out <輸出目錄>
    python cti_llm_pipeline.py --limit 30          # 小量驗證
    python cti_llm_pipeline.py --think             # 思考模式（慢很多）
    python cti_llm_pipeline.py --force             # 忽略既有結果重跑
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from string import Template

BASE = Path(__file__).resolve().parent

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3.8"

DEFAULT_SRC = BASE / "04_transcript"
DEFAULT_OUT = BASE / "05_analysis"
GLOSSARY = BASE / "glossary.json"
LABELS = BASE / "labels.json"
PROMPT_TEMPLATE = BASE / "prompt_system.txt"

# 送進 system prompt 的機構名稱。模型靠它判斷「來電第一句是客服的機構問候語」，
# 也靠它把 whisper 對機構名的誤認拉回來，所以要填實際用的名稱。
# 本 repo 為去識別化版本，預設留佔位字串。
ORG_NAME = os.environ.get("CTI_ORG_NAME", "○○眼科")

# num_ctx 是「輸入＋輸出」的總額度，不是只有輸入。
# 2026-09-02 量測：A院區 3,925 通實測 total_token ≈ 3182 + 4.249 × 逐字稿字元數，
# 舊值 16384 等於把逐字稿長度上限卡在約 3,100 字元 —— 成功通話的 token 總量
# 最大值 16380，離上限只差 4，9 通永久失敗中有 5 通可用超限解釋。
# 模型硬上限為 262144（Qwen3.5 27.3B / Q4_K_M），16384 是我們自己設的軟上限。
# 改 32768：最長那通估 23469 也進得去；KV cache 約 +3 GB（17 GB 權重 + 6 GB ≈ 23.5 GB），
# 32 GB 卡仍有約 9 GB 餘裕。要再往上調前先確認顯存。
NUM_CTX = 32768

# 輸出上限。合法通話實測最多只用到 7,871 tokens（A院區 3,925 通 + 救回的 6 通），
# 這裡給約 50% 餘裕。用途不是省錢，是止血：
# 2026-09-02 實測有少數通話會在 grammar-constrained 輸出時陷入暴走生成
# （逐字稿僅 627 字元卻吐出 4 萬字元 JSON），與逐字稿長短、whisper 重複都無關。
# 不設上限的話，一次暴走會燒滿 num_ctx 再乘以 RETRIES，單通耗時約 17 分鐘。
NUM_PREDICT = 12000
TEMPERATURE = 0.3
TOP_P = 0.9
TIMEOUT = 900
RETRIES = 3

HEADER_RE = re.compile(
    r"診所:\s*(?P<clinic>[^|]+?)\s*\|.*?通話時間:\s*(?P<time>[^|]+?)\s*\|"
    r"\s*方向:\s*(?P<direction>[^|]+?)\s*\|"
    r"(?:\s*分機:\s*(?P<ext>[^|]+?)\s*\|)?"
    r"(?:\s*對方:\s*(?P<other>[^|]+?)\s*\|)?"
    r"\s*長度:\s*(?P<dur>\d+)")
FNAME_RE = re.compile(r"^preset-(\d{8})_(\d{6})-(.+)-(.+)$")


def log(msg: str) -> None:
    print(msg, flush=True)


# ─────────────────────── schema ───────────────────────

def build_schema(lb: dict) -> dict:
    def enum(name):
        return {"type": "string", "enum": lb[name]["values"]}

    def enum_arr(name):
        return {"type": "array", "items": {"type": "string", "enum": lb[name]["values"]}}

    # 順序即生成順序：先寫對話 → 再基於對話下標籤 → 最後回報看不懂的詞
    props = {
        "valid": {"type": "boolean"},
        "invalid_reason": enum("invalid_reason"),
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"speaker": {"type": "string"},
                               "text": {"type": "string"}},
                "required": ["speaker", "text"],
            },
        },
        "summary": {"type": "string"},
        "call_category": enum("call_category"),
        "outcome": enum("outcome"),
        "caller_sentiment": enum("caller_sentiment"),
        "has_complaint_risk": {"type": "boolean"},
        "complaint_severity": enum("complaint_severity"),
        "complaint_reason": {"type": "string"},
        "patient_waiting": {"type": "boolean"},
        "promised_callback": {"type": "boolean"},
        "transferred": {"type": "boolean"},
        "followups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "owner": enum("followup_owner"),
                    "deadline": {"type": "string"},
                },
                "required": ["item", "owner", "deadline"],
            },
        },
        "mentioned_procedures": enum_arr("mentioned_procedures"),
        "mentioned_doctors": {"type": "array", "items": {"type": "string"}},
        "symptoms_mentioned": {"type": "array", "items": {"type": "string"}},
        "price_quoted": {"type": "boolean"},
        "price_detail": {"type": "string"},
        "contains_personal_data": {"type": "boolean"},
        "pii_types": enum_arr("pii_types"),
        "contains_credentials": {"type": "boolean"},
        "service_quality_note": {"type": "string"},
        "speaker_confidence": {"type": "string", "enum": ["高", "中", "低", "不適用"]},
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
                "required": ["from", "to"],
            },
        },
        "unresolved_terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "guess": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["term", "guess", "context"],
            },
        },
    }
    return {"type": "object", "properties": props, "required": list(props.keys())}


# ─────────────────────── prompt ───────────────────────

def build_system_prompt(gl: dict, lb: dict) -> str:
    """把 glossary / labels 的內容填進外部 prompt 樣板。

    樣板刻意放在程式外面：術語對映表與各欄位的判定準則屬於院內資料，
    要能單獨管控、單獨改版，不隨程式碼一起流通。
    樣板用 string.Template 的 $name 佔位符（不是 str.format），
    因為 prompt 裡遲早會出現 JSON 範例的大括號。
    """
    if not PROMPT_TEMPLATE.exists():
        sys.exit("[fatal] 找不到 prompt 樣板：%s\n"
                 "        請由 config/prompt_system.example.txt 複製一份到根目錄，"
                 "填入實際內容後再跑。" % PROMPT_TEMPLATE)

    conf = [c for c in gl["corrections"] if c["confidence"] == "confirmed"]
    likely = [c for c in gl["corrections"] if c["confidence"] == "likely"]

    def fmt(items):
        return "\n".join("  %s ← %s" % (c["correct"], "／".join(c["variants"][:8]))
                         for c in items)

    # 小量測試發現：只給 enum 清單、不給每個值的判定準則，模型會亂選
    #（29 通裡有 17 通把「已當場解決」標成「已轉他人處理」）。定義放 labels.json 維護。
    values = {
        "org_name": ORG_NAME,
        "confirmed_corrections": fmt(conf),
        "likely_corrections": fmt(likely),
        "vocabulary": "\n".join("  %s：%s" % (k, "、".join(v))
                                for k, v in gl["vocabulary"].items()),
        "rules": "\n".join("  - %s：%s" % (r["desc"], r["detail"])
                           for r in gl["rules"]),
        "invalid_markers": "、".join(gl["invalid_call_markers"]["phrases"]),
        "category_tie_break": "\n".join(
            "    %d. %s" % (i, r)
            for i, r in enumerate(lb["call_category"].get("tie_break", []), 1)),
        "outcome_definitions": "\n".join(
            "    - %s：%s" % (k, v)
            for k, v in lb["outcome"].get("definitions", {}).items()),
        "sentiment_note": lb["caller_sentiment"].get("note", ""),
    }

    # rstrip 尾端換行：樣板檔照一般文字檔慣例以換行結尾，但 prompt 本身不該多帶
    # 一個換行進去。少了這行，重構前後送給模型的 prompt 會差一個字元。
    tpl = Template(PROMPT_TEMPLATE.read_text(encoding="utf-8").rstrip("\n"))
    try:
        return tpl.substitute(values)
    except KeyError as e:
        sys.exit("[fatal] prompt 樣板用到未定義的佔位符 $%s" % e.args[0])


def parse_header(text: str, fname: str) -> dict:
    meta = {"direction": "未知", "extension": "", "counterparty": "",
            "duration": 0.0, "clinic": "", "call_time": ""}
    m = HEADER_RE.search(text)
    if m:
        meta.update(clinic=m.group("clinic").strip(), call_time=m.group("time").strip(),
                    direction=m.group("direction").strip(),
                    extension=(m.group("ext") or "").strip(),
                    counterparty=(m.group("other") or "").strip(),
                    duration=float(m.group("dur")))
    f = FNAME_RE.match(fname)
    if f and not meta["call_time"]:
        meta["call_time"] = "%s-%s-%s %s:%s:%s" % (
            f.group(1)[:4], f.group(1)[4:6], f.group(1)[6:8],
            f.group(2)[:2], f.group(2)[2:4], f.group(2)[4:6])
    return meta


def build_user_prompt(header: str, body: list[str]) -> str:
    return ("以下是一通電話的逐字稿。\n\n【通話資訊】\n%s\n\n【逐字稿】\n%s\n"
            % (header.lstrip("# ").strip(), "\n".join(body)))


# ─────────────────────── Ollama ───────────────────────

def ollama_chat(system: str, user: str, schema: dict, think: bool):
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "format": schema,
        "think": think,
        "options": {"temperature": TEMPERATURE, "top_p": TOP_P,
                    "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
    }
    body = json.dumps(payload).encode("utf-8")

    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(
                OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                resp = json.loads(r.read().decode("utf-8"))
            content = (resp.get("message") or {}).get("content") or ""
            n_out = resp.get("eval_count") or 0
            used = (resp.get("prompt_eval_count") or 0) + n_out
            if not content.strip():
                raise ValueError("空回應")
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                # 額度用完時輸出會被硬切斷，症狀是 JSON 解不開（Unterminated string 之類），
                # 看起來像模型亂寫、實際是撞到上限。分開報，免得誤診成模型品質問題。
                # 兩種撞法要分辨：撞 num_predict＝模型暴走（與逐字稿長短無關，見上方註解）；
                # 撞 num_ctx＝逐字稿真的太長，塞不下輸入＋輸出。
                if n_out >= NUM_PREDICT * 0.98:
                    raise ValueError(
                        "輸出撞到 num_predict 上限（%d tokens）疑似暴走生成："
                        "%s" % (NUM_PREDICT, str(e)[:80])) from None
                if used >= NUM_CTX * 0.98:
                    raise ValueError(
                        "輸出被 num_ctx 截斷（已用 %d／上限 %d，逐字稿過長）：%s"
                        % (used, NUM_CTX, str(e)[:80])) from None
                raise
            return data, {
                "eval_count": resp.get("eval_count", 0),
                "prompt_eval_count": resp.get("prompt_eval_count", 0),
            }
        except (urllib.error.URLError, TimeoutError, ValueError,
                json.JSONDecodeError, OSError) as e:
            last = e
            log("      重試 %d/%d：%s" % (attempt, RETRIES, str(e)[:120]))
            time.sleep(3 * attempt)
    raise RuntimeError("Ollama 呼叫失敗：%s" % last)


# ─────────────────────── 主流程 ───────────────────────

def derive_flags(d: dict) -> list[str]:
    """把結構化欄位轉成人可讀的 flags，給報表與 render 用。"""
    flags = []
    if not d.get("valid", True):
        flags.append("排除分析：" + (d.get("invalid_reason") or "無效"))
    if d.get("has_complaint_risk"):
        sev = d.get("complaint_severity") or "低"
        r = (d.get("complaint_reason") or "").strip()
        flags.append("客訴風險（%s）%s" % (sev, "：" + r if r else ""))
    if d.get("patient_waiting"):
        flags.append("病患現場等候")
    if d.get("promised_callback"):
        flags.append("已承諾回電")
    for fu in (d.get("followups") or []):
        item = str(fu.get("item", "")).strip()
        if item:
            owner = fu.get("owner") or "不明"
            dl = (fu.get("deadline") or "").strip()
            flags.append("需追蹤：%s（%s%s）" % (item, owner, "／" + dl if dl else ""))
    if d.get("contains_personal_data"):
        flags.append("含病患個資：" + "、".join(d.get("pii_types") or []))
    if d.get("contains_credentials"):
        flags.append("含憑證資訊")
    return flags


def process_one(path: Path, system: str, schema: dict, think: bool, rel: str) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    header = next((l for l in lines if l.startswith("#")), "")
    body = [l for l in lines if l.strip() and not l.startswith("#")]
    meta = parse_header(header, path.stem)

    if not body:
        return {"file": path.stem, "rel": rel, **meta, "valid": False,
                "invalid_reason": "空白錄音", "call_category": "未接通或無效",
                "summary": "逐字稿無內容。", "speaker_confidence": "不適用",
                "turns": [], "flags": ["排除分析：空白錄音"], "labels": {},
                "_stats": {"eval_count": 0, "skipped": True}}

    data, stats = ollama_chat(system, build_user_prompt(header, body), schema, think)
    # 模型偶爾會在多選陣列裡重複同一個值，去重才能正確聚合
    for k in ("mentioned_procedures", "pii_types",
              "mentioned_doctors", "symptoms_mentioned"):
        v = data.get(k)
        if isinstance(v, list):
            data[k] = list(dict.fromkeys(x for x in v if x))
    labels = {k: v for k, v in data.items() if k not in ("turns", "summary")}
    return {
        "file": path.stem, "rel": rel, **meta,
        "valid": bool(data.get("valid", True)),
        "invalid_reason": data.get("invalid_reason", ""),
        "call_category": data.get("call_category", "其他"),
        "summary": data.get("summary", ""),
        "speaker_confidence": data.get("speaker_confidence", "中"),
        "turns": data.get("turns", []) or [],
        "flags": derive_flags(data),
        "labels": labels,
        "_stats": stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="CTI 逐字稿地端 LLM 分析 v3")
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    globals()["MODEL"] = args.model

    src, out = Path(args.src), Path(args.out)
    files = sorted(src.rglob("*.txt"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit("[fatal] %s 下沒有逐字稿" % src)

    gl = json.loads(GLOSSARY.read_text(encoding="utf-8"))
    lb = json.loads(LABELS.read_text(encoding="utf-8"))
    schema = build_schema(lb)
    system = build_system_prompt(gl, lb)

    out.mkdir(parents=True, exist_ok=True)
    # 每次執行都留一份「實際送出去的 prompt」快照，事後才查得出某批結果是哪個版本產的。
    # 檔名與根目錄的樣板同名但不同層；--out . 會讓快照蓋掉樣板，擋掉。
    snapshot = out / "prompt_system.txt"
    if snapshot.resolve() == PROMPT_TEMPLATE.resolve():
        sys.exit("[fatal] --out 指到專案根目錄，prompt 快照會覆蓋掉樣板檔。請換一個輸出目錄。")
    snapshot.write_text(system, encoding="utf-8")
    (out / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    runlog = out / "run.log"

    log("模型 %s ｜ think=%s ｜ %d 通 ｜ prompt %d 字 ｜ 欄位 %d"
        % (MODEL, args.think, len(files), len(system), len(schema["properties"])))

    t_all = time.time()
    done = fail = skip = 0
    tokens = 0
    for i, f in enumerate(files, 1):
        rel = f.relative_to(src).as_posix()
        dst = (out / "calls" / f.relative_to(src)).with_suffix(".json")
        if dst.exists() and not args.force:
            skip += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            rec = process_one(f, system, schema, args.think, rel)
        except Exception as e:
            fail += 1
            msg = "[%d/%d] 失敗 %s：%s" % (i, len(files), f.stem, str(e)[:200])
            log(msg)
            with runlog.open("a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
            continue
        dst.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        tokens += rec.get("_stats", {}).get("eval_count", 0)
        done += 1
        if done % 25 == 0 or i == len(files):
            el = time.time() - t_all
            rate = done / el if el else 0
            left = (len(files) - skip - done - fail) / rate if rate else 0
            log("[%d/%d] 完成 %d 失敗 %d 略過 %d ｜ %.2f 通/秒 ｜ 預估剩餘 %.1f 小時"
                % (i, len(files), done, fail, skip, rate, left / 3600))
        elif done <= 5:
            log("[%d/%d] %s  %-14s turns=%-3d %4.1fs"
                % (i, len(files), f.stem[16:22], rec["call_category"],
                   len(rec["turns"]), time.time() - t0))

    calls = []
    for f in files:
        p = (out / "calls" / f.relative_to(src)).with_suffix(".json")
        if p.exists():
            rec = json.loads(p.read_text(encoding="utf-8"))
            rec.pop("_stats", None)
            calls.append(rec)
    (out / "results.json").write_text(
        json.dumps({"meta": {"model": MODEL, "think": args.think,
                             "glossary": gl.get("version"), "labels": lb.get("version"),
                             "source": str(src), "processor": "地端 Ollama",
                             "generated": time.strftime("%Y-%m-%d %H:%M:%S")},
                    "calls": calls}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    el = time.time() - t_all
    log("\n完成 %d ｜ 失敗 %d ｜ 略過 %d ｜ 總計 %d 通" % (done, fail, skip, len(calls)))
    log("耗時 %.1f 小時，輸出 %d tokens" % (el / 3600, tokens))
    if done:
        log("平均 %.1f 秒/通（%.0f tok/s）" % (el / done, tokens / el if el else 0))
    log("→ %s" % (out / "results.json"))


if __name__ == "__main__":
    main()
