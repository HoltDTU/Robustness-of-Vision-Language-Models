"""
Produktions-loop for VLM-graf-evaluering (02445) -- FARVE-PROBE.
================================================================
240 grafer. temperature=0.0 (deterministisk).
2 isolerede spoergsmaal pr. graf (antal blaa knuder / antal roede knuder).

Formaal: teste om modellerne faktisk OPFATTER farve. Fejl paa farve-taelling
sammenlignes med fejl paa total-taelling: ligner de hinanden, ser modellen
farve fint, og flaskehalsen er taelling -- ikke perception.
Indbygget konsistenscheck: hvis alle knuder er farvede, skal blaa + roed = total.

ROBUSTHED MOD FORBINDELSESTAB:
  * Ved opstart fjernes ufuldstaendige raekker (skrevet under et tidligere
    netvaerks-udfald), saa de koeres igen.
  * En raekke skrives KUN, naar begge svar er modtaget. Falder forbindelsen,
    stopper scriptet rent uden at skrive halve/tomme raekker -> sikker resume.

OBS: API'et kan kun naas paa DTU's netvaerk (campus eller DTU-VPN).

Brug:
    pip install openai python-dotenv pandas
    python run_study.py
"""
import base64
import csv
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / "API.env")

BASE_URL = "https://api.campusai.compute.dtu.dk/v1"
MODELS = ["google/gemma-4-26b-a4b", "alibaba/qwen-3.6-35b-a3b"]

MANIFEST_CSV = HERE / "correct_graph_answers.csv"
IMAGES_DIR   = HERE / "png_grafer"
PRED_CSV     = HERE / "blue_red_predictions.csv"
RAW_LOG      = HERE / "blue_red_raw_responses.csv"

TEMPERATURE = 0.0
MAX_TOKENS  = 512
EXTRA_BODY  = {"chat_template_kwargs": {"enable_thinking": False}}

# Manifest-kolonner (facit)
COL_ID    = "image_id"
COL_FILE  = "filename"
COL_LEVEL = "node_level"
COL_NODES = "n_nodes_true"
COL_BLUE  = "n_blue_nodes_true"
COL_RED   = "n_red_nodes_true"

TASKS = {
    "blue": {"prompt": "Hvor mange blå knuder (cirkler) er der i denne graf? "
                       "Svar kun med et tal.",
             "kind": "count", "pred_col": "pred_blue"},
    "red":  {"prompt": "Hvor mange røde knuder (cirkler) er der i denne graf? "
                       "Svar kun med et tal.",
             "kind": "count", "pred_col": "pred_red"},
}

PRED_FIELDS = ["image_id", "model", "pred_blue", "pred_red"]
RAW_FIELDS  = ["timestamp", "image_id", "model", "task", "raw_answer",
               "finish_reason", "completion_tokens"]

client = OpenAI(api_key=os.environ["CAMPUSAI_API_KEY"], base_url=BASE_URL, timeout=30.0)


def parse_count(text):
    if not text:
        return None
    m = re.search(r"-?\d+", text.replace(".", "").replace(",", ""))
    return int(m.group()) if m else None


def extract(resp):
    msg = resp.choices[0].message
    content = msg.content
    if not content:
        content = getattr(msg, "reasoning_content", None) or ""
    return content.strip()


def call_with_retry(make_call, retries=4, backoff=3.0):
    """Genproever korte blip. Efter sidste forsoeg KASTES fejlen videre."""
    for attempt in range(retries):
        try:
            return make_call()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"    (forsoeg {attempt+1} fejlede: {e}; venter...)")
            time.sleep(backoff * (attempt + 1))


def ask(model, prompt, b64):
    def make_call():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE, extra_body=EXTRA_BODY,
        )
    resp = call_with_retry(make_call)
    return extract(resp), resp.choices[0].finish_reason, resp.usage.completion_tokens


def resolve_images_dir(manifest):
    first = str(manifest[COL_FILE].iloc[0])
    if (IMAGES_DIR / first).exists():
        return IMAGES_DIR
    for cand in sorted(HERE.rglob(first)):
        return cand.parent
    raise SystemExit(f"\nKunne IKKE finde billederne (ledte efter '{first}'). "
                     f"Saet IMAGES_DIR i toppen af scriptet.\n")


def clean_predictions():
    """Fjern ufuldstaendige raekker (begge taelle-svar tomme) fra et tidligere
    netvaerks-udfald, saa de koeres igen ved resume."""
    if not PRED_CSV.exists() or PRED_CSV.stat().st_size == 0:
        return
    df = pd.read_csv(PRED_CSV)
    bad = df["pred_blue"].isna() & df["pred_red"].isna()
    if bad.any():
        print(f"Oprydning: fjerner {int(bad.sum())} ufuldstaendige raekker "
              f"(forbindelsestab) -- de koeres igen.")
        df[~bad].to_csv(PRED_CSV, index=False)


def load_done():
    done = set()
    if PRED_CSV.exists() and PRED_CSV.stat().st_size > 0:
        with open(PRED_CSV, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["image_id"], row["model"]))
    return done


def main():
    df = pd.read_csv(MANIFEST_CSV)
    images_dir = resolve_images_dir(df)
    found = sum((images_dir / str(r[COL_FILE])).exists() for _, r in df.iterrows())
    print(f"Billedmappe: {images_dir}  ({found}/{len(df)} billeder fundet)")
    if found == 0:
        raise SystemExit("Fandt ingen billeder -- tjek IMAGES_DIR. Stopper.")

    clean_predictions()
    done = load_done()
    total_calls = len(df) * len(MODELS) * len(TASKS)
    print(f"Planlagte kald: {total_calls}. Allerede faerdige: "
          f"{len(done)} (graf, model).  temperature={TEMPERATURE}\n")

    pred_new = not PRED_CSV.exists() or PRED_CSV.stat().st_size == 0
    raw_new  = not RAW_LOG.exists() or RAW_LOG.stat().st_size == 0
    pred_fh = open(PRED_CSV, "a", newline="")
    raw_fh  = open(RAW_LOG, "a", newline="")
    pred_w = csv.DictWriter(pred_fh, fieldnames=PRED_FIELDS)
    raw_w  = csv.DictWriter(raw_fh, fieldnames=RAW_FIELDS)
    if pred_new: pred_w.writeheader()
    if raw_new:  raw_w.writeheader()

    n_rows, t0 = 0, time.time()
    for _, row in df.iterrows():
        image_id = str(row[COL_ID])
        img_path = images_dir / str(row[COL_FILE])
        if not img_path.exists():
            print(f"  ADVARSEL: billede mangler: {img_path} -- springes over")
            continue
        b64 = base64.b64encode(img_path.read_bytes()).decode()

        for model in MODELS:
            if (image_id, model) in done:
                continue
            pred_row = {"image_id": image_id, "model": model}
            raw_rows = []
            try:
                for task_name, task in TASKS.items():
                    raw, finish, ctoks = ask(model, task["prompt"], b64)
                    pred_row[task["pred_col"]] = parse_count(raw)
                    raw_rows.append({
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "image_id": image_id, "model": model, "task": task_name,
                        "raw_answer": raw, "finish_reason": finish, "completion_tokens": ctoks})
            except Exception as e:
                # Vedvarende fejl (typisk tabt forbindelse): stop RENT, skriv ikke.
                pred_fh.close(); raw_fh.close()
                print(f"\nForbindelse tabt ({e}) ved {image_id} / {model}.")
                print("Stoppede uden at skrive en ufuldstaendig raekke.")
                print("Genopret DTU-netvaerket (VPN eller campus) og koer igen "
                      "-- scriptet fortsaetter, hvor det slap.")
                sys.exit(1)

            for rr in raw_rows:
                raw_w.writerow(rr)
            pred_w.writerow(pred_row)
            pred_fh.flush(); raw_fh.flush()
            n_rows += 1
            if n_rows % 10 == 0:
                print(f"  +{n_rows*len(TASKS)} kald denne koersel  ({time.time()-t0:.0f}s)")

    pred_fh.close(); raw_fh.close()
    print(f"\nFaerdig. Forudsigelser i {PRED_CSV}")
    quick_accuracy(df)


def quick_accuracy(manifest):
    if not PRED_CSV.exists():
        return
    pred = pd.read_csv(PRED_CSV)
    if pred.empty:
        return
    m = manifest.merge(pred, on="image_id")
    print("\n== Hurtigt overblik (eksakt korrekt) ==")
    for model, g in m.groupby("model"):
        blue_acc = (g["pred_blue"] == g[COL_BLUE]).mean()
        red_acc  = (g["pred_red"]  == g[COL_RED]).mean()
        # Konsistens: stemmer blaa+roed overens med modellens implicitte total?
        sum_match = ((g["pred_blue"] + g["pred_red"]) == g[COL_NODES]).mean()
        print(f"  {model:30s} | blaa={blue_acc:.0%} | roed={red_acc:.0%} | "
              f"blaa+roed=total={sum_match:.0%}")


if __name__ == "__main__":
    main()