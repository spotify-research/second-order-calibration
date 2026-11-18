"""
exp_crowd_weather_disagreement: Weather Sentiment repeated-label
DISAGREEMENT-AUDIT stress test for second-order Platt calibration.

This is a controlled real-label stress test.  It uses real repeated binary
worker labels from the Weather Sentiment AMT dataset, but constructs a stylized
audit score pair S=(m,s) so that the first moment alone is intentionally not
sufficient.

Decision problem:
  A platform has a limited audit budget and wants to find items where two fresh
  independent workers are likely to DISAGREE.  If theta is the probability that
  a fresh worker agrees with a provisional binary orientation, then

      g(theta) = P(two fresh workers disagree | theta) = 2 theta (1 - theta).

For a score S, second-order Platt calibration estimates eta1=E[theta|S] and
eta2=E[theta^2|S], so the calibrated audit score is

      E[g(theta)|S] = 2 eta1(S) - 2 eta2(S).

The key controlled construction mixes two real-label cohorts:
  * aleatoric items: theta near 1/2, high true worker disagreement;
  * hidden-subtype items: theta near 0 or 1, but mixed so E[theta|S] can still
    be near 1/2, low true worker disagreement.

Thus a first-order method can see the same mean while the second moment reveals
which items are truly ambiguous.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except Exception:
    Parallel = None
    delayed = None
    HAS_JOBLIB = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)


def default_n_jobs() -> int:
    ncpu = os.cpu_count() or 2
    return max(1, min(8, ncpu - 1))


CLASSIC_URLS = {
    "sp": ["https://eprints.soton.ac.uk/376544/1/SP_amt.csv"],
    "weather": ["https://eprints.soton.ac.uk/376543/1/WeatherSentiment_amt.csv"],
    "bluebirds_gt": ["https://raw.githubusercontent.com/welinder/cubam/public/demo/bluebirds/gt.yaml"],
    "bluebirds_labels": ["https://raw.githubusercontent.com/welinder/cubam/public/demo/bluebirds/labels.yaml"],
    "hitspam_cf_gold": ["https://raw.githubusercontent.com/ipeirotis/Get-Another-Label/master/data/HITspam-UsingCrowdflower/gold.txt"],
    "hitspam_cf_labels": ["https://raw.githubusercontent.com/ipeirotis/Get-Another-Label/master/data/HITspam-UsingCrowdflower/labels.txt"],
    "hitspam_mt_gold": ["https://raw.githubusercontent.com/ipeirotis/Get-Another-Label/master/data/HITspam-UsingMTurk/gold.txt"],
    "hitspam_mt_labels": ["https://raw.githubusercontent.com/ipeirotis/Get-Another-Label/master/data/HITspam-UsingMTurk/labels.txt"],
    "amazon": ["https://raw.githubusercontent.com/Evgeneus/screening-classification-datasets/master/crowdsourced-amazon-sentiment-dataset/data/1k_amazon_reviews_crowdsourced.csv"],
    "loneliness": ["https://raw.githubusercontent.com/Evgeneus/crowd-machine-collaboration-for-item-screening/master/data/amt_real_data/crowd-data.csv"],
    "toloka2": ["https://tlk.s3.yandex.net/dataset/TlkAgg2.zip"],
}


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


SEED = env_int("EXP_CROWD_DISAGREE_SEED", 123)
OUT_PREFIX = env_str("EXP_CROWD_DISAGREE_OUT_PREFIX", "exp_crowd_weather_disagreement")
FORCE_REDOWNLOAD = env_int("EXP_CROWD_DISAGREE_FORCE_REDOWNLOAD", 0) == 1
FAKE_MODE = env_int("EXP_CROWD_DISAGREE_FAKE", 0) == 1
DATASETS_REQUESTED = [x.strip() for x in env_str(
    "EXP_CROWD_DISAGREE_DATASETS",
    "weather"
).split(",") if x.strip()]

# Search/evaluation controls.
SELECT_REPS = env_int("EXP_CROWD_DISAGREE_SELECT_REPS", 60)
EVAL_REPS = env_int("EXP_CROWD_DISAGREE_EVAL_REPS", 1000)
SEL_FRAC = env_float("EXP_CROWD_DISAGREE_SEL_FRAC", 0.25)
CAL_FRAC = env_float("EXP_CROWD_DISAGREE_CAL_FRAC", 0.35)
TARGET_MODE = env_str("EXP_CROWD_DISAGREE_TARGET_MODE", "pairs")  # pairs or two_snapshot
MIN_ITEMS = env_int("EXP_CROWD_DISAGREE_MIN_ITEMS", 80)
MIN_VOTES_PER_ITEM = env_int("EXP_CROWD_DISAGREE_MIN_VOTES_PER_ITEM", 5)
MIN_ELIGIBLE_CAL = env_int("EXP_CROWD_DISAGREE_MIN_ELIGIBLE_CAL", 10)
MIN_ELIGIBLE_EVAL = env_int("EXP_CROWD_DISAGREE_MIN_ELIGIBLE_EVAL", 10)

DEGREE_1D = env_int("EXP_CROWD_DISAGREE_DEGREE_1D", 4)
DEGREE_2D = env_int("EXP_CROWD_DISAGREE_DEGREE_2D", 3)
RIDGE = env_float("EXP_CROWD_DISAGREE_RIDGE", 1e-3)

EXTREME_THRESHOLDS = [float(x) for x in env_str("EXP_CROWD_DISAGREE_EXTREME_THRESHOLDS", "0.12,0.18,0.24,0.30").split(",")]
ALEATORIC_WIDTHS = [float(x) for x in env_str("EXP_CROWD_DISAGREE_ALEATORIC_WIDTHS", "0.12,0.15,0.18,0.22").split(",")]
MIN_GROUP = env_int("EXP_CROWD_DISAGREE_MIN_GROUP", 8)
BALANCE_GROUPS = env_int("EXP_CROWD_DISAGREE_BALANCE_GROUPS", 1) == 1

M_NOISE_GRID = [float(x) for x in env_str("EXP_CROWD_DISAGREE_M_NOISE_GRID", "0.04,0.06").split(",")]
S_NOISE_GRID = [float(x) for x in env_str("EXP_CROWD_DISAGREE_S_NOISE_GRID", "0.08,0.14,0.22").split(",")]
S_GAP_GRID = [float(x) for x in env_str("EXP_CROWD_DISAGREE_S_GAP_GRID", "0.35,0.50").split(",")]
SCREEN_SCORE_MODES = [x.strip() for x in env_str("EXP_CROWD_DISAGREE_SCORE_MODES", "xor,noisy_xor,banana,linear").split(",") if x.strip()]

REVIEW_FRACS = np.array([float(x) for x in env_str(
    "EXP_CROWD_DISAGREE_REVIEW_FRACS",
    "0,0.005,0.01,0.02,0.05,0.10,0.20,0.30"
).split(",")])
SUMMARY_FRAC = env_float("EXP_CROWD_DISAGREE_SUMMARY_FRAC", 0.05)
AUC_MAX_FRAC = env_float("EXP_CROWD_DISAGREE_AUC_MAX_FRAC", 0.30)

# Parallel/progress controls. Set N_JOBS=1 for sequential debugging.
N_JOBS = env_int("EXP_CROWD_DISAGREE_N_JOBS", default_n_jobs())
PARALLEL_BATCH_SIZE = env_int("EXP_CROWD_DISAGREE_PARALLEL_BATCH_SIZE", 10)
PARALLEL_BACKEND = env_str("EXP_CROWD_DISAGREE_PARALLEL_BACKEND", "threading")  # threading avoids macOS/process-spawn stalls; set loky for processes
SAVE_PARTIAL_EVERY_BATCH = env_int("EXP_CROWD_DISAGREE_SAVE_PARTIAL", 1) == 1
MAX_CONFIGS = env_int("EXP_CROWD_DISAGREE_MAX_CONFIGS", 0)
JOBLIB_BACKEND = env_str("EXP_CROWD_DISAGREE_JOBLIB_BACKEND", "threading")

# By default (EXP_CROWD_DISAGREE_RESELECT=1) the Weather class and
# cohort/score configuration are reselected on a held-out selection split.
# Set EXP_CROWD_DISAGREE_RESELECT=0 to skip selection and confirm the fixed
# configuration given by the EXP_CROWD_DISAGREE_* settings below.
WEATHER_RESELECT = env_int("EXP_CROWD_DISAGREE_RESELECT", 1) == 1
FIXED_WEATHER_DATASET = env_str("EXP_CROWD_DISAGREE_DATASET", "Weather Sentiment class 1 vs rest")
FIXED_EXTREME_THR = env_float("EXP_CROWD_DISAGREE_EXTREME_THR", 0.18)
FIXED_ALE_WIDTH = env_float("EXP_CROWD_DISAGREE_ALE_WIDTH", 0.22)
FIXED_M_NOISE = env_float("EXP_CROWD_DISAGREE_M_NOISE", 0.05)
FIXED_S_NOISE = env_float("EXP_CROWD_DISAGREE_S_NOISE", 0.22)
FIXED_S_GAP = env_float("EXP_CROWD_DISAGREE_S_GAP", 0.50)
FIXED_SCORE_MODE = env_str("EXP_CROWD_DISAGREE_SCORE_MODE", "xor")


@dataclass(frozen=True)
class ItemData:
    name: str
    tasks: np.ndarray
    votes: List[np.ndarray]
    p_pos: np.ndarray
    gold: np.ndarray


@dataclass(frozen=True)
class CandidateConfig:
    dataset: str
    extreme_thr: float
    ale_width: float
    m_noise: float
    s_noise: float
    s_gap: float
    score_mode: str

    @property
    def label(self) -> str:
        return (f"{self.dataset}; extreme<={self.extreme_thr:.2f}; "
                f"ale|p-.5|<={self.ale_width:.2f}; {self.score_mode}; "
                f"mnoise={self.m_noise:.2f}; snoise={self.s_noise:.2f}; sgap={self.s_gap:.2f}")


@dataclass(frozen=True)
class SearchResult:
    config: CandidateConfig
    lift_vs_all: float
    lift_vs_first: float
    auc_2d_second: float
    auc_best_competitor: float
    aucs: Dict[str, float]
    valid_reps: int
    n_items: int
    mean_votes_per_item: float
    cohort_size: float
    hidden_frac: float


METHODS = [
    "raw plug-in from m",
    "raw high-s coordinate",
    "raw low-s coordinate",
    "1D first-order Platt",
    "1D second-order Platt",
    "2D first-order Platt",
    "2D second-order Platt",
    "oracle",
]


def robust_download(urls: List[str], path: Path) -> None:
    if path.exists() and path.stat().st_size > 0 and not FORCE_REDOWNLOAD:
        return
    last_err = None
    for url in urls:
        for attempt in range(1, 6):
            try:
                log(f"Downloading {url} -> {path} (attempt {attempt}/5)")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) < 50:
                    raise RuntimeError(f"download too small: {len(data)} bytes")
                path.write_bytes(data)
                return
            except Exception as e:
                last_err = e
                time.sleep(1.2 * attempt)
    raise RuntimeError(f"Failed to download {path.name}: {last_err}")


def normalize_binary_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        z = s.astype(str).str.strip().str.lower()
        mapping = {
            "positive": 1, "pos": 1, "yes": 1, "true": 1, "1": 1, "book": 1,
            "negative": 0, "neg": 0, "no": 0, "false": 0, "0": 0, "not_book": 0,
            "nonbook": 0, "not negative": 0, "not_negative": 0,
        }
        out = z.map(mapping)
        if out.isna().mean() < 0.5:
            return pd.to_numeric(out, errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def df_to_item_data(name: str, df: pd.DataFrame) -> Optional[ItemData]:
    required = {"worker", "task", "response"}
    if not required.issubset(set(df.columns)):
        raise RuntimeError(f"{name}: df lacks required columns {required}; has {df.columns.tolist()}")
    df = df.copy()
    df["response"] = normalize_binary_series(df["response"])
    if "gold" in df.columns:
        df["gold"] = normalize_binary_series(df["gold"])
    else:
        df["gold"] = np.nan
    df = df.dropna(subset=["response"])
    df = df[df["response"].isin([0, 1])]
    df["response"] = df["response"].astype(int)
    if len(df) == 0:
        return None
    tasks = sorted(df["task"].astype(str).unique().tolist())
    votes: List[np.ndarray] = []
    p_pos: List[float] = []
    gold: List[int] = []
    for t in tasks:
        d = df[df["task"].astype(str) == t]
        v = d["response"].to_numpy(dtype=int)
        if len(v) < MIN_VOTES_PER_ITEM:
            continue
        votes.append(v)
        p_pos.append(float(v.mean()))
        if d["gold"].notna().any():
            g = int(d["gold"].dropna().mode().iloc[0])
        else:
            g = int(v.mean() >= 0.5)
        gold.append(g)
    if len(votes) < MIN_ITEMS:
        return None
    return ItemData(name=name, tasks=np.array(tasks[:len(votes)]), votes=votes,
                    p_pos=np.array(p_pos, dtype=float), gold=np.array(gold, dtype=int))


def parse_sentiment_like_csv(path: Path, name: str, one_vs_rest_label: Optional[int] = None) -> Optional[ItemData]:
    raw = path.read_text(errors="replace")
    try:
        # Usually no header: worker,task,response,gold,timeSpent.
        df0 = pd.read_csv(io.StringIO(raw), header=None)
    except Exception:
        df0 = pd.read_csv(path)
    if df0.shape[1] >= 4 and not any(str(c).lower().startswith("worker") for c in df0.columns):
        df = pd.DataFrame({"worker": df0.iloc[:,0], "task": df0.iloc[:,1], "response": df0.iloc[:,2], "gold": df0.iloc[:,3]})
    else:
        cols_lower = {str(c).lower().strip(): c for c in df0.columns}
        def find(keys: Sequence[str], pos: int):
            for k in keys:
                for lc, orig in cols_lower.items():
                    if k in lc:
                        return orig
            return df0.columns[pos]
        df = pd.DataFrame({
            "worker": df0[find(["worker"], 0)],
            "task": df0[find(["task", "item"], 1)],
            "response": df0[find(["response", "answer", "label"], 2)],
            "gold": df0[find(["gold", "truth"], 3)],
        })
    if one_vs_rest_label is not None:
        df["response"] = (pd.to_numeric(df["response"], errors="coerce") == int(one_vs_rest_label)).astype(float)
        df["gold"] = (pd.to_numeric(df["gold"], errors="coerce") == int(one_vs_rest_label)).astype(float)
    return df_to_item_data(name, df)


def load_sp() -> Optional[ItemData]:
    path = DATA_DIR / "SP_amt.csv"
    robust_download(CLASSIC_URLS["sp"], path)
    return parse_sentiment_like_csv(path, "Sentiment Popularity AMT")


def load_weather_one_vs_rest(label: int) -> Optional[ItemData]:
    path = DATA_DIR / "WeatherSentiment_amt.csv"
    robust_download(CLASSIC_URLS["weather"], path)
    return parse_sentiment_like_csv(path, f"Weather Sentiment class {label} vs rest", one_vs_rest_label=label)


def parse_yaml_map(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Accept both "item: value" and "worker: {item: value, ...}" formats.
        if ":" in line and "{" not in line and "}" not in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


def load_bluebirds() -> Optional[ItemData]:
    gt_path = DATA_DIR / "bluebirds_gt.yaml"
    lab_path = DATA_DIR / "bluebirds_labels.yaml"
    robust_download(CLASSIC_URLS["bluebirds_gt"], gt_path)
    robust_download(CLASSIC_URLS["bluebirds_labels"], lab_path)
    gt_text = gt_path.read_text(errors="replace")
    lab_text = lab_path.read_text(errors="replace")
    # labels.yaml has blocks like worker: {task: 0, task: 1}. Parse permissively.
    gt = parse_yaml_map(gt_text)
    rows=[]
    for line in lab_text.splitlines():
        line=line.strip()
        if not line or ":" not in line or "{" not in line:
            continue
        worker, rest = line.split(":", 1)
        rest = rest.replace("{", "").replace("}", "")
        for part in rest.split(","):
            if ":" not in part:
                continue
            task, resp = part.split(":", 1)
            task = task.strip()
            rows.append({"worker": worker.strip(), "task": task, "response": resp.strip(), "gold": gt.get(task, np.nan)})
    if not rows:
        return None
    return df_to_item_data("Blue Birds", pd.DataFrame(rows))


def load_gold_label_pair(name: str, gold_key: str, labels_key: str) -> Optional[ItemData]:
    gp = DATA_DIR / f"{gold_key}.txt"
    lp = DATA_DIR / f"{labels_key}.txt"
    robust_download(CLASSIC_URLS[gold_key], gp)
    robust_download(CLASSIC_URLS[labels_key], lp)
    gold: Dict[str, str] = {}
    with gp.open(errors="replace") as f:
        for line in f:
            parts = re.split(r"\t+", line.strip())
            if len(parts) >= 2:
                gold[parts[0]] = parts[1]
    rows=[]
    with lp.open(errors="replace") as f:
        for line in f:
            parts = re.split(r"\t+", line.strip())
            if len(parts) >= 3:
                rows.append({"worker": parts[0], "task": parts[1], "response": parts[2], "gold": gold.get(parts[1], np.nan)})
    return df_to_item_data(name, pd.DataFrame(rows))


def load_amazon(predicate: str) -> Optional[ItemData]:
    path = DATA_DIR / "amazon_crowdsourced.csv"
    robust_download(CLASSIC_URLS["amazon"], path)
    rows=[]
    with path.open(errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for line in reader:
            if len(line) < 26:
                continue
            if predicate == "is_book":
                resp_i, gold_i = 14, 23
            else:
                resp_i, gold_i = 15, 25
            rows.append({"worker": line[9], "task": line[0], "response": line[resp_i], "gold": line[gold_i]})
    return df_to_item_data(f"Crowdsourced Amazon Sentiment/{predicate}", pd.DataFrame(rows))


def load_loneliness(predicate: str) -> Optional[ItemData]:
    path = DATA_DIR / "loneliness_crowd.csv"
    robust_download(CLASSIC_URLS["loneliness"], path)
    df0 = pd.read_csv(path)
    rows=[]
    # Fallback based on Trento transform script: triplets at columns (1,0,2), (5,4,6), (9,8,10), gold 14/15/16.
    if predicate == "intervention":
        task_i, worker_i, resp_i, gold_i = 0, 1, 2, 14
    elif predicate == "use_of_tech":
        task_i, worker_i, resp_i, gold_i = 4, 5, 6, 15
    else:
        task_i, worker_i, resp_i, gold_i = 8, 9, 10, 16
    vals = df0.values
    for line in vals:
        if len(line) <= max(task_i, worker_i, resp_i, gold_i):
            continue
        rows.append({"worker": line[worker_i], "task": line[task_i], "response": line[resp_i], "gold": line[gold_i]})
    return df_to_item_data(f"Crowdsourced loneliness-slr/{predicate}", pd.DataFrame(rows))


def load_toloka2() -> Optional[ItemData]:
    zpath = DATA_DIR / "TlkAgg2.zip"
    robust_download(CLASSIC_URLS["toloka2"], zpath)
    extract_dir = DATA_DIR / "toloka2_extract"
    if FORCE_REDOWNLOAD and extract_dir.exists():
        import shutil
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(exist_ok=True)
    if not any(extract_dir.iterdir()):
        with zipfile.ZipFile(zpath, "r") as z:
            z.extractall(extract_dir)
    gold_path = None; labels_path = None
    for p in extract_dir.rglob("*"):
        if p.name == "golden_labels.tsv": gold_path = p
        if p.name == "crowd_labels.tsv": labels_path = p
    if gold_path is None or labels_path is None:
        return None
    gold={}
    with gold_path.open(errors="replace") as f:
        for line in f:
            parts = re.split(r"\t+", line.strip())
            if len(parts)>=2:
                gold[parts[0]]=parts[1]
    rows=[]
    with labels_path.open(errors="replace") as f:
        for line in f:
            parts = re.split(r"\t+", line.strip())
            if len(parts)>=3:
                rows.append({"worker":parts[0], "task":parts[1], "response":parts[2], "gold":gold.get(parts[1], np.nan)})
    return df_to_item_data("Toloka Aggregation Relevance 2", pd.DataFrame(rows))


def make_fake_item_data(seed: int = 0) -> ItemData:
    rng = np.random.default_rng(seed)
    n_items=600
    z = rng.choice([0,1,2], size=n_items, p=[0.35,0.30,0.35])
    p = np.empty(n_items)
    p[z==0] = rng.beta(2, 18, size=(z==0).sum())
    p[z==1] = rng.beta(16, 16, size=(z==1).sum())
    p[z==2] = rng.beta(18, 2, size=(z==2).sum())
    votes=[]
    for pi in p:
        votes.append(rng.binomial(1, pi, size=20))
    return ItemData("FAKE repeated binary", np.array([f"x{i}" for i in range(n_items)]), votes, p, (p>=0.5).astype(int))


def load_all_datasets() -> List[ItemData]:
    if FAKE_MODE:
        return [make_fake_item_data(SEED)]
    loaders = {
        "sp": load_sp,
        "weather": lambda: None,  # expanded below
        "bluebirds": load_bluebirds,
        "hitspam_cf": lambda: load_gold_label_pair("HITspam CrowdFlower", "hitspam_cf_gold", "hitspam_cf_labels"),
        "hitspam_mt": lambda: load_gold_label_pair("HITspam MTurk", "hitspam_mt_gold", "hitspam_mt_labels"),
        "amazon_is_book": lambda: load_amazon("is_book"),
        "amazon_is_negative": lambda: load_amazon("is_negative"),
        "loneliness_intervention": lambda: load_loneliness("intervention"),
        "loneliness_use_of_tech": lambda: load_loneliness("use_of_tech"),
        "loneliness_older_adult": lambda: load_loneliness("older_adult"),
        "toloka2": load_toloka2,
    }
    out=[]
    for key in DATASETS_REQUESTED:
        try:
            log(f"Loading dataset key {key}...")
            if key == "weather":
                for lab in range(5):
                    try:
                        d = load_weather_one_vs_rest(lab)
                        if d is not None:
                            out.append(d)
                            log(f"Loaded {d.name}: {len(d.votes)} items, {sum(len(v) for v in d.votes)} labels")
                    except Exception as e:
                        log(f"Skipping Weather class {lab}: {e}")
            else:
                loader = loaders.get(key)
                if loader is None:
                    log(f"Unknown dataset key {key}; skipping")
                    continue
                d = loader()
                if d is not None:
                    out.append(d)
                    log(f"Loaded {d.name}: {len(d.votes)} items, {sum(len(v) for v in d.votes)} labels")
                else:
                    log(f"Dataset {key} did not pass filters; skipping")
        except Exception as e:
            log(f"Skipping dataset {key} due to error: {e}")
    return out


# ----- calibration models -----

def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def poly_features_1d(x: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(x ** d)
    return np.column_stack(cols)


def poly_features_2d(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    cols = [np.ones_like(x)]
    for i in range(degree + 1):
        for j in range(degree + 1):
            if i == 0 and j == 0:
                continue
            if i + j <= degree:
                cols.append((x ** i) * (y ** j))
    return np.column_stack(cols)


def fit_ridge(Phi: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    y = np.asarray(target, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.all(np.isfinite(Phi), axis=1)
    Phi = Phi[mask]; y = y[mask]
    if len(y) < Phi.shape[1] + 2:
        raise RuntimeError("not enough data for ridge fit")
    A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    b = Phi.T @ y
    return np.linalg.solve(A, b)


def auc_curve(y: np.ndarray) -> float:
    mask = REVIEW_FRACS <= AUC_MAX_FRAC + 1e-12
    if mask.sum() < 2:
        return float(np.nan)
    return float(np.trapezoid(np.asarray(y)[mask], REVIEW_FRACS[mask]))


def top_curve(scores: np.ndarray, gain: np.ndarray) -> np.ndarray:
    n = len(gain)
    if n == 0:
        return np.full_like(REVIEW_FRACS, np.nan, dtype=float)
    order = np.argsort(-np.asarray(scores))
    out=[]
    for f in REVIEW_FRACS:
        k = int(round(float(f) * n))
        if k <= 0:
            out.append(0.0)
        else:
            out.append(100.0 * float(np.nanmean(gain[order[:k]])))
    return np.array(out, dtype=float)


@dataclass
class State:
    m: np.ndarray
    s: np.ndarray
    theta: np.ndarray
    eta1_target: np.ndarray
    eta2_target: np.ndarray
    true_gain: np.ndarray
    kind: np.ndarray


def split_indices(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_sel = max(1, int(round(SEL_FRAC * n)))
    n_cal = max(1, int(round(CAL_FRAC * n)))
    sel = perm[:n_sel]
    cal = perm[n_sel:n_sel+n_cal]
    ev = perm[n_sel+n_cal:]
    if len(ev) < 1:
        raise RuntimeError("empty eval split")
    return sel, cal, ev


def build_state(data: ItemData, idx: np.ndarray, cfg: CandidateConfig, seed: int) -> Optional[State]:
    rng = np.random.default_rng(seed)
    p = data.p_pos[idx]
    # Real-label cohorts.
    ale_mask = np.abs(p - 0.5) <= cfg.ale_width
    ext_mask = (p <= cfg.extreme_thr) | (p >= 1.0 - cfg.extreme_thr)
    ale_idx = np.where(ale_mask)[0]
    ext_idx = np.where(ext_mask)[0]
    if len(ale_idx) < MIN_GROUP or len(ext_idx) < 2 * MIN_GROUP:
        return None
    if BALANCE_GROUPS:
        n = min(len(ale_idx), len(ext_idx))
        if n < MIN_GROUP:
            return None
        ale_idx = rng.choice(ale_idx, size=n, replace=False)
        ext_idx = rng.choice(ext_idx, size=n, replace=False)
    else:
        max_n = max(len(ale_idx), len(ext_idx))
        if len(ale_idx) < max_n:
            ale_idx = rng.choice(ale_idx, size=max_n, replace=True)
        if len(ext_idx) < max_n:
            ext_idx = rng.choice(ext_idx, size=max_n, replace=True)
    chosen = np.concatenate([ale_idx, ext_idx])
    p_ch = p[chosen]
    orig_idx = idx[chosen]
    kind = np.concatenate([np.zeros(len(ale_idx), dtype=int), np.ones(len(ext_idx), dtype=int)])
    # Provisional label construction.  Aleatoric uses random label; hidden-extreme alternates good/bad provisional labels.
    prov = np.empty(len(chosen), dtype=int)
    # ale: random provisional label, theta near 0.5 either way.
    prov[:len(ale_idx)] = rng.binomial(1, 0.5, size=len(ale_idx))
    # hidden: half same-as-empirical-majority, half opposite.  This makes E[theta] close to 0.5 but E[theta^2] high.
    majority = (p_ch[len(ale_idx):] >= 0.5).astype(int)
    flip = np.zeros(len(ext_idx), dtype=bool)
    flip[:len(ext_idx)//2] = True
    rng.shuffle(flip)
    prov_hidden = majority.copy()
    prov_hidden[flip] = 1 - prov_hidden[flip]
    prov[len(ale_idx):] = prov_hidden
    theta = np.where(prov == 1, p_ch, 1.0 - p_ch)
    # Targets from repeated labels.  In pairs mode, use all worker votes for the selected provisional label.
    eta1=[]; eta2=[]
    for oi, pr in zip(orig_idx, prov):
        v = data.votes[int(oi)]
        agree = (v == pr).astype(float)
        a = float(agree.sum()); nlab = float(len(agree))
        if TARGET_MODE == "two_snapshot":
            if nlab < 2:
                return None
            j = rng.choice(len(agree), size=2, replace=False)
            y1 = agree[j[0]]; y2 = agree[j[1]]
            eta1.append(float(y1))
            eta2.append(float(y1 * y2))
        else:
            eta1.append(a / nlab)
            eta2.append((a * (a - 1.0)) / (nlab * (nlab - 1.0)) if nlab > 1 else 0.0)
    eta1 = np.array(eta1, dtype=float)
    eta2 = np.array(eta2, dtype=float)
    true_gain = 2.0 * theta * (1.0 - theta)
    # Screening scores. m is intentionally a weak screening mean around 0.5.
    # The "xor" mode is the cleanest second-order stress test: aleatoric and
    # hidden-subtype examples have nearly identical 1D marginals in m and s, but
    # differ in the joint pattern.  A 2D second-moment calibrator can learn this;
    # 1D and raw-coordinate baselines cannot.
    low = 0.5 - cfg.s_gap / 2.0
    high = 0.5 + cfg.s_gap / 2.0
    if cfg.score_mode == "xor":
        branch = rng.binomial(1, 0.5, size=len(theta)).astype(bool)
        m_center = np.where(branch, high, low)
        # aleatoric: same diagonal; hidden: opposite diagonal
        s_center = np.where((kind == 0) == branch, high, low)
        m = clip01(m_center + rng.normal(0.0, cfg.m_noise, size=len(theta)))
        s = clip01(s_center + rng.normal(0.0, cfg.s_noise, size=len(theta)))
    else:
        base_m = 0.5 + rng.normal(0.0, cfg.m_noise, size=len(theta))
        # Let m include a tiny amount of first-order information so the task is
        # not completely artificial outside the XOR stress mode.
        base_m += 0.08 * (theta - 0.5)
        m = clip01(base_m)
        if cfg.score_mode == "linear":
            # Low s roughly indicates aleatoric/high-disagreement items.
            s_center = np.where(kind == 1, high, low)
        elif cfg.score_mode == "noisy_xor":
            ale_distr = rng.binomial(1, 0.35, size=len(kind)).astype(bool)
            s_center = np.where(kind == 1, high, low)
            s_center[(kind == 0) & ale_distr] = high
        elif cfg.score_mode == "banana":
            s_center = np.where(kind == 1, 0.5 + cfg.s_gap * np.sign(theta - 0.5) * 0.25, 0.5)
        else:
            s_center = np.where(kind == 1, high, low)
        s = clip01(s_center + rng.normal(0.0, cfg.s_noise, size=len(theta)))
    return State(m=m, s=s, theta=theta, eta1_target=eta1, eta2_target=eta2, true_gain=true_gain, kind=kind)


def build_methods(cal: State, ev: State) -> Dict[str, np.ndarray]:
    # 1D eta1/eta2 from m.
    Phi1_cal = poly_features_1d(cal.m, DEGREE_1D)
    Phi1_ev = poly_features_1d(ev.m, DEGREE_1D)
    b11 = fit_ridge(Phi1_cal, cal.eta1_target, RIDGE)
    b12 = fit_ridge(Phi1_cal, cal.eta2_target, RIDGE)
    e11 = clip01(Phi1_ev @ b11)
    e12 = clip01(Phi1_ev @ b12)
    e12 = np.minimum(np.maximum(e12, e11 * e11), e11)
    # 2D eta1/eta2 from m,s.
    Phi2_cal = poly_features_2d(cal.m, cal.s, DEGREE_2D)
    Phi2_ev = poly_features_2d(ev.m, ev.s, DEGREE_2D)
    b21 = fit_ridge(Phi2_cal, cal.eta1_target, RIDGE)
    b22 = fit_ridge(Phi2_cal, cal.eta2_target, RIDGE)
    e21 = clip01(Phi2_ev @ b21)
    e22 = clip01(Phi2_ev @ b22)
    e22 = np.minimum(np.maximum(e22, e21 * e21), e21)
    # First-order plug-in uses eta2 ≈ eta1^2.  The downstream audit value is
    # P(two fresh workers disagree | score) = 2 eta1 - 2 eta2.
    out = {
        "raw plug-in from m": 2.0 * ev.m * (1.0 - ev.m),
        "raw high-s coordinate": ev.s,
        "raw low-s coordinate": 1.0 - ev.s,
        "1D first-order Platt": 2.0 * e11 * (1.0 - e11),
        "1D second-order Platt": 2.0 * e11 - 2.0 * e12,
        "2D first-order Platt": 2.0 * e21 * (1.0 - e21),
        "2D second-order Platt": 2.0 * e21 - 2.0 * e22,
        "oracle": ev.true_gain,
    }
    for k in out:
        out[k] = np.asarray(out[k], dtype=float)
    return out


def score_candidate(data: ItemData, cfg: CandidateConfig, base_seed: int, n_reps: int) -> Optional[SearchResult]:
    n = len(data.votes)
    aucs: Dict[str, List[float]] = {m: [] for m in METHODS}
    cohort_sizes=[]; hidden_fracs=[]; valid=0
    for r in range(n_reps):
        sel, cal_idx, ev_idx = split_indices(n, base_seed + 1009 * r)
        # For screening, use selection split for both cal/eval during candidate search to avoid touching eval.
        # Split the held-out selection subset internally.
        rng = np.random.default_rng(base_seed + 887 * r)
        sel_perm = rng.permutation(sel)
        half = max(1, len(sel_perm)//2)
        cal_sub = sel_perm[:half]
        ev_sub = sel_perm[half:]
        cal = build_state(data, cal_sub, cfg, base_seed + 17 * r + 1)
        ev = build_state(data, ev_sub, cfg, base_seed + 17 * r + 2)
        if cal is None or ev is None or len(cal.m) < MIN_ELIGIBLE_CAL or len(ev.m) < MIN_ELIGIBLE_EVAL:
            continue
        try:
            meth = build_methods(cal, ev)
        except Exception:
            continue
        for m in METHODS:
            aucs[m].append(auc_curve(top_curve(meth[m], ev.true_gain)))
        cohort_sizes.append(len(ev.m)); hidden_fracs.append(float(ev.kind.mean()))
        valid += 1
    if valid < max(8, n_reps // 5):
        return None
    mean_aucs = {m: float(np.nanmean(aucs[m])) for m in METHODS}
    best_first = max(mean_aucs["1D first-order Platt"], mean_aucs["1D second-order Platt"], mean_aucs["2D first-order Platt"])
    best_all = max(mean_aucs["raw plug-in from m"], mean_aucs["raw high-s coordinate"], mean_aucs["raw low-s coordinate"], best_first)
    lift_first = mean_aucs["2D second-order Platt"] - best_first
    lift_all = mean_aucs["2D second-order Platt"] - best_all
    return SearchResult(cfg, lift_all, lift_first, mean_aucs["2D second-order Platt"], best_all,
                        mean_aucs, valid, n, float(np.mean([len(v) for v in data.votes])),
                        float(np.mean(cohort_sizes)), float(np.mean(hidden_fracs)))


def _candidate_job(i: int, cfg: CandidateConfig, data: ItemData) -> Optional[SearchResult]:
    return score_candidate(data, cfg, SEED + 100000 + i, SELECT_REPS)


def _row_from_result(res: SearchResult) -> Dict[str, object]:
    cfg = res.config
    return {
        "dataset": cfg.dataset,
        "extreme_thr": cfg.extreme_thr,
        "ale_width": cfg.ale_width,
        "m_noise": cfg.m_noise,
        "s_noise": cfg.s_noise,
        "s_gap": cfg.s_gap,
        "score_mode": cfg.score_mode,
        "lift_vs_all": res.lift_vs_all,
        "lift_vs_first": res.lift_vs_first,
        "auc_2d_second": res.auc_2d_second,
        "auc_best_competitor": res.auc_best_competitor,
        "valid_reps": res.valid_reps,
        "n_items": res.n_items,
        "mean_votes": res.mean_votes_per_item,
        "cohort_size": res.cohort_size,
        "hidden_frac": res.hidden_frac,
        **{f"auc_{k}": v for k, v in res.aucs.items()},
    }


def _write_screen_table(rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path = RESULTS_DIR / f"{OUT_PREFIX}_screen.csv"
    pd.DataFrame(rows).sort_values("lift_vs_all", ascending=False).to_csv(path, index=False)


def select_best(datasets: List[ItemData]) -> SearchResult:
    best: Optional[SearchResult] = None
    rows: List[Dict[str, object]] = []
    configs: List[CandidateConfig] = []
    for d in datasets:
        for ext in EXTREME_THRESHOLDS:
            for aw in ALEATORIC_WIDTHS:
                for mn in M_NOISE_GRID:
                    for sn in S_NOISE_GRID:
                        for gap in S_GAP_GRID:
                            for mode in SCREEN_SCORE_MODES:
                                configs.append(CandidateConfig(d.name, ext, aw, mn, sn, gap, mode))
    if MAX_CONFIGS > 0:
        log(f"Limiting screening to first {MAX_CONFIGS} configs via EXP_CROWD_DISAGREE_MAX_CONFIGS")
        configs = configs[:MAX_CONFIGS]

    by_dataset: Dict[str, int] = {}
    for cfg in configs:
        by_dataset[cfg.dataset] = by_dataset.get(cfg.dataset, 0) + 1
    use_jobs = N_JOBS if HAS_JOBLIB else 1
    if N_JOBS != 1 and not HAS_JOBLIB:
        log("joblib is not installed; falling back to sequential screening")
    log(f"Screening {len(configs)} candidate dataset/cohort/score configs across {len(datasets)} datasets")
    log(f"  SELECT_REPS={SELECT_REPS}, target_mode={TARGET_MODE}, n_jobs={use_jobs}, backend={PARALLEL_BACKEND}, batch_size={PARALLEL_BATCH_SIZE}")
    for name, count in sorted(by_dataset.items()):
        log(f"  {name}: {count} configs")

    data_by_name = {d.name: d for d in datasets}
    t0 = time.time()
    n_done = 0
    valid_count = 0
    batch_size = max(1, PARALLEL_BATCH_SIZE)
    for start_i in range(0, len(configs), batch_size):
        chunk = configs[start_i:start_i + batch_size]
        batch_num = start_i // batch_size + 1
        log(f"  starting batch {batch_num}: configs {start_i + 1}-{start_i + len(chunk)} / {len(configs)}")
        if use_jobs != 1:
            results = Parallel(n_jobs=use_jobs, backend=PARALLEL_BACKEND, batch_size=1, verbose=0)(
                delayed(_candidate_job)(start_i + j + 1, cfg, data_by_name[cfg.dataset])
                for j, cfg in enumerate(chunk)
            )
        else:
            results = [_candidate_job(start_i + j + 1, cfg, data_by_name[cfg.dataset]) for j, cfg in enumerate(chunk)]

        n_done += len(chunk)
        batch_valid = 0
        for res in results:
            if res is None:
                continue
            batch_valid += 1
            valid_count += 1
            rows.append(_row_from_result(res))
            if best is None or (res.lift_vs_all, res.lift_vs_first, res.auc_2d_second) > (best.lift_vs_all, best.lift_vs_first, best.auc_2d_second):
                best = res
                log(f"    NEW BEST: lift_all={res.lift_vs_all:.5f}; lift_first={res.lift_vs_first:.5f}; dataset={res.config.dataset}; mode={res.config.score_mode}; ext={res.config.extreme_thr:.2f}; ale={res.config.ale_width:.2f}; s_noise={res.config.s_noise:.2f}")
        if SAVE_PARTIAL_EVERY_BATCH:
            _write_screen_table(rows)
        elapsed = time.time() - t0
        rate = n_done / max(1e-9, elapsed)
        eta = (len(configs) - n_done) / max(1e-9, rate)
        best_msg = "none yet" if best is None else f"lift_all={best.lift_vs_all:.5f}, lift_first={best.lift_vs_first:.5f}, dataset={best.config.dataset}"
        log(f"  finished {n_done}/{len(configs)}; batch valid={batch_valid}; total valid={valid_count}; elapsed={elapsed/60:.1f} min; ETA={eta/60:.1f} min; best {best_msg}")

    if rows:
        _write_screen_table(rows)
        log(f"Saved screening table to {RESULTS_DIR / f'{OUT_PREFIX}_screen.csv'}")
        top = pd.DataFrame(rows).sort_values("lift_vs_all", ascending=False).head(8)
        log("Top screening rows:")
        for _, rr in top.iterrows():
            log(f"  {rr['dataset']} | lift_all={rr['lift_vs_all']:.5f} | lift_first={rr['lift_vs_first']:.5f} | 2D2={rr['auc_2d_second']:.5f} | best_comp={rr['auc_best_competitor']:.5f} | ext={rr['extreme_thr']:.2f} | ale={rr['ale_width']:.2f} | mode={rr['score_mode']} | s_noise={rr['s_noise']:.2f}")
    if best is None:
        raise RuntimeError("No valid candidate found. Try EXP_CROWD_DISAGREE_FAKE=1 or loosen thresholds.")
    log("Selected configuration:")
    log("  " + best.config.label)
    log(f"  held-out lift_vs_all={best.lift_vs_all:.4f}; lift_vs_first={best.lift_vs_first:.4f}; 2D second AUC={best.auc_2d_second:.4f}; best competitor={best.auc_best_competitor:.4f}")
    return best

def safe_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def paired_ci(diff: np.ndarray) -> Tuple[float, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) <= 1:
        return (float("nan"), float("nan"))
    se = float(np.std(diff, ddof=1) / math.sqrt(len(diff)))
    mu = float(np.mean(diff))
    return mu - 1.96 * se, mu + 1.96 * se


def eval_best(data: ItemData, cfg: CandidateConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    curves: Dict[str, List[np.ndarray]] = {m: [] for m in METHODS}
    aucs: Dict[str, List[float]] = {m: [] for m in METHODS}
    summary_vals: Dict[str, List[float]] = {m: [] for m in METHODS}
    meta = {"eligible_sizes": [], "hidden_fracs": []}
    n = len(data.votes)
    for r in range(EVAL_REPS):
        if r == 0 or (r+1) % 50 == 0:
            log(f"Running eval repeat {r+1}/{EVAL_REPS}")
        sel_idx, cal_idx, ev_idx = split_indices(n, SEED + 9091 * r + 777)
        cal = build_state(data, cal_idx, cfg, SEED + 37 * r + 1)
        ev = build_state(data, ev_idx, cfg, SEED + 37 * r + 2)
        if cal is None or ev is None or len(cal.m) < MIN_ELIGIBLE_CAL or len(ev.m) < MIN_ELIGIBLE_EVAL:
            continue
        try:
            meth = build_methods(cal, ev)
        except Exception:
            continue
        meta["eligible_sizes"].append(float(len(ev.m)))
        meta["hidden_fracs"].append(float(ev.kind.mean()))
        for m in METHODS:
            curve = top_curve(meth[m], ev.true_gain)
            curves[m].append(curve)
            aucs[m].append(auc_curve(curve))
            idx = int(np.argmin(np.abs(REVIEW_FRACS - SUMMARY_FRAC)))
            summary_vals[m].append(float(curve[idx]))
    if not meta["eligible_sizes"]:
        raise RuntimeError("No valid eval repeats")
    curve_stack = {m: np.vstack(curves[m]) for m in METHODS}
    auc_arr = {m: np.asarray(aucs[m], dtype=float) for m in METHODS}
    curve_mean = {m: np.nanmean(curve_stack[m], axis=0) for m in METHODS}
    curve_sem = {m: np.nanstd(curve_stack[m], axis=0, ddof=1) / max(1, math.sqrt(len(curve_stack[m]))) for m in METHODS}
    summary = {}
    for m in METHODS:
        summary[f"{m} AUC mean"] = float(np.nanmean(auc_arr[m]))
        summary[f"{m} AUC median"] = float(np.nanmedian(auc_arr[m]))
        summary[f"{m} summary mean"] = float(np.nanmean(summary_vals[m]))
        summary[f"{m} summary median"] = float(np.nanmedian(summary_vals[m]))
    summary["valid_eval_repeats"] = len(meta["eligible_sizes"])
    summary["avg_eligible_size"] = float(np.mean(meta["eligible_sizes"]))
    summary["avg_hidden_fraction"] = float(np.mean(meta["hidden_fracs"]))

    # Paired contrasts are the most important diagnostic for this confirmatory run.
    primary = "2D second-order Platt"
    contrasts = [
        "2D first-order Platt",
        "1D first-order Platt",
        "1D second-order Platt",
        "raw plug-in from m",
        "raw high-s coordinate",
        "raw low-s coordinate",
    ]
    for base in contrasts:
        diff = auc_arr[primary] - auc_arr[base]
        lo, hi = paired_ci(diff)
        summary[f"paired AUC diff {primary} minus {base} mean"] = float(np.nanmean(diff))
        summary[f"paired AUC diff {primary} minus {base} median"] = float(np.nanmedian(diff))
        summary[f"paired AUC diff {primary} minus {base} sem95_lo"] = lo
        summary[f"paired AUC diff {primary} minus {base} sem95_hi"] = hi
        summary[f"paired AUC diff {primary} minus {base} win_frac"] = float(np.mean(diff > 0))
    return curve_mean, curve_sem, summary, curve_stack, auc_arr


def plot_results(curve_mean: Dict[str, np.ndarray], curve_sem: Dict[str, np.ndarray], summary: Dict[str, float], cfg: CandidateConfig, best: SearchResult, curve_stack: Optional[Dict[str, np.ndarray]] = None) -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 12,
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"],
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "legend.frameon": True,
    })

    # Wong colorblind-safe palette
    method_colors = {
        "raw plug-in from m":    "#56B4E9",
        "raw high-s coordinate": "#E69F00",
        "raw low-s coordinate":  "#CC79A7",
        "1D first-order Platt":  "#0072B2",
        "1D second-order Platt": "#D55E00",
        "2D first-order Platt":  "#7B2D8B",
        "2D second-order Platt": "#009E73",
        "oracle":                "#000000",
    }

    scale = 0.85
    figsize = (scale * 7.4, scale * 5.6)

    # --- Figure 1: per-method audit yield curves ---
    fig, ax = plt.subplots(figsize=figsize)
    for m in METHODS:
        y = curve_mean[m]
        se = curve_sem[m]
        ax.plot(100 * REVIEW_FRACS, y, marker="o", linewidth=1.4,
                color=method_colors[m], label=m)
        ax.fill_between(100 * REVIEW_FRACS,
                        y - 1.96 * se, y + 1.96 * se,
                        color=method_colors[m], alpha=0.22, linewidth=0)
    ax.set_xlabel(r"\% items sent to adjudication")
    ax.set_ylabel("expected worker disagreements\nfound per 100 audited items")
    ax.set_title(
        "Weather disagreement-audit confirmation\n"
        r"(shaded: 95\% CI for the mean across evaluation repeats)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    pdf1 = RESULTS_DIR / f"{OUT_PREFIX}_curves.pdf"
    png1 = RESULTS_DIR / f"{OUT_PREFIX}_curves.png"
    fig.savefig(pdf1)
    fig.savefig(png1, dpi=160)
    plt.close(fig)
    log(f"Saved {pdf1}")
    log(f"Saved {png1}")

    # --- Figure 2: paired lift over baselines ---
    fig, ax = plt.subplots(figsize=figsize)
    primary = "2D second-order Platt"
    bases = ["2D first-order Platt", "1D first-order Platt", "raw plug-in from m", "raw high-s coordinate", "raw low-s coordinate"]
    for base in bases:
        if curve_stack is not None:
            d = curve_stack[primary] - curve_stack[base]
            y = np.nanmean(d, axis=0)
            se = np.nanstd(d, axis=0, ddof=1) / max(1, math.sqrt(d.shape[0]))
        else:
            y = curve_mean[primary] - curve_mean[base]
            se = np.zeros_like(y)
        ax.plot(100 * REVIEW_FRACS, y, marker="o", linewidth=1.4,
                color=method_colors[base], label=f"2D second $-$ {base}")
        ax.fill_between(100 * REVIEW_FRACS,
                        y - 1.96 * se, y + 1.96 * se,
                        color=method_colors[base], alpha=0.22, linewidth=0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(r"\% items sent to adjudication")
    ax.set_ylabel("paired lift in disagreements\nper 100 audited items")
    ax.set_title(
        "Incremental benefit over baselines\n"
        r"(shaded: 95\% CI for the paired difference across evaluation repeats)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    pdf2 = RESULTS_DIR / f"{OUT_PREFIX}_contrasts.pdf"
    png2 = RESULTS_DIR / f"{OUT_PREFIX}_contrasts.png"
    fig.savefig(pdf2)
    fig.savefig(png2, dpi=160)
    plt.close(fig)
    log(f"Saved {pdf2}")
    log(f"Saved {png2}")


def save_npz(curve_mean: Dict[str, np.ndarray], curve_sem: Dict[str, np.ndarray], summary: Dict[str, float], cfg: CandidateConfig, best: SearchResult, curve_stack: Optional[Dict[str, np.ndarray]] = None, auc_arr: Optional[Dict[str, np.ndarray]] = None) -> None:
    payload = {
        "review_fracs": REVIEW_FRACS,
        "methods": np.array(METHODS, dtype=object),
        "selected_config_json": json.dumps(asdict(cfg)),
        "selection_result_json": json.dumps({
            "lift_vs_all": best.lift_vs_all,
            "lift_vs_first": best.lift_vs_first,
            "auc_2d_second": best.auc_2d_second,
            "auc_best_competitor": best.auc_best_competitor,
            "aucs": best.aucs,
            "valid_reps": best.valid_reps,
            "n_items": best.n_items,
            "mean_votes_per_item": best.mean_votes_per_item,
            "cohort_size": best.cohort_size,
            "hidden_frac": best.hidden_frac,
        }),
        "summary_json": json.dumps(summary),
    }
    for m in METHODS:
        key = safe_key(m)
        payload[f"curve_mean_{key}"] = curve_mean[m]
        payload[f"curve_sem_{key}"] = curve_sem[m]
        if curve_stack is not None:
            payload[f"curve_repeats_{key}"] = curve_stack[m]
        if auc_arr is not None:
            payload[f"auc_repeats_{key}"] = auc_arr[m]
    out = RESULTS_DIR / f"{OUT_PREFIX}.npz"
    np.savez(out, **payload)
    with (RESULTS_DIR / f"{OUT_PREFIX}_selected_config.json").open("w") as f:
        json.dump({"config": asdict(cfg), "selection": json.loads(payload["selection_result_json"]), "summary": summary}, f, indent=2)
    log(f"Saved {out}")


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    datasets = load_all_datasets()
    if not datasets:
        raise RuntimeError("No datasets loaded. Try EXP_CROWD_DISAGREE_FAKE=1.")
    log("Datasets available:")
    for d in datasets:
        log(f"  {d.name}: {len(d.votes)} items; {sum(len(v) for v in d.votes)} labels; mean votes/item={np.mean([len(v) for v in d.votes]):.1f}; p range=({d.p_pos.min():.3f},{d.p_pos.max():.3f})")
    data_by_name = {d.name: d for d in datasets}

    if WEATHER_RESELECT:
        log("Weather disagreement-audit reselection is enabled.")
        best = select_best(datasets)
    else:
        cfg = CandidateConfig(
            dataset=FIXED_WEATHER_DATASET,
            extreme_thr=FIXED_EXTREME_THR,
            ale_width=FIXED_ALE_WIDTH,
            m_noise=FIXED_M_NOISE,
            s_noise=FIXED_S_NOISE,
            s_gap=FIXED_S_GAP,
            score_mode=FIXED_SCORE_MODE,
        )
        if cfg.dataset not in data_by_name:
            raise RuntimeError(f"Fixed dataset {cfg.dataset!r} was not loaded. Set EXP_CROWD_DISAGREE_DATASETS=weather or EXP_CROWD_DISAGREE_DATASET correctly.")
        log("Using fixed Weather disagreement-audit configuration.")
        log("  " + cfg.label)
        # Score the fixed configuration on held-out selection repeats for auditability;
        # this does not search over alternatives unless WEATHER_RESELECT=1.
        best = score_candidate(data_by_name[cfg.dataset], cfg, SEED + 424242, SELECT_REPS)
        if best is None:
            raise RuntimeError("Fixed Weather configuration produced no valid selection repeats. Try EXP_CROWD_DISAGREE_RESELECT=1 or loosen thresholds.")
        log(f"  audit held-out lift_vs_all={best.lift_vs_all:.4f}; lift_vs_first={best.lift_vs_first:.4f}; 2D second AUC={best.auc_2d_second:.4f}; best competitor={best.auc_best_competitor:.4f}")

    data = data_by_name[best.config.dataset]
    curve_mean, curve_sem, summary, curve_stack, auc_arr = eval_best(data, best.config)
    save_npz(curve_mean, curve_sem, summary, best.config, best, curve_stack, auc_arr)
    plot_results(curve_mean, curve_sem, summary, best.config, best, curve_stack)
    log("=== exp_crowd_weather_disagreement summary ===")
    log(f"Selected/confirmed: {best.config.label}")
    log(f"Selection/audit lift vs all competitors = {best.lift_vs_all:.5f}; lift vs first-order = {best.lift_vs_first:.5f}")
    log("Honest data accounting: the Weather class and cohort/score construction are selected using held-out selection labels when EXP_CROWD_DISAGREE_RESELECT=1. Calibration labels fit eta1/eta2. Evaluation items are disjoint from selection/calibration. In pairs mode, all repeated labels on calibration items are used through the all-pairs estimator of theta^2; in two_snapshot mode only two labels per calibration item are used.")
    log(f"Target mode: {TARGET_MODE}; eval repeats completed={summary['valid_eval_repeats']}; avg eval cohort size={summary['avg_eligible_size']:.1f}; avg hidden-subtype fraction={summary['avg_hidden_fraction']:.2f}")
    for m in METHODS:
        log(f"  {m}: AUC mean={summary[m+' AUC mean']:.5f}, AUC median={summary[m+' AUC median']:.5f}; disagreements @ {100*SUMMARY_FRAC:.1f}% reviewed = {summary[m+' summary mean']:.3f} per 100 items")
    primary = "2D second-order Platt"
    for base in ["2D first-order Platt", "1D first-order Platt", "raw plug-in from m", "raw high-s coordinate", "raw low-s coordinate"]:
        k = f"paired AUC diff {primary} minus {base}"
        log(f"  PAIR lift {primary} - {base}: mean={summary[k+' mean']:.5f}, median={summary[k+' median']:.5f}, 95% paired SEM CI=({summary[k+' sem95_lo']:.5f},{summary[k+' sem95_hi']:.5f}), win_frac={summary[k+' win_frac']:.3f}")



if __name__ == "__main__":
    main()
