"""Public LaLonde / NSW data in the Dehejia and Wahba (1999) version.

Source: Rajeev Dehejia's data page at NBER, https://users.nber.org/~rdehejia/data/
Files are cached in data/raw/ and checked against a SHA-256 digest, so every run uses
the same bytes. If a file is missing it is downloaded once and then verified.

Citations:
  LaLonde, R. (1986). Evaluating the econometric evaluations of training programs with
    experimental data. American Economic Review 76(4), 604-620.
  Dehejia, R. and Wahba, S. (1999). Causal effects in nonexperimental studies:
    reevaluating the evaluation of training programs. JASA 94(448), 1053-1062.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
BASE_URL = "https://users.nber.org/~rdehejia/data/"

SOURCES = {
    "nsw_dw.dta": "d1bd2680a1c6f799f1c6d2455bf29633fdf19be01cb19490621c20a560b4e072",
    "psid_controls.dta": "7beebae8928035d6abf662b994ce77a4fbea7ae6575037ff94e92fe099b43e14",
    "cps_controls.dta": "80f0123eaf723870bd060ba9f8569617a7cb519a462dc20badbfb3443cd8374b",
}

COVARIATES = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]
OUTCOME = "re78"

SAMPLES = {
    "experimental": "NSW treated vs NSW randomised controls (the benchmark)",
    "psid": "NSW treated vs PSID-1 survey controls (observational)",
    "cps": "NSW treated vs CPS-1 survey controls (observational)",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(raw_dir: Path = RAW, download: bool = True) -> dict[str, Path]:
    """Return verified local paths for the three files, downloading any that are missing."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, digest in SOURCES.items():
        path = raw_dir / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"{path} missing and download disabled")
            urllib.request.urlretrieve(BASE_URL + name, path)  # noqa: S310 (fixed https URL)
        got = sha256(path)
        if got != digest:
            raise ValueError(f"checksum mismatch for {name}: {got} != {digest}")
        out[name] = path
    return out


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_stata(path)
    df = df.drop(columns=["data_id"]).astype("float64")
    df["treat"] = df["treat"].astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["u74"] = (df["re74"] == 0).astype(float)
    df["u75"] = (df["re75"] == 0).astype(float)
    return df


def load(sample: str, raw_dir: Path = RAW) -> pd.DataFrame:
    """Load one analysis sample with columns treat, re78, covariates, u74, u75."""
    if sample not in SAMPLES:
        raise KeyError(sample)
    paths = fetch(raw_dir)
    nsw = _read(paths["nsw_dw.dta"])
    if sample == "experimental":
        df = nsw
    else:
        controls = _read(paths[f"{sample}_controls.dta"])
        df = pd.concat([nsw[nsw.treat == 1], controls], ignore_index=True)
    df = add_features(df)
    assert set(np.unique(df.treat)) == {0, 1}
    return df.reset_index(drop=True)
