"""
Comparative Analysis of Data Mining Techniques for Customer Segmentation in E-commerce
Prototype pipeline (v0.1)

Dataset : UCI Online Retail II (Chen, 2019) - online_retail_II.xlsx
Stages  : load -> clean -> RFM feature engineering -> transform/scale
          -> K-Means / Agglomerative (Ward) / GMM / DBSCAN
          -> internal validation (Silhouette, Davies-Bouldin, Calinski-Harabasz)
          -> stability (bootstrap ARI) -> segment profiles

Usage   : python segmentation_pipeline.py --data online_retail_II.xlsx --out results
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

SEED = 42


# ---------------------------------------------------------------- 1. Load
def load_data(path):
    """Read both yearly sheets of Online Retail II and concatenate them."""
    if path.endswith(".csv"):
        df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    else:
        sheets = pd.read_excel(path, sheet_name=None)       # dict of DataFrames
        df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = [c.strip().replace(" ", "") for c in df.columns]  # 'Customer ID' -> 'CustomerID'
    # also accept the older UCI "Online Retail" (2010-11) column names
    df = df.rename(columns={"InvoiceNo": "Invoice", "UnitPrice": "Price"})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    print(f"  loaded {len(df):,} rows, {df['InvoiceDate'].min():%d %b %Y} to {df['InvoiceDate'].max():%d %b %Y}")
    return df


# ---------------------------------------------------------------- 2. Clean
def clean_data(df, log=print):
    """Apply documented cleaning rules; returns cleaned frame + audit log."""
    audit = [("raw rows", len(df))]
    df = df.dropna(subset=["CustomerID"])
    audit.append(("drop missing CustomerID", len(df)))
    df = df[~df["Invoice"].astype(str).str.startswith("C")]          # cancellations
    audit.append(("drop cancelled invoices", len(df)))
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    audit.append(("drop non-positive qty/price", len(df)))
    non_product = ["POST", "D", "M", "BANK CHARGES", "DOT", "C2", "PADS", "AMAZONFEE"]
    df = df[~df["StockCode"].astype(str).str.upper().isin(non_product)]
    audit.append(("drop non-product stock codes", len(df)))
    df = df.drop_duplicates()
    audit.append(("drop exact duplicates", len(df)))
    df = df.copy()
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["LineTotal"] = df["Quantity"] * df["Price"]
    for step, n in audit:
        log(f"  {step:<32}{n:>10,}")
    return df, audit


# ---------------------------------------------------------------- 3. RFM
def build_rfm(df):
    """Recency (days since last purchase), Frequency (distinct invoices), Monetary (total spend)."""
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("LineTotal", "sum"),
    )
    return rfm


def transform_rfm(rfm, cap_q=0.99):
    """Winsorise extreme values, log1p to reduce skew, then z-score scale."""
    X = rfm.copy()
    for c in X.columns:
        X[c] = X[c].clip(upper=X[c].quantile(cap_q))
    X = np.log1p(X)
    return StandardScaler().fit_transform(X)


# ---------------------------------------------------------------- 4. Models
def choose_k(X, k_range=range(2, 11), out=None):
    """Elbow (inertia) + silhouette across k for K-Means."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X, km.labels_, sample_size=min(5000, len(X)),
                                                    random_state=SEED)})
    res = pd.DataFrame(rows)
    if out:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot(res.k, res.inertia, "o-"); ax[0].set(title="Elbow (inertia)", xlabel="k")
        ax[1].plot(res.k, res.silhouette, "o-"); ax[1].set(title="Silhouette", xlabel="k")
        fig.tight_layout(); fig.savefig(os.path.join(out, "k_selection.png"), dpi=150)
    return res


def dbscan_eps(X, min_samples=8, out=None):
    """k-distance heuristic (Ester et al., 1996): eps ~ knee of sorted k-NN distances."""
    d, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
    kd = np.sort(d[:, -1])
    eps = float(np.percentile(kd, 95))
    if out:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(kd); ax.axhline(eps, ls="--"); ax.set(title=f"{min_samples}-distance plot", ylabel="distance")
        fig.tight_layout(); fig.savefig(os.path.join(out, "dbscan_kdist.png"), dpi=150)
    return eps


def fit_models(X, k, eps, min_samples=8):
    models = {
        "K-Means": lambda: KMeans(n_clusters=k, n_init=10, random_state=SEED),
        "Agglomerative (Ward)": lambda: AgglomerativeClustering(n_clusters=k, linkage="ward"),
        "GMM": lambda: GaussianMixture(n_components=k, covariance_type="full", random_state=SEED),
        "DBSCAN": lambda: DBSCAN(eps=eps, min_samples=min_samples),
    }
    labels, times = {}, {}
    for name, make in models.items():
        t0 = time.perf_counter()
        m = make()
        labels[name] = m.fit_predict(X)
        times[name] = time.perf_counter() - t0
    return labels, times


# ---------------------------------------------------------------- 5. Evaluate
def evaluate(X, labels, times):
    rows = []
    for name, lab in labels.items():
        mask = lab != -1                                   # exclude DBSCAN noise from internal indices
        n_clusters = len(set(lab[mask]))
        row = {"Model": name, "Clusters": n_clusters,
               "Noise %": round(100 * (~mask).mean(), 1), "Fit time (s)": round(times[name], 3)}
        if n_clusters >= 2:
            row["Silhouette"] = silhouette_score(X[mask], lab[mask], sample_size=min(5000, mask.sum()),
                                                 random_state=SEED)
            row["Davies-Bouldin"] = davies_bouldin_score(X[mask], lab[mask])
            row["Calinski-Harabasz"] = calinski_harabasz_score(X[mask], lab[mask])
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def stability(X, k, n_boot=20, frac=0.8):
    """Mean ARI between full-data K-Means/GMM labels and labels re-fitted on bootstrap subsamples."""
    rng = np.random.default_rng(SEED)
    out = {}
    for name, make in {"K-Means": lambda s: KMeans(k, n_init=10, random_state=s),
                       "GMM": lambda s: GaussianMixture(k, random_state=s)}.items():
        ref = make(SEED).fit(X)
        aris = []
        for b in range(n_boot):
            idx = rng.choice(len(X), int(frac * len(X)), replace=False)
            m = make(b).fit(X[idx])
            aris.append(adjusted_rand_score(ref.predict(X[idx]), m.predict(X[idx])))
        out[name] = (np.mean(aris), np.std(aris))
    return out


def profile(rfm, lab):
    p = rfm.assign(Segment=lab).groupby("Segment").agg(
        Customers=("Recency", "size"), Recency=("Recency", "median"),
        Frequency=("Frequency", "median"), Monetary=("Monetary", "median"))
    p["Share %"] = (100 * p.Customers / p.Customers.sum()).round(1)
    return p.round(1)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--k", type=int, default=None, help="override number of clusters")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("[1] Loading");   df = load_data(a.data)
    print("[2] Cleaning");  df, audit = clean_data(df)
    print("[3] RFM");       rfm = build_rfm(df); X = transform_rfm(rfm)
    print(rfm.describe().round(1))
    print("[4] Selecting k")
    ks = choose_k(X, out=a.out); print(ks.round(3))
    k = a.k or int(ks.loc[ks.silhouette.idxmax(), "k"])
    eps = dbscan_eps(X, out=a.out)
    print(f"    k={k}, eps={eps:.3f}")
    print("[5] Fitting models"); labels, times = fit_models(X, k, eps)
    res = evaluate(X, labels, times); print(res.to_string(index=False))
    print("[6] Stability (bootstrap ARI)")
    for n, (m, s) in stability(X, k).items():
        print(f"    {n:<10} mean ARI={m:.3f} (sd {s:.3f})")
    print("[7] K-Means segment profile"); print(profile(rfm, labels["K-Means"]))

    res.to_csv(os.path.join(a.out, "model_comparison.csv"), index=False)
    pd.DataFrame(audit, columns=["step", "rows"]).to_csv(os.path.join(a.out, "cleaning_audit.csv"), index=False)
    rfm.assign(**{n: l for n, l in labels.items()}).to_csv(os.path.join(a.out, "rfm_labels.csv"))


if __name__ == "__main__":
    main()
