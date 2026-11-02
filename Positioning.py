#!/usr/bin/env python3
"""
FX Positioning Intelligence Terminal — single-file Flask application.

The analytical foundation is `cot_engine.py` (unmodified). This module
contains the Flask backend AND the entire frontend (HTML + CSS + JS)
embedded inline — no templates/ or static/ folders required.

Run:   python Positioning.py
Open:  http://127.0.0.1:5000
"""

import os
import sys

# ----------------------------------------------------------------------
# sys.path bootstrap (helps when run via Spyder's %runfile --wdir)
# ----------------------------------------------------------------------
_HERE = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import io
import math
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request
from pymongo import MongoClient
import certifi

import cot_engine as engine

# ----------------------------------------------------------------------
# Inlined app config
# ----------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 5000
DEBUG = False

app = Flask(__name__)


# ======================================================================
# Serialisation helpers
# ======================================================================
def _clean(v):
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.strftime("%Y-%m-%d")
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, str):
        return v
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _row_to_dict(row):
    return {k: _clean(v) for k, v in row.items()}


def _df_to_records(df):
    if df is None or df.empty:
        return []
    return [_row_to_dict(r) for _, r in df.iterrows()]


def _parse_date_safe(s):
    if not s:
        return None
    try:
        return pd.Timestamp(s)
    except Exception:
        return None


# ======================================================================
# Cache
# ======================================================================
class Cache:
    def __init__(self):
        self.lock = threading.Lock()
        self.currency_df = None
        self.pair_df = None
        self.loaded_at = None
        self.loading = False
        self.error = None
        self.version = 0
        self.data_source = None      # 'mongo' or 'engine'
        self.storage = Storage()

    @property
    def ready(self):
        return self.currency_df is not None and self.pair_df is not None

    def _ensure_files(self):
        engine.ensure_directories()
        raw_dir = engine.CONFIG["raw_dir"]
        has_raw = os.path.exists(raw_dir) and any(
            f.endswith(".csv") for f in os.listdir(raw_dir)
        )
        if not has_raw:
            engine.download_all()
        processed = os.path.join(
            engine.CONFIG["processed_dir"], "cot_currency_weekly.csv"
        )
        if not os.path.exists(processed):
            engine.process_data()

    def _try_mongo(self):
        """Return True if a fresh snapshot was loaded from Mongo."""
        if not self.storage.connected:
            return False
        print("[cache] Trying Mongo snapshot...")
        currency_df, pair_df, _meta = self.storage.load_snapshot()
        if currency_df is None or currency_df.empty:
            print("[cache] Mongo empty — will run engine")
            return False
        latest = currency_df["report_date"].max()
        days_old = (datetime.now() - latest).days
        print(f"[cache] Mongo snapshot: latest report {latest.date()}, "
              f"{days_old} days old, {len(currency_df)} currency rows")
        if days_old > 10:
            print("[cache] Mongo snapshot is stale — will run engine")
            return False
        self.currency_df = currency_df
        self.pair_df = pair_df
        self.loaded_at = datetime.now()
        self.data_source = "mongo"
        self.version += 1
        self.error = None
        print(f"[cache] Loaded from Mongo: {len(currency_df)} + {len(pair_df)} rows")
        return True

    def _build_from_engine(self):
        """Run the engine end-to-end and repopulate the cache."""
        print("[cache] Running engine...")
        self._ensure_files()
        processed = os.path.join(
            engine.CONFIG["processed_dir"], "cot_currency_weekly.csv"
        )
        if not os.path.exists(processed):
            raise RuntimeError("No processed COT data available.")

        currency_df = pd.read_csv(processed, parse_dates=["report_date"])
        currency_df = currency_df.sort_values(
            ["currency", "report_date"]).reset_index(drop=True)

        pair_history = engine.generate_pair_history(
            currency_df, engine.CANONICAL_PAIRS
        )
        if not pair_history.empty:
            pair_history = engine.calculate_pair_trajectory(
                pair_history, periods=[1, 4, 12])
            pair_history["nc_status"] = pair_history["nc_diff"].apply(
                engine.classify_pair_status)
            pair_history["comm_status"] = pair_history["comm_diff"].apply(
                engine.classify_pair_status)
            pair_history["nr_status"] = pair_history["nr_diff"].apply(
                engine.classify_pair_status)
            pair_history["opposition"] = pair_history.apply(
                lambda r: engine.classify_opposition(r["nc_diff"], r["comm_diff"]),
                axis=1)
            pair_history["nc_trajectory"] = pair_history[
                "nc_diff_4w_change"].apply(engine.classify_trajectory)
            pair_history["comm_trajectory"] = pair_history[
                "comm_diff_4w_change"].apply(engine.classify_trajectory)
            pair_history["context"] = pair_history.apply(
                engine.classify_positioning_context, axis=1)
            pair_history = pair_history.sort_values(
                ["pair", "report_date"]).reset_index(drop=True)

        self.currency_df = currency_df
        self.pair_df = pair_history
        self.loaded_at = datetime.now()
        self.data_source = "engine"
        self.version += 1
        self.error = None

        if self.storage.connected:
            self.storage.save_snapshot(currency_df, pair_history)

    def _build(self):
        if self._try_mongo():
            return
        self._build_from_engine()

    def load(self):
        if self.loading:
            return
        self.loading = True
        try:
            with self.lock:
                self._build()
        except Exception as e:
            self.error = str(e)
            print(f"[cache] Load failed: {e}")
        finally:
            self.loading = False

    def refresh(self):
        if self.loading:
            return False
        self.loading = True
        try:
            with self.lock:
                engine.download_all()
                engine.process_data()
                self._build_from_engine()
            return True
        except Exception as e:
            self.error = str(e)
            return False
        finally:
            self.loading = False
            
            
# ======================================================================
# MongoDB storage layer — persists engine output across ephemeral restarts
# ======================================================================
class Storage:
    """Wraps a small, fixed set of MongoDB collections.

    If MONGO_URI is not set, or the connection fails, the app falls back
    to disk-only behaviour identical to the pre-Mongo version. This means
    local development works without a running MongoDB instance.
    """

    def __init__(self):
        self.client = None
        self.db = None
        self.connected = False
        self.last_error = None
        self._connect()

    def _connect(self):
        uri = os.environ.get("MONGO_URI")
        if not uri:
            print("[storage] MONGO_URI not set — disk-only mode")
            return
        db_name = os.environ.get("MONGO_DB", "fx_intel")
        try:
            self.client = MongoClient(
                uri,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=8000,
            )
            self.client.admin.command("ping")
            self.db = self.client[db_name]
            self.connected = True
            print(f"[storage] Connected to MongoDB database '{db_name}'")
        except Exception as e:
            self.last_error = str(e)
            print(f"[storage] Mongo connection failed: {e}")
            self.client = None
            self.db = None

    @staticmethod
    def _df_to_records(df):
        if df is None or df.empty:
            return []
        records = df.to_dict(orient="records")
        for rec in records:
            for k, v in list(rec.items()):
                if isinstance(v, pd.Timestamp):
                    rec[k] = None if pd.isna(v) else v.strftime("%Y-%m-%d")
                elif isinstance(v, (np.integer,)):
                    rec[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    rec[k] = None if (np.isnan(v) or np.isinf(v)) else float(v)
                elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    rec[k] = None
                elif isinstance(v, np.bool_):
                    rec[k] = bool(v)
        return records

    def save_snapshot(self, currency_df, pair_df):
        if not self.connected:
            return False
        try:
            currency_records = self._df_to_records(currency_df)
            pair_records = self._df_to_records(pair_df)
            latest_report = (
                currency_df["report_date"].max().strftime("%Y-%m-%d")
                if currency_df is not None and not currency_df.empty else None
            )
            now = datetime.utcnow()
            self.db.currency_history.replace_one(
                {"_id": "snapshot"},
                {"_id": "snapshot", "rows": currency_records,
                 "row_count": len(currency_records),
                 "latest_report_date": latest_report, "updated_at": now},
                upsert=True,
            )
            self.db.pair_history.replace_one(
                {"_id": "snapshot"},
                {"_id": "snapshot", "rows": pair_records,
                 "row_count": len(pair_records), "updated_at": now},
                upsert=True,
            )
            self.db.metadata.replace_one(
                {"_id": "status"},
                {"_id": "status", "latest_report_date": latest_report,
                 "updated_at": now,
                 "currency_rows": len(currency_records),
                 "pair_rows": len(pair_records)},
                upsert=True,
            )
            print(f"[storage] Saved {len(currency_records)} currency rows, "
                  f"{len(pair_records)} pair rows to Mongo")
            return True
        except Exception as e:
            self.last_error = str(e)
            print(f"[storage] save_snapshot failed: {e}")
            return False

    def load_snapshot(self):
        if not self.connected:
            return None, None, None
        try:
            currency_doc = self.db.currency_history.find_one({"_id": "snapshot"})
            pair_doc = self.db.pair_history.find_one({"_id": "snapshot"})
            if not currency_doc or not pair_doc:
                return None, None, None
            currency_df = pd.DataFrame(currency_doc["rows"])
            pair_df = pd.DataFrame(pair_doc["rows"])
            if not currency_df.empty and "report_date" in currency_df.columns:
                currency_df["report_date"] = pd.to_datetime(currency_df["report_date"])
                currency_df = currency_df.sort_values(
                    ["currency", "report_date"]).reset_index(drop=True)
            if not pair_df.empty and "report_date" in pair_df.columns:
                pair_df["report_date"] = pd.to_datetime(pair_df["report_date"])
                pair_df = pair_df.sort_values(
                    ["pair", "report_date"]).reset_index(drop=True)
            meta = {
                "updated_at": currency_doc.get("updated_at"),
                "latest_report_date": currency_doc.get("latest_report_date"),
                "currency_rows": len(currency_df),
                "pair_rows": len(pair_df),
            }
            return currency_df, pair_df, meta
        except Exception as e:
            self.last_error = str(e)
            print(f"[storage] load_snapshot failed: {e}")
            return None, None, None

        def info(self):
        return {
            "connected": self.connected,
            "database": self.db.name if self.db is not None else None,
            "last_error": self.last_error,
        }


CACHE = Cache()


# ======================================================================
# Shared helpers
# ======================================================================
def _require_data():
    if not CACHE.ready:
        return jsonify({
            "error": "Data not ready",
            "loading": CACHE.loading,
            "detail": CACHE.error,
        }), 503
    return None


def _compute_extremes(cur_df, lookback_years):
    if cur_df is None or cur_df.empty:
        return None
    latest = cur_df.iloc[-1]
    latest_date = latest["report_date"]
    cutoff = latest_date - pd.Timedelta(days=lookback_years * 365)
    window = cur_df[cur_df["report_date"] >= cutoff]
    if window.empty:
        return None

    result = {
        "window_years": lookback_years,
        "window_start": _clean(window["report_date"].min()),
        "window_end": _clean(latest_date),
        "window_weeks": int(len(window)),
        "metrics": {},
    }

    for col in [
        "noncommercial_net",
        "commercial_net",
        "noncommercial_positioning_score",
        "commercial_positioning_score",
    ]:
        if col not in window.columns:
            continue
        high_val = window[col].max()
        low_val = window[col].min()
        if pd.isna(high_val) or pd.isna(low_val):
            continue
        high_idx = window[col].idxmax()
        low_idx = window[col].idxmin()
        high_date = window.loc[high_idx, "report_date"]
        low_date = window.loc[low_idx, "report_date"]
        current = latest[col]

        tol_hi = 1e-6 * max(abs(float(high_val)), 1.0)
        tol_lo = 1e-6 * max(abs(float(low_val)), 1.0)
        is_high = pd.notna(current) and abs(float(current) - float(high_val)) <= tol_hi
        is_low = pd.notna(current) and abs(float(current) - float(low_val)) <= tol_lo

        weeks_since_high = 0 if is_high else int((latest_date - high_date).days // 7)
        weeks_since_low = 0 if is_low else int((latest_date - low_date).days // 7)

        rng = float(high_val) - float(low_val)
        pos_in_range = None
        if rng > 0 and pd.notna(current):
            pos_in_range = (float(current) - float(low_val)) / rng * 100

        result["metrics"][col] = {
            "current": _clean(current),
            "high": _clean(high_val),
            "high_date": _clean(high_date),
            "low": _clean(low_val),
            "low_date": _clean(low_date),
            "is_current_high": bool(is_high),
            "is_current_low": bool(is_low),
            "distance_from_high": _clean(float(current) - float(high_val))
            if pd.notna(current) else None,
            "distance_from_low": _clean(float(current) - float(low_val))
            if pd.notna(current) else None,
            "weeks_since_high": weeks_since_high,
            "weeks_since_low": weeks_since_low,
            "range": _clean(rng),
            "position_in_range": _clean(pos_in_range),
        }
    return result


def _interpret_currency(latest):
    score = latest.get("noncommercial_positioning_score")
    pct = latest.get("noncommercial_percentile")
    chg_1w = latest.get("noncommercial_score_1w_change")
    chg_4w = latest.get("noncommercial_score_4w_change")
    chg_12w = latest.get("noncommercial_score_12w_change")
    div = latest.get("commercial_noncommercial_divergence")
    parts = []

    if pd.notna(score):
        if score >= 60:
            parts.append(f"Positioning is historically bullish (score {score:+.0f}).")
        elif score >= 20:
            parts.append(f"Positioning leans moderately bullish (score {score:+.0f}).")
        elif score > -20:
            parts.append(f"Positioning is essentially neutral (score {score:+.0f}).")
        elif score > -60:
            parts.append(f"Positioning leans moderately bearish (score {score:+.0f}).")
        else:
            parts.append(f"Positioning is historically bearish (score {score:+.0f}).")

    if pd.notna(pct):
        if pct >= 90:
            parts.append(f"Position sits in the top decile of its rolling percentile ({pct:.0f}th).")
        elif pct <= 10:
            parts.append(f"Position sits in the bottom decile of its rolling percentile ({pct:.0f}th).")

    if pd.notna(chg_4w):
        if chg_4w > 10:
            parts.append(f"The 4-week score has risen sharply ({chg_4w:+.0f}).")
        elif chg_4w < -10:
            parts.append(f"The 4-week score has fallen sharply ({chg_4w:+.0f}).")

    if pd.notna(chg_1w) and pd.notna(chg_4w) and chg_4w != 0:
        avg4 = chg_4w / 4
        if np.sign(chg_1w) == np.sign(chg_4w) and abs(chg_1w) > abs(avg4) * 1.2:
            parts.append("Recent weekly movement is accelerating the 4-week trend.")
        elif np.sign(chg_1w) == np.sign(chg_4w) and abs(chg_1w) < abs(avg4) * 0.8:
            parts.append("Recent weekly movement is decelerating the 4-week trend.")
        elif np.sign(chg_1w) != np.sign(chg_4w):
            parts.append("The most recent weekly change is reversing the 4-week trend.")

    if pd.notna(div) and abs(div) >= 60:
        parts.append(f"Commercial and non-commercial positioning diverge significantly (spread {div:+.0f}).")

    if pd.notna(chg_12w) and pd.notna(chg_4w):
        if np.sign(chg_12w) == np.sign(chg_4w):
            parts.append("Recent movement agrees with the 12-week trend.")
        else:
            parts.append("Recent movement conflicts with the 12-week trend.")

    return " ".join(parts) if parts else "Insufficient data for interpretation."


def _normalize_pair(p):
    if not p:
        return None
    p = p.upper().replace("/", "").replace("-", "").replace("_", "").strip()
    if len(p) != 6:
        return None
    return f"{p[:3]}/{p[3:]}"


PERIOD_DAYS = {
    "1m": 30, "3m": 90, "6m": 180,
    "1y": 365, "2y": 730, "3y": 1095, "5y": 1825, "max": 99999,
}


def _slice_period(df, period, date_col="report_date"):
    days = PERIOD_DAYS.get(period, 365)
    if days >= 99999 or df.empty:
        return df
    latest = df[date_col].max()
    return df[df[date_col] >= latest - pd.Timedelta(days=days)]


# ======================================================================
# API routes
# ======================================================================
@app.route("/api/status")
def api_status():
    report_date = None
    if CACHE.ready:
        report_date = _clean(CACHE.currency_df["report_date"].max())
    return jsonify({
        "ready": CACHE.ready,
        "loading": CACHE.loading,
        "error": CACHE.error,
        "loaded_at": _clean(CACHE.loaded_at) if CACHE.loaded_at else None,
        "report_date": report_date,
        "data_source": CACHE.data_source,
        "version": CACHE.version,
        "storage": CACHE.storage.info(),
    })


@app.route("/api/cache-info")
def api_cache_info():
    info = {
        "ready": CACHE.ready,
        "loading": CACHE.loading,
        "data_source": CACHE.data_source,
        "loaded_at": _clean(CACHE.loaded_at) if CACHE.loaded_at else None,
        "storage": CACHE.storage.info(),
    }
    if CACHE.ready:
        latest = CACHE.currency_df["report_date"].max()
        days_old = (datetime.now() - latest).days
        info["latest_report_date"] = _clean(latest)
        info["days_old"] = days_old
        info["currency_rows"] = len(CACHE.currency_df)
        info["pair_rows"] = len(CACHE.pair_df)
        info["currencies"] = sorted(
            CACHE.currency_df["currency"].unique().tolist())
        info["pairs_count"] = int(CACHE.pair_df["pair"].nunique())
    return jsonify(info)


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    if CACHE.loading:
        return jsonify({"status": "already_loading"})
    threading.Thread(target=CACHE.refresh, daemon=True).start()
    return jsonify({"status": "refresh_started"})


@app.route("/api/overview")
def api_overview():
    err = _require_data()
    if err:
        return err
    df = CACHE.currency_df
    latest_date = df["report_date"].max()
    snap = df[df["report_date"] == latest_date].copy()

    def _change(cur_df, col, weeks):
        if len(cur_df) < weeks + 1:
            return None
        return float(cur_df[col].iloc[-1] - cur_df[col].iloc[-(weeks + 1)])

    rows_1w, rows_4w, rows_12w = [], [], []
    for cur in engine.CONFIG["target_currencies"]:
        cdf = df[df["currency"] == cur].sort_values("report_date")
        if cdf.empty:
            continue
        for weeks, bucket in [(1, rows_1w), (4, rows_4w), (12, rows_12w)]:
            v = _change(cdf, "noncommercial_net", weeks)
            if v is not None:
                bucket.append({"currency": cur, "change": v})

    def _top(bucket):
        if not bucket:
            return None
        return max(bucket, key=lambda r: abs(r["change"]))

    sorted_snap = snap.sort_values("noncommercial_positioning_score", ascending=False)
    strongest = _row_to_dict(sorted_snap.iloc[0]) if not sorted_snap.empty else None
    weakest = _row_to_dict(sorted_snap.iloc[-1]) if not sorted_snap.empty else None

    crowd_long = crowd_short = None
    if snap["noncommercial_percentile"].notna().any():
        crowd_long = _row_to_dict(snap.loc[snap["noncommercial_percentile"].idxmax()])
        crowd_short = _row_to_dict(snap.loc[snap["noncommercial_percentile"].idxmin()])

    approaching = []
    for cur in engine.CONFIG["target_currencies"]:
        cdf = df[df["currency"] == cur].sort_values("report_date")
        ext = _compute_extremes(cdf, lookback_years=3)
        if not ext:
            continue
        m = ext["metrics"].get("noncommercial_positioning_score")
        if not m or m.get("position_in_range") is None:
            continue
        p = m["position_in_range"]
        if p >= 85 or p <= 15:
            approaching.append({
                "currency": cur, "score": m["current"], "position_in_range": p,
                "direction": "bullish" if p >= 85 else "bearish",
                "high": m["high"], "low": m["low"],
            })

    pair_latest = CACHE.pair_df[
        CACHE.pair_df["report_date"] == CACHE.pair_df["report_date"].max()
    ].copy()
    pair_latest["abs_nc"] = pair_latest["nc_diff"].abs()
    interesting = _df_to_records(
        pair_latest.sort_values("abs_nc", ascending=False).head(6)
    )

    return jsonify({
        "report_date": _clean(latest_date),
        "currencies": _df_to_records(snap),
        "strongest": strongest,
        "weakest": weakest,
        "crowded_long": crowd_long,
        "crowded_short": crowd_short,
        "largest_1w_change": _top(rows_1w),
        "largest_4w_change": _top(rows_4w),
        "largest_12w_change": _top(rows_12w),
        "approaching_extremes": approaching,
        "interesting_pairs": interesting,
        "pair_count": int(len(pair_latest)),
    })


@app.route("/api/currencies")
def api_currencies():
    err = _require_data()
    if err:
        return err
    df = CACHE.currency_df
    latest_date = df["report_date"].max()
    snap = df[df["report_date"] == latest_date]
    return jsonify({
        "report_date": _clean(latest_date),
        "currencies": _df_to_records(snap),
    })


@app.route("/api/currency/<code>")
def api_currency(code):
    err = _require_data()
    if err:
        return err
    code = code.upper()
    if code not in engine.CONFIG["target_currencies"]:
        return jsonify({"error": f"Unknown currency: {code}"}), 404
    cdf = CACHE.currency_df[CACHE.currency_df["currency"] == code].sort_values("report_date")
    if cdf.empty:
        return jsonify({"error": "No data for currency"}), 404
    latest = cdf.iloc[-1]
    return jsonify({
        "currency": code,
        "latest": _row_to_dict(latest),
        "extremes_3y": _compute_extremes(cdf, 3),
        "extremes_5y": _compute_extremes(cdf, 5),
        "interpretation": _interpret_currency(latest),
    })


@app.route("/api/currency/<code>/history")
def api_currency_history(code):
    err = _require_data()
    if err:
        return err
    code = code.upper()
    cdf = CACHE.currency_df[CACHE.currency_df["currency"] == code].sort_values("report_date").copy()
    if cdf.empty:
        return jsonify({"error": "No data"}), 404
    period = request.args.get("period", "1y")
    start = _parse_date_safe(request.args.get("start"))
    end = _parse_date_safe(request.args.get("end"))
    if period and not start:
        days = PERIOD_DAYS.get(period, 365)
        if days < 99999:
            start = cdf["report_date"].max() - pd.Timedelta(days=days)
    if start is not None:
        cdf = cdf[cdf["report_date"] >= start]
    if end is not None:
        cdf = cdf[cdf["report_date"] <= end]

    cols = [
        "open_interest",
        "noncommercial_long", "noncommercial_short", "noncommercial_net",
        "commercial_long", "commercial_short", "commercial_net",
        "nonreportable_long", "nonreportable_short", "nonreportable_net",
        "noncommercial_net_pct_oi", "commercial_net_pct_oi",
        "noncommercial_percentile", "noncommercial_positioning_score",
        "commercial_percentile", "commercial_positioning_score",
        "nonreportable_percentile", "nonreportable_positioning_score",
        "commercial_noncommercial_divergence",
    ]
    cols = [c for c in cols if c in cdf.columns]
    series = {c: [_clean(v) for v in cdf[c].tolist()] for c in cols}
    return jsonify({
        "currency": code,
        "dates": cdf["report_date"].dt.strftime("%Y-%m-%d").tolist(),
        "series": series,
    })


@app.route("/api/pairs")
def api_pairs():
    err = _require_data()
    if err:
        return err
    df = CACHE.pair_df
    latest_date = df["report_date"].max()
    snap = df[df["report_date"] == latest_date]
    return jsonify({
        "report_date": _clean(latest_date),
        "pairs": _df_to_records(snap),
    })


@app.route("/api/pair/<pair>/history")
def api_pair_history(pair):
    err = _require_data()
    if err:
        return err
    norm = _normalize_pair(pair)
    if not norm:
        return jsonify({"error": f"Invalid pair '{pair}'"}), 400
    pdf = CACHE.pair_df[CACHE.pair_df["pair"] == norm].sort_values("report_date").copy()
    if pdf.empty:
        return jsonify({"error": f"Pair {norm} not found"}), 404
    period = request.args.get("period", "1y")
    start = _parse_date_safe(request.args.get("start"))
    end = _parse_date_safe(request.args.get("end"))
    if period and not start:
        days = PERIOD_DAYS.get(period, 365)
        if days < 99999:
            start = pdf["report_date"].max() - pd.Timedelta(days=days)
    if start is not None:
        pdf = pdf[pdf["report_date"] >= start]
    if end is not None:
        pdf = pdf[pdf["report_date"] <= end]

    cols = [
        "nc_diff", "comm_diff", "nr_diff", "divergence",
        "nc_diff_1w_change", "nc_diff_4w_change", "nc_diff_12w_change",
        "comm_diff_1w_change", "comm_diff_4w_change", "comm_diff_12w_change",
    ]
    series = {c: [_clean(v) for v in pdf[c].tolist()] for c in cols if c in pdf.columns}
    latest = pdf.iloc[-1] if not pdf.empty else None
    return jsonify({
        "pair": norm,
        "latest": _row_to_dict(latest) if latest is not None else None,
        "dates": pdf["report_date"].dt.strftime("%Y-%m-%d").tolist(),
        "series": series,
    })


@app.route("/api/extremes")
def api_extremes():
    err = _require_data()
    if err:
        return err
    df = CACHE.currency_df
    latest_date = df["report_date"].max()
    extremes, rapid, divergences = [], [], []
    for cur in engine.CONFIG["target_currencies"]:
        cdf = df[df["currency"] == cur].sort_values("report_date")
        if cdf.empty:
            continue
        latest = cdf.iloc[-1]
        for years in (3, 5):
            ext = _compute_extremes(cdf, years)
            if not ext:
                continue
            for metric, m in ext["metrics"].items():
                if m.get("is_current_high") or m.get("is_current_low"):
                    extremes.append({
                        "currency": cur, "window_years": years, "metric": metric,
                        "type": "high" if m["is_current_high"] else "low",
                        "current": m["current"],
                        "extreme_value": m["high"] if m["is_current_high"] else m["low"],
                        "extreme_date": m["high_date"] if m["is_current_high"] else m["low_date"],
                        "weeks_since": m["weeks_since_high"] if m["is_current_high"] else m["weeks_since_low"],
                        "range": m["range"],
                    })
        if "noncommercial_net_change" in cdf.columns:
            chg_series = cdf["noncommercial_net_change"].dropna()
            chg = latest.get("noncommercial_net_change")
            if pd.notna(chg) and len(chg_series) > 20:
                std = float(chg_series.std())
                if std > 0 and abs(chg) > 2 * std:
                    rapid.append({
                        "currency": cur, "weekly_change": _clean(chg),
                        "z_score": _clean(chg / std), "std": _clean(std),
                    })
        div = latest.get("commercial_noncommercial_divergence")
        if pd.notna(div) and abs(div) >= 60:
            divergences.append({
                "currency": cur, "divergence": _clean(div),
                "commercial_score": _clean(latest.get("commercial_positioning_score")),
                "noncommercial_score": _clean(latest.get("noncommercial_positioning_score")),
            })
    return jsonify({
        "report_date": _clean(latest_date),
        "extremes": extremes,
        "rapid_changes": rapid,
        "divergences": divergences,
    })


@app.route("/api/heatmap")
def api_heatmap():
    err = _require_data()
    if err:
        return err
    df = CACHE.currency_df
    metric = request.args.get("metric", "positioning_score")
    date_str = request.args.get("date")
    metric_map = {
        "positioning_score": "noncommercial_positioning_score",
        "percentile": "noncommercial_percentile",
        "net": "noncommercial_net",
        "weekly_change": "noncommercial_net_change",
        "4w_change": "noncommercial_score_4w_change",
        "12w_change": "noncommercial_score_12w_change",
        "net_pct_oi": "noncommercial_net_pct_oi",
        "commercial_score": "commercial_positioning_score",
        "commercial_percentile": "commercial_percentile",
    }
    col = metric_map.get(metric, metric)
    if col not in df.columns:
        return jsonify({"error": f"Unknown metric '{metric}'"}), 400

    if date_str:
        target = _parse_date_safe(date_str)
        if target is None:
            return jsonify({"error": "Invalid date"}), 400
        snap = df[df["report_date"] == target]
        if snap.empty:
            snap = df[df["report_date"] <= target]
            if snap.empty:
                return jsonify({"error": "No data on or before that date"}), 404
            snap = snap[snap["report_date"] == snap["report_date"].max()]
    else:
        snap = df[df["report_date"] == df["report_date"].max()]

    data = [{"currency": r["currency"], "value": _clean(r[col])} for _, r in snap.iterrows()]
    all_dates = sorted(df["report_date"].dt.strftime("%Y-%m-%d").unique())
    return jsonify({
        "metric": metric,
        "metric_column": col,
        "date": _clean(snap["report_date"].max()),
        "data": data,
        "available_dates": all_dates,
    })


@app.route("/api/momentum")
def api_momentum():
    err = _require_data()
    if err:
        return err
    df = CACHE.currency_df
    latest_date = df["report_date"].max()
    results = []
    for cur in engine.CONFIG["target_currencies"]:
        cdf = df[df["currency"] == cur].sort_values("report_date")
        if cdf.empty:
            continue
        latest = cdf.iloc[-1]
        dL = dS = dNet = None
        if len(cdf) >= 5:
            past = cdf.iloc[-5]
            dL = float(latest["noncommercial_long"] - past["noncommercial_long"])
            dS = float(latest["noncommercial_short"] - past["noncommercial_short"])
            dNet = float(latest["noncommercial_net"] - past["noncommercial_net"])
        if dL is None:
            label = "INSUFFICIENT DATA"
        elif dL > 0 and dS < 0:
            label = "STRONG BULLISH"
        elif dL < 0 and dS > 0:
            label = "STRONG BEARISH"
        elif dL > 0 and dS > 0:
            label = "LONG BUILD" if dL > dS else "MIXED BUILD"
        elif dL < 0 and dS < 0:
            label = "SHORT COVERING" if abs(dS) > abs(dL) else "LONG LIQUIDATION"
        else:
            label = "FLAT"

        chg_1w = latest.get("noncommercial_score_1w_change")
        chg_4w = latest.get("noncommercial_score_4w_change")
        if pd.notna(chg_1w) and pd.notna(chg_4w) and chg_4w != 0:
            avg4 = chg_4w / 4
            if np.sign(chg_1w) == np.sign(chg_4w) and abs(chg_1w) > abs(avg4) * 1.2:
                accel = "ACCELERATING"
            elif np.sign(chg_1w) == np.sign(chg_4w) and abs(chg_1w) < abs(avg4) * 0.8:
                accel = "DECELERATING"
            elif np.sign(chg_1w) != np.sign(chg_4w):
                accel = "REVERSING"
            else:
                accel = "STABLE"
        else:
            accel = "N/A"

        results.append({
            "currency": cur, "label": label, "acceleration": accel,
            "change_1w": _clean(chg_1w), "change_4w": _clean(chg_4w),
            "change_12w": _clean(latest.get("noncommercial_score_12w_change")),
            "d_long_4w": _clean(dL), "d_short_4w": _clean(dS), "d_net_4w": _clean(dNet),
            "score": _clean(latest.get("noncommercial_positioning_score")),
            "percentile": _clean(latest.get("noncommercial_percentile")),
        })
    return jsonify({"report_date": _clean(latest_date), "momentum": results})


@app.route("/api/comparison")
def api_comparison():
    err = _require_data()
    if err:
        return err
    df = CACHE.currency_df
    raw = request.args.get("currencies", "GBP,USD").split(",")
    currencies = [c.strip().upper() for c in raw if c.strip()]
    currencies = [c for c in currencies if c in engine.CONFIG["target_currencies"]]
    if not currencies:
        return jsonify({"error": "No valid currencies requested"}), 400
    period = request.args.get("period", "1y")
    filtered = _slice_period(df, period)
    filtered = filtered[filtered["currency"].isin(currencies)]

    metrics = [
        "noncommercial_positioning_score", "noncommercial_percentile",
        "noncommercial_net", "noncommercial_net_pct_oi",
    ]
    series = {}
    for cur in currencies:
        cdf = filtered[filtered["currency"] == cur].sort_values("report_date")
        entry = {"dates": cdf["report_date"].dt.strftime("%Y-%m-%d").tolist()}
        for m in metrics:
            if m in cdf.columns:
                entry[m] = [_clean(v) for v in cdf[m].tolist()]
        series[cur] = entry

    latest_date = df["report_date"].max()
    snap = df[(df["report_date"] == latest_date) & (df["currency"].isin(currencies))]
    return jsonify({
        "period": period, "currencies": currencies,
        "report_date": _clean(latest_date),
        "series": series,
        "snapshot": _df_to_records(snap),
    })


@app.route("/api/divergence")
def api_divergence():
    err = _require_data()
    if err:
        return err
    df = CACHE.currency_df
    latest_date = df["report_date"].max()
    period = request.args.get("period", "1y")
    filtered = _slice_period(df, period)

    rows = []
    for cur in engine.CONFIG["target_currencies"]:
        cdf = df[df["currency"] == cur].sort_values("report_date")
        if cdf.empty:
            continue
        latest = cdf.iloc[-1]
        strong = cdf[cdf["commercial_noncommercial_divergence"].abs() >= 80]
        rows.append({
            "currency": cur,
            "commercial_score": _clean(latest.get("commercial_positioning_score")),
            "noncommercial_score": _clean(latest.get("noncommercial_positioning_score")),
            "nonreportable_score": _clean(latest.get("nonreportable_positioning_score")),
            "divergence": _clean(latest.get("commercial_noncommercial_divergence")),
            "periods_strong_disagreement": int(len(strong)),
            "last_strong_disagreement": _clean(strong["report_date"].max()) if not strong.empty else None,
        })

    charts = {}
    for cur in engine.CONFIG["target_currencies"]:
        cdf = filtered[filtered["currency"] == cur].sort_values("report_date")
        if cdf.empty:
            continue
        charts[cur] = {
            "dates": cdf["report_date"].dt.strftime("%Y-%m-%d").tolist(),
            "commercial": [_clean(v) for v in cdf["commercial_positioning_score"].tolist()],
            "noncommercial": [_clean(v) for v in cdf["noncommercial_positioning_score"].tolist()],
            "divergence": [_clean(v) for v in cdf["commercial_noncommercial_divergence"].tolist()],
        }
    return jsonify({
        "report_date": _clean(latest_date),
        "current": rows,
        "charts": charts,
    })


@app.route("/api/market-regime")
def api_market_regime():
    err = _require_data()
    if err:
        return err
    df = CACHE.currency_df
    latest_date = df["report_date"].max()
    snap = df[df["report_date"] == latest_date].set_index("currency")

    def _score(cur):
        if cur not in snap.index:
            return None
        return _clean(snap.loc[cur, "noncommercial_positioning_score"])

    usd_score = _score("USD")
    risk_currencies = [c for c in ["AUD", "NZD", "CAD"] if c in snap.index]
    safe_currencies = [c for c in ["JPY", "CHF"] if c in snap.index]
    risk_avg = (_clean(snap.loc[risk_currencies, "noncommercial_positioning_score"].mean())
                if risk_currencies else None)
    safe_avg = (_clean(snap.loc[safe_currencies, "noncommercial_positioning_score"].mean())
                if safe_currencies else None)

    if usd_score is None:
        usd_regime = "UNKNOWN"
    elif usd_score >= 60:
        usd_regime = "STRONG USD BULLISH"
    elif usd_score >= 20:
        usd_regime = "MODERATE USD BULLISH"
    elif usd_score > -20:
        usd_regime = "NEUTRAL USD"
    elif usd_score > -60:
        usd_regime = "MODERATE USD BEARISH"
    else:
        usd_regime = "STRONG USD BEARISH"

    if risk_avg is None or safe_avg is None:
        risk_regime = "UNKNOWN"
    elif risk_avg > 20 and risk_avg > safe_avg:
        risk_regime = "RISK-ON BIAS"
    elif risk_avg < -20 and risk_avg < safe_avg:
        risk_regime = "RISK-OFF BIAS"
    else:
        risk_regime = "MIXED / RANGE"

    ranked = snap["noncommercial_positioning_score"].dropna().sort_values(ascending=False)
    leadership = [
        {"currency": c, "score": _clean(v),
         "percentile": _clean(snap.loc[c, "noncommercial_percentile"])}
        for c, v in ranked.items()
    ]

    past_date = latest_date - pd.Timedelta(days=12 * 7)
    past = df[df["report_date"] <= past_date]
    past_leadership = []
    if not past.empty:
        past_snap = past[past["report_date"] == past["report_date"].max()].set_index("currency")
        pr = past_snap["noncommercial_positioning_score"].dropna().sort_values(ascending=False)
        past_leadership = [{"currency": c, "score": _clean(v)} for c, v in pr.items()]

    rotation = []
    current_rank = {r["currency"]: i for i, r in enumerate(leadership)}
    past_rank = {r["currency"]: i for i, r in enumerate(past_leadership)}
    for c in current_rank:
        if c in past_rank:
            delta = past_rank[c] - current_rank[c]
            if abs(delta) >= 2:
                rotation.append({"currency": c, "rank_change": int(delta)})

    return jsonify({
        "report_date": _clean(latest_date),
        "usd_score": usd_score, "usd_regime": usd_regime,
        "risk_avg": risk_avg, "safe_avg": safe_avg, "risk_regime": risk_regime,
        "current_leadership": leadership,
        "past_leadership_12w": past_leadership,
        "rotation": rotation,
    })


@app.route("/api/report/currency/<code>")
def api_report_currency(code):
    err = _require_data()
    if err:
        return err
    code = code.upper()
    cdf = CACHE.currency_df[CACHE.currency_df["currency"] == code].sort_values("report_date")
    if cdf.empty:
        return jsonify({"error": "No data"}), 404
    latest = cdf.iloc[-1]
    return jsonify({
        "type": "currency", "currency": code,
        "report_date": _clean(latest["report_date"]),
        "profile": _row_to_dict(latest),
        "extremes_3y": _compute_extremes(cdf, 3),
        "extremes_5y": _compute_extremes(cdf, 5),
        "interpretation": _interpret_currency(latest),
        "methodology": (
            "Non-commercial positioning percentile is computed over a rolling "
            "5-year window from the CFTC Legacy Futures report. Positioning score = "
            "2 x percentile - 100. Extreme classification thresholds: "
            ">=60 strong long, >=80 very extreme long, symmetric for shorts. "
            "These are descriptive metrics, not forecasts."
        ),
    })


@app.route("/api/report/market")
def api_report_market():
    err = _require_data()
    if err:
        return err
    df = CACHE.currency_df
    latest_date = df["report_date"].max()
    snap = df[df["report_date"] == latest_date]
    pair_latest = CACHE.pair_df[
        CACHE.pair_df["report_date"] == CACHE.pair_df["report_date"].max()
    ].copy()
    pair_latest["abs_nc"] = pair_latest["nc_diff"].abs()
    return jsonify({
        "type": "market", "report_date": _clean(latest_date),
        "currencies": _df_to_records(snap),
        "top_pairs_long": _df_to_records(pair_latest.sort_values("nc_diff", ascending=False).head(5)),
        "top_pairs_short": _df_to_records(pair_latest.sort_values("nc_diff", ascending=True).head(5)),
        "largest_divergences": _df_to_records(pair_latest.sort_values("abs_nc", ascending=False).head(5)),
        "methodology": (
            "Market positioning report generated from CFTC Legacy Futures data. "
            "Pair differentials = base currency score - quote currency score. "
            "All values are descriptive and should not be treated as trade signals."
        ),
    })


@app.route("/api/export")
def api_export():
    err = _require_data()
    if err:
        return err
    export_type = request.args.get("type", "currencies")
    fmt = request.args.get("format", "csv")
    period = request.args.get("period", "1y")
    currencies_arg = request.args.get("currencies", "")

    if export_type == "currencies":
        df = CACHE.currency_df
        out = df[df["report_date"] == df["report_date"].max()].copy()
    elif export_type == "pairs":
        out = CACHE.pair_df[CACHE.pair_df["report_date"] == CACHE.pair_df["report_date"].max()].copy()
    elif export_type == "history":
        out = _slice_period(CACHE.currency_df, period).copy()
        if currencies_arg:
            cur_list = [c.strip().upper() for c in currencies_arg.split(",") if c.strip()]
            out = out[out["currency"].isin(cur_list)]
    elif export_type == "pairs_history":
        out = _slice_period(CACHE.pair_df, period).copy()
    else:
        return jsonify({"error": f"Unknown export type: {export_type}"}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if fmt == "csv":
        buf = io.StringIO()
        out.to_csv(buf, index=False)
        return Response(
            buf.getvalue(), mimetype="text/csv",
            headers={"Content-Disposition":
                     f"attachment; filename=cot_{export_type}_{timestamp}.csv"},
        )
    if fmt == "json":
        return jsonify({"data": _df_to_records(out)})
    return jsonify({"error": f"Unknown format: {fmt}"}), 400


# ======================================================================
# FRONTEND (embedded HTML + CSS + JS — no templates/static needed)
# ======================================================================
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FX Positioning Intelligence Terminal</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#05070d;--surface:#0a0e1a;--surface2:#10162a;--border:#1e2740;--border2:#2a3350;
--text:#e6ecf5;--muted:#8892a7;--dim:#5a647a;--cyan:#22d3ee;--violet:#a855f7;
--green:#22c55e;--red:#ef4444;--amber:#f59e0b;--glow:0 0 12px rgba(34,211,238,.25)}
html,body{height:100%;background:var(--bg);color:var(--text);
font-family:'Inter',-apple-system,sans-serif;font-size:14px;line-height:1.5;overflow:hidden}
body:before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
background:radial-gradient(ellipse 80% 50% at 20% 0%,rgba(34,211,238,.06),transparent 60%),
radial-gradient(ellipse 80% 50% at 80% 100%,rgba(168,85,247,.05),transparent 60%)}
.app{display:grid;grid-template-columns:240px 1fr;height:100vh;position:relative;z-index:1}
.sidebar{background:rgba(10,14,26,.85);backdrop-filter:blur(20px);
border-right:1px solid var(--border);display:flex;flex-direction:column;
padding:20px 0;overflow-y:auto}
.brand{padding:0 20px 20px;border-bottom:1px solid var(--border);margin-bottom:14px}
.brand-name{font-size:16px;font-weight:700;letter-spacing:.15em;color:var(--text)}
.brand-name span{color:var(--cyan)}
.brand-tag{font-size:9px;letter-spacing:.2em;color:var(--dim);margin-top:4px}
.nav{display:flex;flex-direction:column;gap:2px;padding:0 8px;flex:1}
.nav-label{font-size:9px;letter-spacing:.2em;color:var(--dim);
padding:14px 12px 6px;font-weight:600}
.nav-item{padding:9px 14px;border-radius:6px;color:var(--muted);
font-size:13px;cursor:pointer;transition:all .15s;border-left:2px solid transparent}
.nav-item:hover{background:var(--surface2);color:var(--text)}
.nav-item.active{background:linear-gradient(90deg,rgba(34,211,238,.15),rgba(168,85,247,.05));
color:var(--text);border-left-color:var(--cyan)}
.sidebar-footer{padding:14px 20px 0;border-top:1px solid var(--border);
display:flex;align-items:center;gap:8px;font-size:11px;color:var(--muted)}
.status-dot{width:7px;height:7px;border-radius:50%;background:var(--dim)}
.status-dot.ok{background:var(--green);box-shadow:0 0 6px var(--green)}
.status-dot.loading{background:var(--amber);animation:pulse 1.5s infinite}
.status-dot.err{background:var(--red)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.main{display:flex;flex-direction:column;overflow:hidden}
.topbar{height:56px;padding:0 24px;border-bottom:1px solid var(--border);
background:rgba(10,14,26,.7);backdrop-filter:blur(20px);
display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
.topbar-left{display:flex;align-items:center;gap:12px}
.topbar-title{font-size:15px;font-weight:600}
.topbar-right{display:flex;align-items:center;gap:10px}
.chip{display:flex;flex-direction:column;padding:6px 12px;
background:var(--surface);border:1px solid var(--border);border-radius:6px;font-size:11px}
.chip-label{font-size:9px;letter-spacing:.15em;color:var(--dim);margin-bottom:2px}
.btn,.btn-icon{background:var(--surface2);border:1px solid var(--border2);
color:var(--text);padding:8px 16px;border-radius:6px;font-size:12px;
cursor:pointer;font-family:inherit;transition:all .15s}
.btn:hover{border-color:var(--cyan);box-shadow:var(--glow)}
.btn-icon{padding:6px 10px;font-size:14px;display:none}
.content{flex:1;overflow-y:auto;padding:24px;position:relative}
.view{display:none}
.view.active{display:block;animation:fadein .3s}
@keyframes fadein{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.panel{background:linear-gradient(180deg,rgba(16,22,42,.7),rgba(10,14,26,.5));
backdrop-filter:blur(20px);border:1px solid var(--border);border-radius:10px;
padding:20px;margin-bottom:20px}
.panel h2{font-size:12px;letter-spacing:.1em;color:var(--cyan);
margin-bottom:16px;font-weight:500;text-transform:uppercase}
.panel-head{display:flex;justify-content:space-between;align-items:center;
margin-bottom:16px;flex-wrap:wrap;gap:10px}
.panel-head h2{margin-bottom:0}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
gap:16px;margin-bottom:20px}
.kpi{background:linear-gradient(180deg,rgba(16,22,42,.8),rgba(10,14,26,.4));
border:1px solid var(--border);border-radius:10px;padding:18px;position:relative;overflow:hidden}
.kpi:before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--border2)}
.kpi.green:before{background:var(--green)}
.kpi.red:before{background:var(--red)}
.kpi.cyan:before{background:var(--cyan)}
.kpi.violet:before{background:var(--violet)}
.kpi-label{font-size:9px;letter-spacing:.15em;color:var(--dim);margin-bottom:8px;font-weight:600}
.kpi-value{font-size:24px;font-weight:700;font-family:'JetBrains Mono',monospace;margin-bottom:4px}
.kpi-sub{font-size:11px;color:var(--muted)}
.kpi.green .kpi-value{color:var(--green)}
.kpi.red .kpi-value{color:var(--red)}
.kpi.cyan .kpi-value{color:var(--cyan)}
.kpi.violet .kpi-value{color:var(--violet)}
.data-table{width:100%;border-collapse:collapse;font-size:12px;
font-family:'JetBrains Mono',monospace}
.data-table th{text-align:left;padding:10px 12px;border-bottom:1px solid var(--border2);
font-size:10px;letter-spacing:.1em;color:var(--dim);text-transform:uppercase;
font-weight:500;font-family:'Inter',sans-serif}
.data-table td{padding:10px 12px;border-bottom:1px solid var(--border)}
.data-table tr:hover{background:rgba(34,211,238,.04)}
.data-table tr.clickable{cursor:pointer}
.pos{color:var(--green)!important}
.neg{color:var(--red)!important}
.neu{color:var(--muted)!important}
.cc{font-weight:600;color:var(--text);font-family:'JetBrains Mono',monospace}
.badge{display:inline-block;padding:3px 8px;border-radius:4px;
font-size:10px;font-weight:600;letter-spacing:.05em}
.badge.green{background:rgba(34,197,94,.15);color:var(--green)}
.badge.red{background:rgba(239,68,68,.15);color:var(--red)}
.badge.cyan{background:rgba(34,211,238,.15);color:var(--cyan)}
.badge.violet{background:rgba(168,85,247,.15);color:var(--violet)}
.badge.amber{background:rgba(245,158,11,.15);color:var(--amber)}
.badge.neu{background:rgba(136,146,167,.15);color:var(--muted)}
.toolbar{display:flex;gap:12px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
.toolbar select,.toolbar input{background:var(--surface2);
border:1px solid var(--border2);color:var(--text);padding:8px 12px;
border-radius:6px;font-family:inherit;font-size:12px}
.tabs{display:flex;gap:4px;flex-wrap:wrap}
.tab{background:var(--surface2);border:1px solid var(--border);color:var(--muted);
padding:8px 14px;border-radius:6px;cursor:pointer;font-size:12px;
font-family:inherit;font-weight:500;transition:all .15s}
.tab:hover{color:var(--text);border-color:var(--border2)}
.tab.active{background:linear-gradient(135deg,rgba(34,211,238,.2),rgba(168,85,247,.15));
border-color:var(--cyan);color:var(--text)}
.chart{width:100%;height:380px}
.loading{padding:40px;text-align:center;color:var(--muted);font-size:13px}
.error{padding:20px;background:rgba(239,68,68,.1);border:1px solid var(--red);
border-radius:8px;color:var(--red)}
.muted{color:var(--muted)}
.mini-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.mini{background:var(--surface2);border:1px solid var(--border);
border-radius:8px;padding:14px}
.mini-label{font-size:9px;letter-spacing:.15em;color:var(--dim);margin-bottom:4px}
.mini-value{font-size:18px;font-weight:700;color:var(--cyan);
font-family:'JetBrains Mono',monospace}
.mini-sub{font-size:11px;color:var(--muted);margin-top:2px}
.list{display:flex;flex-direction:column;gap:8px}
.list-item{padding:12px 16px;background:var(--surface2);
border:1px solid var(--border);border-radius:6px;
display:flex;justify-content:space-between;align-items:center;
gap:12px;font-size:12px}
.list-item.green{border-left:3px solid var(--green)}
.list-item.red{border-left:3px solid var(--red)}
.interp{background:linear-gradient(135deg,rgba(34,211,238,.05),rgba(168,85,247,.05));
border:1px solid var(--border2);border-left:3px solid var(--cyan);
border-radius:8px;padding:16px 20px;margin-bottom:20px}
.interp-label{font-size:10px;letter-spacing:.15em;color:var(--cyan);
margin-bottom:8px;font-weight:600}
.interp-text{color:var(--text);line-height:1.6;font-size:13px}
.hm-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px}
.hm-cell{padding:16px;border-radius:8px;text-align:center;
border:1px solid var(--border);transition:transform .15s}
.hm-cell:hover{transform:scale(1.03)}
.hm-code{font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:700;margin-bottom:6px}
.hm-val{font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:600}
.hm-pct{font-size:10px;opacity:.7;margin-top:2px}
.pairs-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px}
.pair-card{padding:12px;background:var(--surface2);border:1px solid var(--border);
border-radius:8px;cursor:pointer;transition:all .15s;text-align:center}
.pair-card:hover{border-color:var(--border2)}
.pair-card.selected{border-color:var(--cyan);box-shadow:var(--glow)}
.pair-card.green{border-top:2px solid var(--green)}
.pair-card.red{border-top:2px solid var(--red)}
.pair-code{font-family:'JetBrains Mono',monospace;font-weight:600;font-size:13px;margin-bottom:4px}
.pair-val{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;color:var(--cyan)}
.pair-sub{font-size:10px;color:var(--muted);margin-top:2px}
.report{background:var(--surface);border:1px solid var(--border);
border-radius:8px;padding:16px;font-family:'JetBrains Mono',monospace;
font-size:11px;max-height:520px;overflow-y:auto;color:var(--muted);
white-space:pre-wrap;line-height:1.7}
@media(max-width:900px){.app{grid-template-columns:1fr}
.sidebar{position:fixed;left:-260px;top:0;bottom:0;width:260px;z-index:100;
transition:left .25s}.sidebar.open{left:0}.btn-icon{display:block}
.content{padding:16px}.kpi-grid{grid-template-columns:repeat(2,1fr)}}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:var(--dim)}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar" id="sidebar">
    <div class="brand">
      <div class="brand-name">POSITION<span>INTEL</span></div>
      <div class="brand-tag">INSTITUTIONAL FX TERMINAL</div>
    </div>
    <nav class="nav">
      <div class="nav-label">COMMAND</div>
      <a class="nav-item active" data-page="overview">Overview</a>
      <a class="nav-item" data-page="currency">Currency Intelligence</a>
      <a class="nav-item" data-page="comparison">Currency Comparison</a>
      <a class="nav-item" data-page="historical">Historical Positioning</a>
      <div class="nav-label">SIGNALS</div>
      <a class="nav-item" data-page="extremes">Extreme Scanner</a>
      <a class="nav-item" data-page="heatmap">Positioning Heatmap</a>
      <a class="nav-item" data-page="momentum">Momentum &amp; Reversals</a>
      <a class="nav-item" data-page="flows">Institutional Flows</a>
      <div class="nav-label">ANALYSIS</div>
      <a class="nav-item" data-page="pairs">FX Pair Analysis</a>
      <a class="nav-item" data-page="divergence">Commercial vs Spec</a>
      <a class="nav-item" data-page="regime">Market Regime</a>
      <div class="nav-label">SYSTEM</div>
      <a class="nav-item" data-page="reports">Research Reports</a>
      <a class="nav-item" data-page="data">Data Center</a>
      <a class="nav-item" data-page="settings">Settings</a>
    </nav>
    <div class="sidebar-footer">
      <div class="status-dot" id="status-dot"></div>
      <span id="status-text">Loading…</span>
    </div>
  </aside>
  <div class="main">
    <header class="topbar">
      <div class="topbar-left">
        <button class="btn-icon" id="menu-toggle">&#9776;</button>
        <span class="topbar-title" id="page-title">Overview</span>
      </div>
      <div class="topbar-right">
        <div class="chip"><span class="chip-label">REPORT DATE</span><span id="report-date">—</span></div>
        <button class="btn" id="refresh-btn">Refresh</button>
      </div>
    </header>
    <main class="content" id="content"></main>
  </div>
</div>
<script>
// ============================================================
// FX Positioning Intelligence Terminal — frontend controller
// ============================================================
const CURRENCIES=['USD','EUR','GBP','JPY','CHF','CAD','AUD','NZD'];
const DARK={
  paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
  font:{color:'#e6ecf5',family:'Inter,sans-serif',size:12},
  xaxis:{gridcolor:'#1c2233',zerolinecolor:'#2a3348',linecolor:'#2a3348'},
  yaxis:{gridcolor:'#1c2233',zerolinecolor:'#2a3348',linecolor:'#2a3348'},
  margin:{l:50,r:20,t:40,b:40},
  hoverlabel:{bgcolor:'#11162a',bordercolor:'#22d3ee',font:{color:'#e6ecf5'}},
  legend:{bgcolor:'rgba(0,0,0,0)',font:{color:'#8892a7'}}
};
const DCFG={displayModeBar:false,responsive:true};
const S={page:'overview',currency:'USD',period:'1y',
  compCurs:['GBP','USD'],compPeriod:'1y',selectedPair:'EUR/USD'};

const api={
  status:()=>fetch('/api/status').then(r=>r.json()),
  overview:()=>fetch('/api/overview').then(r=>r.json()),
  currency:(c)=>fetch('/api/currency/'+c).then(r=>r.json()),
  cHist:(c,p)=>fetch('/api/currency/'+c+'/history?period='+p).then(r=>r.json()),
  pairs:()=>fetch('/api/pairs').then(r=>r.json()),
  pairHist:(p,per)=>fetch('/api/pair/'+p.replace(/\//g,'')+'/history?period='+per).then(r=>{
  if(!r.ok) return r.json().then(j=>Promise.reject(new Error(j.error||('HTTP '+r.status))));
  return r.json();
}),
  extremes:()=>fetch('/api/extremes').then(r=>r.json()),
  heatmap:(m,d)=>fetch('/api/heatmap?metric='+m+(d?'&date='+d:'')).then(r=>r.json()),
  momentum:()=>fetch('/api/momentum').then(r=>r.json()),
  comparison:(cs,p)=>fetch('/api/comparison?currencies='+cs.join(',')+'&period='+p).then(r=>r.json()),
  divergence:()=>fetch('/api/divergence').then(r=>r.json()),
  regime:()=>fetch('/api/market-regime').then(r=>r.json()),
  reportCur:(c)=>fetch('/api/report/currency/'+c).then(r=>r.json()),
  reportMarket:()=>fetch('/api/report/market').then(r=>r.json()),
  refresh:()=>fetch('/api/refresh',{method:'POST'}).then(r=>r.json())
};

const fmt=(v,d=1)=>v==null?'—':Number(v).toFixed(d);
const pct=(v)=>v==null?'—':Number(v).toFixed(0)+'%';
const esc=(s)=>String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function kpi(label,value,sub,cls){
  return `<div class="kpi ${cls||''}"><div class="kpi-label">${label}</div>
    <div class="kpi-value">${value}</div><div class="kpi-sub">${sub}</div></div>`;
}
function plot(id,traces,layout){
  const el=document.getElementById(id);if(!el)return;
  Plotly.newPlot(el,traces,Object.assign({},DARK,layout||{}),DCFG);
}

// ---------- Routing ----------
const PAGE_TITLES={
  overview:'Overview',currency:'Currency Intelligence',comparison:'Currency Comparison',
  historical:'Historical Positioning',extremes:'Extreme Scanner',heatmap:'Positioning Heatmap',
  momentum:'Momentum & Reversals',flows:'Institutional Flows',pairs:'FX Pair Analysis',
  divergence:'Commercial vs Speculative',regime:'Market Regime',reports:'Research Reports',
  data:'Data Center',settings:'Settings'
};
async function route(page){
  S.page=page;
  document.getElementById('page-title').textContent=PAGE_TITLES[page]||page;
  document.querySelectorAll('.nav-item').forEach(n=>n.classList.toggle('active',n.dataset.page===page));
  document.getElementById('sidebar').classList.remove('open');
  const content=document.getElementById('content');
  content.innerHTML='<div class="loading">Loading…</div>';
  try{
    const fn=RENDER[page];
    if(fn) await fn(content);
  }catch(e){
    content.innerHTML='<div class="error">Error: '+esc(e.message)+'</div>';
  }
}

// ========== VIEWS ==========
const RENDER={};

RENDER.overview=async(el)=>{
  const d=await api.overview();
  let h=`<div class="kpi-grid">
    ${kpi('STRONGEST',d.strongest?.currency||'—',`Score ${fmt(d.strongest?.noncommercial_positioning_score)}`,'green')}
    ${kpi('WEAKEST',d.weakest?.currency||'—',`Score ${fmt(d.weakest?.noncommercial_positioning_score)}`,'red')}
    ${kpi('CROWDED LONG',d.crowded_long?.currency||'—',`${pct(d.crowded_long?.noncommercial_percentile)} pctile`,'cyan')}
    ${kpi('CROWDED SHORT',d.crowded_short?.currency||'—',`${pct(d.crowded_short?.noncommercial_percentile)} pctile`,'violet')}
  </div>`;

  h+=`<section class="panel"><h2>Currency Positioning Matrix</h2>
    <table class="data-table"><thead><tr>
      <th>Currency</th><th>Net</th><th>Score</th><th>Percentile</th>
      <th>1W</th><th>4W</th><th>Status</th></tr></thead><tbody>`;
  for(const c of d.currencies||[]){
    const cls=c.noncommercial_positioning_score>=20?'pos':c.noncommercial_positioning_score<=-20?'neg':'neu';
    h+=`<tr class="clickable" data-cur="${c.currency}">
      <td><span class="cc">${c.currency}</span></td>
      <td class="${(c.noncommercial_net||0)>=0?'pos':'neg'}">${fmt(c.noncommercial_net,0)}</td>
      <td>${fmt(c.noncommercial_positioning_score)}</td>
      <td>${pct(c.noncommercial_percentile)}</td>
      <td>${fmt(c.noncommercial_score_1w_change)}</td>
      <td>${fmt(c.noncommercial_score_4w_change)}</td>
      <td><span class="badge ${cls}">${esc(c.noncommercial_extreme_flag||'—')}</span></td>
    </tr>`;
  }
  h+=`</tbody></table></section>`;

  const chg=(l,v)=>v?`<div class="mini"><div class="mini-label">${l}</div>
    <div class="mini-value">${v.currency}</div>
    <div class="mini-sub">${fmt(v.change,0)} contracts</div></div>`:'';
  h+=`<section class="panel"><h2>Largest Position Changes</h2>
    <div class="mini-grid">${chg('1-WEEK',d.largest_1w_change)}
      ${chg('4-WEEK',d.largest_4w_change)}${chg('12-WEEK',d.largest_12w_change)}</div></section>`;

  if((d.approaching_extremes||[]).length){
    h+=`<section class="panel"><h2>Currencies Near 3Y Extremes</h2><div class="list">`;
    for(const a of d.approaching_extremes){
      h+=`<div class="list-item ${a.direction==='bullish'?'green':'red'}">
        <span class="cc">${a.currency}</span>
        <span>${a.direction} — ${pct(a.position_in_range)} of 3Y range</span>
        <span class="muted">Score ${fmt(a.score)}</span></div>`;
    }
    h+=`</div></section>`;
  }

  if((d.interesting_pairs||[]).length){
    h+=`<section class="panel"><h2>Most Interesting Pairs</h2>
      <table class="data-table"><thead><tr><th>Pair</th><th>NC Diff</th>
      <th>Comm Diff</th><th>Status</th><th>Context</th></tr></thead><tbody>`;
    for(const p of d.interesting_pairs){
      h+=`<tr><td>${p.pair}</td>
        <td class="${(p.nc_diff||0)>=0?'pos':'neg'}">${fmt(p.nc_diff)}</td>
        <td class="${(p.comm_diff||0)>=0?'pos':'neg'}">${fmt(p.comm_diff)}</td>
        <td>${esc(p.nc_status||'—')}</td><td>${esc(p.context||'—')}</td></tr>`;
    }
    h+=`</tbody></table></section>`;
  }

  el.innerHTML=h;
  el.querySelectorAll('tr[data-cur]').forEach(tr=>{
    tr.addEventListener('click',()=>{S.currency=tr.dataset.cur;route('currency');});
  });
};

RENDER.currency=async(el)=>{
  el.innerHTML=`<div class="toolbar"><div class="tabs">
    ${CURRENCIES.map(c=>`<button class="tab ${c===S.currency?'active':''}" data-cur="${c}">${c}</button>`).join('')}
    </div></div><div id="cur-body"><div class="loading">Loading…</div></div>`;
  el.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{
    S.currency=b.dataset.cur;RENDER.currency(el);
  }));

  const info=await api.currency(S.currency);
  const p=info.latest;
  const body=el.querySelector('#cur-body');
  let h=`<div class="kpi-grid">
    ${kpi('NC SCORE',fmt(p.noncommercial_positioning_score),p.noncommercial_extreme_flag||'',
      p.noncommercial_positioning_score>=0?'green':'red')}
    ${kpi('PERCENTILE',pct(p.noncommercial_percentile),'rolling 5Y','')}
    ${kpi('NC NET',fmt(p.noncommercial_net,0),'contracts',p.noncommercial_net>=0?'green':'red')}
    ${kpi('NET % OI',fmt(p.noncommercial_net_pct_oi)+'%','of open interest','')}
    ${kpi('COMM SCORE',fmt(p.commercial_positioning_score),p.commercial_extreme_flag||'','')}
    ${kpi('NC/COMM DIV',fmt(p.commercial_noncommercial_divergence),'spread','')}
  </div>`;
  h+=`<div class="interp"><div class="interp-label">POSITIONING INTERPRETATION</div>
    <div class="interp-text">${esc(info.interpretation||'—')}</div></div>`;

  h+=`<section class="panel"><h2>Current Profile</h2>
    <table class="data-table"><thead><tr><th>Group</th><th>Long</th><th>Short</th>
    <th>Net</th><th>% OI</th><th>Pctile</th><th>Score</th><th>Status</th>
    </tr></thead><tbody>`;
  for(const g of ['commercial','noncommercial','nonreportable']){
    h+=`<tr><td>${g}</td>
      <td class="pos">${fmt(p[g+'_long'],0)}</td>
      <td class="neg">${fmt(p[g+'_short'],0)}</td>
      <td class="${(p[g+'_net']||0)>=0?'pos':'neg'}">${fmt(p[g+'_net'],0)}</td>
      <td>${fmt(p[g+'_net_pct_oi'])}%</td>
      <td>${pct(p[g+'_percentile'])}</td>
      <td>${fmt(p[g+'_positioning_score'])}</td>
      <td>${esc(p[g+'_extreme_flag']||'—')}</td></tr>`;
  }
  h+=`</tbody></table></section>`;

  h+=`<section class="panel"><div class="panel-head">
    <h2>Historical Positioning</h2><div class="toolbar">
      <select id="m-sel">
        <option value="noncommercial_positioning_score">NC Score</option>
        <option value="commercial_positioning_score">Comm Score</option>
        <option value="noncommercial_net">NC Net</option>
        <option value="commercial_net">Comm Net</option>
        <option value="noncommercial_net_pct_oi">NC Net % OI</option>
        <option value="noncommercial_percentile">NC Percentile</option>
      </select>
      <select id="p-sel">
        ${['1m','3m','6m','1y','2y','3y','5y','max'].map(pp=>
          `<option value="${pp}" ${pp===S.period?'selected':''}>${pp}</option>`).join('')}
      </select></div></div>
    <div id="h-chart" class="chart"></div></section>`;

  h+=renderExtTable(info.extremes_3y,'3-Year',S.currency);
  h+=renderExtTable(info.extremes_5y,'5-Year',S.currency);

  body.innerHTML=h;

  const draw=async(metric,period)=>{
    const hist=await api.cHist(S.currency,period);
    plot('h-chart',[{
      x:hist.dates,y:hist.series[metric],type:'scatter',mode:'lines',
      line:{color:'#22d3ee',width:2},fill:'tozeroy',
      fillcolor:'rgba(34,211,238,.08)',name:metric
    }],{title:{text:`${S.currency} — ${metric}`,font:{color:'#8892a7',size:11}}});
  };
  draw('noncommercial_positioning_score',S.period);

  body.querySelector('#m-sel').addEventListener('change',e=>{
    draw(e.target.value,body.querySelector('#p-sel').value);
  });
  body.querySelector('#p-sel').addEventListener('change',e=>{
    S.period=e.target.value;
    draw(body.querySelector('#m-sel').value,e.target.value);
  });
};

function renderExtTable(ext,label,cur){
  if(!ext||!ext.metrics) return '';
  let h=`<section class="panel"><h2>${label} Extremes — ${cur}</h2>
    <div class="muted" style="margin-bottom:10px;font-size:12px">
      Window: ${ext.window_start} → ${ext.window_end} (${ext.window_weeks} weeks)</div>
    <table class="data-table"><thead><tr><th>Metric</th><th>Current</th>
    <th>High</th><th>High Date</th><th>Low</th><th>Low Date</th>
    <th>Pos in Range</th><th>Flag</th></tr></thead><tbody>`;
  for(const[k,m]of Object.entries(ext.metrics)){
    const flag=m.is_current_high?'NEW HIGH':m.is_current_low?'NEW LOW':'';
    const cls=m.is_current_high?'green':m.is_current_low?'red':'neu';
    h+=`<tr><td>${k}</td><td>${fmt(m.current,0)}</td>
      <td>${fmt(m.high,0)}</td><td>${m.high_date||'—'}</td>
      <td>${fmt(m.low,0)}</td><td>${m.low_date||'—'}</td>
      <td>${pct(m.position_in_range)}</td>
      <td>${flag?`<span class="badge ${cls}">${flag}</span>`:'—'}</td></tr>`;
  }
  return h+'</tbody></table></section>';
}

RENDER.comparison=async(el)=>{
  if(!S.compCurs||!S.compCurs.length) S.compCurs=['GBP','USD'];
  el.innerHTML=`<div class="toolbar">
    <div class="tabs">${CURRENCIES.map(c=>
      `<button class="tab ${S.compCurs.includes(c)?'active':''}" data-cur="${c}">${c}</button>`).join('')}
    </div>
    <select id="c-p">${['3m','6m','1y','2y','3y','5y','max'].map(pp=>
      `<option value="${pp}" ${pp===S.compPeriod?'selected':''}>${pp}</option>`).join('')}
    </select></div>
    <section class="panel"><div id="c-chart" class="chart" style="height:420px"></div></section>
    <section class="panel"><h2>Snapshot</h2><div id="c-snap"></div></section>`;

  const redraw=async()=>{
    const d=await api.comparison(S.compCurs,S.compPeriod);
    const traces=S.compCurs.map(c=>({
      x:d.series[c].dates,y:d.series[c].noncommercial_positioning_score,
      type:'scatter',mode:'lines',name:c,line:{width:2}
    }));
    plot('c-chart',traces,{title:{text:'NC Positioning Score',font:{size:11,color:'#8892a7'}}});
    let h=`<table class="data-table"><thead><tr>
      <th>Currency</th><th>Score</th><th>Pctile</th><th>Net</th>
      </tr></thead><tbody>`;
    for(const s of d.snapshot){
      h+=`<tr><td class="cc">${s.currency}</td>
        <td>${fmt(s.noncommercial_positioning_score)}</td>
        <td>${pct(s.noncommercial_percentile)}</td>
        <td>${fmt(s.noncommercial_net,0)}</td></tr>`;
    }
    el.querySelector('#c-snap').innerHTML=h+'</tbody></table>';
  };

  el.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{
    const c=b.dataset.cur;
    if(S.compCurs.includes(c)){
      if(S.compCurs.length>1) S.compCurs=S.compCurs.filter(x=>x!==c);
    } else S.compCurs.push(c);
    RENDER.comparison(el);
  }));
  el.querySelector('#c-p').addEventListener('change',e=>{S.compPeriod=e.target.value;redraw();});
  redraw();
};

RENDER.historical=async(el)=>{
  el.innerHTML=`<div class="toolbar"><select id="h-p">
    ${['6m','1y','2y','3y','5y','max'].map(pp=>`<option value="${pp}">${pp}</option>`).join('')}
    </select></div><section class="panel">
    <div id="h-all" class="chart" style="height:520px"></div></section>`;
  const draw=async()=>{
    const period=el.querySelector('#h-p').value;
    const d=await api.comparison(CURRENCIES,period);
    const traces=CURRENCIES.map(c=>({
      x:d.series[c].dates,y:d.series[c].noncommercial_positioning_score,
      type:'scatter',mode:'lines',name:c,line:{width:1.5}
    }));
    plot('h-all',traces,{title:{text:'All Currencies — NC Score History',
      font:{size:11,color:'#8892a7'}}});
  };
  el.querySelector('#h-p').addEventListener('change',draw);
  draw();
};

RENDER.extremes=async(el)=>{
  const d=await api.extremes();
  let h=`<section class="panel"><h2>Historical Highs &amp; Lows (Current)</h2>`;
  if(!(d.extremes||[]).length){
    h+=`<div class="muted">No currencies currently at 3Y or 5Y extremes.</div>`;
  } else {
    h+=`<table class="data-table"><thead><tr><th>Currency</th><th>Window</th>
      <th>Metric</th><th>Type</th><th>Current</th><th>Extreme</th>
      <th>Date</th><th>Weeks</th></tr></thead><tbody>`;
    for(const e of d.extremes){
      h+=`<tr><td class="cc">${e.currency}</td><td>${e.window_years}Y</td>
        <td>${e.metric}</td>
        <td><span class="badge ${e.type==='high'?'green':'red'}">${e.type.toUpperCase()}</span></td>
        <td>${fmt(e.current,0)}</td><td>${fmt(e.extreme_value,0)}</td>
        <td>${e.extreme_date}</td><td>${e.weeks_since}</td></tr>`;
    }
    h+=`</tbody></table>`;
  }
  h+=`</section>`;

  if((d.rapid_changes||[]).length){
    h+=`<section class="panel"><h2>Rapid Changes (&gt;2σ weekly)</h2>
      <table class="data-table"><thead><tr><th>Currency</th><th>Weekly</th>
      <th>Z-Score</th><th>Std</th></tr></thead><tbody>`;
    for(const r of d.rapid_changes){
      h+=`<tr><td class="cc">${r.currency}</td>
        <td class="${r.weekly_change>=0?'pos':'neg'}">${fmt(r.weekly_change,0)}</td>
        <td>${fmt(r.z_score,2)}</td><td>${fmt(r.std,0)}</td></tr>`;
    }
    h+=`</tbody></table></section>`;
  }

  if((d.divergences||[]).length){
    h+=`<section class="panel"><h2>Divergences</h2>
      <table class="data-table"><thead><tr><th>Currency</th><th>Divergence</th>
      <th>Comm</th><th>NC</th></tr></thead><tbody>`;
    for(const r of d.divergences){
      h+=`<tr><td class="cc">${r.currency}</td>
        <td>${fmt(r.divergence)}</td>
        <td>${fmt(r.commercial_score)}</td>
        <td>${fmt(r.noncommercial_score)}</td></tr>`;
    }
    h+=`</tbody></table></section>`;
  }
  el.innerHTML=h;
};

RENDER.heatmap=async(el)=>{
  el.innerHTML=`<div class="toolbar"><select id="hm-m">
    <option value="positioning_score">Positioning Score</option>
    <option value="percentile">Percentile</option>
    <option value="net">NC Net</option>
    <option value="weekly_change">1W Change</option>
    <option value="4w_change">4W Change</option>
    <option value="12w_change">12W Change</option>
    <option value="net_pct_oi">Net % OI</option></select>
    <input type="date" id="hm-d"></div>
    <section class="panel"><h2 id="hm-t">Heatmap</h2>
    <div id="hm-g" class="hm-grid"></div></section>`;

  const draw=async()=>{
    const m=el.querySelector('#hm-m').value;
    const d=el.querySelector('#hm-d').value;
    const r=await api.heatmap(m,d);
    el.querySelector('#hm-t').textContent=m.replace(/_/g,' ')+' — '+r.date;
    const vals=(r.data||[]).map(x=>x.value).filter(v=>v!=null);
    const lo=vals.length?Math.min(...vals):0,hi=vals.length?Math.max(...vals):1;
    const rng=(hi-lo)||1;
    el.querySelector('#hm-g').innerHTML=(r.data||[]).map(x=>{
      const t=x.value==null?.5:(x.value-lo)/rng;
      const hue=t*120;
      const bg=x.value==null?'#1c2233':`hsl(${hue},60%,${24+t*20}%)`;
      return `<div class="hm-cell" style="background:${bg};color:#e6ecf5">
        <div class="hm-code">${x.currency}</div>
        <div class="hm-val">${x.value==null?'—':Number(x.value).toFixed(1)}</div>
        <div class="hm-pct">${Math.round(t*100)}%</div></div>`;
    }).join('');
  };
  el.querySelector('#hm-m').addEventListener('change',draw);
  el.querySelector('#hm-d').addEventListener('change',draw);
  draw();
};

RENDER.momentum=async(el)=>{
  const d=await api.momentum();
  let h=`<section class="panel"><h2>Positioning Momentum</h2>
    <table class="data-table"><thead><tr><th>Currency</th><th>Label</th>
    <th>Accel</th><th>1W</th><th>4W</th><th>12W</th>
    <th>ΔLong 4W</th><th>ΔShort 4W</th><th>Score</th></tr></thead><tbody>`;
  for(const m of d.momentum){
    const ac=m.acceleration==='ACCELERATING'?'green':
             m.acceleration==='REVERSING'?'amber':'neu';
    h+=`<tr><td class="cc">${m.currency}</td><td>${esc(m.label)}</td>
      <td><span class="badge ${ac}">${m.acceleration}</span></td>
      <td>${fmt(m.change_1w)}</td><td>${fmt(m.change_4w)}</td><td>${fmt(m.change_12w)}</td>
      <td class="pos">${fmt(m.d_long_4w,0)}</td>
      <td class="neg">${fmt(m.d_short_4w,0)}</td>
      <td>${fmt(m.score)}</td></tr>`;
  }
  el.innerHTML=h+'</tbody></table></section>';
};

RENDER.flows=async(el)=>{
  const d=await api.divergence();
  let h=`<section class="panel"><h2>Commercial vs Speculative Flows</h2>
    <table class="data-table"><thead><tr><th>Currency</th>
    <th>Comm</th><th>NC</th><th>Divergence</th></tr></thead><tbody>`;
  for(const r of d.current){
    h+=`<tr><td class="cc">${r.currency}</td>
      <td>${fmt(r.commercial_score)}</td>
      <td>${fmt(r.noncommercial_score)}</td>
      <td>${fmt(r.divergence)}</td></tr>`;
  }
  el.innerHTML=h+'</tbody></table></section>';
};

RENDER.pairs=async(el)=>{
  if(!S.selectedPair) S.selectedPair='EUR/USD';
  const d=await api.pairs();
  const pairs=d.pairs||[];

  // ---- summary table of all 28 pairs ----
  let h=`<section class="panel"><h2>All FX Pairs (${pairs.length})</h2>
    <table class="data-table"><thead><tr>
      <th>Pair</th><th>NC Diff</th><th>Status</th>
      <th>1W Δ</th><th>4W Δ</th><th>12W Δ</th>
      <th>Opposition</th><th>Context</th>
    </tr></thead><tbody>`;
  const sorted=[...pairs].sort((a,b)=>(b.nc_diff||0)-(a.nc_diff||0));
  for(const p of sorted){
    const cls=p.nc_diff>20?'pos':p.nc_diff<-20?'neg':'neu';
    h+=`<tr class="clickable pair-row" data-pair="${p.pair}">
      <td><span class="cc">${p.pair}</span></td>
      <td class="${cls}">${fmt(p.nc_diff)}</td>
      <td>${esc(p.nc_status||'—')}</td>
      <td>${fmt(p.nc_diff_1w_change)}</td>
      <td>${fmt(p.nc_diff_4w_change)}</td>
      <td>${fmt(p.nc_diff_12w_change)}</td>
      <td>${esc(p.opposition||'—')}</td>
      <td>${esc(p.context||'—')}</td>
    </tr>`;
  }
  h+=`</tbody></table></section>`;

  // ---- selected pair detail + chart ----
  h+=`<section class="panel">
    <div class="panel-head">
      <h2>Pair Analysis — <span id="pname">${S.selectedPair}</span></h2>
      <select id="p-per">${['3m','6m','1y','2y','3y','5y','max']
        .map(pp=>`<option value="${pp}">${pp}</option>`).join('')}</select>
    </div>
    <div id="p-details" style="margin-bottom:16px"></div>
    <div id="p-chart" class="chart" style="height:400px"></div>
    <div class="muted" style="font-size:11px;margin-top:8px">
      COT positioning measures institutional futures exposure, not a spot price forecast.</div>
  </section>`;

  el.innerHTML=h;

  // render detail table for the currently selected pair
  const renderDetails=(p)=>{
    const box=el.querySelector('#p-details');
    if(!p){ box.innerHTML='<div class="muted">No snapshot for this pair.</div>'; return; }
    box.innerHTML=`<table class="data-table"><thead><tr>
      <th>NC Diff</th><th>Comm Diff</th><th>Divergence</th>
      <th>Status</th><th>1W Δ</th><th>4W Δ</th><th>12W Δ</th>
      <th>Opposition</th><th>Context</th>
    </tr></thead><tbody><tr>
      <td class="${p.nc_diff>=0?'pos':'neg'}">${fmt(p.nc_diff)}</td>
      <td class="${p.comm_diff>=0?'pos':'neg'}">${fmt(p.comm_diff)}</td>
      <td>${fmt(p.divergence)}</td>
      <td>${esc(p.nc_status||'—')}</td>
      <td>${fmt(p.nc_diff_1w_change)}</td>
      <td>${fmt(p.nc_diff_4w_change)}</td>
      <td>${fmt(p.nc_diff_12w_change)}</td>
      <td>${esc(p.opposition||'—')}</td>
      <td>${esc(p.context||'—')}</td>
    </tr></tbody></table>`;
  };

  const currentPair=pairs.find(p=>p.pair===S.selectedPair)||pairs[0];
  if(currentPair) S.selectedPair=currentPair.pair;
  renderDetails(currentPair);
  el.querySelector('#pname').textContent=S.selectedPair;

  // draw the history chart with proper error handling
  const draw=async()=>{
    const chartDiv=el.querySelector('#p-chart');
    chartDiv.innerHTML='<div class="loading">Loading history…</div>';
    try{
      const hist=await api.pairHist(S.selectedPair,el.querySelector('#p-per').value||'1y');
      if(!hist||!hist.dates||!hist.dates.length){
        chartDiv.innerHTML='<div class="muted" style="padding:20px">No historical observations for '+S.selectedPair+' in the selected period.</div>';
        return;
      }
      chartDiv.innerHTML='';
      plot('p-chart',[
        {x:hist.dates,y:hist.series.nc_diff,name:'NC Diff',type:'scatter',
         mode:'lines',line:{color:'#22d3ee',width:2}},
        {x:hist.dates,y:hist.series.comm_diff,name:'Comm Diff',type:'scatter',
         mode:'lines',line:{color:'#a855f7',width:2}}
      ],{title:{text:S.selectedPair+' — differential history',
        font:{color:'#8892a7',size:11}}});
    }catch(e){
      chartDiv.innerHTML='<div class="error">Could not load history for '+esc(S.selectedPair)+': '+esc(e.message)+'</div>';
    }
  };
  draw();

  // wire up clicks + period change
  el.querySelectorAll('.pair-row').forEach(row=>row.addEventListener('click',()=>{
    S.selectedPair=row.dataset.pair;
    const p=pairs.find(x=>x.pair===S.selectedPair);
    renderDetails(p);
    el.querySelector('#pname').textContent=S.selectedPair;
    draw();
  }));
  el.querySelector('#p-per').addEventListener('change',draw);
};

RENDER.divergence=async(el)=>{
  const d=await api.divergence();
  let h=`<section class="panel"><h2>Current Divergence</h2>
    <table class="data-table"><thead><tr><th>Currency</th>
    <th>Comm</th><th>NC</th><th>NR</th><th>Divergence</th>
    <th>Strong Disagreement Periods</th></tr></thead><tbody>`;
  for(const r of d.current){
    const cls=r.divergence>30?'pos':r.divergence<-30?'neg':'neu';
    h+=`<tr><td class="cc">${r.currency}</td>
      <td>${fmt(r.commercial_score)}</td>
      <td>${fmt(r.noncommercial_score)}</td>
      <td>${fmt(r.nonreportable_score)}</td>
      <td class="${cls}">${fmt(r.divergence)}</td>
      <td>${r.periods_strong_disagreement}</td></tr>`;
  }
  el.innerHTML=h+'</tbody></table></section>';
};

RENDER.regime=async(el)=>{
  const d=await api.regime();
  let h=`<div class="kpi-grid">
    ${kpi('USD REGIME',esc(d.usd_regime),`Score ${fmt(d.usd_score)}`,'')}
    ${kpi('RISK REGIME',esc(d.risk_regime),`Risk avg ${fmt(d.risk_avg)}`,'')}
    ${kpi('SAFE HAVEN',fmt(d.safe_avg),'JPY + CHF avg','')}
  </div><section class="panel"><h2>Positioning Leadership</h2>
    <table class="data-table"><thead><tr><th>Rank</th><th>Currency</th>
    <th>Score</th><th>Pctile</th></tr></thead><tbody>`;
  (d.current_leadership||[]).forEach((r,i)=>{
    h+=`<tr><td>${i+1}</td><td class="cc">${r.currency}</td>
      <td>${fmt(r.score)}</td><td>${pct(r.percentile)}</td></tr>`;
  });
  h+=`</tbody></table></section>`;

  if((d.rotation||[]).length){
    h+=`<section class="panel"><h2>12-Week Rotation</h2><div class="list">`;
    for(const r of d.rotation){
      const cls=r.rank_change>0?'green':'red';
      h+=`<div class="list-item ${cls}"><span class="cc">${r.currency}</span>
        <span>${r.rank_change>0?'Improved':'Declined'} ${Math.abs(r.rank_change)} positions</span></div>`;
    }
    h+=`</div></section>`;
  }
  el.innerHTML=h;
};

RENDER.reports=async(el)=>{
  el.innerHTML=`<div class="toolbar">
    <select id="r-cur">${CURRENCIES.map(c=>`<option value="${c}">${c}</option>`).join('')}</select>
    <button class="btn" id="r-gen-cur">Currency Report</button>
    <button class="btn" id="r-gen-mkt">Market Report</button></div>
    <section class="panel"><h2>Report Output</h2>
    <div id="r-out" class="report">Select a report to generate.</div></section>`;

  el.querySelector('#r-gen-cur').addEventListener('click',async()=>{
    const c=el.querySelector('#r-cur').value;
    const r=await api.reportCur(c);
    el.querySelector('#r-out').textContent=JSON.stringify(r,null,2);
  });
  el.querySelector('#r-gen-mkt').addEventListener('click',async()=>{
    const r=await api.reportMarket();
    el.querySelector('#r-out').textContent=JSON.stringify(r,null,2);
  });
};

RENDER.data=async(el)=>{
  el.innerHTML=`<section class="panel"><h2>Data Exports</h2>
    <div class="toolbar">
      <button class="btn" data-exp="currencies">Current Currencies (CSV)</button>
      <button class="btn" data-exp="pairs">Current Pairs (CSV)</button>
      <button class="btn" data-exp="history">Historical Currencies (CSV)</button>
    </div></section>
    <section class="panel"><h2>Data Source</h2>
    <div class="muted" style="line-height:1.8">
      Source: CFTC Commitments of Traders — Legacy Futures report<br>
      8 currencies: USD, EUR, GBP, JPY, CHF, CAD, AUD, NZD<br>
      28 canonical pairs derived from base/quote positioning differentials<br>
      All values calculated by cot_engine.py — no synthetic data
    </div></section>`;
  el.querySelectorAll('[data-exp]').forEach(b=>b.addEventListener('click',()=>{
    const t=b.dataset.exp;
    window.location.href='/api/export?type='+t+'&format=csv';
  }));
};

RENDER.settings=async(el)=>{
  el.innerHTML=`<section class="panel"><h2>Application Settings</h2>
    <div class="muted" style="line-height:1.8">
      <b style="color:#e6ecf5">Host:</b> ${location.host}<br>
      <b style="color:#e6ecf5">Engine:</b> cot_engine.py (unmodified)<br>
      <b style="color:#e6ecf5">Data source:</b> CFTC Legacy Futures<br>
      <b style="color:#e6ecf5">Methodology:</b> Rolling 5Y percentile, score = 2 × pct − 100<br>
      <b style="color:#e6ecf5">Status:</b> ${CACHE_READY?'Ready':'Loading'}
    </div></section>`;

  await checkStatus();
};

// ========== Status bar ==========
let CACHE_READY=false;
async function checkStatus(){
  try{
    const s=await api.status();
    CACHE_READY=s.ready;
    const dot=document.getElementById('status-dot');
    const txt=document.getElementById('status-text');
    const rpt=document.getElementById('report-date');
    if(s.ready){
      dot.className='status-dot ok';txt.textContent='Ready';
    } else if(s.loading){
      dot.className='status-dot loading';txt.textContent='Loading…';
    } else if(s.error){
      dot.className='status-dot err';txt.textContent='Error';
    } else {
      dot.className='status-dot loading';txt.textContent='Starting…';
    }
    if(s.loaded_at) rpt.textContent=s.loaded_at.slice(0,10);
    return s;
  }catch(e){
    document.getElementById('status-dot').className='status-dot err';
    document.getElementById('status-text').textContent='Offline';
    return null;
  }
}

// ========== Boot ==========
document.querySelectorAll('.nav-item').forEach(n=>n.addEventListener('click',()=>route(n.dataset.page)));
document.getElementById('menu-toggle').addEventListener('click',()=>{
  document.getElementById('sidebar').classList.toggle('open');
});
document.getElementById('refresh-btn').addEventListener('click',async()=>{
  await api.refresh();
  setTimeout(async()=>{
    const s=await checkStatus();
    if(s&&s.ready) route(S.page);
  },1500);
});

async function boot(){
  const s=await checkStatus();
  if(s&&s.ready){
    route('overview');
  } else {
    document.getElementById('content').innerHTML=
      `<div class="loading">Engine is loading COT data — this may take a minute on first run.<br>
       <small class="muted">Polling /api/status…</small></div>`;
    const poll=setInterval(async()=>{
      const s=await checkStatus();
      if(s&&s.ready){
        clearInterval(poll);
        route('overview');
      }
    },2000);
  }
}
boot();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


# ======================================================================
# Bootstrap
# ======================================================================
def _initial_load():
    CACHE.load()


# Start the load thread at module level so it runs under gunicorn
# (imported as `Positioning`) as well as under `python Positioning.py`.
_load_thread_lock = threading.Lock()
_load_thread_started = False


def _ensure_load_started():
    global _load_thread_started
    with _load_thread_lock:
        if _load_thread_started:
            return
        _load_thread_started = True
    threading.Thread(target=_initial_load, daemon=True).start()


_ensure_load_started()


if __name__ == "__main__":
    try:
        os.chdir(_HERE)
    except Exception:
        pass
    port = int(os.environ.get("PORT", PORT))
    host = "0.0.0.0" if os.environ.get("PORT") else HOST
    print("\n  FX Positioning Intelligence Terminal")
    print(f"  -> http://{host}:{port}\n")
    app.run(host=host, port=port, debug=DEBUG,
            threaded=True, use_reloader=False)
