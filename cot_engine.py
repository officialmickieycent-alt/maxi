#!/usr/bin/env python3
"""
COT FX Positioning Engine – Professional Edition
Downloads, processes, and analyses CFTC Commitments of Traders data for
8 currencies (USD, EUR, GBP, JPY, CHF, CAD, AUD, NZD) and 28 pairs.
Uses the `cot_reports` library for data retrieval.
"""

import argparse
import logging
import os
import sys
import re
import webbrowser
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

try:
    import cot_reports as cot
except ImportError:
    raise ImportError(
        "The 'cot_reports' package is required. Install it with:\n"
        "    pip install cot_reports"
    )

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
CONFIG = {
    "target_currencies": ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"],
    "lookback_years": 5,
    "extreme_threshold": 80,
    "strong_threshold": 60,
    "moderate_threshold": 20,
    # Trajectory thresholds (4-week change based)
    "trajectory_strong_rise": 20,
    "trajectory_rise": 5,
    "trajectory_fall": -5,
    "trajectory_strong_fall": -20,
    # Pair status thresholds
    "pair_extreme_long": 120,
    "pair_strong_long": 80,
    "pair_moderate_long": 40,
    "pair_moderate_short": -40,
    "pair_strong_short": -80,
    "pair_extreme_short": -120,
    # Opposition thresholds
    "opposition_strong": 80,
    "opposition_moderate": 40,
    # Data directories
    "data_dir": "data",
    "raw_dir": "data/raw",
    "processed_dir": "data/processed",
    "output_dir": "data/output",
    "chart_dir": "charts",
    "report_type": "legacy_fut",
    "start_year": 2020,
    "end_year": 2026,
}

# ----------------------------------------------------------------------
# Canonical 28 FX pairs
# ----------------------------------------------------------------------
CANONICAL_PAIRS = [
    ("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY"), ("USD", "CHF"),
    ("AUD", "USD"), ("USD", "CAD"), ("NZD", "USD"),
    ("EUR", "GBP"), ("EUR", "JPY"), ("EUR", "CHF"), ("EUR", "AUD"),
    ("EUR", "CAD"), ("EUR", "NZD"),
    ("GBP", "JPY"), ("GBP", "CHF"), ("GBP", "AUD"), ("GBP", "CAD"), ("GBP", "NZD"),
    ("CHF", "JPY"),
    ("AUD", "JPY"), ("CAD", "JPY"), ("NZD", "JPY"),
    ("AUD", "CHF"), ("CAD", "CHF"), ("NZD", "CHF"),
    ("AUD", "CAD"), ("AUD", "NZD"), ("NZD", "CAD"),
]

# ----------------------------------------------------------------------
# Currency mapping
# ----------------------------------------------------------------------
CURRENCY_PATTERNS = {
    "EUR": ["EURO FX", "EURO"],
    "GBP": ["BRITISH POUND", "POUND STERLING", "POUND"],
    "JPY": ["JAPANESE YEN", "YEN"],
    "CHF": ["SWISS FRANC", "FRANC"],
    "CAD": ["CANADIAN DOLLAR", "CANADIAN"],
    "AUD": ["AUSTRALIAN DOLLAR", "AUSSIE"],
    "NZD": ["NZ DOLLAR", "NEW ZEALAND DOLLAR", "NZ"],
    "USD": ["USD INDEX", "U.S. DOLLAR INDEX", "USD INDEX - ICE FUTURES U.S."],
}

def map_currency(market_name: str) -> Optional[str]:
    if not isinstance(market_name, str):
        return None
    name_upper = market_name.upper()
    for currency, patterns in CURRENCY_PATTERNS.items():
        for pat in patterns:
            if pat in name_upper:
                return currency
    return None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cot_engine")

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def ensure_directories():
    for dir_path in [CONFIG["raw_dir"], CONFIG["processed_dir"], CONFIG["output_dir"], CONFIG["chart_dir"]]:
        os.makedirs(dir_path, exist_ok=True)

def get_year_list() -> List[int]:
    end_year = CONFIG["end_year"] or datetime.now().year
    if CONFIG["start_year"] is None:
        start_year = datetime.now().year - CONFIG["lookback_years"] - 1
    else:
        start_year = CONFIG["start_year"]
    return list(range(start_year, end_year + 1))

def standardize_cot_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a clean DataFrame by directly selecting the required columns
    using exact normalized column names.
    """
    df = df.copy()

    def norm(col):
        return re.sub(r'[^a-z0-9]', '', col.lower())

    norm_to_orig = {norm(c): c for c in df.columns}

    exact_mapping = {
        "market_name": ["marketandexchangenames"],
        "report_date": ["asofdateinformyyyymmdd", "asofdateinformyymmdd"],
        "open_interest": ["openinterestall"],
        "noncommercial_long": ["noncommercialpositionslongall", "noncommpositionslongall"],
        "noncommercial_short": ["noncommercialpositionsshortall", "noncommpositionsshortall"],
        "commercial_long": ["commercialpositionslongall", "commpositionslongall"],
        "commercial_short": ["commercialpositionsshortall", "commpositionsshortall"],
        "nonreportable_long": ["nonreportablepositionslongall", "nonreptpositionslongall"],
        "nonreportable_short": ["nonreportablepositionsshortall", "nonreptpositionsshortall"],
        "cftc_contract_code": ["cftccontractmarketcode", "cftccontractmarketcodequotes"],
        "cftc_market_code": ["cftcmarketcodeininitials", "cftcmarketcode"],
        "cftc_region_code": ["cftcregioncode"],
        "commodity_code": ["cftccommoditycode", "cftccommoditycodequotes"],
    }

    new_data = {}
    for internal, exact_names in exact_mapping.items():
        orig_col = None
        for exact in exact_names:
            if exact in norm_to_orig:
                orig_col = norm_to_orig[exact]
                break
        if orig_col is None:
            logger.error(f"Required column '{internal}' not found. Normalized columns: {list(norm_to_orig.keys())}")
            return pd.DataFrame()
        new_data[internal] = df[orig_col]

    clean_df = pd.DataFrame(new_data)

    clean_df["currency"] = clean_df["market_name"].apply(map_currency)
    clean_df = clean_df.dropna(subset=["currency"])
    clean_df = clean_df[clean_df["currency"].isin(CONFIG["target_currencies"])]

    clean_df["report_date"] = pd.to_datetime(clean_df["report_date"], errors="coerce")

    numeric_cols = [
        "open_interest", "noncommercial_long", "noncommercial_short",
        "commercial_long", "commercial_short", "nonreportable_long", "nonreportable_short"
    ]
    for col in numeric_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    return clean_df

def save_raw_data(df: pd.DataFrame, year: int):
    path = os.path.join(CONFIG["raw_dir"], f"cot_{year}.csv")
    df.to_csv(path, index=False)
    logger.info(f"Saved raw data for {year} to {path}")

def load_raw_data() -> pd.DataFrame:
    all_files = [os.path.join(CONFIG["raw_dir"], f) for f in os.listdir(CONFIG["raw_dir"]) if f.endswith(".csv")]
    if not all_files:
        return pd.DataFrame()
    df_list = [pd.read_csv(f, parse_dates=["report_date"]) for f in all_files]
    combined = pd.concat(df_list, ignore_index=True)
    logger.info(f"Loaded {len(combined)} raw rows from {len(all_files)} files.")
    if not combined.empty:
        logger.info(f"Currencies in raw data: {combined['currency'].unique()}")
    return combined.sort_values(["currency", "report_date"]).reset_index(drop=True)

# ----------------------------------------------------------------------
# Data Cleaning
# ----------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data...")
    df = df.drop_duplicates(subset=["currency", "report_date"], keep="last")
    df = df.sort_values(["currency", "report_date"]).reset_index(drop=True)
    numeric_cols = ["noncommercial_long", "noncommercial_short",
                    "commercial_long", "commercial_short",
                    "nonreportable_long", "nonreportable_short",
                    "open_interest"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    essential = ["noncommercial_long", "noncommercial_short",
                 "commercial_long", "commercial_short",
                 "nonreportable_long", "nonreportable_short"]
    df = df.dropna(subset=essential, how="all")
    missing = df[essential].isna().sum()
    if missing.any():
        logger.warning("Missing values in essential columns:\n%s", missing)
    dupes = df.duplicated(subset=["currency", "report_date"]).sum()
    if dupes:
        logger.error(f"Found {dupes} duplicate rows after cleaning.")
    return df

# ----------------------------------------------------------------------
# Net Position Calculations
# ----------------------------------------------------------------------
def calculate_net_positions(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating net positions...")
    df = df.copy()
    df["commercial_net"] = df["commercial_long"] - df["commercial_short"]
    df["noncommercial_net"] = df["noncommercial_long"] - df["noncommercial_short"]
    df["nonreportable_net"] = df["nonreportable_long"] - df["nonreportable_short"]
    return df

def calculate_net_changes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["commercial_net", "noncommercial_net", "nonreportable_net"]:
        df[f"{col}_change"] = df.groupby("currency")[col].diff()
    return df

# ----------------------------------------------------------------------
# Rolling Percentile Engine
# ----------------------------------------------------------------------
def calculate_rolling_percentile(df: pd.DataFrame, window_years: int = None) -> pd.DataFrame:
    if window_years is None:
        window_years = CONFIG["lookback_years"]
    logger.info(f"Calculating rolling percentiles with {window_years}-year window...")
    df = df.copy()
    window_days = window_years * 365
    for group in ["commercial", "noncommercial", "nonreportable"]:
        net_col = f"{group}_net"
        pct_col = f"{group}_percentile"
        df[pct_col] = np.nan
        for currency in df["currency"].unique():
            mask = df["currency"] == currency
            idx = df.loc[mask].index
            dates = df.loc[idx, "report_date"]
            nets = df.loc[idx, net_col]
            for i, (date, net) in enumerate(zip(dates, nets)):
                past_mask = (dates < date) & (dates >= date - pd.Timedelta(days=window_days))
                past_nets = nets[past_mask].dropna()
                if len(past_nets) > 0:
                    percentile = (past_nets < net).mean() * 100
                    df.loc[idx[i], pct_col] = percentile
    return df

# ----------------------------------------------------------------------
# Positioning Score and Extremity
# ----------------------------------------------------------------------
def calculate_positioning_scores(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating positioning scores...")
    df = df.copy()
    for group in ["commercial", "noncommercial", "nonreportable"]:
        pct_col = f"{group}_percentile"
        score_col = f"{group}_positioning_score"
        df[score_col] = 2 * df[pct_col] - 100
    return df

def classify_extreme(score: float) -> str:
    if pd.isna(score):
        return "UNKNOWN"
    if score >= CONFIG["extreme_threshold"]:
        return "VERY EXTREME LONG"
    elif score >= CONFIG["strong_threshold"]:
        return "STRONG LONG"
    elif score >= CONFIG["moderate_threshold"]:
        return "MODERATE LONG"
    elif score > -CONFIG["moderate_threshold"]:
        return "NEUTRAL"
    elif score > -CONFIG["strong_threshold"]:
        return "MODERATE SHORT"
    elif score > -CONFIG["extreme_threshold"]:
        return "STRONG SHORT"
    else:
        return "VERY EXTREME SHORT"

def calculate_extreme_flags(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Classifying extremity...")
    df = df.copy()
    for group in ["commercial", "noncommercial", "nonreportable"]:
        score_col = f"{group}_positioning_score"
        flag_col = f"{group}_extreme_flag"
        df[flag_col] = df[score_col].apply(classify_extreme)
    return df

# ----------------------------------------------------------------------
# Divergence (currency-level)
# ----------------------------------------------------------------------
def calculate_divergences(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating divergences...")
    df = df.copy()
    df["commercial_noncommercial_divergence"] = (
        df["noncommercial_positioning_score"] - df["commercial_positioning_score"]
    )
    df["nonreportable_noncommercial_divergence"] = (
        df["nonreportable_positioning_score"] - df["noncommercial_positioning_score"]
    )
    df["nonreportable_commercial_divergence"] = (
        df["nonreportable_positioning_score"] - df["commercial_positioning_score"]
    )
    return df

# ----------------------------------------------------------------------
# Trajectory Calculations (per currency)
# ----------------------------------------------------------------------
def calculate_score_trajectory(df: pd.DataFrame, periods=[1, 4, 12]) -> pd.DataFrame:
    """
    For each group, compute the change in positioning score over
    the specified number of weeks using shift.
    """
    logger.info(f"Calculating score trajectories for periods {periods}...")
    df = df.copy()
    df = df.sort_values(["currency", "report_date"]).reset_index(drop=True)
    for group in ["commercial", "noncommercial", "nonreportable"]:
        score_col = f"{group}_positioning_score"
        for period in periods:
            new_col = f"{group}_score_{period}w_change"
            prev = df.groupby("currency")[score_col].shift(period)
            df[new_col] = df[score_col] - prev
    return df

def classify_trajectory(change_4w: float) -> str:
    if pd.isna(change_4w):
        return "N/A"
    if change_4w > CONFIG["trajectory_strong_rise"]:
        return "STRONGLY RISING"
    elif change_4w > CONFIG["trajectory_rise"]:
        return "RISING"
    elif change_4w > CONFIG["trajectory_fall"]:
        return "STABLE"
    elif change_4w > CONFIG["trajectory_strong_fall"]:
        return "FALLING"
    else:
        return "STRONGLY FALLING"

# ----------------------------------------------------------------------
# Net as % of OI (safe division)
# ----------------------------------------------------------------------
def calculate_net_pct_oi(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating net as % of OI...")
    df = df.copy()
    for group in ["commercial", "noncommercial", "nonreportable"]:
        net_col = f"{group}_net"
        pct_col = f"{group}_net_pct_oi"
        df[pct_col] = df[net_col] / df["open_interest"].replace(0, np.nan) * 100
    return df

# ----------------------------------------------------------------------
# Pair Engine
# ----------------------------------------------------------------------
def generate_all_pairs(currencies: List[str]) -> List[Dict[str, str]]:
    """
    Return the fixed canonical list of 28 pairs as dictionaries.
    """
    pairs = []
    for base, quote in CANONICAL_PAIRS:
        pairs.append({
            "pair": f"{base}/{quote}",
            "base_currency": base,
            "quote_currency": quote,
            "category": "major" if base == "USD" or quote == "USD" else "cross"
        })
    return pairs

def generate_pair_history(df_currency: pd.DataFrame, canonical_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Generate a time series of pair differentials for all canonical pairs.
    """
    logger.info("Generating pair history...")
    score_cols = ["noncommercial_positioning_score", "commercial_positioning_score", "nonreportable_positioning_score"]
    pivot = df_currency.pivot_table(index="report_date", columns="currency", values=score_cols, aggfunc="last")
    records = []
    for base, quote in canonical_pairs:
        for score_type in score_cols:
            if (score_type, base) not in pivot.columns or (score_type, quote) not in pivot.columns:
                logger.warning(f"Missing score data for {base}/{quote} {score_type}")
                continue
            diff_series = pivot[(score_type, base)] - pivot[(score_type, quote)]
            temp = pd.DataFrame({
                "pair": f"{base}/{quote}",
                "base_currency": base,
                "quote_currency": quote,
                "score_type": score_type,
                "report_date": pivot.index,
                "differential": diff_series.values
            })
            records.append(temp)
    if not records:
        return pd.DataFrame()
    pair_history = pd.concat(records, ignore_index=True)
    pair_wide = pair_history.pivot_table(index=["pair", "base_currency", "quote_currency", "report_date"],
                                         columns="score_type", values="differential", aggfunc="last").reset_index()
    pair_wide = pair_wide.rename(columns={
        "noncommercial_positioning_score": "nc_diff",
        "commercial_positioning_score": "comm_diff",
        "nonreportable_positioning_score": "nr_diff"
    })
    pair_wide = pair_wide.sort_values(["pair", "report_date"]).reset_index(drop=True)
    return pair_wide

def calculate_pair_trajectory(pair_history: pd.DataFrame, periods=[1, 4, 12]) -> pd.DataFrame:
    df = pair_history.copy()
    for prefix in ["nc", "comm", "nr"]:
        diff_col = f"{prefix}_diff"
        for period in periods:
            new_col = f"{prefix}_diff_{period}w_change"
            prev = df.groupby("pair")[diff_col].shift(period)
            df[new_col] = df[diff_col] - prev
    df["divergence"] = df["nc_diff"] - df["comm_diff"]
    for period in periods:
        prev_div = df.groupby("pair")["divergence"].shift(period)
        df[f"divergence_{period}w_change"] = df["divergence"] - prev_div
    return df

# ----------------------------------------------------------------------
# Classification Helpers (for pair status)
# ----------------------------------------------------------------------
def classify_pair_status(diff: float) -> str:
    if pd.isna(diff):
        return "N/A"
    if diff >= CONFIG["pair_extreme_long"]:
        return "EXTREME LONG"
    elif diff >= CONFIG["pair_strong_long"]:
        return "STRONG LONG"
    elif diff >= CONFIG["pair_moderate_long"]:
        return "MODERATE LONG"
    elif diff > CONFIG["pair_moderate_short"]:
        return "NEUTRAL"
    elif diff > CONFIG["pair_strong_short"]:
        return "MODERATE SHORT"
    elif diff > CONFIG["pair_extreme_short"]:
        return "STRONG SHORT"
    else:
        return "EXTREME SHORT"

def classify_opposition(nc_diff: float, comm_diff: float) -> str:
    if pd.isna(nc_diff) or pd.isna(comm_diff):
        return "UNKNOWN"
    nc_long = nc_diff > 0
    nc_short = nc_diff < 0
    comm_long = comm_diff > 0
    comm_short = comm_diff < 0
    if (nc_long and comm_long) or (nc_short and comm_short):
        return "ALIGNMENT"
    abs_comm = abs(comm_diff)
    if abs_comm >= CONFIG["opposition_strong"]:
        return "STRONG OPPOSITION"
    elif abs_comm >= CONFIG["opposition_moderate"]:
        return "MODERATE OPPOSITION"
    else:
        return "LOW OPPOSITION"

def classify_positioning_context(row: pd.Series) -> str:
    nc_diff = row.get("nc_diff", np.nan)
    nc_4w = row.get("nc_diff_4w_change", np.nan)
    opposition = row.get("opposition", "UNKNOWN")

    if pd.isna(nc_diff):
        return "NEUTRAL"

    nc_abs = abs(nc_diff)
    nc_extreme = nc_abs >= CONFIG["pair_strong_long"]

    nc_traj = classify_trajectory(nc_4w)

    if nc_extreme and nc_traj in ["FALLING", "STRONGLY FALLING"] and opposition in ["STRONG OPPOSITION", "MODERATE OPPOSITION"]:
        return "REVERSAL-WATCH"
    if nc_extreme and nc_traj in ["FALLING", "STRONGLY FALLING"]:
        return "EXTREME / WATCH"
    if nc_traj in ["RISING", "STRONGLY RISING"] and opposition in ["ALIGNMENT", "LOW OPPOSITION"]:
        return "CONTINUATION-FAVORABLE"
    if nc_traj in ["RISING", "STRONGLY RISING"] and opposition == "MODERATE OPPOSITION":
        return "CONTINUATION-FAVORABLE"
    return "MIXED"

# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def validate_data(df: pd.DataFrame) -> List[str]:
    errors = []
    logger.info("Running validation checks...")
    if df.empty:
        errors.append("DataFrame is empty.")
        return errors
    if set(df["currency"].unique()) != set(CONFIG["target_currencies"]):
        errors.append(f"Missing currencies: {set(CONFIG['target_currencies']) - set(df['currency'].unique())}")
    dupes = df.duplicated(subset=["currency", "report_date"]).sum()
    if dupes:
        errors.append(f"Duplicate rows: {dupes}")
    for group in ["commercial", "noncommercial", "nonreportable"]:
        long_col = f"{group}_long"
        short_col = f"{group}_short"
        net_col = f"{group}_net"
        mask = ~(df[long_col].isna() | df[short_col].isna())
        if not np.allclose(df.loc[mask, net_col], df.loc[mask, long_col] - df.loc[mask, short_col], equal_nan=True):
            errors.append(f"{group}_net mismatch")
    for group in ["commercial", "noncommercial", "nonreportable"]:
        pct_col = f"{group}_percentile"
        if (df[pct_col] < 0).any() or (df[pct_col] > 100).any():
            errors.append(f"{pct_col} out of range")
        score_col = f"{group}_positioning_score"
        if (df[score_col] < -100).any() or (df[score_col] > 100).any():
            errors.append(f"{score_col} out of range")
    # Canonical pair validation
    pairs = generate_all_pairs(CONFIG["target_currencies"])
    if len(pairs) != 28:
        errors.append(f"Pair count incorrect: expected 28, got {len(pairs)}")
    pair_set = set()
    for p in pairs:
        pair_str = p["pair"]
        rev_pair = f"{p['quote_currency']}/{p['base_currency']}"
        if pair_str in pair_set or rev_pair in pair_set:
            errors.append(f"Duplicate or reversed pair: {pair_str}")
        pair_set.add(pair_str)
    canonical_set = {f"{b}/{q}" for b, q in CANONICAL_PAIRS}
    for p in pairs:
        if p["pair"] not in canonical_set:
            errors.append(f"Non-canonical pair: {p['pair']}")
    pair_counts = {}
    for p in pairs:
        pair_counts[p["base_currency"]] = pair_counts.get(p["base_currency"], 0) + 1
        pair_counts[p["quote_currency"]] = pair_counts.get(p["quote_currency"], 0) + 1
    for cur in CONFIG["target_currencies"]:
        if pair_counts.get(cur, 0) != 7:
            errors.append(f"{cur} appears {pair_counts.get(cur, 0)} times, expected 7")
    return errors

# ----------------------------------------------------------------------
# Reporting / Output
# ----------------------------------------------------------------------
def generate_outputs(df: pd.DataFrame):
    logger.info("Generating output files...")
    df.to_csv(os.path.join(CONFIG["processed_dir"], "cot_currency_weekly.csv"), index=False)
    try:
        df_for_parquet = df.copy()
        for col in df_for_parquet.columns:
            if df_for_parquet[col].dtype == object:
                df_for_parquet[col] = df_for_parquet[col].astype(str)
        df_for_parquet.to_parquet(os.path.join(CONFIG["processed_dir"], "cot_currency_weekly.parquet"), index=False)
    except Exception as e:
        logger.warning(f"Could not write parquet file: {e}")

def print_currency_table(latest_currency: pd.DataFrame):
    print("\n" + "="*100)
    print("1. CURRENCY POSITIONING")
    print("="*100)
    cols = [
        "currency",
        "noncommercial_positioning_score", "noncommercial_extreme_flag",
        "noncommercial_score_1w_change", "noncommercial_score_4w_change", "noncommercial_score_12w_change",
        "commercial_positioning_score", "commercial_extreme_flag",
        "commercial_score_1w_change", "commercial_score_4w_change", "commercial_score_12w_change",
        "commercial_noncommercial_divergence",
        "nonreportable_positioning_score", "nonreportable_score_1w_change", "nonreportable_score_4w_change"
    ]
    for col in cols:
        if col not in latest_currency.columns:
            latest_currency[col] = np.nan
    display = latest_currency[cols].rename(columns={
        "noncommercial_positioning_score": "NC Score",
        "noncommercial_extreme_flag": "NC Status",
        "noncommercial_score_1w_change": "NC 1W",
        "noncommercial_score_4w_change": "NC 4W",
        "noncommercial_score_12w_change": "NC 12W",
        "commercial_positioning_score": "Comm Score",
        "commercial_extreme_flag": "Comm Status",
        "commercial_score_1w_change": "Comm 1W",
        "commercial_score_4w_change": "Comm 4W",
        "commercial_score_12w_change": "Comm 12W",
        "commercial_noncommercial_divergence": "NC/Comm Div",
        "nonreportable_positioning_score": "NR Score",
        "nonreportable_score_1w_change": "NR 1W",
        "nonreportable_score_4w_change": "NR 4W",
    })
    print(display.to_string(index=False, float_format=lambda x: f"{x:+.1f}"))

def print_pair_table(pair_df: pd.DataFrame, title: str, sort_by: Optional[str] = None, ascending: bool = False):
    print("\n" + "="*100)
    print(title)
    print("="*100)
    if sort_by:
        pair_df = pair_df.sort_values(sort_by, ascending=ascending)
    display_cols = [
        "pair", "nc_diff", "nc_status", "nc_diff_1w_change", "nc_diff_4w_change", "nc_diff_12w_change",
        "nc_trajectory", "comm_diff", "comm_status", "comm_diff_1w_change", "comm_diff_4w_change", "comm_diff_12w_change",
        "comm_trajectory", "divergence", "divergence_4w_change", "nr_diff", "nr_status", "opposition", "context"
    ]
    for col in display_cols:
        if col not in pair_df.columns:
            pair_df[col] = "N/A"
    print(pair_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:+.1f}"))

def print_dashboard(df_currency_latest: pd.DataFrame, pair_latest: pd.DataFrame):
    """Print professional multi‑section dashboard."""
    print("\n" + "="*100)
    print("COT POSITIONING ANALYSIS")
    print(f"Report Date: {df_currency_latest['report_date'].max().date()}")
    print("="*100)

    print_currency_table(df_currency_latest)

    print_pair_table(pair_latest, "2. STRONGEST NC LONG PAIRS", sort_by="nc_diff", ascending=False)
    print_pair_table(pair_latest, "3. STRONGEST NC SHORT PAIRS", sort_by="nc_diff", ascending=True)

    cont = pair_latest[pair_latest["context"] == "CONTINUATION-FAVORABLE"].sort_values("nc_diff", ascending=False)
    print_pair_table(cont, "4. CONTINUATION-FAVORABLE CONFIGURATIONS")

    rev = pair_latest[pair_latest["context"] == "REVERSAL-WATCH"].sort_values("nc_diff", ascending=False)
    print_pair_table(rev, "5. REVERSAL-WATCH CONFIGURATIONS")

    pair_latest["abs_divergence"] = pair_latest["divergence"].abs()
    print_pair_table(pair_latest, "6. LARGEST NC/COMMERCIAL DIVERGENCES", sort_by="abs_divergence", ascending=False)

    print_pair_table(pair_latest, "7. FULL 28-PAIR POSITIONING MATRIX", sort_by="pair", ascending=True)

    print("\n" + "="*100)
    print("8. MARKET POSITIONING SUMMARY")
    print("="*100)
    top_long_cur = df_currency_latest.loc[df_currency_latest["noncommercial_positioning_score"].idxmax()]
    top_short_cur = df_currency_latest.loc[df_currency_latest["noncommercial_positioning_score"].idxmin()]
    print(f"* {top_long_cur['currency']} has the strongest NC positioning ({top_long_cur['noncommercial_positioning_score']:+.1f}) and trajectory {classify_trajectory(top_long_cur.get('noncommercial_score_4w_change', np.nan))}.")
    print(f"* {top_short_cur['currency']} has the weakest NC positioning ({top_short_cur['noncommercial_positioning_score']:+.1f}) and trajectory {classify_trajectory(top_short_cur.get('noncommercial_score_4w_change', np.nan))}.")
    max_div_pair = pair_latest.loc[pair_latest["abs_divergence"].idxmax()]
    print(f"* {max_div_pair['pair']} shows the largest NC/Commercial divergence ({max_div_pair['divergence']:+.1f}).")
    print("\n(These are descriptive observations, not trade recommendations.)")

# ----------------------------------------------------------------------
# HTML Report Generation (dark theme, whole numbers)
# ----------------------------------------------------------------------
def generate_html_dashboard(currency_latest: pd.DataFrame, pair_latest: pd.DataFrame, output_path: str):
    """Generate a compact, dark-themed HTML dashboard and save to output_path."""
    # Prepare currency table
    currency_cols = [
        "currency",
        "noncommercial_positioning_score", "noncommercial_extreme_flag",
        "noncommercial_score_1w_change", "noncommercial_score_4w_change", "noncommercial_score_12w_change",
        "commercial_positioning_score", "commercial_extreme_flag",
        "commercial_score_1w_change", "commercial_score_4w_change", "commercial_score_12w_change",
        "commercial_noncommercial_divergence",
        "nonreportable_positioning_score", "nonreportable_score_1w_change", "nonreportable_score_4w_change"
    ]
    for col in currency_cols:
        if col not in currency_latest.columns:
            currency_latest[col] = np.nan
    currency_display = currency_latest[currency_cols].rename(columns={
        "noncommercial_positioning_score": "NC Score",
        "noncommercial_extreme_flag": "NC Status",
        "noncommercial_score_1w_change": "NC 1W",
        "noncommercial_score_4w_change": "NC 4W",
        "noncommercial_score_12w_change": "NC 12W",
        "commercial_positioning_score": "Comm Score",
        "commercial_extreme_flag": "Comm Status",
        "commercial_score_1w_change": "Comm 1W",
        "commercial_score_4w_change": "Comm 4W",
        "commercial_score_12w_change": "Comm 12W",
        "commercial_noncommercial_divergence": "NC/Comm Div",
        "nonreportable_positioning_score": "NR Score",
        "nonreportable_score_1w_change": "NR 1W",
        "nonreportable_score_4w_change": "NR 4W",
    })

    # Pair table columns
    pair_cols = [
        "pair", "nc_diff", "nc_status", "nc_diff_1w_change", "nc_diff_4w_change", "nc_diff_12w_change",
        "nc_trajectory", "comm_diff", "comm_status", "comm_diff_1w_change", "comm_diff_4w_change", "comm_diff_12w_change",
        "comm_trajectory", "divergence", "divergence_4w_change", "nr_diff", "nr_status", "opposition", "context"
    ]
    for col in pair_cols:
        if col not in pair_latest.columns:
            pair_latest[col] = "N/A"

    # Create pair subsets
    strongest_nc_long = pair_latest.sort_values("nc_diff", ascending=False)
    strongest_nc_short = pair_latest.sort_values("nc_diff", ascending=True)
    cont = pair_latest[pair_latest["context"] == "CONTINUATION-FAVORABLE"].sort_values("nc_diff", ascending=False)
    rev = pair_latest[pair_latest["context"] == "REVERSAL-WATCH"].sort_values("nc_diff", ascending=False)
    pair_latest["abs_divergence"] = pair_latest["divergence"].abs()
    largest_div = pair_latest.sort_values("abs_divergence", ascending=False)
    full_matrix = pair_latest.sort_values("pair", ascending=True)

    # Summary text
    top_long_cur = currency_latest.loc[currency_latest["noncommercial_positioning_score"].idxmax()]
    top_short_cur = currency_latest.loc[currency_latest["noncommercial_positioning_score"].idxmin()]
    max_div_pair = pair_latest.loc[pair_latest["abs_divergence"].idxmax()]
    summary_text = f"""
    <ul>
        <li>{top_long_cur['currency']} has the strongest NC positioning ({top_long_cur['noncommercial_positioning_score']:+.0f}) and trajectory {classify_trajectory(top_long_cur.get('noncommercial_score_4w_change', np.nan))}.</li>
        <li>{top_short_cur['currency']} has the weakest NC positioning ({top_short_cur['noncommercial_positioning_score']:+.0f}) and trajectory {classify_trajectory(top_short_cur.get('noncommercial_score_4w_change', np.nan))}.</li>
        <li>{max_div_pair['pair']} shows the largest NC/Commercial divergence ({max_div_pair['divergence']:+.0f}).</li>
        <li><em>These are descriptive observations, not trade recommendations.</em></li>
    </ul>
    """

    # Helper to convert DataFrame to HTML with integer formatting
    def df_to_html(df, columns):
        return df[columns].to_html(index=False, classes='dataframe', float_format=lambda x: f"{x:.0f}")

    # Build HTML with dark theme
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>COT Positioning Analysis</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #111;
                color: #ddd;
                margin: 20px;
            }}
            h1, h2 {{ color: #f0f0f0; }}
            table.dataframe {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 30px;
                font-size: 12px;
                border: 1px solid #444;
            }}
            table.dataframe th, table.dataframe td {{
                border: 1px solid #444;
                padding: 4px 6px;
                text-align: left;
                white-space: nowrap;
            }}
            table.dataframe th {{
                background-color: #222;
                color: #fff;
            }}
            table.dataframe tr:nth-child(even) {{
                background-color: #1a1a1a;
            }}
            table.dataframe tr:hover {{
                background-color: #333;
            }}
            .section {{ margin-bottom: 40px; }}
            .summary {{
                background-color: #1e2a38;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #5b9bd5;
            }}
            .summary ul {{ margin: 0; padding-left: 20px; }}
        </style>
    </head>
    <body>
        <h1>COT POSITIONING ANALYSIS</h1>
        <p><strong>Report Date:</strong> {currency_latest['report_date'].max().date()}</p>

        <div class="section">
            <h2>1. Currency Positioning</h2>
            {df_to_html(currency_display, currency_display.columns)}
        </div>

        <div class="section">
            <h2>2. Strongest NC Long Pairs</h2>
            {df_to_html(strongest_nc_long, pair_cols)}
        </div>

        <div class="section">
            <h2>3. Strongest NC Short Pairs</h2>
            {df_to_html(strongest_nc_short, pair_cols)}
        </div>

        <div class="section">
            <h2>4. Continuation-Favorable Configurations</h2>
            {df_to_html(cont, pair_cols)}
        </div>

        <div class="section">
            <h2>5. Reversal-Watch Configurations</h2>
            {df_to_html(rev, pair_cols)}
        </div>

        <div class="section">
            <h2>6. Largest NC/Commercial Divergences</h2>
            {df_to_html(largest_div, pair_cols)}
        </div>

        <div class="section">
            <h2>7. Full 28-Pair Positioning Matrix</h2>
            {df_to_html(full_matrix, pair_cols)}
        </div>

        <div class="section">
            <h2>8. Market Positioning Summary</h2>
            <div class="summary">{summary_text}</div>
        </div>
    </body>
    </html>
    """

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"HTML dashboard saved to {output_path}")

def open_html_dashboard(output_path: str):
    """Open the HTML dashboard in the default browser."""
    if os.path.exists(output_path):
        try:
            if sys.platform == "win32":
                os.startfile(output_path)
            else:
                webbrowser.open('file://' + os.path.realpath(output_path))
            logger.info(f"Opened dashboard in browser: {output_path}")
        except Exception as e:
            logger.error(f"Could not open browser: {e}")
    else:
        logger.error(f"HTML dashboard not found: {output_path}")

# ----------------------------------------------------------------------
# Main Processing Pipeline
# ----------------------------------------------------------------------
def process_data():
    ensure_directories()
    raw_df = load_raw_data()
    if raw_df.empty:
        logger.error("No raw data found. Run 'download' first.")
        return pd.DataFrame(), pd.DataFrame()
    cleaned = clean_data(raw_df)
    df = calculate_net_positions(cleaned)
    df = calculate_net_changes(df)
    df = calculate_rolling_percentile(df)
    df = calculate_positioning_scores(df)
    df = calculate_extreme_flags(df)
    df = calculate_divergences(df)
    df = calculate_score_trajectory(df, periods=[1, 4, 12])
    df = calculate_net_pct_oi(df)

    errors = validate_data(df)
    if errors:
        logger.error("Validation errors:")
        for e in errors:
            logger.error(f"  - {e}")
    else:
        logger.info("Validation passed.")

    generate_outputs(df)

    pair_history = generate_pair_history(df, CANONICAL_PAIRS)
    if not pair_history.empty:
        pair_history = calculate_pair_trajectory(pair_history, periods=[1, 4, 12])
        pair_history["nc_status"] = pair_history["nc_diff"].apply(classify_pair_status)
        pair_history["comm_status"] = pair_history["comm_diff"].apply(classify_pair_status)
        pair_history["nr_status"] = pair_history["nr_diff"].apply(classify_pair_status)
        pair_history["opposition"] = pair_history.apply(lambda r: classify_opposition(r["nc_diff"], r["comm_diff"]), axis=1)
        pair_history["nc_trajectory"] = pair_history["nc_diff_4w_change"].apply(classify_trajectory)
        pair_history["comm_trajectory"] = pair_history["comm_diff_4w_change"].apply(classify_trajectory)
        pair_history["context"] = pair_history.apply(classify_positioning_context, axis=1)

        latest_date = pair_history["report_date"].max()
        pair_latest = pair_history[pair_history["report_date"] == latest_date].copy()
    else:
        pair_latest = pd.DataFrame()

    latest_currency_date = df["report_date"].max()
    currency_latest = df[df["report_date"] == latest_currency_date].copy()
    currency_latest["nc_trajectory"] = currency_latest["noncommercial_score_4w_change"].apply(classify_trajectory)
    currency_latest["comm_trajectory"] = currency_latest["commercial_score_4w_change"].apply(classify_trajectory)

    return currency_latest, pair_latest

# ----------------------------------------------------------------------
# Downloader using cot_reports
# ----------------------------------------------------------------------
def download_all():
    """
    Download CFTC data using cot_reports.

    Workflow:
      - Past years are downloaded once and then cached.
      - The current year is ALWAYS refreshed so the latest weekly report
        is fetched every time this function runs.
    """
    ensure_directories()
    current_year = datetime.now().year

    for year in get_year_list():
        save_path = os.path.join(CONFIG["raw_dir"], f"cot_{year}.csv")

        # If it's a past year and the file already exists, skip it.
        if os.path.exists(save_path) and year != current_year:
            logger.info(f"Raw data for {year} already exists, skipping.")
            continue

        # If it's the current year, delete stale cache and re-download.
        if year == current_year and os.path.exists(save_path):
            logger.info(f"Refreshing current year's data ({year})...")
            try:
                os.remove(save_path)
            except Exception as e:
                logger.warning(f"Could not remove stale file {save_path}: {e}")

        logger.info(f"Fetching COT data for {year} using cot_reports...")
        try:
            try:
                df = cot.cot_year(year, cot_report_type=CONFIG["report_type"], store_txt=False)
            except TypeError:
                df = cot.cot_year(year, cot_report_type=CONFIG["report_type"])
        except Exception as e:
            logger.error(f"Failed to fetch {year}: {e}")
            continue

        if os.path.exists("annual.txt"):
            os.remove("annual.txt")

        if df is None or df.empty:
            logger.warning(f"No data returned for {year}")
            continue
        df = standardize_cot_df(df)
        if not df.empty:
            save_raw_data(df, year)
            logger.info(f"Saved {len(df)} rows for {year}")
        else:
            logger.warning(f"No target currency data for {year}")

# ----------------------------------------------------------------------
# CLI Entry Point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="COT FX Positioning Engine - Professional")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Common arguments for subcommands that can generate HTML
    html_parser = argparse.ArgumentParser(add_help=False)
    html_parser.add_argument("--html", action="store_true", help="Generate HTML dashboard and open in browser")

    # Download
    subparsers.add_parser("download", help="Download CFTC data")
    # Process
    parser_process = subparsers.add_parser("process", parents=[html_parser], help="Process raw data")
    # Update
    parser_update = subparsers.add_parser("update", parents=[html_parser], help="Download and process")
    # Dashboard
    parser_dashboard = subparsers.add_parser("dashboard", parents=[html_parser], help="Show latest dashboard")
    # Validate
    subparsers.add_parser("validate", help="Run validation only")
    # All
    parser_all = subparsers.add_parser("all", parents=[html_parser], help="Complete pipeline (download, process, dashboard)")

    # Additional options for start/end year and lookback
    for p in [parser_process, parser_update, parser_dashboard, parser_all]:
        p.add_argument("--start-year", type=int, help="Override start year")
        p.add_argument("--end-year", type=int, help="Override end year")
        p.add_argument("--lookback", type=int, help="Override lookback years")

    args, _ = parser.parse_known_args()

    # Safely check for optional attributes (fixes AttributeError for 'download')
    if getattr(args, "start_year", None):
        CONFIG["start_year"] = args.start_year
    if getattr(args, "end_year", None):
        CONFIG["end_year"] = args.end_year
    if getattr(args, "lookback", None):
        CONFIG["lookback_years"] = args.lookback

    if args.command == "download":
        download_all()
    elif args.command == "process":
        currency_latest, pair_latest = process_data()
        if getattr(args, "html", False):
            html_path = os.path.join(CONFIG["output_dir"], "cot_dashboard.html")
            generate_html_dashboard(currency_latest, pair_latest, html_path)
            open_html_dashboard(html_path)
    elif args.command == "update":
        download_all()
        currency_latest, pair_latest = process_data()
        if getattr(args, "html", False):
            html_path = os.path.join(CONFIG["output_dir"], "cot_dashboard.html")
            generate_html_dashboard(currency_latest, pair_latest, html_path)
            open_html_dashboard(html_path)
    elif args.command == "dashboard":
        processed_file = os.path.join(CONFIG["processed_dir"], "cot_currency_weekly.csv")
        if os.path.exists(processed_file):
            df = pd.read_csv(processed_file, parse_dates=["report_date"])
            pair_history = generate_pair_history(df, CANONICAL_PAIRS)
            pair_history = calculate_pair_trajectory(pair_history, periods=[1,4,12])
            pair_history["nc_status"] = pair_history["nc_diff"].apply(classify_pair_status)
            pair_history["comm_status"] = pair_history["comm_diff"].apply(classify_pair_status)
            pair_history["nr_status"] = pair_history["nr_diff"].apply(classify_pair_status)
            pair_history["opposition"] = pair_history.apply(lambda r: classify_opposition(r["nc_diff"], r["comm_diff"]), axis=1)
            pair_history["nc_trajectory"] = pair_history["nc_diff_4w_change"].apply(classify_trajectory)
            pair_history["comm_trajectory"] = pair_history["comm_diff_4w_change"].apply(classify_trajectory)
            pair_history["context"] = pair_history.apply(classify_positioning_context, axis=1)
            latest_date = df["report_date"].max()
            currency_latest = df[df["report_date"] == latest_date].copy()
            pair_latest = pair_history[pair_history["report_date"] == latest_date].copy()
            print_dashboard(currency_latest, pair_latest)
            if getattr(args, "html", False):
                html_path = os.path.join(CONFIG["output_dir"], "cot_dashboard.html")
                generate_html_dashboard(currency_latest, pair_latest, html_path)
                open_html_dashboard(html_path)
        else:
            logger.error("No processed data found. Run 'process' first.")
    elif args.command == "validate":
        processed_file = os.path.join(CONFIG["processed_dir"], "cot_currency_weekly.csv")
        if os.path.exists(processed_file):
            df = pd.read_csv(processed_file, parse_dates=["report_date"])
            errors = validate_data(df)
            if errors:
                print("Validation errors:")
                for e in errors:
                    print(f" - {e}")
            else:
                print("All validation checks passed.")
        else:
            logger.error("No processed data found.")
    elif args.command == "all":
        download_all()
        currency_latest, pair_latest = process_data()
        print_dashboard(currency_latest, pair_latest)
        if getattr(args, "html", False):
            html_path = os.path.join(CONFIG["output_dir"], "cot_dashboard.html")
            generate_html_dashboard(currency_latest, pair_latest, html_path)
            open_html_dashboard(html_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()