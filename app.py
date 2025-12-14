# app.py
import os
import io
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# Optional solver (recommended)
try:
    import pulp  # type: ignore
    HAS_PULP = True
except Exception:
    pulp = None
    HAS_PULP = False


# ----------------------------
# Config / Constants
# ----------------------------

SALARY_CAP = 50000

ROSTER_SLOTS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]

POS_REQUIRED = {
    "QB": 1,
    "DST": 1,
}
POS_MIN = {"RB": 2, "WR": 3, "TE": 1}
POS_MAX = {"RB": 3, "WR": 4, "TE": 2}

SKILL_POS = {"QB", "RB", "WR", "TE"}  # not DST


@dataclass
class UIState:
    num_lineups: int = 50
    min_salary: int = 48000

    # Build rules
    require_qb_stack: bool = True
    require_bring_back: bool = False
    bring_back_games: int = 1
    no_te_in_flex: bool = False
    avoid_opp_dst: bool = True

    # Pool / rules textareas
    pool_raw: str = ""
    dnu_raw: str = ""
    required_combo_raw: str = ""
    required_combo_min: int = 1
    restrict_to_pool: bool = False  # only restrict if explicitly enabled

    # Simulation / payout thresholds
    auto_payout: bool = False
    cash_line: float = 135.0
    top1_line: float = 165.0
    win_line: float = 185.0
    cash_quantile: float = 0.60
    top1_quantile: float = 0.99
    win_quantile: float = 0.999

    # Monte Carlo
    sim_trials: int = 6000

    # EV/ROI payout knobs (lightweight, not exact DK)
    entry_fee: float = 20.0
    prize_cash: float = 30.0
    prize_top1: float = 200.0
    prize_win: float = 5000.0

    # Sorting
    sort_by: str = "ev"  # ev/roi/p50/p95/proj/stars


@dataclass
class SimConfig:
    # number of Monte Carlo trials per lineup
    trials: int = 6000

    # lineup quantile for p95
    ceiling_quantile: float = 0.95

    # top X% of trials used for per-player ceiling contributions
    tail_share: float = 0.05

    # how much scripts push outcomes
    script_strength: float = 0.55

    # additional team-level correlation (QB + pass catchers, etc.)
    team_corr_pass_sigma: float = 0.16
    team_corr_rush_sigma: float = 0.13
    team_corr_power: float = 0.65


# ----------------------------
# Helpers: parsing
# ----------------------------

def _infer_column(cols: List[str], candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    # fuzzy
    for c in cols:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def parse_pool_caps(pool_raw: str) -> Dict[str, float]:
    """
    pool_raw lines: Name|max%
    returns {Name: max_pct}
    """
    caps: Dict[str, float] = {}
    if not pool_raw:
        return caps
    for line in pool_raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        name = parts[0].strip()
        if not name:
            continue
        pct = 100.0
        if len(parts) > 1:
            try:
                pct = float(parts[1])
            except Exception:
                pct = 100.0
        caps[name] = max(0.0, min(100.0, pct))
    return caps


def parse_comma_list(txt: str) -> List[str]:
    if not txt:
        return []
    out = []
    for t in txt.split(","):
        nm = t.strip()
        if nm:
            out.append(nm)
    return out


def infer_teams_from_game(game: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Accepts formats like:
      "ARI@HOU", "ARI vs HOU", "ARI-HOU", "ARI/HOU"
    Returns (team_a, team_b) if detected else (None, None)
    """
    if not game:
        return (None, None)
    g = str(game).strip().upper()
    # normalize separators
    for sep in [" VS ", " V ", " @ ", "@", " VS.", " - ", "-", "/", "|"]:
        if sep.strip() in g:
            # handle both "@" and " vs " styles
            if "@" in g:
                parts = g.split("@")
            elif "VS" in g:
                parts = g.replace("VS.", "VS").split("VS")
            elif "-" in g:
                parts = g.split("-")
            elif "/" in g:
                parts = g.split("/")
            elif "|" in g:
                parts = g.split("|")
            else:
                parts = [g]
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) >= 2:
                return (parts[0][:3], parts[1][:3])
            break
    # last-chance regex 3 letters + nonletters + 3 letters
    import re
    m = re.search(r"([A-Z]{2,4})[^A-Z]+([A-Z]{2,4})", g)
    if m:
        return (m.group(1)[:3], m.group(2)[:3])
    return (None, None)


def canonical_game_key(team_a: str, team_b: str) -> str:
    a = (team_a or "").upper().strip()[:3]
    b = (team_b or "").upper().strip()[:3]
    if not a or not b:
        return ""
    # canonical, order independent
    return " vs ".join(sorted([a, b]))


def load_projections_from_upload(file_storage) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Optional[str]]:
    """
    Returns (df, players_preview, warn)
    df always has: PlayerName, Team, Pos, Salary, Projection, Game (optional), Opp (computed), VegasSpread, VegasTotal
    """
    if not file_storage:
        return pd.DataFrame(), [], "No file uploaded."

    raw = file_storage.read()
    if not raw:
        return pd.DataFrame(), [], "Uploaded file was empty."

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        return pd.DataFrame(), [], f"Could not read CSV: {e}"

    cols = list(df.columns)

    col_player = _infer_column(cols, ["Player", "PlayerName", "Name"])
    col_team = _infer_column(cols, ["Team", "Tm"])
    col_pos = _infer_column(cols, ["Pos", "Position"])
    col_salary = _infer_column(cols, ["Salary", "Sal"])
    col_proj = _infer_column(cols, ["Projection", "Proj", "Fpts", "Points"])
    col_opp = _infer_column(cols, ["Opp", "Opponent"])
    col_game = _infer_column(cols, ["Game", "Matchup"])

    missing = []
    for req, col in [("Player", col_player), ("Team", col_team), ("Pos", col_pos), ("Salary", col_salary), ("Projection", col_proj)]:
        if not col:
            missing.append(req)
    if missing:
        return pd.DataFrame(), [], f"Missing required columns: {', '.join(missing)}"

    out = pd.DataFrame()
    out["PlayerName"] = df[col_player].astype(str).str.strip()
    out["Team"] = df[col_team].astype(str).str.strip().str.upper().str[:3]
    out["Pos"] = df[col_pos].astype(str).str.strip().str.upper()
    out["Salary"] = pd.to_numeric(df[col_salary], errors="coerce").fillna(0).astype(int)
    out["Projection"] = pd.to_numeric(df[col_proj], errors="coerce").fillna(0.0).astype(float)

    if col_game:
        out["GameRaw"] = df[col_game].astype(str).fillna("").str.strip()
    else:
        out["GameRaw"] = ""

    if col_opp:
        out["OppRaw"] = df[col_opp].astype(str).fillna("").str.strip().str.upper().str[:3]
    else:
        out["OppRaw"] = ""

    # Normalize games + opp
    team_a_list = []
    team_b_list = []
    game_key_list = []
    opp_list = []

    for _, r in out.iterrows():
        game_raw = str(r["GameRaw"] or "")
        ta, tb = infer_teams_from_game(game_raw)
        if ta and tb:
            gk = canonical_game_key(ta, tb)
            opp = tb if r["Team"] == ta else (ta if r["Team"] == tb else (r["OppRaw"] or ""))
        else:
            # try OppRaw
            if r["OppRaw"]:
                ta = r["Team"]
                tb = r["OppRaw"]
                gk = canonical_game_key(ta, tb)
                opp = tb
            else:
                ta = r["Team"]
                tb = ""
                gk = ""
                opp = ""
        team_a_list.append(ta or "")
        team_b_list.append(tb or "")
        game_key_list.append(gk)
        opp_list.append(opp or "")

    out["GameKey"] = game_key_list
    out["Opp"] = opp_list

    # Build a nicer display string if possible
    out["Game"] = out.apply(lambda r: (r["GameKey"] if r["GameKey"] else ""), axis=1)

    # Clean positions into DFS buckets
    out.loc[out["Pos"].isin(["D", "DEF", "DST"]), "Pos"] = "DST"

    # Drop any rows missing essentials
    out = out[(out["PlayerName"] != "") & (out["Team"] != "") & (out["Salary"] > 0)]

    # Default Vegas columns (filled later)
    out["VegasSpread"] = np.nan
    out["VegasTotal"] = np.nan
    out["FavTeam"] = ""
    out["DogTeam"] = ""

    out = out.reset_index(drop=True)

    # Preview list sorted by projection desc
    players_preview = out.sort_values("Projection", ascending=False).head(250).to_dict(orient="records")
    return out, players_preview, None


# ----------------------------
# DraftKings salaries mapping (optional, for DK upload export)
# ----------------------------

import re as _re

_SUFFIX_RE = _re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", _re.IGNORECASE)


def normalize_player_name(name: str) -> str:
    """
    Normalize a player name for matching across projection feeds and DK salaries.
    - lowercases
    - strips punctuation
    - strips common suffixes (Jr, Sr, II, III...)
    """
    s = (name or "").strip().lower()
    s = _SUFFIX_RE.sub("", s)
    s = _re.sub(r"[^a-z0-9 ]+", "", s)
    s = _re.sub(r"\s+", " ", s).strip()
    return s


def make_player_key(name: str, team: str, pos: str, salary: int) -> str:
    return f"{(name or '').strip()}|{(team or '').strip().upper()[:3]}|{(pos or '').strip().upper()}|{int(salary or 0)}"


def load_dk_salaries_from_upload(file_storage) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """
    Reads a DK Salaries CSV and returns (dk_map, warn).

    dk_map keys:
      - for non-DST:  "<norm_name>|<TEAM>" -> {id, name_id, name, team, pos}
      - for DST:      "DST|<TEAM>"        -> {id, name_id, name, team, pos}

    We keep the mapping intentionally lightweight to embed into the page.
    """
    if not file_storage:
        return {}, None

    raw = file_storage.read()
    if not raw:
        return {}, None

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        return {}, f"Could not read DK salaries CSV: {e}"

    # expected columns (but tolerate variants)
    cols = list(df.columns)
    col_name = _infer_column(cols, ["Name", "PlayerName"])
    col_name_id = _infer_column(cols, ["Name + ID", "Name+ID"])
    col_id = _infer_column(cols, ["ID", "Player ID", "PlayerId"])
    col_team = _infer_column(cols, ["TeamAbbrev", "Team", "TeamAbbrev "])
    col_pos = _infer_column(cols, ["Position", "Pos"])

    missing = []
    for req, col in [("Name", col_name), ("ID", col_id), ("Team", col_team), ("Position", col_pos)]:
        if not col:
            missing.append(req)
    if missing:
        return {}, f"DK salaries CSV missing required columns: {', '.join(missing)}"

    dk_map: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        name = str(r.get(col_name) or "").strip()
        team = str(r.get(col_team) or "").strip().upper()[:3]
        pos = str(r.get(col_pos) or "").strip().upper()
        try:
            pid = str(int(r.get(col_id)))
        except Exception:
            pid = str(r.get(col_id) or "").strip()

        if not name or not team or not pid:
            continue

        name_id = str(r.get(col_name_id) or "").strip()
        if not name_id:
            name_id = f"{name} ({pid})"

        # DST rows on DK are typically like "49ers" with Position DST and TeamAbbrev SF
        if pos in ("DST", "D", "DEF", "D/ST"):
            key = f"DST|{team}"
        else:
            key = f"{normalize_player_name(name)}|{team}"

        dk_map[key] = {
            "id": pid,
            "name_id": name_id,
            "name": name,
            "team": team,
            "pos": pos,
        }

    if not dk_map:
        return {}, "DK salaries CSV loaded, but no rows could be parsed."

    return dk_map, None


def lookup_dk_player(dk_map: Dict[str, Dict[str, Any]], name: str, team: str, pos: str) -> Optional[Dict[str, Any]]:
    if not dk_map:
        return None
    team3 = (team or "").strip().upper()[:3]
    posu = (pos or "").strip().upper()

    if posu == "DST":
        return dk_map.get(f"DST|{team3}")

    key = f"{normalize_player_name(name)}|{team3}"
    hit = dk_map.get(key)
    if hit:
        return hit

    # fallback: try name only if unique (rarely safe, but helps some feeds)
    nkey = normalize_player_name(name)
    candidates = [v for k, v in dk_map.items() if k.startswith(nkey + "|")]
    if len(candidates) == 1:
        return candidates[0]
    return None




def build_dk_export_map_for_results(results: Optional[List[Dict[str, Any]]], dk_map: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Build a lightweight mapping for client-side DK upload export.

    We intentionally only include players that appear in the *currently rendered* results,
    rather than embedding the entire DK slate mapping into the page.

    Keys match JS: PlayerName|Team|Pos|Salary
    """
    dk_export_map: Dict[str, Dict[str, Any]] = {}
    dk_stats: Dict[str, int] = {"mapped": 0, "missing": 0}

    if not results or not dk_map:
        return dk_export_map, dk_stats

    seen: Set[str] = set()
    for r in results:
        for p in (r.get("players") or []):
            key = make_player_key(p.get("PlayerName"), p.get("Team"), p.get("Pos"), int(p.get("Salary") or 0))
            if key in seen:
                continue
            seen.add(key)
            hit = lookup_dk_player(dk_map, p.get("PlayerName"), p.get("Team"), p.get("Pos"))
            if hit:
                dk_export_map[key] = {"id": hit.get("id"), "name_id": hit.get("name_id")}
                dk_stats["mapped"] += 1
            else:
                dk_stats["missing"] += 1

    return dk_export_map, dk_stats

def apply_projection_adjustments(df: pd.DataFrame, adj_map: Dict[str, float]) -> pd.DataFrame:
    """Apply +/- percentage adjustments to Projection based on key Player|Team|Pos|Salary."""
    if df is None or df.empty or not adj_map:
        return df

    out = df.copy()
    out["PlayerKey"] = out["PlayerName"].astype(str) + "|" + out["Team"].astype(str) + "|" + out["Pos"].astype(str) + "|" + out["Salary"].astype(str)
    out["AdjPct"] = out["PlayerKey"].map(adj_map).fillna(0.0).astype(float)
    out.loc[:, "Projection"] = out["Projection"].astype(float) * (1.0 + out["AdjPct"] / 100.0)
    return out


# ----------------------------
# Vegas odds / game scripts
# ----------------------------

DEFAULT_SCRIPT_WEIGHTS = {
    "fav_pct": 28.0,
    "dog_pct": 22.0,
    "shootout_pct": 25.0,
    "slog_pct": 15.0,
    "weird_pct": 10.0,
}

def load_vegas_odds() -> Dict[str, Dict[str, Any]]:
    """
    Reads vegas_odds.json if present.
    Expected structure: { "ARI vs HOU": { "spread": -3.5, "total": 47.5, "fav_team": "ARI", "dog_team":"HOU", ... } }
    Keys can be any matchup string; we attempt to normalize.
    """
    path = None
    for p in ["vegas_odds.json", "static/vegas_odds.json"]:
        if os.path.exists(p):
            path = p
            break
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            # allow list of objects with "Game" key
            out: Dict[str, Dict[str, Any]] = {}
            for row in raw:
                if not isinstance(row, dict):
                    continue
                g = row.get("Game") or row.get("game") or ""
                if not g:
                    continue
                out[str(g)] = row
            raw = out
        if not isinstance(raw, dict):
            return {}
        return raw
    except Exception:
        return {}

def _normalize_vegas_key(key: str) -> str:
    ta, tb = infer_teams_from_game(key)
    if ta and tb:
        return canonical_game_key(ta, tb)
    return str(key).strip()

def merge_vegas_overrides(base: Dict[str, Dict[str, Any]], overrides: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    overrides keyed by canonical game key (e.g., "ARI vs HOU").
    """
    if not overrides:
        return base
    out = {k: dict(v) for k, v in base.items()}
    # map normalized base keys
    norm_map: Dict[str, str] = {}
    for k in list(out.keys()):
        norm_map[_normalize_vegas_key(k)] = k

    for gk, ov in overrides.items():
        if not isinstance(ov, dict):
            continue
        target_key = norm_map.get(gk) or gk
        existing = out.get(target_key, {})
        merged = dict(existing)
        merged.update(ov)
        out[target_key] = merged
    return out

def build_games_summary(vegas_odds: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    for k, info in vegas_odds.items():
        gk = _normalize_vegas_key(k)
        spread = float(info.get("spread") or 0.0)
        total = float(info.get("total") or 44.0)
        fav_team = str(info.get("fav_team") or info.get("favorite") or "").upper().strip()[:3]
        dog_team = str(info.get("dog_team") or info.get("underdog") or "").upper().strip()[:3]
        # fallback derive favorite/dog from spread sign if teams unknown
        if not (fav_team and dog_team):
            ta, tb = infer_teams_from_game(k)
            if ta and tb:
                # if spread is negative, assume first team is favorite (common format),
                # else assume first team is favorite anyway (we just need labels).
                fav_team = ta
                dog_team = tb

        row = {
            "Game": gk,
            "spread": spread,
            "total": total,
            "fav_team": fav_team,
            "dog_team": dog_team,
        }
        for w in DEFAULT_SCRIPT_WEIGHTS:
            row[w] = float(info.get(w) or DEFAULT_SCRIPT_WEIGHTS[w])
        summary.append(row)

    # stable display: higher totals first
    summary.sort(key=lambda r: (-(r.get("total") or 0.0), abs(r.get("spread") or 0.0)))
    return summary

def build_game_config_map(df: pd.DataFrame, vegas_odds: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Returns map canonical game key -> config dict (spread,total,teams,script weights)
    Also fills df Vegas columns in-place when possible.
    """
    if df.empty:
        return {}

    # Normalize vegas keys -> info
    vegas_norm: Dict[str, Dict[str, Any]] = {}
    for k, info in vegas_odds.items():
        vegas_norm[_normalize_vegas_key(k)] = info

    game_cfg: Dict[str, Dict[str, Any]] = {}

    for gk in df["GameKey"].dropna().unique().tolist():
        gk = str(gk).strip()
        if not gk:
            continue
        info = vegas_norm.get(gk, {})
        spread = float(info.get("spread") or 0.0)
        total = float(info.get("total") or 44.0)

        ta, tb = infer_teams_from_game(gk)
        fav_team = str(info.get("fav_team") or info.get("favorite") or ta or "").upper().strip()[:3]
        dog_team = str(info.get("dog_team") or info.get("underdog") or tb or "").upper().strip()[:3]

        cfg = {
            "Game": gk,
            "spread": spread,
            "total": total,
            "fav_team": fav_team,
            "dog_team": dog_team,
            "fav_pct": float(info.get("fav_pct") or DEFAULT_SCRIPT_WEIGHTS["fav_pct"]),
            "dog_pct": float(info.get("dog_pct") or DEFAULT_SCRIPT_WEIGHTS["dog_pct"]),
            "shootout_pct": float(info.get("shootout_pct") or DEFAULT_SCRIPT_WEIGHTS["shootout_pct"]),
            "slog_pct": float(info.get("slog_pct") or DEFAULT_SCRIPT_WEIGHTS["slog_pct"]),
            "weird_pct": float(info.get("weird_pct") or DEFAULT_SCRIPT_WEIGHTS["weird_pct"]),
        }
        game_cfg[gk] = cfg

    # Fill per-player vegas columns
    if game_cfg:
        spread_map = {gk: cfg["spread"] for gk, cfg in game_cfg.items()}
        total_map = {gk: cfg["total"] for gk, cfg in game_cfg.items()}
        fav_map = {gk: cfg["fav_team"] for gk, cfg in game_cfg.items()}
        dog_map = {gk: cfg["dog_team"] for gk, cfg in game_cfg.items()}
        df.loc[:, "VegasSpread"] = df["GameKey"].map(spread_map)
        df.loc[:, "VegasTotal"] = df["GameKey"].map(total_map)
        df.loc[:, "FavTeam"] = df["GameKey"].map(fav_map).fillna("")
        df.loc[:, "DogTeam"] = df["GameKey"].map(dog_map).fillna("")

    return game_cfg


# ----------------------------
# Player p95 estimate (used by sim + optimizer)
# ----------------------------

def estimate_player_p95(row: pd.Series) -> float:
    proj = float(row.get("Projection") or 0.0)
    if proj <= 0:
        return 0.0
    pos = str(row.get("Pos") or "").upper()
    total = float(row.get("VegasTotal") or 44.0)
    team = str(row.get("Team") or "")
    fav = str(row.get("FavTeam") or "")
    dog = str(row.get("DogTeam") or "")
    is_fav = fav and team and fav == team

    # Base ratio: cap the studs a bit, let lower-proj guys be spikier.
    if pos == "QB":
        ratio = 1.35 if proj >= 18 else 1.45
        if total >= 50:
            ratio += 0.05
        ratio = min(ratio, 1.55)
    elif pos in ("WR", "TE"):
        if proj >= 20:
            ratio = 1.38
        elif proj >= 12:
            ratio = 1.55
        elif proj >= 8:
            ratio = 1.95
        else:
            ratio = 2.60  # allow low-proj WR/TE to have a real TD+yards ceiling
        if total >= 50:
            ratio += 0.07
        if total <= 40:
            ratio -= 0.05
        ratio = min(max(ratio, 1.25), 2.85)
    elif pos == "RB":
        if proj >= 20:
            ratio = 1.35
        elif proj >= 12:
            ratio = 1.50
        elif proj >= 8:
            ratio = 1.85
        else:
            ratio = 2.30
        if is_fav:
            ratio += 0.05
        if total >= 50:
            ratio += 0.05
        if total <= 40:
            ratio -= 0.05
        ratio = min(max(ratio, 1.20), 2.60)
    elif pos == "DST":
        ratio = 1.85 if proj >= 6 else 2.20
        ratio = min(max(ratio, 1.40), 2.70)
    else:
        ratio = 1.50

    p95 = proj * ratio

    # Hard-ish caps to avoid truly absurd single-player outputs
    if pos == "QB":
        p95 = min(p95, 44.0)
    elif pos in ("WR", "TE", "RB"):
        p95 = min(p95, 48.0 if proj >= 18 else 42.0 if proj >= 10 else 30.0)
    elif pos == "DST":
        p95 = min(p95, 25.0)

    return max(proj + 0.5, p95)  # ensure > proj slightly


# ----------------------------
# Simulation (Monte Carlo with game scripts)
# ----------------------------

def _z95() -> float:
    return 1.6448536269514722

def _lognormal_sigma_for_mean_and_p95(mean: float, p95: float) -> float:
    """
    Returns a moderate sigma solution (smaller root) for a lognormal
    with given mean and 95th percentile. If infeasible or mean<=0, returns 0.6
    """
    if mean <= 0 or p95 <= 0:
        return 0.6
    ratio = p95 / mean
    ratio = max(1.01, min(ratio, 3.75))
    z = _z95()
    # mean=1 simplifies; but keep general
    A = math.log(mean)
    B = math.log(p95)
    # equation: 0.5*s^2 - z*s + (B-A) = 0
    disc = z*z - 2.0*(B - A)
    if disc <= 1e-9:
        return 0.9
    s = z - math.sqrt(disc)  # smaller root => less insane tails
    if s <= 1e-6:
        s = 0.25
    return float(min(max(s, 0.12), 1.10))

def _lognormal_mu_for_mean(mean: float, sigma: float) -> float:
    if mean <= 0:
        return -10.0
    return math.log(mean) - 0.5 * sigma * sigma


def sample_lineup_with_scripts(
    lineup: List[Dict[str, Any]],
    game_cfg_map: Dict[str, Dict[str, Any]],
    sim_cfg: SimConfig,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns (totals, contrib_matrix[nTrials x nPlayers], scripts_by_game)

    - Player distributions are lognormal calibrated to (Projection mean, P95 estimate)
    - Per-trial game scripts shift outcomes (fav leads / dog leads / shootout / slog / weird)
    - A small team-level latent factor adds correlation (QB ↔ WR/TE, RB ↔ game control)
    """
    n = len(lineup)
    T = int(sim_cfg.trials)
    if n == 0:
        return np.zeros(T), np.zeros((T, 0)), {}

    base_proj = np.array([float(p.get("Projection") or 0.0) for p in lineup], dtype=float)
    base_p95 = np.array([float(p.get("P95") or max(p.get("Projection", 0.0) * 1.5, 0.0)) for p in lineup], dtype=float)
    pos = [str(p.get("Pos") or "").upper() for p in lineup]
    team = [str(p.get("Team") or "").upper()[:3] for p in lineup]
    game = [str(p.get("GameKey") or "") for p in lineup]

    # Precompute per-player lognormal params
    sigmas = np.zeros(n, dtype=float)
    mus = np.zeros(n, dtype=float)
    caps = np.zeros(n, dtype=float)

    for i in range(n):
        mean = max(0.1, base_proj[i])
        p95 = max(mean + 0.5, base_p95[i])
        sigma = _lognormal_sigma_for_mean_and_p95(mean, p95)
        mu = _lognormal_mu_for_mean(mean, sigma)
        sigmas[i] = sigma
        mus[i] = mu

        # Cap: keep extreme outliers under control
        cap = p95 * (1.35 if pos[i] == "QB" else 1.45)
        if pos[i] == "QB":
            cap = min(cap, 50.0)
        elif pos[i] in ("WR", "TE", "RB"):
            cap = min(cap, 55.0)
        elif pos[i] == "DST":
            cap = min(cap, 28.0)
        caps[i] = max(p95, cap)

    # Script + team multipliers
    mult = np.ones((T, n), dtype=float)

    SCRIPT_LABELS = {
        0: "Fav leads",
        1: "Dog leads",
        2: "Shootout",
        3: "Low & slow",
        4: "Weird",
    }

    scripts_by_game: Dict[str, np.ndarray] = {}

    # Group players by game to apply shared script selection
    game_to_idxs: Dict[str, List[int]] = {}
    for i, g in enumerate(game):
        game_to_idxs.setdefault(g, []).append(i)

    # Mean-1 lognormal helper (so we can multiply without shifting mean too much)
    def _mean1_lognormal(sigma: float, size: int) -> np.ndarray:
        sigma = float(max(0.001, sigma))
        mu = -0.5 * sigma * sigma
        return rng.lognormal(mean=mu, sigma=sigma, size=size)

    for gk, idxs in game_to_idxs.items():
        cfg = game_cfg_map.get(gk, None)
        if not cfg:
            continue

        # weights
        w = np.array([
            float(cfg.get("fav_pct") or 0.0),
            float(cfg.get("dog_pct") or 0.0),
            float(cfg.get("shootout_pct") or 0.0),
            float(cfg.get("slog_pct") or 0.0),
            float(cfg.get("weird_pct") or 0.0),
        ], dtype=float)
        if w.sum() <= 0:
            w = np.array([28, 22, 25, 15, 10], dtype=float)
        w = w / w.sum()

        # sample script id per trial for this game
        script_id = rng.choice(5, size=T, p=w)
        scripts_by_game[gk] = script_id

        fav = str(cfg.get("fav_team") or "").upper()[:3]
        dog = str(cfg.get("dog_team") or "").upper()[:3]

        # multiplier templates by script
        # index: 0=fav leads, 1=dog leads, 2=shootout, 3=slog, 4=weird
        def pos_mult(p: str, is_fav: bool, sid: int) -> float:
            if sid == 2:  # shootout
                base = {"QB": 1.12, "WR": 1.10, "TE": 1.08, "RB": 1.04, "DST": 0.92}
            elif sid == 0:  # fav leads
                base = {"QB": 0.98, "WR": 0.95, "TE": 0.96, "RB": 1.10, "DST": 1.08}
            elif sid == 1:  # dog leads
                base = {"QB": 1.06, "WR": 1.05, "TE": 1.03, "RB": 0.98, "DST": 1.02}
            elif sid == 3:  # slog
                base = {"QB": 0.90, "WR": 0.88, "TE": 0.90, "RB": 1.02, "DST": 1.06}
            else:  # weird
                base = {"QB": 1.00, "WR": 1.00, "TE": 1.00, "RB": 1.00, "DST": 1.00}

            m = base.get(p, 1.00)
            if p == "RB" and is_fav and sid == 0:
                m *= 1.03
            if p in ("QB", "WR", "TE") and (not is_fav) and sid in (1, 2):
                m *= 1.02
            return float(m)

        # Team correlation (mean ~1) — adds correlation between QB and pass-catchers
        pass_fav = _mean1_lognormal(sim_cfg.team_corr_pass_sigma, T)
        pass_dog = _mean1_lognormal(sim_cfg.team_corr_pass_sigma, T)
        rush_fav = _mean1_lognormal(sim_cfg.team_corr_rush_sigma, T)
        rush_dog = _mean1_lognormal(sim_cfg.team_corr_rush_sigma, T)

        # Script nudges on team factors
        # (kept small to avoid blowing up tails; these are *additional* to pos_mult)
        for sid in range(5):
            mask = script_id == sid
            if not np.any(mask):
                continue
            if sid == 2:  # shootout
                pass_fav[mask] *= 1.05
                pass_dog[mask] *= 1.05
                rush_fav[mask] *= 1.02
                rush_dog[mask] *= 1.02
            elif sid == 3:  # slog
                pass_fav[mask] *= 0.93
                pass_dog[mask] *= 0.93
                rush_fav[mask] *= 0.97
                rush_dog[mask] *= 0.97
            elif sid == 0:  # fav leads
                rush_fav[mask] *= 1.06
                pass_fav[mask] *= 0.96
                pass_dog[mask] *= 1.02
            elif sid == 1:  # dog leads
                pass_dog[mask] *= 1.06
                pass_fav[mask] *= 1.03
                rush_fav[mask] *= 0.97

        # fill multipliers
        for j in idxs:
            t = team[j]
            is_fav_team = bool(fav and t and t == fav)

            # base script multiplier per trial
            mvec = np.ones(T, dtype=float)
            for sid in range(5):
                mask = script_id == sid
                if not np.any(mask):
                    continue
                mvec[mask] = pos_mult(pos[j], is_fav_team, sid)

            # team correlation factor
            tf = np.ones(T, dtype=float)
            if t and fav and t == fav:
                if pos[j] in ("QB", "WR", "TE"):
                    tf = pass_fav
                elif pos[j] == "RB":
                    tf = 0.65 * rush_fav + 0.35 * pass_fav
            elif t and dog and t == dog:
                if pos[j] in ("QB", "WR", "TE"):
                    tf = pass_dog
                elif pos[j] == "RB":
                    tf = 0.65 * rush_dog + 0.35 * pass_dog

            # power keeps correlation moderate
            tf = np.power(tf, float(max(0.0, sim_cfg.team_corr_power)))
            mult[:, j] = mvec * tf

    # Sample lognormal per player
    contrib = np.zeros((T, n), dtype=float)
    for i in range(n):
        draw = rng.lognormal(mean=mus[i], sigma=sigmas[i], size=T)
        draw = draw * mult[:, i]
        draw = np.clip(draw, 0.0, caps[i])
        contrib[:, i] = draw

    totals = contrib.sum(axis=1)
    return totals, contrib, scripts_by_game


def _lineup_stack_features(lineup: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Basic game-theory features used to slightly adjust EV/⭐."""
    qb = next((p for p in lineup if str(p.get('Pos','')).upper() == 'QB'), None)
    if not qb:
        return {"stack_level": 0, "bringback": 0, "qb_team": ""}

    qb_team = str(qb.get('Team') or '').upper()[:3]
    qb_game = str(qb.get('GameKey') or '')

    passcatch = [p for p in lineup if str(p.get('Pos','')).upper() in ('WR','TE') and str(p.get('Team') or '').upper()[:3] == qb_team]
    stack_level = len(passcatch)

    bringback = 0
    if qb_game:
        for p in lineup:
            if str(p.get('GameKey') or '') != qb_game:
                continue
            if str(p.get('Team') or '').upper()[:3] != qb_team and str(p.get('Pos','')).upper() != 'DST':
                bringback = 1
                break

    return {
        "stack_level": int(stack_level),
        "bringback": int(bringback),
        "qb_team": qb_team,
    }


def evaluate_lineup_from_samples(
    lineup: List[Dict[str, Any]],
    totals: np.ndarray,
    contrib: np.ndarray,
    scripts_by_game: Dict[str, np.ndarray],
    ui: UIState,
    sim_cfg: SimConfig,
    cash_line: float,
    top1_line: float,
    win_line: float,
) -> Dict[str, Any]:
    """Compute percentiles, rates, EV/ROI, script summaries, and per-player ceiling contributions."""

    if totals.size == 0:
        return {
            "p50": 0.0, "p80": 0.0, "p95": 0.0,
            "cash_rate": 0.0, "top1_rate": 0.0, "win_rate": 0.0,
            "ev": 0.0, "roi": 0.0, "star_rating": 0.0,
            "archetype": "N/A",
            "stack": {"stack_level": 0, "bringback": 0, "qb_team": ""},
            "tail_scripts": {},
            "script_summary": "",
            "ceiling_script_summary": "",
            "ceiling_players": [],
        }

    p50 = float(np.quantile(totals, 0.50))
    p80 = float(np.quantile(totals, 0.80))
    p95 = float(np.quantile(totals, sim_cfg.ceiling_quantile))

    cash_rate = float(np.mean(totals >= cash_line))
    top1_rate = float(np.mean(totals >= top1_line))
    win_rate = float(np.mean(totals >= win_line))

    # Game-theory nudges (kept small)
    stack = _lineup_stack_features(lineup)
    stack_mult = 1.0 + 0.03 * min(2, stack.get("stack_level", 0)) + 0.03 * stack.get("bringback", 0)
    stack_mult = float(min(1.12, max(0.95, stack_mult)))

    adj_top1 = min(1.0, top1_rate * stack_mult)
    adj_win = min(1.0, win_rate * stack_mult)

    # Expected payout (stepwise to avoid double-counting)
    prize_cash = float(ui.prize_cash)
    prize_top1 = float(ui.prize_top1)
    prize_win = float(ui.prize_win)
    entry_fee = float(ui.entry_fee)

    expected_payout = prize_cash * cash_rate + (prize_top1 - prize_cash) * adj_top1 + (prize_win - prize_top1) * adj_win
    ev = expected_payout - entry_fee
    roi = (expected_payout - entry_fee) / max(entry_fee, 1e-9)

    archetype = "Aggressive upside" if p95 >= top1_line else "Balanced" if p80 >= cash_line else "Thin"

    # Tail scripts summary: which game scripts show up most in top outcomes
    tail = max(1, int(sim_cfg.trials * sim_cfg.tail_share))
    idx = np.argpartition(totals, -tail)[-tail:]

    SCRIPT_LABELS = {
        0: "Fav leads",
        1: "Dog leads",
        2: "Shootout",
        3: "Low & slow",
        4: "Weird",
    }

    tail_scripts: Dict[str, Dict[str, float]] = {}
    parts = []
    for gk, sid in scripts_by_game.items():
        try:
            vals = sid[idx]
            counts = np.bincount(vals, minlength=5).astype(float)
            denom = float(counts.sum() or 1.0)
            counts = counts / denom
            dist = {SCRIPT_LABELS[i]: float(counts[i]) for i in range(5) if counts[i] >= 0.05}
            tail_scripts[gk] = dist
            # pick dominant
            dom_i = int(np.argmax(counts))
            parts.append(f"{gk}: {SCRIPT_LABELS[dom_i]} {counts[dom_i]*100:.0f}%")
        except Exception:
            continue

    script_summary = "; ".join(parts)

    # Per-player ceiling contributions: avg contribution in the top tail
    hi_contrib = contrib[idx, :]
    avg_hi = hi_contrib.mean(axis=0) if hi_contrib.size else np.zeros(contrib.shape[1])

    ceiling_players = []
    for i, p in enumerate(lineup):
        proj = float(p.get("Projection") or 0.0)
        avh = float(avg_hi[i]) if i < len(avg_hi) else 0.0
        diff_pct = None
        if proj > 0 and avh > 0:
            diff_pct = (avh / proj - 1.0) * 100.0
        ceiling_players.append({
            "PlayerName": p.get("PlayerName"),
            "Team": p.get("Team"),
            "Pos": p.get("Pos"),
            "avg_hi": avh,
            "diff_pct": diff_pct,
        })

    return {
        "p50": p50,
        "p80": p80,
        "p95": p95,
        "cash_rate": cash_rate,
        "top1_rate": top1_rate,
        "win_rate": win_rate,
        "ev": float(ev),
        "roi": float(roi),
        # filled later after we know batch ranks
        "star_rating": 0.0,
        "archetype": archetype,
        "stack": stack,
        "stack_mult": float(stack_mult),
        "tail_scripts": tail_scripts,
        "script_summary": script_summary,
        "ceiling_script_summary": "Player ceilings based on top-tail outcomes (p95+)",
        "ceiling_players": ceiling_players,
    }
def ceiling_map_for_lineup(ceiling_players: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    key: PlayerName|Team|Pos -> {avg_hi, diff_pct}
    """
    out: Dict[str, Dict[str, Any]] = {}
    for p in ceiling_players:
        key = f"{p.get('PlayerName','')}|{p.get('Team','')}|{p.get('Pos','')}"
        out[key] = p
    return out


# ----------------------------
# Optimizer (PuLP)
# ----------------------------

def build_single_lineup(
    df: pd.DataFrame,
    ui: UIState,
    pool_caps: Dict[str, float],
    dnu: Set[str],
    required_combo: Set[str],
    required_combo_min: int,
    banned_players: Set[str],
    previous_lineups: List[Set[int]],
) -> Optional[List[int]]:
    """
    Returns list of df indices selected for lineup (length 9), or None.
    Uses PuLP if available, otherwise falls back to heuristic sampling.
    """
    # If PuLP missing, fall back early with a helpful message at call site
    if not HAS_PULP:
        return None

    # candidate mask
    cand = df.copy()
    cand = cand[~cand["PlayerName"].isin(banned_players)]
    cand = cand[~cand["PlayerName"].isin(dnu)]

    if cand.empty:
        return None

    # Restrict-to-pool ONLY if user enabled it
    pool_names = set(pool_caps.keys())
    if ui.restrict_to_pool and pool_names:
        cand = cand[cand["PlayerName"].isin(pool_names)]
        if cand.empty:
            return None

    # indices in original df
    idxs = cand.index.tolist()
    n = len(idxs)

    prob = pulp.LpProblem("DFS", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", idxs, lowBound=0, upBound=1, cat="Binary")

    # Objective: projection + upside weight (p95 - proj) with small cheap penalty
    upside_w = 0.35
    cheap_penalty = 0.10  # small; we also add hard limits below

    obj_terms = []
    for i in idxs:
        proj = float(df.at[i, "Projection"])
        p95 = float(df.at[i, "P95"])
        sal = int(df.at[i, "Salary"])
        pos = str(df.at[i, "Pos"]).upper()
        team = str(df.at[i, "Team"]).upper()

        # Small environment bump for high-total games + small favorite RB/DST bump.
        env = 0.0
        tot = df.at[i, "VegasTotal"] if "VegasTotal" in df.columns else float('nan')
        if pd.notna(tot):
            env = (float(tot) - 44.0) / 10.0  # ~[-0.6, +1.2]
        fav_bias = 0.0
        fav = str(df.at[i, "FavTeam"]).upper() if "FavTeam" in df.columns and pd.notna(df.at[i, "FavTeam"]) else ""
        spread = df.at[i, "VegasSpread"] if "VegasSpread" in df.columns else float('nan')
        if fav and pd.notna(spread):
            is_fav = team == fav
            s = min(14.0, abs(float(spread)))
            if is_fav and pos == 'RB':
                fav_bias = 0.10 * (s / 10.0)
            if is_fav and pos in ('DST', 'D', 'DEF', 'D/ST'):
                fav_bias = 0.08 * (s / 10.0)

        cheap = 1.0 if sal <= 3800 else 0.0
        score = (
            proj
            + upside_w * max(0.0, (p95 - proj))
            + 0.60 * env
            + 1.00 * fav_bias
            - cheap_penalty * cheap
        )
        obj_terms.append(score * x[i])

    prob += pulp.lpSum(obj_terms)

    # Salary constraints
    prob += pulp.lpSum([int(df.at[i, "Salary"]) * x[i] for i in idxs]) <= SALARY_CAP
    prob += pulp.lpSum([int(df.at[i, "Salary"]) * x[i] for i in idxs]) >= int(ui.min_salary)

    # Total roster size
    prob += pulp.lpSum([x[i] for i in idxs]) == 9

    # Required positions
    for pos, cnt in POS_REQUIRED.items():
        pos_idxs = [i for i in idxs if str(df.at[i, "Pos"]).upper() == pos]
        prob += pulp.lpSum([x[i] for i in pos_idxs]) == cnt

    # Skill positions mins/maxs
    for pos, cnt in POS_MIN.items():
        pos_idxs = [i for i in idxs if str(df.at[i, "Pos"]).upper() == pos]
        prob += pulp.lpSum([x[i] for i in pos_idxs]) >= cnt
    for pos, cnt in POS_MAX.items():
        pos_idxs = [i for i in idxs if str(df.at[i, "Pos"]).upper() == pos]
        prob += pulp.lpSum([x[i] for i in pos_idxs]) <= cnt

    # FLEX rules: total of RB/WR/TE must be 7 (QB + DST fixed)
    skill_idxs = [i for i in idxs if str(df.at[i, "Pos"]).upper() in ("RB", "WR", "TE")]
    prob += pulp.lpSum([x[i] for i in skill_idxs]) == 7

    if ui.no_te_in_flex:
        # TE count fixed to 1
        te_idxs = [i for i in idxs if str(df.at[i, "Pos"]).upper() == "TE"]
        prob += pulp.lpSum([x[i] for i in te_idxs]) == 1

    # Exposure caps: applied ONLY for names listed in pool_caps (doesn't restrict overall player universe)
    # If a player is capped at 0, treat as DNU
    # (actual enforcement across N lineups occurs in build_lineups, here we only keep feasibility)

    # Required combo constraint
    if required_combo and required_combo_min > 0:
        combo_idxs = [i for i in idxs if df.at[i, "PlayerName"] in required_combo]
        if combo_idxs:
            prob += pulp.lpSum([x[i] for i in combo_idxs]) >= int(required_combo_min)

    # QB stack: if QB chosen from team, require >=1 WR/TE same team
    if ui.require_qb_stack:
        teams = cand["Team"].dropna().unique().tolist()
        for t in teams:
            qb_idxs = [i for i in idxs if df.at[i, "Pos"] == "QB" and df.at[i, "Team"] == t]
            passcatch_idxs = [i for i in idxs if df.at[i, "Pos"] in ("WR", "TE") and df.at[i, "Team"] == t]
            if qb_idxs and passcatch_idxs:
                prob += pulp.lpSum([x[i] for i in passcatch_idxs]) >= pulp.lpSum([x[i] for i in qb_idxs])

    # Bring-back: require at least 1 player from both sides of at least K games
    if ui.require_bring_back and ui.bring_back_games > 0:
        games = [g for g in cand["GameKey"].dropna().unique().tolist() if g]
        y = pulp.LpVariable.dicts("bb", games, lowBound=0, upBound=1, cat="Binary")
        for g in games:
            g_idxs = [i for i in idxs if df.at[i, "GameKey"] == g and df.at[i, "Pos"] != "DST"]
            if not g_idxs:
                prob += y[g] == 0
                continue
            # Need at least 2 different teams in that game
            teams_in_g = sorted(set(df.at[i, "Team"] for i in g_idxs))
            if len(teams_in_g) < 2:
                prob += y[g] == 0
                continue
            t1, t2 = teams_in_g[0], teams_in_g[1]
            t1_idxs = [i for i in g_idxs if df.at[i, "Team"] == t1]
            t2_idxs = [i for i in g_idxs if df.at[i, "Team"] == t2]
            # If y[g]=1, pick at least 1 from each team
            prob += pulp.lpSum([x[i] for i in t1_idxs]) >= y[g]
            prob += pulp.lpSum([x[i] for i in t2_idxs]) >= y[g]
            # If y[g]=0, no constraint; but prevent y[g]=1 unless there are >=2 players in game
            prob += pulp.lpSum([x[i] for i in g_idxs]) >= 2 * y[g]

        prob += pulp.lpSum([y[g] for g in games]) >= int(ui.bring_back_games)

    # Avoid DST vs opponent skill players
    if ui.avoid_opp_dst:
        dst_idxs = [i for i in idxs if df.at[i, "Pos"] == "DST"]
        for d in dst_idxs:
            dst_team = df.at[d, "Team"]
            opp = df.at[d, "Opp"]
            if not opp:
                continue
            opp_skill_idxs = [i for i in idxs if df.at[i, "Team"] == opp and str(df.at[i, "Pos"]).upper() in SKILL_POS]
            for i in opp_skill_idxs:
                prob += x[d] + x[i] <= 1

    # Limit number of "boom" low-tier plays (cheap + spiky)
    boom_idxs = []
    cheap_idxs = []
    for i in idxs:
        sal = int(df.at[i, "Salary"])
        proj = float(df.at[i, "Projection"])
        p95 = float(df.at[i, "P95"])
        if sal <= 4000:
            cheap_idxs.append(i)
        if sal <= 4500 and proj <= 9 and proj > 0 and (p95 / proj) >= 2.0 and df.at[i, "Pos"] != "DST":
            boom_idxs.append(i)

    if cheap_idxs:
        prob += pulp.lpSum([x[i] for i in cheap_idxs]) <= 3
    if boom_idxs:
        prob += pulp.lpSum([x[i] for i in boom_idxs]) <= 2

    # Diversity constraints vs previous lineups: differ by at least 2 players
    for prev in previous_lineups:
        prev_in_idxs = [i for i in prev if i in idxs]
        if prev_in_idxs:
            prob += pulp.lpSum([x[i] for i in prev_in_idxs]) <= 7

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    chosen = [i for i in idxs if pulp.value(x[i]) > 0.5]
    if len(chosen) != 9:
        return None
    return chosen


def assign_slots(selected_rows: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Assigns DFS slots (QB,RB1,RB2,WR1..,TE,FLEX,DST) and returns list of player dicts in slot order.
    """
    rows = selected_rows.to_dict(orient="records")

    def take_one(pos: str) -> Dict[str, Any]:
        for k, r in enumerate(rows):
            if str(r.get("Pos", "")).upper() == pos:
                return rows.pop(k)
        return {}

    qb = take_one("QB")
    dst = take_one("DST")

    rbs = [r for r in rows if str(r.get("Pos")).upper() == "RB"]
    wrs = [r for r in rows if str(r.get("Pos")).upper() == "WR"]
    tes = [r for r in rows if str(r.get("Pos")).upper() == "TE"]

    # remove taken
    rows = [r for r in rows if r not in rbs + wrs + tes]

    # Pick TE first (1 required)
    te = tes[0] if tes else {}
    if te in tes:
        tes.remove(te)

    # RBs
    rbs = sorted(rbs, key=lambda r: -float(r.get("Projection") or 0.0))
    rb1 = rbs[0] if len(rbs) > 0 else {}
    rb2 = rbs[1] if len(rbs) > 1 else {}
    rem = []
    if rb1 in rbs: rem.append(rb1)
    if rb2 in rbs: rem.append(rb2)
    for r in rem:
        if r in rbs: rbs.remove(r)

    # WRs
    wrs = sorted(wrs, key=lambda r: -float(r.get("Projection") or 0.0))
    wr1 = wrs[0] if len(wrs) > 0 else {}
    wr2 = wrs[1] if len(wrs) > 1 else {}
    wr3 = wrs[2] if len(wrs) > 2 else {}
    rem = []
    if wr1 in wrs: rem.append(wr1)
    if wr2 in wrs: rem.append(wr2)
    if wr3 in wrs: rem.append(wr3)
    for r in rem:
        if r in wrs: wrs.remove(r)

    # FLEX from leftovers (RB/WR/TE)
    flex_pool = []
    flex_pool.extend(rbs)
    flex_pool.extend(wrs)
    flex_pool.extend(tes)
    flex_pool = sorted(flex_pool, key=lambda r: -float(r.get("Projection") or 0.0))
    flex = flex_pool[0] if flex_pool else {}

    slot_map = {
        "QB": qb,
        "RB1": rb1,
        "RB2": rb2,
        "WR1": wr1,
        "WR2": wr2,
        "WR3": wr3,
        "TE": te,
        "FLEX": flex,
        "DST": dst,
    }

    out = []
    for slot in ROSTER_SLOTS:
        r = dict(slot_map.get(slot) or {})
        r["Slot"] = slot
        out.append(r)
    return out


def build_lineups(
    df: pd.DataFrame,
    ui: UIState,
    pool_caps: Dict[str, float],
    dnu: Set[str],
    required_combo: Set[str],
    required_combo_min: int,
) -> Tuple[List[List[Dict[str, Any]]], Optional[str]]:
    """
    Build up to ui.num_lineups lineups.
    Returns (list_of_lineup_players, warn_msg)
    """
    # quick feasibility checks
    pos_counts = df["Pos"].value_counts().to_dict()
    for pos, need in POS_REQUIRED.items():
        if pos_counts.get(pos, 0) < need:
            return [], f"Not enough {pos} players in projections to build a lineup."
    if df[df["Pos"] == "RB"].shape[0] < 2 or df[df["Pos"] == "WR"].shape[0] < 3 or df[df["Pos"] == "TE"].shape[0] < 1:
        return [], "Not enough RB/WR/TE players in projections to build a lineup."

    if not HAS_PULP:
        return [], "PuLP is not installed, so the solver can't run. Install it with: pip install pulp"

    lineups: List[List[Dict[str, Any]]] = []
    prev_sets: List[Set[int]] = []
    banned_players: Set[str] = set()

    # exposure tracking
    exposure_counts: Dict[str, int] = {}

    max_attempts = max(50, ui.num_lineups * 3)
    attempts = 0
    while len(lineups) < ui.num_lineups and attempts < max_attempts:
        attempts += 1

        chosen = build_single_lineup(
            df=df,
            ui=ui,
            pool_caps=pool_caps,
            dnu=dnu,
            required_combo=required_combo,
            required_combo_min=required_combo_min,
            banned_players=banned_players,
            previous_lineups=prev_sets,
        )
        if not chosen:
            break

        # Apply exposure caps: if any player exceeds their cap, ban them for future lineups
        # (only for players listed in pool_caps)
        for i in chosen:
            nm = str(df.at[i, "PlayerName"])
            exposure_counts[nm] = exposure_counts.get(nm, 0) + 1
            cap = pool_caps.get(nm, None)
            if cap is not None:
                max_ct = int(math.floor((cap / 100.0) * ui.num_lineups + 1e-9))
                if max_ct <= 0:
                    banned_players.add(nm)
                elif exposure_counts[nm] >= max_ct:
                    banned_players.add(nm)

        prev_sets.append(set(chosen))

        sel = df.loc[chosen].copy()
        players = assign_slots(sel)
        lineups.append(players)

    if not lineups:
        # Provide more actionable message
        hint = "Unable to build any valid lineups with the current rules. "
        hint += "Try: lowering min salary, unchecking 'Avoid DST vs opponent skill players', or disabling stack/bring-back rules."
        return [], hint

    if len(lineups) < ui.num_lineups:
        return lineups, f"Built {len(lineups)} lineups (could not reach {ui.num_lineups} under current constraints)."
    return lineups, None


def compute_exposures(lineups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    counts: Dict[Tuple[str, str, str], int] = {}
    total = len(lineups)
    for lu in lineups:
        for p in lu:
            key = (p.get("Pos",""), p.get("PlayerName",""), p.get("Team",""))
            counts[key] = counts.get(key, 0) + 1
    rows = []
    for (pos, name, team), ct in counts.items():
        rows.append({
            "Pos": pos,
            "PlayerName": name,
            "Team": team,
            "LineupCount": ct,
            "Exposure": (ct / total) * 100.0 if total else 0.0
        })
    rows.sort(key=lambda r: (-r["Exposure"], r["Pos"], r["PlayerName"]))
    return rows


# ----------------------------
# Flask app
# ----------------------------

app = Flask(__name__, template_folder='.', static_folder='static')
app.secret_key = os.environ.get("SECRET_KEY", "dev")

CACHE: Dict[str, Any] = {
    "df": pd.DataFrame(),
    "players_preview": [],
    "games_summary": [],
    "vegas_odds": {},
    "dk_map": {},

    # Last generated outputs (so DK salaries can be uploaded later without rerunning)
    "last_results": None,
    "last_exposures": None,
    "last_dk_export_map": {},
    "last_dk_stats": {},
}

def _ui_from_request(req: Any, prev: UIState) -> UIState:
    ui = UIState(**asdict(prev))
    form = req.form

    def as_int(name: str, default: int) -> int:
        try:
            return int(form.get(name, default))
        except Exception:
            return default

    def as_float(name: str, default: float) -> float:
        try:
            return float(form.get(name, default))
        except Exception:
            return default

    def as_bool(name: str) -> bool:
        return form.get(name) in ("on", "true", "1", "yes")

    ui.num_lineups = max(1, min(500, as_int("num_lineups", ui.num_lineups)))
    ui.min_salary = max(0, min(SALARY_CAP, as_int("min_salary", ui.min_salary)))

    ui.require_qb_stack = as_bool("require_qb_stack")
    ui.require_bring_back = as_bool("require_bring_back")
    ui.bring_back_games = max(1, min(4, as_int("bring_back_games", ui.bring_back_games)))
    ui.no_te_in_flex = as_bool("no_te_in_flex")
    ui.avoid_opp_dst = as_bool("avoid_opp_dst")
    ui.restrict_to_pool = as_bool("restrict_to_pool")

    ui.pool_raw = form.get("pool_raw", ui.pool_raw)
    ui.dnu_raw = form.get("dnu_raw", ui.dnu_raw)
    ui.required_combo_raw = form.get("required_combo_raw", ui.required_combo_raw)
    ui.required_combo_min = max(1, min(9, as_int("required_combo_min", ui.required_combo_min)))

    ui.auto_payout = as_bool("auto_payout")

    ui.cash_line = as_float("cash_line", ui.cash_line)
    ui.top1_line = as_float("top1_line", ui.top1_line)
    ui.win_line = as_float("win_line", ui.win_line)

    # Auto-threshold quantiles (used if auto_payout enabled)
    ui.cash_quantile = max(0.30, min(0.90, as_float("cash_quantile", ui.cash_quantile)))
    ui.top1_quantile = max(0.80, min(0.999, as_float("top1_quantile", ui.top1_quantile)))
    ui.win_quantile = max(0.90, min(0.9999, as_float("win_quantile", ui.win_quantile)))
    if ui.win_quantile <= ui.top1_quantile:
        ui.win_quantile = min(0.9999, ui.top1_quantile + 0.001)

    # Monte Carlo trials
    ui.sim_trials = max(500, min(20000, as_int("sim_trials", ui.sim_trials)))

    # EV/ROI knobs (optional)
    ui.entry_fee = max(0.0, as_float("entry_fee", ui.entry_fee))
    ui.prize_cash = max(0.0, as_float("prize_cash", ui.prize_cash))
    ui.prize_top1 = max(ui.prize_cash, as_float("prize_top1", ui.prize_top1))
    ui.prize_win = max(ui.prize_top1, as_float("prize_win", ui.prize_win))

    ui.sort_by = form.get("sort_by", ui.sort_by)

    return ui


UI = UIState()

@app.route("/", methods=["GET", "POST"])
def index():
    global UI, CACHE

    message = ""
    warn = ""

    def render(exposures=None, results=None, dk_export_map=None, dk_stats=None):
        return render_template(
            "index.html",
            ui_state=UI,
            message=message or None,
            warn=warn or None,
            players_preview=CACHE.get("players_preview") or [],
            games_summary=CACHE.get("games_summary") or [],
            exposures=exposures,
            results=results,
            solver_ready=HAS_PULP,
            dk_ready=bool(CACHE.get("dk_map")),
            dk_export_map=dk_export_map or {},
            dk_stats=dk_stats or {},
        )

    if request.method == "POST":
        action = request.form.get("action") or ""
        UI = _ui_from_request(request, UI)

        # vegas overrides from editable table
        vegas_overrides_raw = request.form.get("vegas_overrides_raw", "").strip()
        vegas_overrides = {}
        if vegas_overrides_raw:
            try:
                vegas_overrides = json.loads(vegas_overrides_raw)
                if not isinstance(vegas_overrides, dict):
                    vegas_overrides = {}
            except Exception:
                vegas_overrides = {}

        # player projection adjustments (client-side bumps)
        adj_raw = request.form.get("player_adjustments_raw", "").strip()
        adj_map: Dict[str, float] = {}
        if adj_raw:
            try:
                adj_map = json.loads(adj_raw)
                if not isinstance(adj_map, dict):
                    adj_map = {}
            except Exception:
                adj_map = {}

        # DK salaries upload (separate action supported)
        dk_file = request.files.get("dk_salaries_file")
        dk_uploaded = False
        if dk_file and getattr(dk_file, "filename", ""):
            dk_map, dk_warn = load_dk_salaries_from_upload(dk_file)
            if dk_warn:
                warn = (warn + " " + dk_warn).strip()
            if dk_map:
                CACHE["dk_map"] = dk_map
                dk_uploaded = True
                message = (message + f" Loaded DK salaries ({len(dk_map)} keys). ").strip()

        # If user explicitly clicked DK upload but didn't choose a file
        if action == "dk_upload" and not dk_uploaded and not CACHE.get("dk_map"):
            warn = (warn + " Choose a DraftKings salaries CSV to upload.").strip()

        if action == "dk_upload":
            # Do NOT touch cached projections. Just (optionally) update DK mapping and re-render last results.
            results_cached = CACHE.get("last_results")
            exposures_cached = CACHE.get("last_exposures")
            dk_export_map, dk_stats = build_dk_export_map_for_results(results_cached, CACHE.get("dk_map") or {})
            CACHE["last_dk_export_map"] = dk_export_map
            CACHE["last_dk_stats"] = dk_stats
            if dk_stats.get("missing"):
                warn = (warn + f" DK mapping missing for {dk_stats['missing']} players in the generated lineups.").strip()
            return render(exposures=exposures_cached, results=results_cached, dk_export_map=dk_export_map, dk_stats=dk_stats)

        if action == "preview":
            f = request.files.get("projections_file")

            # If no new file is selected, keep cached projections (critical when uploading DK salaries separately)
            if (not f) or (not getattr(f, "filename", "")):
                df_existing = CACHE.get("df")
                if df_existing is None or getattr(df_existing, "empty", True):
                    warn = warn or "No projections uploaded yet. Choose a projections CSV first."
                    return render(exposures=CACHE.get("last_exposures"), results=CACHE.get("last_results"),
                                  dk_export_map=CACHE.get("last_dk_export_map"), dk_stats=CACHE.get("last_dk_stats"))

                # Rebuild Vegas + P95 on the cached df (so overrides still apply)
                df = df_existing.copy()
                base_vegas = load_vegas_odds()
                vegas = merge_vegas_overrides(base_vegas, vegas_overrides)
                games_summary = build_games_summary(vegas)
                _ = build_game_config_map(df, vegas)
                df["P95"] = df.apply(estimate_player_p95, axis=1)

                CACHE["df"] = df
                CACHE["players_preview"] = df.sort_values("Projection", ascending=False).head(250).to_dict(orient="records")
                CACHE["games_summary"] = games_summary
                CACHE["vegas_odds"] = vegas

                if not message:
                    message = f"Using cached projections ({len(df)} players)."

                # If we have existing results, re-emit DK map too (in case DK salaries were just uploaded)
                dk_export_map, dk_stats = build_dk_export_map_for_results(CACHE.get("last_results"), CACHE.get("dk_map") or {})
                CACHE["last_dk_export_map"] = dk_export_map
                CACHE["last_dk_stats"] = dk_stats
                if dk_stats.get("missing"):
                    warn = (warn + f" DK mapping missing for {dk_stats['missing']} players in the generated lineups.").strip()

                return render(exposures=CACHE.get("last_exposures"), results=CACHE.get("last_results"),
                              dk_export_map=dk_export_map, dk_stats=dk_stats)

            df, preview, w = load_projections_from_upload(f)
            if w:
                # Do not nuke a previously loaded slate if the new upload fails
                warn = (warn + " " + w).strip() if warn else w
                if CACHE.get("df") is None or getattr(CACHE.get("df"), "empty", True):
                    CACHE["df"] = pd.DataFrame()
                    CACHE["players_preview"] = []
                    CACHE["games_summary"] = []
                    CACHE["vegas_odds"] = {}
                return render(exposures=CACHE.get("last_exposures"), results=CACHE.get("last_results"),
                              dk_export_map=CACHE.get("last_dk_export_map"), dk_stats=CACHE.get("last_dk_stats"))

            # New projections loaded => clear prior lineup outputs
            CACHE["last_results"] = None
            CACHE["last_exposures"] = None
            CACHE["last_dk_export_map"] = {}
            CACHE["last_dk_stats"] = {}

            # vegas
            base_vegas = load_vegas_odds()
            vegas = merge_vegas_overrides(base_vegas, vegas_overrides)
            games_summary = build_games_summary(vegas)
            _ = build_game_config_map(df, vegas)  # fills df vegas columns

            # Fill estimated p95
            df["P95"] = df.apply(estimate_player_p95, axis=1)

            preview = df.sort_values("Projection", ascending=False).head(250).to_dict(orient="records")

            CACHE["df"] = df
            CACHE["players_preview"] = preview
            CACHE["games_summary"] = games_summary
            CACHE["vegas_odds"] = vegas
            message = f"Loaded {len(df)} players."
            return render(exposures=None, results=None)

        if action == "run":
            df_cached = CACHE.get("df")
            if df_cached is None or getattr(df_cached, "empty", True):
                # Allow running directly if a projections file is provided on this POST
                f = request.files.get("projections_file")
                if f and getattr(f, "filename", ""):
                    df, preview, w = load_projections_from_upload(f)
                    if w:
                        warn = w
                        return render(exposures=CACHE.get("last_exposures"), results=CACHE.get("last_results"),
                                      dk_export_map=CACHE.get("last_dk_export_map"), dk_stats=CACHE.get("last_dk_stats"))

                    base_vegas = load_vegas_odds()
                    vegas = merge_vegas_overrides(base_vegas, vegas_overrides)
                    games_summary = build_games_summary(vegas)
                    _ = build_game_config_map(df, vegas)
                    df["P95"] = df.apply(estimate_player_p95, axis=1)
                    CACHE["df"] = df
                    CACHE["players_preview"] = df.sort_values("Projection", ascending=False).head(250).to_dict(orient="records")
                    CACHE["games_summary"] = games_summary
                    CACHE["vegas_odds"] = vegas
                    df_cached = df
                    message = (message + f" Loaded {len(df)} players. ").strip()
                else:
                    warn = "Upload projections first (Preview)."
                    return render(exposures=CACHE.get("last_exposures"), results=CACHE.get("last_results"),
                                  dk_export_map=CACHE.get("last_dk_export_map"), dk_stats=CACHE.get("last_dk_stats"))

            # work on a copy so we don't double-apply adjustments
            df = df_cached.copy()

            # vegas
            base_vegas = CACHE.get("vegas_odds") or load_vegas_odds()
            vegas = merge_vegas_overrides(base_vegas, vegas_overrides)
            games_summary = build_games_summary(vegas)
            game_cfg_map = build_game_config_map(df, vegas)
            CACHE["games_summary"] = games_summary
            CACHE["vegas_odds"] = vegas

            # Apply projection adjustments (copy) + recompute P95
            df_adj = apply_projection_adjustments(df, adj_map)
            df_adj["P95"] = df_adj.apply(estimate_player_p95, axis=1)

            # refresh preview (post-override)
            CACHE["players_preview"] = df_adj.sort_values("Projection", ascending=False).head(250).to_dict(orient="records")

            pool_caps = parse_pool_caps(UI.pool_raw)
            dnu = set(parse_comma_list(UI.dnu_raw))
            required_combo = set(parse_comma_list(UI.required_combo_raw))

            lineups, w = build_lineups(
                df=df_adj,
                ui=UI,
                pool_caps=pool_caps,
                dnu=dnu,
                required_combo=required_combo,
                required_combo_min=UI.required_combo_min,
            )
            if w:
                warn = (warn + " " + w).strip()

            if not lineups:
                warn = warn or "No lineups were built."
                return render(exposures=None, results=None)

            # Monte Carlo simulate each lineup (solver + MC)
            rng = np.random.default_rng(1337)
            sim_cfg = SimConfig(trials=int(UI.sim_trials))

            sampled = []
            all_totals = []
            for idx, lu in enumerate(lineups, start=1):
                totals, contrib, scripts_by_game = sample_lineup_with_scripts(lu, game_cfg_map, sim_cfg, rng)
                sampled.append((idx, lu, totals, contrib, scripts_by_game))
                all_totals.append(totals)

            # Auto-generate thresholds from the simulated distribution (optional)
            cash_line = float(UI.cash_line)
            top1_line = float(UI.top1_line)
            win_line = float(UI.win_line)
            if UI.auto_payout and all_totals:
                import numpy as _np
                dist = _np.concatenate(all_totals)
                if dist.size >= 1000:
                    cash_line = float(_np.quantile(dist, UI.cash_quantile))
                    top1_line = float(_np.quantile(dist, UI.top1_quantile))
                    win_line = float(_np.quantile(dist, UI.win_quantile))
                    # round to nearest 0.5
                    cash_line = round(cash_line * 2) / 2.0
                    top1_line = round(top1_line * 2) / 2.0
                    win_line = round(win_line * 2) / 2.0
                    # enforce monotonic
                    if top1_line <= cash_line:
                        top1_line = cash_line + 1.0
                    if win_line <= top1_line:
                        win_line = top1_line + 1.0

                    UI.cash_line, UI.top1_line, UI.win_line = cash_line, top1_line, win_line
                    message = f"Auto thresholds: Cash {cash_line:.1f}, Top 1% {top1_line:.1f}, Win {win_line:.1f}."
                else:
                    warn = (warn + " Not enough simulation samples to auto-generate thresholds.").strip()

            # Evaluate lineups (EV/ROI/stars)
            results = []
            for idx, lu, totals, contrib, scripts_by_game in sampled:
                sim = evaluate_lineup_from_samples(
                    lu,
                    totals,
                    contrib,
                    scripts_by_game,
                    UI,
                    sim_cfg,
                    cash_line=cash_line,
                    top1_line=top1_line,
                    win_line=win_line,
                )
                ceil_map = ceiling_map_for_lineup(sim.get("ceiling_players") or [])

                total_salary = sum(int(p.get("Salary") or 0) for p in lu)
                total_proj = sum(float(p.get("Projection") or 0.0) for p in lu)

                total_ceil = 0.0
                for p in lu:
                    key = f"{p.get('PlayerName','')}|{p.get('Team','')}|{p.get('Pos','')}"
                    cp = ceil_map.get(key)
                    if cp and cp.get("avg_hi") is not None:
                        total_ceil += float(cp["avg_hi"])
                    else:
                        total_ceil += float(p.get("P95") or (p.get("Projection") or 0.0) * 1.5)

                results.append({
                    "idx": idx,
                    "players": lu,
                    "total_salary": total_salary,
                    "total_proj": total_proj,
                    "total_ceil": total_ceil,
                    "sim": sim,
                    "ceil_map": ceil_map,
                    "fav_p95": sim.get("p95", 0.0),
                    "dog_p95": sim.get("p95", 0.0),
                    "shootout_p95": sim.get("p95", 0.0),
                })

            # Add star ratings (harder 5/5): combine absolute + batch rank
            evs = [float(r["sim"].get("ev") or 0.0) for r in results]
            if evs:
                # percentile rank (0..1), higher is better
                order = sorted(range(len(evs)), key=lambda i: evs[i])
                ranks = [0.0] * len(evs)
                for rank, i in enumerate(order):
                    ranks[i] = rank / max(1, (len(evs) - 1))

                for i, r in enumerate(results):
                    sim = r["sim"]
                    ev_rank = float(ranks[i])
                    top1 = float(sim.get("top1_rate") or 0.0)
                    winr = float(sim.get("win_rate") or 0.0)
                    p95 = float(sim.get("p95") or 0.0)
                    cash = float(sim.get("cash_rate") or 0.0)

                    abs_score = 0.55 * min(1.0, top1 / 0.02) + 0.35 * min(1.0, winr / 0.003) + 0.10 * min(1.0, max(0.0, (p95 - cash_line) / max(1e-9, (win_line - cash_line))))
                    abs_score *= float(sim.get("stack_mult") or 1.0)
                    score = 0.70 * abs_score + 0.30 * ev_rank
                    score = max(0.0, min(1.0, score))

                    stars = 5.0 * (score ** 1.6)
                    # Make 5/5 *very* hard
                    if stars > 4.95:
                        if winr < 0.0025 or top1 < 0.02:
                            stars = 4.8
                        else:
                            stars = 5.0
                    sim["star_rating"] = float(round(max(0.0, min(5.0, stars)), 2))

            exposures = compute_exposures(lineups)

            # DK export map for just the players in these results
            dk_export_map, dk_stats = build_dk_export_map_for_results(results, CACHE.get("dk_map") or {})
            if dk_stats.get("missing"):
                warn = (warn + f" DK mapping missing for {dk_stats['missing']} players in the generated lineups.").strip()

            # Cache the last outputs so DK salaries can be uploaded later without rerunning
            CACHE["last_results"] = results
            CACHE["last_exposures"] = exposures
            CACHE["last_dk_export_map"] = dk_export_map
            CACHE["last_dk_stats"] = dk_stats
            # server-side sort
            sort_by = (UI.sort_by or "ev").lower()
            key_map = {
                "ev": lambda r: float(r["sim"].get("ev") or 0.0),
                "roi": lambda r: float(r["sim"].get("roi") or 0.0),
                "proj": lambda r: float(r.get("total_proj") or 0.0),
                "p50": lambda r: float(r["sim"].get("p50") or 0.0),
                "p95": lambda r: float(r["sim"].get("p95") or 0.0),
                "stars": lambda r: float(r["sim"].get("star_rating") or 0.0),
            }
            sort_key = key_map.get(sort_by, key_map["ev"])
            results.sort(key=sort_key, reverse=True)

            return render(exposures=exposures, results=results, dk_export_map=dk_export_map, dk_stats=dk_stats)

    # GET
    return render(exposures=None, results=None)
if __name__ == "__main__":
    app.run(debug=True)
