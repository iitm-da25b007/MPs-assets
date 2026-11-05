# Import required libraries
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re

# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def normalize_constituency(c):
    return re.sub(r"\s*\((SC|ST)\)\s*$", "", str(c).strip())

def load(path):
    df = pd.read_csv(path)
    df["constituency"] = df["constituency"].map(normalize_constituency)
    return df

# Load dataset files
from pathlib import Path

# Directory containing this Python script
BASE_DIR = Path(__file__).resolve().parent

# Pipeline folders:
#   raw/                        scraped CSVs (some asset cells are image URLs)
#   interim/ocr_data/           CSVs with image-only values recovered by OCR
#   interim/cleaned_raw_data/   csv with assets / liabilities / age parsed to ints (-1 = missing)
#   interim/cleaned_data/       csv after dropping rows with missing values (and duplicates)
#   clean/                      dropped files + winner, constituency and Recontester links (used here)
RAW_DIR = BASE_DIR / "raw"
OCR_DIR = BASE_DIR / "interim" / "ocr_data"
CLEANED_RAW_DIR = BASE_DIR / "interim" / "cleaned_raw_data"
DROPPED_DIR = BASE_DIR / "interim" / "cleaned_data"
DATA_DIR = BASE_DIR / "clean"

andhra_14 = load(DATA_DIR / "andhra_2014.csv")
andhra_19 = load(DATA_DIR / "andhra_2019.csv")
andhra_24 = load(DATA_DIR / "andhra_2024.csv")

bihar_14 = load(DATA_DIR / "bihar_2014.csv")
bihar_19 = load(DATA_DIR / "bihar_2019.csv")
bihar_24 = load(DATA_DIR / "bihar_2024.csv")

gujarat_14 = load(DATA_DIR / "gujarat_2014.csv")
gujarat_19 = load(DATA_DIR / "gujarat_2019.csv")
gujarat_24 = load(DATA_DIR / "gujarat_2024.csv")

# Cleaning-stage data: scraped CSV -> OCR-enriched CSV -> parsed csv -> missing rows dropped
STATES = {"Andhra Pradesh": "andhra", "Bihar": "bihar", "Gujarat": "gujarat"}
YEARS = [2014, 2019, 2024]
CLEANED = {
    ("Andhra Pradesh", 2014): andhra_14, ("Andhra Pradesh", 2019): andhra_19, ("Andhra Pradesh", 2024): andhra_24,
    ("Bihar", 2014): bihar_14, ("Bihar", 2019): bihar_19, ("Bihar", 2024): bihar_24,
    ("Gujarat", 2014): gujarat_14, ("Gujarat", 2019): gujarat_19, ("Gujarat", 2024): gujarat_24,
}
MISSING_COLUMNS = ["Age", "Total Assets", "Liabilities"]  # -1 marks a missing value after parsing
RAW, OCR, CLEANED_RAW, DROPPED = {}, {}, {}, {}
for _state, _slug in STATES.items():
    for _year in YEARS:
        RAW[(_state, _year)] = pd.read_csv(RAW_DIR / f"{_slug}_{_year}.csv", encoding="utf-8-sig")
        OCR[(_state, _year)] = pd.read_csv(OCR_DIR / f"{_slug}_{_year}.csv", encoding="utf-8-sig")
        CLEANED_RAW[(_state, _year)] = pd.read_csv(CLEANED_RAW_DIR / f"{_slug}_{_year}.csv")
        DROPPED[(_state, _year)] = pd.read_csv(DROPPED_DIR / f"{_slug}_{_year}_dropped.csv")

def rupees(value):
    """Parse a scraped asset string such as 'Rs 1,23,456 ~ 1 Lacs+' into a number."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (float, int, np.number)):
        return float(value)
    text = str(value).strip()
    match = re.search(r"(?:Rs\.?|INR|₹)\s*([0-9,]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", ""))
    text = text.replace(",", "").strip()
    try:
        return float(text) if text not in {"", "-", "N/A", "nan", "None"} else np.nan
    except ValueError:
        return np.nan

# ----------------------------------------------------------------------------
# Constituency-level asset change (map data)
# ----------------------------------------------------------------------------
import json
from functools import lru_cache

WINNERS_CSV = RAW_DIR / "loksabha_winners_2014_2019_2024_normalized.csv"
GEO_PATH = BASE_DIR / "geo" / "india_pc_2019_simplified.geojson"
CRORE = 10_000_000
MAP_PERIODS = {
    "2014–2019": (2014, 2019),
    "2019–2024": (2019, 2024),
}

def _as_bool(value):
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)

ELECTIONS = {state: {year: CLEANED[(state, year)] for year in YEARS} for state in STATES}
INDEXED = {
    state: {year: frame.set_index("Sno.", drop=False) for year, frame in frames.items()}
    for state, frames in ELECTIONS.items()
}

# Spelling variants between sources map to one key. Add pairs here if the
# data-quality tiles report unmatched boundary names.
NAME_ALIASES = {
    # DataMeet boundary spelling -> election-results spelling (Andhra Pradesh).
    "anakapalli": "anakapalle",
    "anantapuramu": "anantapur",
    "narsapuram": "narasapuram",
}

def normalize_name(name):
    """Lower-case, drop reservation tags such as "(SC)", keep letters only."""
    text = re.sub(r"\(.*?\)", "", str(name)).lower()
    key = re.sub(r"[^a-z]", "", text)
    return NAME_ALIASES.get(key, key)

OFFICIAL_WINNERS = pd.read_csv(WINNERS_CSV)
OFFICIAL_WINNERS["key"] = OFFICIAL_WINNERS["Constituency"].map(normalize_name)

def _linked_row(state, row, from_year, to_year):
    """Follow Recontester links from ``from_year`` to ``to_year`` (either direction).

    Returns the linked row, or ``(None, year)`` naming the election where the chain broke.
    """
    if to_year > from_year:
        for year in [y for y in YEARS if from_year < y <= to_year]:
            frame = ELECTIONS[state][year]
            nxt = frame[frame["Recontester"] == int(row["Sno."])]
            if nxt.empty:
                return None, year
            row = nxt.iloc[0]
        return row, None
    for year in [y for y in reversed(YEARS) if to_year <= y < from_year]:
        prior = int(row.get("Recontester", -1))
        if prior == -1 or prior not in INDEXED[state][year].index:
            return None, year
        row = INDEXED[state][year].loc[prior]
    return row, None

@lru_cache(maxsize=32)
def constituency_changes(period, basis):
    """Asset change attributed to each official constituency.

    basis="start": the candidate who won the seat in the period's first election
                   (the sitting MP), followed forward to the last election.
    basis="end":   the candidate who won the seat in the period's last election,
                   followed back to the first election.
    """
    start, end = MAP_PERIODS[period]
    seat_year = start if basis == "start" else end
    records = []
    for state in STATES:
        frame = ELECTIONS[state][seat_year]
        winners = frame[frame["winner"].map(_as_bool)].copy()
        winners["key"] = winners["constituency"].map(normalize_name)
        official = OFFICIAL_WINNERS[
            (OFFICIAL_WINNERS["State"] == state) & (OFFICIAL_WINNERS["Election Year"] == seat_year)
        ]
        for _, seat in official.iterrows():
            item = {
                "State": state, "No.": seat["Constituency No."], "Constituency": seat["Constituency"],
                "key": seat["key"], "Winner": seat["Winner Candidate"], "Party": seat["Winner Party"],
                "Start assets": np.nan, "End assets": np.nan,
                "Change (₹ crore)": np.nan, "Change (%)": np.nan, "Status": "Linked",
            }
            match = winners[winners["key"] == seat["key"]]
            if match.empty:
                item["Status"] = f"Winner not found in {seat_year} candidate file"
                records.append(item)
                continue
            seat_row = match.iloc[0]
            other_year = end if basis == "start" else start
            other_row, broken = _linked_row(state, seat_row, seat_year, other_year)
            if other_row is None:
                item["Status"] = f"No Recontester link to {broken}"
                records.append(item)
                continue
            start_row, end_row = (seat_row, other_row) if basis == "start" else (other_row, seat_row)
            first, last = float(start_row["Total Assets"]), float(end_row["Total Assets"])
            item["Start assets"] = first / CRORE
            item["End assets"] = last / CRORE
            item["Change (₹ crore)"] = (last - first) / CRORE
            item["Change (%)"] = (last - first) / first * 100 if first > 0 else np.nan
            records.append(item)
    return pd.DataFrame.from_records(records)

def load_geometry():
    """Return {state: FeatureCollection} with feature ids "State|key", or None."""
    if not GEO_PATH.exists():
        return None
    collection = json.loads(GEO_PATH.read_text(encoding="utf-8"))
    features = collection.get("features", [])
    if not features:
        return None
    props = {k.lower(): k for k in features[0].get("properties", {})}
    name_key = next((props[k] for k in ("pc_name", "pcname", "pc_name_1", "name") if k in props), None)
    state_key = next((props[k] for k in ("st_name", "state_name", "st_nm", "state", "state_ut") if k in props), None)
    if not name_key or not state_key:
        return None
    wanted = {normalize_name(s): s for s in STATES}
    by_state = {s: {"type": "FeatureCollection", "features": []} for s in STATES}
    for feature in features:
        properties = feature.get("properties", {})
        state = wanted.get(normalize_name(properties.get(state_key, "")))
        if state is None:
            continue
        feature = dict(feature)
        feature["id"] = f"{state}|{normalize_name(properties.get(name_key, ''))}"
        feature["properties"] = dict(properties, display_name=str(properties.get(name_key, "")).title())
        by_state[state]["features"].append(feature)
    return by_state if any(fc["features"] for fc in by_state.values()) else None

GEOMETRY = load_geometry()

# ----------------------------------------------------------------------------
# Design tokens (dark theme: navy page, raised cards, teal winners / indigo losers)
# ----------------------------------------------------------------------------
BG = "#0B1020"        # page
CARD = "#131A2E"      # panels, sidebar
CARD_HI = "#1B2440"   # selected buttons, hover labels
BORDER = "#232C47"
TEXT = "#E6E9F2"
TEXT_2 = "#B4BBCC"
MUTED = "#8B93A7"
GRID = "#1F2842"
ACCENT = "#2DD4BF"

# Greyscale-safe palette: every pair that shares a chart differs strongly in lightness
# (not just hue), and the darker member also carries a hatch pattern where plotly allows.
WIN = "#FCD34D"      # winners: light gold (luminance ~0.68)
LOSE = "#3B82F6"     # losers / non-winners: mid blue (luminance ~0.24), hatched in charts
LOSE_PATTERN = "/"
PERIOD_COLORS = {"2014-2019": "#A7F3D0", "2019-2024": "#8B5CF6"}  # pale mint vs deep violet
PERIOD_PATTERNS = {"2014-2019": "", "2019-2024": "/"}
# Ordered light -> dark, with alternating hatches so neighbours stay apart in greyscale
PIE_COLORS = ["#FEF9C3", "#FDE047", "#2DD4BF", "#F59E0B", "#EC4899", "#3B82F6", "#7C3AED"]
PIE_PATTERNS = ["", "/", "", "\\", "", ".", "-"]
# Extra colours/hatches so every party can keep one fixed colour across all pies
PIE_COLORS_EXTRA = ["#A7F3D0", "#F97316", "#38BDF8", "#C4B5FD", "#22C55E", "#DC2626", "#F9A8D4", "#A16207"]
PIE_PATTERNS_EXTRA = ["", "/", "", "\\", ".", "-", "", "/"]
OTHERS_COLOR = "#64748B"
OTHERS_PATTERN = "x"
GOOD = "#2DD4BF"
WARN = "#F59E0B"
HIGHLIGHT = "#FF4D8D"  # font colour for spotlighted KPI values
# Decline (orange) -> no change (near white) -> growth (blue), for the constituency map.
# The light centre and mid-tone ends all stand clear of the navy page background.
DIVERGING = [[0, "#EA580C"], [0.25, "#FDBA74"], [0.5, "#F8FAFC"], [0.75, "#7DD3FC"], [1, "#0284C7"]]

FONT = "Inter, -apple-system, 'Segoe UI', Roboto, sans-serif"
SIDEBAR_W = 264

CSS = f"""
body {{ background:{BG}; color:{TEXT}; font-family:{FONT}; }}
h1, h2, h3, h4, h5 {{ color:{TEXT}; }}
.app-container {{ display:flex; min-height:100vh; }}

/* ---------- Sidebar ---------- */
.sidebar {{ position:fixed; top:0; left:0; bottom:0; width:{SIDEBAR_W}px; background:{CARD};
            border-right:1px solid {BORDER}; padding:26px 16px 18px; display:flex;
            flex-direction:column; overflow-y:auto; z-index:10; }}
.brand {{ padding:0 10px; }}
.brand-name {{ font-weight:800; font-size:1.4rem; letter-spacing:.16em; margin:0;
               background:linear-gradient(90deg,{WIN},{LOSE}); -webkit-background-clip:text;
               background-clip:text; color:transparent; }}
.brand-subtitle {{ font-size:.66rem; letter-spacing:.16em; color:{MUTED}; margin:4px 0 0; font-weight:600; }}
.sidebar hr {{ border-color:{BORDER}; opacity:1; margin:18px 4px; }}
.nav-heading {{ font-size:.66rem; letter-spacing:.16em; color:{MUTED}; font-weight:700; margin:0 0 8px 12px; }}
.sidebar .nav-link {{ display:flex; align-items:center; gap:12px; color:{MUTED}; border-radius:10px;
                      padding:9px 12px; font-weight:500; font-size:.92rem; margin-bottom:2px;
                      transition:all .15s ease; }}
.sidebar .nav-link i {{ font-size:1rem; width:18px; text-align:center; }}
.sidebar .nav-link:hover {{ background:{CARD_HI}; color:{TEXT}; }}
.sidebar .nav-link.active, .sidebar .nav-pills .nav-link.active {{ background:{CARD_HI}; color:#fff;
                      font-weight:600; box-shadow:inset 3px 0 0 {WIN}; }}
.sidebar .nav-link.active i {{ color:{WIN}; }}
.sidebar-footer {{ margin-top:auto; padding:14px 16px; border-radius:12px; background:{BG}; border:1px solid {BORDER}; }}
.sidebar-footer p {{ font-size:.66rem; letter-spacing:.14em; color:{MUTED}; margin:0; font-weight:600; }}
.sidebar-footer h5 {{ margin:4px 0; font-weight:700; font-size:1.05rem; }}

/* ---------- Main content ---------- */
.main-content {{ margin-left:{SIDEBAR_W}px; flex:1; min-width:0; padding:34px 40px 56px; }}
.page {{ max-width:1320px; }}
.page-header {{ display:flex; justify-content:space-between; align-items:flex-start; gap:20px;
                flex-wrap:wrap; margin-bottom:24px; }}
.eyebrow {{ font-size:.72rem; letter-spacing:.16em; color:{WIN}; font-weight:700; margin:0 0 6px; }}
.page h1 {{ font-weight:700; font-size:2rem; letter-spacing:-.02em; margin:0 0 8px; }}
.page-description {{ color:{MUTED}; max-width:760px; margin:0; line-height:1.6; }}
.data-status {{ display:inline-flex; align-items:center; gap:8px; background:{CARD}; border:1px solid {BORDER};
                border-radius:999px; padding:7px 14px; font-size:.7rem; letter-spacing:.1em; font-weight:600;
                color:{MUTED}; white-space:nowrap; }}
.status-dot {{ color:{WIN}; font-size:.8rem; line-height:1; }}
.section-heading {{ font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; color:{MUTED};
                    font-weight:600; margin:30px 0 4px; }}
.section-description {{ color:{MUTED}; font-size:.88rem; margin:0 0 14px; }}

/* ---------- KPI cards ---------- */
.kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; margin-bottom:20px; }}
.kpi-card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:14px; height:100%;
             transition:transform .15s ease, border-color .15s ease; }}
.kpi-stack {{ display:flex; flex-direction:column; gap:12px; height:100%; }}
.kpi-stack .kpi-card {{ height:auto; flex:1; }}
.kpi-card:hover {{ transform:translateY(-2px); border-color:#34406a; }}
.kpi-card .card-body {{ padding:16px 18px; }}
.kpi-title {{ font-size:.72rem; letter-spacing:.08em; color:{MUTED}; font-weight:600; margin:0 0 8px; text-transform:uppercase; }}
.kpi-value {{ font-size:1.6rem; font-weight:700; margin:0; letter-spacing:-.01em; color:{TEXT}; }}
.kpi-description {{ font-size:.8rem; color:{MUTED}; margin:6px 0 0; }}
.kpi-win {{ border-left:3px solid {WIN}; }}
.kpi-lose {{ border-left:3px solid {LOSE}; }}
.kpi-neutral {{ border-left:3px solid #38BDF8; }}
.kpi-win .kpi-value {{ color:{WIN}; }}
.kpi-lose .kpi-value {{ color:{LOSE}; }}
.kpi-neutral .kpi-value {{ color:#38BDF8; }}
.kpi-card .kpi-value.kpi-highlight {{ color:{HIGHLIGHT}; font-weight:800;
                                      text-shadow:0 0 12px {HIGHLIGHT}88; }}
.kpi-card:has(.kpi-highlight) {{ border-color:{HIGHLIGHT}; border-left-color:{HIGHLIGHT};
                                 box-shadow:0 0 0 1px {HIGHLIGHT}55; }}

/* ---------- Panels, filters, charts ---------- */
.content-panel {{ background:{CARD}; border:1px solid {BORDER}; border-radius:14px; padding:18px 20px;
                  margin-bottom:20px; }}
.row > * > .content-panel {{ height:100%; margin-bottom:0; }}
.content-panel h3 {{ font-size:1.02rem; font-weight:600; margin:0 0 4px; }}
.chart-description {{ color:{MUTED}; font-size:.86rem; margin:0 0 10px; }}
.chart-panel {{ padding:10px 10px 4px; overflow:hidden; }}
.chart-scroll {{ max-height:620px; overflow-y:auto; }}
.filters {{ display:flex; flex-wrap:wrap; gap:16px 30px; align-items:flex-end; padding:16px 20px 18px; }}
.filter-group {{ display:flex; flex-direction:column; gap:8px; }}
.filter-label {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; font-weight:600; color:{MUTED}; }}
.filters .form-switch {{ margin-bottom:10px; }}
.filters .form-check-label {{ font-size:.88rem; color:{MUTED}; font-weight:500; }}
.filters .form-check-input {{ background-color:{BG}; border-color:{MUTED}; }}
.filters .form-check-input:checked {{ background-color:{WIN}; border-color:{WIN}; }}

/* Segmented button selector (used for every choice control) */
.segmented {{ display:inline-flex; flex-wrap:wrap; background:{BG}; border:1px solid {BORDER};
              border-radius:12px; padding:4px; gap:2px; }}
.segmented .form-check {{ padding:0; margin:0; }}
.segmented-btn {{ background:transparent !important; color:{MUTED} !important; border:1px solid transparent !important;
                  border-radius:9px !important; padding:7px 18px; font-size:.9rem; font-weight:500; }}
.segmented-btn:hover {{ color:{TEXT} !important; }}
.segmented-btn.active {{ background:{CARD_HI} !important; color:#fff !important; font-weight:600;
                         border-color:{WIN}55 !important; }}

.chart-note {{ color:{MUTED}; font-size:.85rem; line-height:1.6; margin:4px 12px 10px;
               padding-top:10px; border-top:1px solid {BORDER}; }}
.plot-summary {{ color:{TEXT_2}; font-size:.85rem; line-height:1.55; margin:4px 12px 10px;
                 padding:8px 12px; border-left:3px solid {ACCENT}; background:{BG}; border-radius:8px; }}
.plot-summary strong {{ color:{TEXT}; }}
.note-panel {{ background:{CARD}; border:1px solid {BORDER}; border-left:3px solid {WARN};
               border-radius:14px; padding:14px 18px; color:{MUTED}; line-height:1.6;
               margin-top:20px; font-size:.9rem; }}
.note-panel ul {{ margin:6px 0 0; padding-left:18px; }}
.note-panel strong {{ color:{TEXT}; }}
.callout {{ background:{CARD}; border:1px solid {BORDER}; border-left:3px solid {WIN}; border-radius:12px;
            padding:12px 16px; font-size:.9rem; line-height:1.55; margin-bottom:20px; color:{TEXT}; }}
.callout-detail {{ display:block; margin-top:4px; color:{MUTED}; }}

/* Data-quality tiles */
.quality-tiles {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:12px; }}
.q-tile {{ padding:10px 12px; border-radius:10px; border:1px solid {BORDER}; background:{BG}; }}
.q-value {{ font-size:1.35rem; font-weight:700; }}
.q-value-small {{ font-size:1rem; }}
.q-label {{ font-size:.78rem; color:{MUTED}; line-height:1.4; }}
.q-good .q-value {{ color:{GOOD}; }}
.q-good .q-value::before {{ content:"✓ "; font-size:.95rem; }}
.q-warn .q-value {{ color:{WARN}; }}
.q-warn .q-value::before {{ content:"! "; font-size:.95rem; }}

.overview-list {{ margin:8px 0 0; padding-left:18px; color:{MUTED}; line-height:1.7; }}
.content-panel p {{ color:{MUTED}; line-height:1.6; }}
.content-panel p:last-child {{ margin-bottom:0; }}

@media (max-width: 991px) {{
    .sidebar {{ position:static; width:auto; border-right:none; border-bottom:1px solid {BORDER}; }}
    .app-container {{ flex-direction:column; }}
    .main-content {{ margin-left:0; padding:24px 18px 40px; }}
}}
"""

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
    ],
    title="Candidate Asset Trends",
    suppress_callback_exceptions=True,  # pages are rendered on demand by the router
)

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>__CSS__</style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>""".replace("__CSS__", CSS)

# ----------------------------------------------------------------------------
# UI building blocks
# ----------------------------------------------------------------------------
def kpi(card_id, title, kind="win", description=None):
    body = [
        html.P(title, className="kpi-title"),
        html.H2("-", id=card_id, className="kpi-value"),
    ]
    if description:
        body.append(html.P(description, className="kpi-description"))
    return dbc.Card(dbc.CardBody(body), className=f"kpi-card kpi-{kind}")

def static_kpi(title, value, description):
    return dbc.Card(dbc.CardBody([
        html.P(title, className="kpi-title"),
        html.H2(value, className="kpi-value"),
        html.P(description, className="kpi-description"),
    ]), className="kpi-card kpi-neutral")

def plot_summary(text):
    """Short plain-language explanation shown under a chart."""
    return html.P([html.Strong("What this shows: "), text], className="plot-summary")

def chart(graph_id, summary, scroll=False):
    graph = dcc.Graph(id=graph_id, config={"displayModeBar": False})
    return html.Div([
        html.Div(graph, className="chart-scroll") if scroll else graph,
        plot_summary(summary),
    ], className="content-panel chart-panel")

def page_header(eyebrow, title, description, status):
    return html.Div([
        html.Div([
            html.P(eyebrow, className="eyebrow"),
            html.H1(title),
            html.P(description, className="page-description"),
        ]),
        html.Div([html.Span("●", className="status-dot"), html.Span(status)], className="data-status"),
    ], className="page-header")

def section(title, description=None):
    items = [html.H3(title, className="section-heading")]
    if description:
        items.append(html.P(description, className="section-description"))
    return html.Div(items)

def segmented(component_id, options, value):
    """Side-by-side button selector (one active option), used for every choice control."""
    return dbc.RadioItems(
        id=component_id,
        options=options,
        value=value,
        className="btn-group segmented",
        inputClassName="btn-check",
        labelClassName="btn segmented-btn",
        labelCheckedClassName="active",
    )

def filter_group(label, control, label_id=None):
    label_el = html.Span(label, id=label_id, className="filter-label") if label_id else html.Span(label, className="filter-label")
    return html.Div([label_el, control], className="filter-group")

def filters(*groups):
    return html.Div(list(groups), className="content-panel filters")

PERIOD_OPTIONS = [
    {"label": "2014 → 2019", "value": "2014-2019"},
    {"label": "2019 → 2024", "value": "2019-2024"},
    {"label": "Both intervals", "value": "Both"},
]
YEAR_OPTIONS = [{"label": y, "value": y} for y in ("2014", "2019", "2024")]
STATUS = "3 ELECTIONS · 2014–2024"

# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
N_SEATS = int(OFFICIAL_WINNERS[OFFICIAL_WINNERS["Election Year"] == 2024].shape[0])
N_RECORDS = sum(len(f) for f in CLEANED.values())
N_RETURNING = int(sum(
    (pd.to_numeric(CLEANED[(s, y)]["Recontester"], errors="coerce").fillna(-1) >= 0).sum()
    for s in STATES for y in (2019, 2024)
))

def overview_page():
    return html.Div([
        page_header(
            "ANALYTICS / OVERVIEW", "Candidate Asset Trends",
            "How declared assets of Lok Sabha candidates changed between elections, comparing winners "
            "with non-winners across Andhra Pradesh, Bihar and Gujarat.", STATUS,
        ),
        html.Div([
            static_kpi("STUDY STATES", "03", "Andhra Pradesh, Bihar, Gujarat"),
            static_kpi("CONSTITUENCIES", f"{N_SEATS}", "Lok Sabha seats across the three states"),
            static_kpi("CANDIDATE RECORDS", f"{N_RECORDS:,}", "Cleaned affidavits, all three elections"),
            static_kpi("RETURNING CANDIDATES", f"{N_RETURNING:,}", "Linked to their earlier declaration"),
        ], className="kpi-grid"),
        html.Div([
            html.H3("Understanding the study"),
            html.P("Candidates contesting Lok Sabha elections file sworn affidavits declaring their total assets "
                   "and liabilities. When a candidate contests again, the dashboard links the new declaration to "
                   "the earlier one through the Recontester reference and measures how their declared wealth "
                   "changed, separately for those who won and those who did not."),
        ], className="content-panel"),
        html.Div([
            html.H3("Explore the analysis"),
            html.Ul([
                html.Li("State pages: asset change of returning winners and losers, per constituency."),
                html.Li("Wealth analytics: candidate counts, total assets and each party's share of declared wealth."),
                html.Li("Constituency map: where sitting MPs' declared assets grew or fell, with data-quality checks."),
                html.Li("Relationships: how assets relate to age, criminal cases, liabilities, education and past assets."),
                html.Li("Cleaning check: how scraping, OCR recovery and dropping incomplete rows shaped the final data."),
            ], className="overview-list"),
        ], className="content-panel"),
    ], className="page")

def state_page(prefix, name):
    return html.Div([
        page_header(
            f"ANALYTICS / {name.upper()}", f"Winners vs Losers: {name}",
            f"Change in declared assets for candidates in {name} who contested again, split by whether "
            "they won the earlier election.", STATUS,
        ),
        filters(filter_group("Election interval", segmented(f"{prefix}-period", PERIOD_OPTIONS, "2014-2019"))),

        section("Returning candidate overview"),
        dbc.Row([
            dbc.Col(kpi(f"{prefix}-winners-median-pchange", "Median % change · winners", "win", "Relative to earlier declaration"), md=6, xl=3),
            dbc.Col(kpi(f"{prefix}-winners-median-change", "Median change · winners", "win", "Absolute change in ₹"), md=6, xl=3),
            dbc.Col(kpi(f"{prefix}-losers-median-pchange", "Median % change · losers", "lose", "Relative to earlier declaration"), md=6, xl=3),
            dbc.Col(kpi(f"{prefix}-losers-median-change", "Median change · losers", "lose", "Absolute change in ₹"), md=6, xl=3),
        ], className="g-3 mb-4"),

        dbc.Row([
            dbc.Col(chart(
                f"{prefix}-asset-graph",
                f"The average amount by which declared assets changed for returning candidates in {name}, "
                "comparing those who won the earlier election with those who lost. A taller bar means a "
                "bigger average gain; a bar below zero means assets fell on average.",
            ), xl=5),
            dbc.Col(chart(
                f"{prefix}-asset-histogram",
                "How the individual asset changes are spread out. Each bar is a range of change and its "
                "height is the share of winners (gold) or losers (blue, hatched) in that range. Bars to the "
                "right of the dotted zero line are gains, bars to the left are losses.",
            ), xl=7),
        ], className="g-3 mb-4"),

        section("Winners and losers across elections",
                "Where returning candidates ended up: did last time's winners keep their seats, and did last time's losers win?"),
        html.Div([
            dcc.Graph(id=f"{prefix}-outcome-sankey", config={"displayModeBar": False}),
            plot_summary(f"Each column is an election and each band follows candidates in {name} who contested "
                         "again. Gold boxes are winners and blue boxes are non-winners. Grey-toned bands show candidates "
                         "whose result stayed the same; the pink band is winners who lost next time and the teal band is "
                         "non-winners who won next time. Band thickness is the number of candidates."),
            html.P(id=f"{prefix}-outcome-sankey-note", className="chart-note"),
        ], className="content-panel chart-panel"),

        section("Constituency breakdown", "Mean asset change of returning winners in each constituency they won."),
        chart(
            f"{prefix}-asset-graph-constituency",
            "One bar per constituency, showing how much the winning candidate's declared assets changed "
            "by the time they contested again. Longer bars to the right are larger gains; bars to the "
            "left are losses. The scale is compressed so both small and very large changes stay visible.",
            scroll=True,
        ),
    ], className="page")

def parties_page():
    return html.Div([
        page_header(
            "ANALYTICS / WEALTH", "Wealth Analytics",
            "Candidate counts, total declared assets by election result, and each party's share of "
            "declared wealth in every state.", STATUS,
        ),
        filters(filter_group("Election year", segmented("assets-period", YEAR_OPTIONS, "2014"))),

        section("Candidates who appeared"),
        dbc.Row([
            dbc.Col(kpi("ap-candidate-count", "Andhra Pradesh", "neutral", "Number of candidates"), md=4),
            dbc.Col(kpi("br-candidate-count", "Bihar", "neutral", "Number of candidates"), md=4),
            dbc.Col(kpi("gj-candidate-count", "Gujarat", "neutral", "Number of candidates"), md=4),
        ], className="g-3"),

        section("Total candidate assets by election result"),
        # one column per state: winners card on top, non-winners card below it
        dbc.Row([
            dbc.Col(html.Div([
                kpi(f"{prefix}-total-winner-assets", f"{state} · winners", "win", "Total declared assets"),
                kpi(f"{prefix}-total-loser-assets", f"{state} · non-winners", "lose", "Total declared assets"),
            ], className="kpi-stack"), md=4)
            for prefix, state in (("ap", "Andhra Pradesh"), ("br", "Bihar"), ("gj", "Gujarat"))
        ], className="g-3 mb-4"),

        chart(
            "median-assets-comparison-graph",
            "The typical (median) declared wealth of a winning candidate (gold) versus a candidate who "
            "lost (blue, hatched) in each state. The scale is logarithmic, so each step up the axis "
            "means ten times more wealth; a gap between the bars shows how much richer winners tend to be.",
        ),

        section("Asset share by party", "Top seven parties by total declared assets; independents are grouped with Others."),
        dbc.Row([
            dbc.Col(chart(
                f"{prefix}-asset-parties-graph",
                f"How the total declared wealth of all candidates in {state} is split between parties. "
                "A bigger slice means that party's candidates together declared more assets. Slices are "
                "told apart by shade and hatch pattern; hover for the exact amount.",
            ), xl=4)
            for prefix, state in (("ap", "Andhra Pradesh"), ("br", "Bihar"), ("gj", "Gujarat"))
        ], className="g-3"),
    ], className="page")

CLEAN_MODE_OPTIONS = [
    {"label": "Compare states", "value": "state"},
    {"label": "Compare years", "value": "year"},
]

def clean_scope_options(mode):
    if mode == "year":
        return [{"label": "All states (pooled)", "value": "all"}] + [{"label": s, "value": s} for s in STATES]
    return [{"label": "All elections (pooled)", "value": "all"}] + [{"label": str(y), "value": str(y)} for y in YEARS]

def cleaning_page():
    return html.Div([
        page_header(
            "PROJECT / DATA CLEANING", "What Cleaning Changed",
            "Source CSV declarations compared with the cleaned files behind the dashboard, after rows "
            "with missing values were dropped.", "SCRAPE · OCR · CLEAN · DROP",
        ),
        filters(
            filter_group("Analysis mode", segmented("clean-mode", CLEAN_MODE_OPTIONS, "state")),
            filter_group("Election year", segmented("clean-scope", clean_scope_options("state"), "all"), label_id="clean-scope-label"),
        ),

        section("Pipeline overview"),
        dbc.Row([
            dbc.Col(kpi("clean-source", "Source rows", "neutral", "Rows in the scraped CSVs"), md=4, xl=2),
            dbc.Col(kpi("clean-parsed", "Raw values parsed", "neutral", "Readable from source CSV"), md=4, xl=2),
            dbc.Col(kpi("clean-recovered", "Recovered by OCR", "win", "OCR-enriched CSVs"), md=4, xl=2),
            dbc.Col(kpi("clean-missing", "Rows with gaps", "lose", "Age, assets or liabilities missing"), md=4, xl=2),
            dbc.Col(kpi("clean-dropped", "Rows dropped", "lose", "Missing values or duplicates"), md=4, xl=2),
            dbc.Col(kpi("clean-final", "Final rows", "neutral", "Cleaned csv rows"), md=4, xl=2),
        ], className="g-3 mb-4"),

        html.Div([
            dcc.Graph(id="clean-plot", config={"displayModeBar": False}),
            plot_summary("Each box shows the range of declared asset values at one stage of the data "
                         "pipeline: the raw scraped file (light), after reading image-only values with OCR "
                         "(orange) and after dropping incomplete rows (blue). The line inside a box is the "
                         "middle value. Boxes that look alike mean cleaning did not distort the data."),
            html.P(id="clean-note", className="chart-note"),
        ], className="content-panel chart-panel"),

        html.Div("Caveat: the source CSV is the scraped input; the final table is after cleaning and "
                 "deduplication, with every row missing age, assets or liabilities dropped rather than "
                 "imputed. All values shown are declarations. The chart compares positive declared values "
                 "on a log scale.",
                 className="note-panel"),
    ], className="page")

def map_page():
    return html.Div([
        page_header(
            "ANALYTICS / GEOGRAPHY", "Constituency Map",
            "Where did winners' declared assets change most? Each constituency is represented by one winner, "
            "followed through the Recontester links to the other election of the selected period.", STATUS,
        ),
        filters(
            filter_group("State", segmented("map-state", [{"label": s, "value": s} for s in STATES], "Bihar")),
            filter_group("Comparison period", segmented(
                "map-period", [{"label": p.replace("–", " → "), "value": p} for p in MAP_PERIODS], "2019–2024")),
            filter_group("Change measure", segmented(
                "map-measure", [{"label": "₹ crore", "value": "abs"},
                                {"label": "% of first-year assets", "value": "pct"}], "abs")),
        ),
        html.Div(id="map-definition", className="callout"),

        dbc.Row([
            dbc.Col(html.Div([
                dcc.Graph(id="constituency-map", config={"displayModeBar": False}),
                plot_summary("Each shape is a Lok Sabha constituency, coloured by how much its winner's "
                             "declared assets changed by the next election. Blue means assets grew, orange "
                             "means they fell, and near-white means little change; deeper colours are bigger "
                             "changes. Dark outlined seats have no data."),
                html.P(id="constituency-map-note", className="chart-note"),
            ], className="content-panel chart-panel"), xl=8),
            dbc.Col(html.Div([
                html.H3("Data quality for this view"),
                html.P("Seat counts behind the map for the selected state and period.", className="chart-description"),
                html.Div(id="quality-tiles", className="quality-tiles"),
            ], className="content-panel"), xl=4),
        ], className="g-3"),

        html.Div([
            html.Strong("Notes on this map"),
            html.Ul([
                html.Li("Winner definition: each constituency is represented by the candidate who won it in the "
                        "first election of the selected period (the sitting MP), followed forward to the later "
                        "election through Recontester links."),
                html.Li("A seat is left unfilled when that winner did not contest the later election or could not be "
                        "linked."),
                html.Li("Candidates with a missing asset declaration were dropped during cleaning, so a winner "
                        "dropped in either election leaves the seat unfilled."),
            ]),
        ], className="note-panel"),
    ], className="page")

PAGES = {
    "/": overview_page,
    "/andhra-pradesh": lambda: state_page("ap", "Andhra Pradesh"),
    "/bihar": lambda: state_page("br", "Bihar"),
    "/gujarat": lambda: state_page("gj", "Gujarat"),
    "/parties": parties_page,
    "/constituency-map": map_page,
    "/cleaning": cleaning_page,
    "/relationships": lambda: relationships_page(),
}

# ----------------------------------------------------------------------------
# Sidebar + application layout
# ----------------------------------------------------------------------------
def nav_link(icon, label, href):
    return dbc.NavLink([html.I(className=f"bi {icon}"), html.Span(label)], href=href, active="exact")

sidebar = html.Div([
    html.Div([
        html.H3(["MP", html.Span("·"), "ASSETS"], className="brand-name"),
        html.P("LOK SABHA AFFIDAVIT ANALYTICS", className="brand-subtitle"),
    ], className="brand"),
    html.Hr(),

    html.P("ANALYSIS", className="nav-heading"),
    dbc.Nav([
        nav_link("bi-grid-1x2", "Overview", "/"),
        nav_link("bi-geo-alt", "Andhra Pradesh", "/andhra-pradesh"),
        nav_link("bi-geo-alt", "Bihar", "/bihar"),
        nav_link("bi-geo-alt", "Gujarat", "/gujarat"),
    ], vertical=True, pills=True, className="navigation"),

    html.Hr(),
    html.P("COMPARISONS", className="nav-heading"),
    dbc.Nav([
        nav_link("bi-pie-chart", "Wealth analytics", "/parties"),
        nav_link("bi-map", "Constituency map", "/constituency-map"),
        nav_link("bi-diagram-3", "Relationships", "/relationships"),
    ], vertical=True, pills=True, className="navigation"),

    html.Hr(),
    html.P("PROJECT INFORMATION", className="nav-heading"),
    dbc.Nav([
        nav_link("bi-clipboard-check", "Cleaning check", "/cleaning"),
    ], vertical=True, pills=True, className="navigation"),

    html.Div([
        html.P("DATA PERIOD"),
        html.H5("2014 — 2024"),
        html.P("ANDHRA · BIHAR · GUJARAT"),
    ], className="sidebar-footer"),
], className="sidebar")

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    sidebar,
    html.Main(html.Div(id="page-content"), className="main-content"),
], className="app-container")

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    return PAGES.get(pathname, overview_page)()

# ----------------------------------------------------------------------------
# Figure helpers
# ----------------------------------------------------------------------------
def symlog_transform(values):
    return np.sign(values) * np.log1p(np.abs(values))

def money_short(value):
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    sign = "−" if value < 0 else ""
    value = abs(value)
    if value >= 10_000_000:
        return f"{sign}₹{value / 10_000_000:.1f} Crores"
    if value >= 100_000:
        return f"{sign}₹{value / 100_000:.1f} Lakhs"
    return f"{sign}₹{value:,.0f}"

def tick_label(v):
    sign = "−" if v < 0 else ""
    v = abs(v)
    if v >= 1e7:
        return f"{sign}₹{v / 1e7:g}Cr"
    if v >= 1e5:
        return f"{sign}₹{v / 1e5:g}L"
    if v >= 1e3:
        return f"{sign}₹{v / 1e3:g}K"
    return f"{sign}₹{v:g}"

def symlog_ticks(raw_values):
    """Tick positions/labels so a symlog axis reads in rupees instead of log units."""
    raw = np.asarray(raw_values, dtype=float)
    raw = raw[~np.isnan(raw)]
    if raw.size == 0 or np.abs(raw).max() == 0:
        return [0], ["₹0"]
    kmax = int(np.ceil(np.log10(np.abs(raw).max())))
    kmin = 4
    step = 1 if kmax - kmin <= 4 else 2
    powers = [10.0 ** k for k in range(kmin, max(kmax, kmin) + 1, step)]
    values = [0.0] + powers
    if raw.min() < 0:
        values = [-p for p in reversed(powers)] + values
    return list(symlog_transform(np.array(values))), [tick_label(v) for v in values]

def linear_ticks(raw_values, target=6):
    """Round-number ticks for a linear axis, labelled in ₹ K / L / Cr."""
    raw = np.asarray(raw_values, dtype=float)
    raw = raw[~np.isnan(raw)]
    if raw.size == 0 or raw.min() == raw.max():
        return [0], ["₹0"]
    span = raw.max() - raw.min()
    mag = 10.0 ** np.floor(np.log10(span / target))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= span / target)
    vals = np.arange(np.floor(raw.min() / step) * step, raw.max() + step, step)
    vals = [0.0 if abs(v) < step * 1e-9 else float(v) for v in vals]
    return vals, [tick_label(v) if v != 0 else "₹0" for v in vals]

def log_ticks(raw_values):
    """Power-of-ten ticks for a log axis, labelled in ₹ K / L / Cr."""
    raw = np.asarray(raw_values, dtype=float)
    raw = raw[(~np.isnan(raw)) & (raw > 0)]
    if raw.size == 0:
        return [1], ["₹1"]
    lo, hi = int(np.floor(np.log10(raw.min()))), int(np.ceil(np.log10(raw.max())))
    ks = list(range(lo, hi + 1))
    return [10.0 ** k for k in ks], [tick_label(10.0 ** k) for k in ks]

def style_fig(fig, height=None):
    """Apply the shared light theme to a figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT, size=12),
        title=dict(font=dict(size=15, color=TEXT), x=0.01, xanchor="left"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            title_text="", font=dict(color=MUTED),
        ),
        hoverlabel=dict(bgcolor=CARD_HI, bordercolor=BORDER, font=dict(family=FONT, color=TEXT)),
        margin=dict(t=70, l=60, r=24, b=50),
    )
    axis = dict(
        gridcolor=GRID, linecolor=BORDER,
        zerolinecolor=MUTED, zerolinewidth=1.5,
        tickfont=dict(color=MUTED), title_font=dict(color=MUTED),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    # Hatched bars keep their solid fill with a dark hatch on top, so they read in greyscale
    fig.update_traces(marker_pattern_fillmode="overlay", marker_pattern_fgcolor=BG,
                      marker_pattern_solidity=0.3, marker_pattern_size=8, selector=dict(type="bar"))
    try:  # rounded bars (newer plotly only)
        fig.update_layout(barcornerradius=6)
    except ValueError:
        pass
    if height:
        fig.update_layout(height=height)
    return fig

# ----------------------------------------------------------------------------
# Statistics + figures
# ----------------------------------------------------------------------------
def calculate_asset_changes(previous_df: pd.DataFrame, current_df: pd.DataFrame):
    previous_df = previous_df.set_index("Sno.")
    records = []

    for _, candidate in current_df.iterrows():
        previous_sno = int(candidate["Recontester"])
        if previous_sno == -1:
            continue

        prev_candidate = previous_df.loc[previous_sno]

        previous_assets = prev_candidate["Total Assets"]
        current_assets = candidate["Total Assets"]
        winner = prev_candidate["winner"]
        constituency = prev_candidate["constituency"]

        records.append({
            "Candidate": candidate["Candidate"],
            "Previous_assets": previous_assets,
            "Asset Change": current_assets - previous_assets,
            "Winner": winner,
            "Constituency": constituency
        })

    return pd.DataFrame(records)

def nice_step(raw_step):
    """Round a bin width to the nearest 1, 2, 2.5 or 5 times a power of ten."""
    mag = 10.0 ** np.floor(np.log10(raw_step))
    candidates = [m * mag for m in (1, 2, 2.5, 5, 10)]
    return min(candidates, key=lambda c: abs(np.log(c / raw_step)))

def histogram_bin_edges(values, min_bins=16, max_bins=30):
    """Equal, round-width bins covering the bulk of the data.

    The window is the Tukey fences (quartiles +/- 1.5 IQR), which stays tight even with a
    few dozen candidates; values outside it are folded into the first/last bin instead of
    stretching the axis. The bin count grows with sample size (Rice rule).
    """
    values = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lo, hi = max(values.min(), q1 - 1.5 * iqr), min(values.max(), q3 + 1.5 * iqr)
    if hi <= lo:
        lo, hi = values.min(), values.max()
    if hi <= lo:
        lo, hi = lo - 1, hi + 1
    n_bins = int(np.clip(np.ceil(2 * values.size ** (1 / 3)), min_bins, max_bins))
    step = nice_step((hi - lo) / n_bins)
    start, stop = np.floor(lo / step) * step, np.ceil(hi / step) * step
    return np.arange(start, stop + step / 2, step), step

def create_asset_histogram(changes):
    fig = go.Figure()
    values = changes["Asset Change"].to_numpy(dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return style_fig(fig, height=440)

    edges, step = histogram_bin_edges(values)
    centers = (edges[:-1] + edges[1:]) / 2
    clipped_low, clipped_high = values.min() < edges[0], values.max() > edges[-1]
    ranges = [f"{money_short(a)} to {money_short(b)}" for a, b in zip(edges[:-1], edges[1:])]
    if clipped_low:
        ranges[0] = f"below {money_short(edges[1])}"
    if clipped_high:
        ranges[-1] = f"above {money_short(edges[-2])}"

    groups = [
        ("Winners", True, WIN, "rgba(252,211,77,0.45)", ""),
        ("Losers", False, LOSE, "rgba(59,130,246,0.45)", LOSE_PATTERN),
    ]
    for label, flag, color, fill, pattern in groups:
        group = changes.loc[changes["Winner"] == flag, "Asset Change"].dropna().to_numpy(dtype=float)
        if group.size == 0:
            continue
        # fold outliers into the end bins
        counts, _ = np.histogram(np.clip(group, edges[0], edges[-1]), bins=edges)
        share = counts / group.size
        fig.add_trace(go.Bar(
            x=centers, y=share, width=step * 0.9, name=f"{label} (n={group.size})",
            marker=dict(color=fill, line=dict(color=color, width=1.5),
                        pattern=dict(shape=pattern)),
            customdata=np.column_stack([ranges, counts]),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Range: %{customdata[0]}<br>"
                "Candidates: %{customdata[1]}<br>"
                "Share of group: %{y:.1%}"
                "<extra></extra>"
            ),
        ))

    tv, tt = linear_ticks(edges, target=8)
    style_fig(fig, height=440)
    fig.update_layout(
        barmode="overlay",
        bargap=0,
        title="Distribution of Asset Changes: Winners vs Losers",
        hovermode="closest",
    )
    outlier_note = "" if (clipped_low or clipped_high) else ""
    fig.update_xaxes(title=f"Asset change (bins of {tick_label(step)}){outlier_note}",
                     range=[edges[0], edges[-1]], tickvals=tv, ticktext=tt, zeroline=False)
    fig.update_yaxes(title="Share of group", tickformat=".0%", zeroline=False, rangemode="tozero")
    if edges[0] < 0 < edges[-1]:
        fig.add_vline(x=0, line=dict(color=MUTED, width=1, dash="dot"))
    return fig

def calculate_statistics(changes: pd.DataFrame):
    winners = changes[changes["Winner"] == True]
    denom = np.where(winners["Previous_assets"] > 0, winners["Previous_assets"], 1)
    winners_percen = (winners["Asset Change"] / denom) * 100

    losers = changes[changes["Winner"] == False]
    denom_losers = np.where(losers["Previous_assets"] > 0, losers["Previous_assets"], 1)
    losers_percen = (losers["Asset Change"] / denom_losers) * 100

    n_cons = max(winners["Constituency"].nunique(), 1)

    if "Period" in changes.columns:
        period_order = ["2014-2019", "2019-2024"]

        df1 = changes.groupby(["Period", "Winner"])["Asset Change"].mean().reset_index()
        df1["Winner"] = df1["Winner"].replace({True: "Winners", False: "Losers"})
        df1["Formatted Asset Change"] = df1["Asset Change"].apply(money_short)

        bar_graph1 = px.bar(
            df1,
            x="Winner",
            y="Asset Change",
            color="Period",
            barmode="group",
            custom_data=["Formatted Asset Change"],
            category_orders={"Winner": ["Winners", "Losers"], "Period": period_order},
            color_discrete_map=PERIOD_COLORS,
            pattern_shape="Period",
            pattern_shape_map=PERIOD_PATTERNS,
            title="Mean Asset Change: Winners vs Losers"
        )
        bar_graph1.update_traces(
            hovertemplate=(
                "Category: %{x}<br>"
                "Period: %{fullData.name}<br>"
                "Mean Asset Change: %{customdata[0]}"
                "<extra></extra>"
            )
        )
        tv, tt = symlog_ticks(df1["Asset Change"])
        bar_graph1.update_layout(xaxis_title="Election outcome", yaxis_title="Mean asset change")
        bar_graph1.update_yaxes(tickvals=tv, ticktext=tt)

        df2 = winners.groupby(["Constituency", "Period"])["Asset Change"].mean().reset_index()
        constituency_data = df2.pivot(
            index="Constituency", columns="Period", values="Asset Change"
        ).reindex(columns=period_order)

        bar_graph2 = go.Figure()
        for period in period_order:
            actual_values = constituency_data[period]
            bar_graph2.add_trace(go.Bar(
                x=symlog_transform(actual_values),
                y=constituency_data.index,
                orientation="h",
                name=period,
                marker=dict(color=PERIOD_COLORS[period],
                            pattern=dict(shape=PERIOD_PATTERNS[period])),
                customdata=actual_values.apply(money_short),
                hovertemplate=(
                    "Constituency: %{y}<br>"
                    f"Period: {period}<br>"
                    "Actual Asset Change: %{customdata}"
                    "<extra></extra>"
                )
            ))
        tv2, tt2 = symlog_ticks(constituency_data.to_numpy().ravel())
        bar_graph2.update_layout(
            barmode="group",
            title="Constituency-wise Asset Change of Winners",
            xaxis_title="Asset change (symlog scale)",
            yaxis_title="",
        )
        bar_graph2.update_xaxes(tickvals=tv2, ticktext=tt2)
        height2 = 38 * n_cons
        cons_order = (symlog_transform(constituency_data.fillna(0)).sum(axis=1)
                      .sort_values().index.tolist())
    else:
        df1 = changes.groupby("Winner")["Asset Change"].mean().reset_index()
        df1["Winner"] = df1["Winner"].replace({True: "Winners", False: "Losers"})
        df1["Formatted Asset Change"] = df1["Asset Change"].apply(money_short)

        bar_graph1 = px.bar(
            df1,
            x="Winner",
            y="Asset Change",
            color="Winner",
            custom_data=["Formatted Asset Change"],
            category_orders={"Winner": ["Winners", "Losers"]},
            color_discrete_map={"Losers": LOSE, "Winners": WIN},
            pattern_shape="Winner",
            pattern_shape_map={"Winners": "", "Losers": LOSE_PATTERN},
            title="Mean Asset Change: Winners vs Losers"
        )
        bar_graph1.update_traces(
            hovertemplate=(
                "Category: %{x}<br>"
                "Mean Asset Change: %{customdata[0]}"
                "<extra></extra>"
            )
        )
        tv, tt = symlog_ticks(df1["Asset Change"])
        bar_graph1.update_layout(
            xaxis_title="Election outcome",
            yaxis_title="Mean asset change",
            showlegend=True,
        )
        bar_graph1.update_yaxes(tickvals=tv, ticktext=tt)

        df2 = winners.groupby("Constituency")["Asset Change"].mean().reset_index()
        df2["Scaled Asset Change"] = symlog_transform(df2["Asset Change"])
        df2["Transformed Asset Change"] = df2["Asset Change"].apply(money_short)

        bar_graph2 = px.bar(
            df2,
            y="Constituency",
            x="Scaled Asset Change",
            orientation="h",
            custom_data=["Transformed Asset Change"],
            color_discrete_sequence=[WIN],
            title="Constituency-wise Asset Change of Winners",
        )
        bar_graph2.update_traces(
            hovertemplate=(
                "Constituency: %{y}<br>"
                "Actual Asset Change: %{customdata[0]}"
                "<extra></extra>"
            )
        )
        tv2, tt2 = symlog_ticks(df2["Asset Change"])
        bar_graph2.update_layout(
            xaxis_title="Asset change (symlog scale)", yaxis_title="", showlegend=False
        )
        bar_graph2.update_xaxes(tickvals=tv2, ticktext=tt2)
        height2 = 26 * n_cons
        cons_order = df2.sort_values("Asset Change")["Constituency"].tolist()

    histogram_graph = create_asset_histogram(changes)

    style_fig(bar_graph1, height=440)
    style_fig(bar_graph2, height=max(580, height2))
    bar_graph1.update_xaxes(categoryorder="array", categoryarray=["Winners", "Losers"])
    # Explicit category list + range: plotly.js otherwise keeps constituencies from the
    # previously shown period as empty rows when the figure is re-rendered.
    bar_graph2.update_yaxes(categoryorder="array", categoryarray=cons_order,
                            range=[-0.5, len(cons_order) - 0.5], autorange=False,
                            dtick=1, automargin=True, tickfont=dict(size=11, color=MUTED))
    bar_graph2.update_layout(margin=dict(t=70, l=130, r=24, b=50))

    return (
        f"{winners_percen.median():,.2f}%" if not winners.empty else "N/A",
        money_short(winners["Asset Change"].median()) if not winners.empty else "N/A",
        f"{losers_percen.median():,.2f}%" if not losers.empty else "N/A",
        money_short(losers["Asset Change"].median()) if not losers.empty else "N/A",
        bar_graph1,
        histogram_graph,
        bar_graph2
    )

def summarize_election_assets(data):
    """Summarize declared assets for winners and non-winners in one election."""
    winners = data.loc[data["winner"] == True, "Total Assets"].dropna()
    non_winners = data.loc[data["winner"] == False, "Total Assets"].dropna()

    return {
        "total_winners": money_short(winners.sum()) if not winners.empty else "N/A",
        "total_non_winners": money_short(non_winners.sum()) if not non_winners.empty else "N/A",
    }

def create_median_assets_comparison(ap_data, br_data, gj_data, year):
    """Compare median declared assets of winners and non-winners across states."""
    records = []

    for state, data in [
        ("Andhra Pradesh", ap_data),
        ("Bihar", br_data),
        ("Gujarat", gj_data),
    ]:
        winners = data.loc[data["winner"] == True, "Total Assets"].dropna()
        non_winners = data.loc[data["winner"] == False, "Total Assets"].dropna()

        records.extend([
            {"State": state, "Candidate Result": "Winners",
             "Median Assets": winners.median() if not winners.empty else 0},
            {"State": state, "Candidate Result": "Non-winners",
             "Median Assets": non_winners.median() if not non_winners.empty else 0},
        ])

    chart_data = pd.DataFrame(records)
    chart_data["Formatted Assets"] = chart_data["Median Assets"].apply(money_short)

    fig = px.bar(
        chart_data,
        x="State",
        y="Median Assets",
        color="Candidate Result",
        barmode="group",
        title=f"Median Candidate Assets: Winners vs Non-winners ({year})",
        custom_data=["Formatted Assets"],
        color_discrete_map={"Winners": WIN, "Non-winners": LOSE},
        pattern_shape="Candidate Result",
        pattern_shape_map={"Winners": "", "Non-winners": LOSE_PATTERN},
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "%{fullData.name}: %{customdata[0]}"
            "<extra></extra>"
        )
    )

    tv, tt = log_ticks(chart_data["Median Assets"])
    style_fig(fig, height=420)
    fig.update_layout(bargap=0.35, bargroupgap=0.08)
    fig.update_xaxes(title="", showgrid=False)
    fig.update_yaxes(title="Median assets (log scale)", type="log", tickvals=tv, ticktext=tt)
    return fig

def short_party(name, max_len=10):
    """Abbreviate long party names to initials so legends stay compact."""
    name = str(name)
    if len(name) <= max_len or " " not in name:
        return name
    return "".join(w[0].upper() for w in re.split(r"[\s\-]+", name) if w)

PIE_TOP_N = 7

def _top_parties(data, top_n=PIE_TOP_N):
    """Legend labels of the parties (excluding independents) that get their own slice in a pie."""
    assets = data.groupby("Party")["Total Assets"].sum().drop("IND", errors="ignore")
    return [short_party(p) for p in assets.sort_values(ascending=False).head(top_n).index]

def build_party_styles(frames, top_n=PIE_TOP_N):
    """One fixed (colour, hatch) per party, shared by every pie.

    Keyed by the legend label (``short_party``), so spellings that display the same
    (e.g. "YSRCP" and "Yuvajana Sramika Rythu Congress Party") share one colour.

    Parties are taken in a stable order (number of pies they appear in, then total
    declared assets across all elections, then name). Parties shown in more than one
    pie get a colour of their own; the rest take the first slot not used by a recurring
    party or by anyone they share a pie with, so slices never clash within a pie and a
    party never changes colour.
    """
    palette = list(zip(PIE_COLORS + PIE_COLORS_EXTRA, PIE_PATTERNS + PIE_PATTERNS_EXTRA))
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["Party"].notna() & (combined["Party"] != "IND")]
    totals = combined.groupby(combined["Party"].map(short_party))["Total Assets"].sum()

    neighbours = {p: set() for p in totals.index}
    appearances = {p: 0 for p in totals.index}
    for frame in frames:
        shown = _top_parties(frame, top_n)
        for p in shown:
            appearances[p] += 1
            neighbours[p].update(q for q in shown if q != p)
    order = sorted(totals.index, key=lambda p: (-min(appearances[p], 2), -totals[p], str(p)))

    slots, reserved = {}, set()
    for i, party in enumerate(order):
        taken = {slots[q] for q in neighbours[party] if q in slots}
        free = [k for k in range(len(palette)) if k not in taken]
        preferred = [k for k in free if k not in reserved]
        slots[party] = (preferred or free or [i % len(palette)])[0]
        if appearances[party] > 1:
            reserved.add(slots[party])
    styles = {p: palette[k] for p, k in slots.items()}
    styles["IND"] = styles["Others"] = (OTHERS_COLOR, OTHERS_PATTERN)  # independents sit in Others
    return styles

PARTY_STYLES = build_party_styles(list(CLEANED.values()))
PARTY_COLORS = {party: color for party, (color, _) in PARTY_STYLES.items()}  # legend label -> colour

def _is_light(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5

def create_party_assets_pie(data, state, year, top_n=PIE_TOP_N):
    party_assets = (
        data.groupby("Party")["Total Assets"]
        .sum()
        .sort_values(ascending=False)
    )

    independent_assets = party_assets.get("IND", 0)
    party_assets = party_assets.drop("IND", errors="ignore")
    top_parties = party_assets.head(top_n).copy()
    other_assets = party_assets.iloc[top_n:].sum() + independent_assets

    if other_assets > 0:
        top_parties.loc["Others"] = other_assets
    top_parties = top_parties.reset_index(name="Total Assets")
    top_parties["Formatted"] = top_parties["Total Assets"].apply(money_short)
    top_parties["Short"] = top_parties["Party"].map(short_party)

    # fixed per-party colours from PARTY_STYLES, so a party looks the same in every pie
    color_map, patterns = {}, []
    for full, short in zip(top_parties["Party"], top_parties["Short"]):
        color, pattern = PARTY_STYLES.get("Others" if full == "Others" else short,
                                          (OTHERS_COLOR, OTHERS_PATTERN))
        color_map[short] = color
        patterns.append(pattern)
    # dark labels on light slices, white labels on dark slices
    label_colors = ["#0B1020" if _is_light(color_map[s]) else "#FFFFFF" for s in top_parties["Short"]]

    fig = px.pie(
        data_frame=top_parties,
        names="Short",
        values="Total Assets",
        title=f"{state} ({year})",
        hole=0.55,
        color="Short",
        color_discrete_map=color_map,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent",
        textfont=dict(color=label_colors, size=12),
        marker=dict(line=dict(color=CARD, width=2),
                    pattern=dict(shape=patterns, fillmode="overlay", fgcolor=BG, solidity=0.25, size=8)),
        hovertemplate="<b>%{label}</b><br>Assets: %{customdata}<br>Share: %{percent}<extra></extra>",
        customdata=top_parties["Party"],
        sort=False,
        domain=dict(x=[0.05, 0.95], y=[0.2, 0.92]),  # same donut size in every card
    )

    style_fig(fig, height=400)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=0, xanchor="center", x=0.5),
        margin=dict(t=50, l=10, r=10, b=10),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    try:  # keep the legend from resizing the plot area (plotly >= 5.x container refs)
        fig.update_layout(legend=dict(yref="container"))
    except ValueError:
        pass
    return fig

# ----------------------------------------------------------------------------
# Winner / loser transitions (Sankey)
# ----------------------------------------------------------------------------
SANKEY_LINKS = {
    # (was a winner earlier, is a winner now): (name, link colour)
    (True, True): ("Won again", "rgba(252,211,77,0.38)"),
    (True, False): ("Won, then lost", "rgba(255,77,141,0.60)"),
    (False, True): ("Lost, then won", "rgba(45,212,191,0.60)"),
    (False, False): ("Lost again", "rgba(59,130,246,0.38)"),
}

def calculate_outcome_flows(previous_df: pd.DataFrame, current_df: pd.DataFrame):
    """Count returning candidates by (won earlier, won now), using the Recontester links."""
    counts = {key: 0 for key in SANKEY_LINKS}
    now = pd.DataFrame({
        "prev": pd.to_numeric(current_df["Recontester"], errors="coerce"),
        "won_now": current_df["winner"].map(_as_bool),
    })
    now = now[now["prev"].notna() & (now["prev"] >= 0)]
    before = pd.DataFrame({
        "prev": pd.to_numeric(previous_df["Sno."], errors="coerce"),
        "won_before": previous_df["winner"].map(_as_bool),
    }).drop_duplicates("prev")
    merged = now.merge(before, on="prev", how="inner")
    for (won_before, won_now), n in merged.groupby(["won_before", "won_now"]).size().items():
        counts[(bool(won_before), bool(won_now))] = int(n)
    return counts

def create_outcome_sankey(stages):
    """stages: list of (from_year, to_year, counts) in election order."""
    years = [stages[0][0]] + [to_year for _, to_year, _ in stages]
    sources, targets, values, colors, link_names, link_info = [], [], [], [], [], []
    for col, (from_year, to_year, counts) in enumerate(stages):
        for won_before in (True, False):
            group_total = counts[(won_before, True)] + counts[(won_before, False)]
            for won_now in (True, False):
                n = counts[(won_before, won_now)]
                if n == 0:
                    continue
                name, color = SANKEY_LINKS[(won_before, won_now)]
                sources.append(2 * col + (0 if won_before else 1))
                targets.append(2 * (col + 1) + (0 if won_now else 1))
                values.append(n)
                colors.append(color)
                link_names.append(name)
                who = "winners" if won_before else "non-winners"
                link_info.append(f"{n / group_total:.0%} of the {group_total} {from_year} {who} who contested again in {to_year}")

    fig = go.Figure()
    if not values:
        fig.add_annotation(xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           text="No returning candidates could be linked for this interval",
                           font=dict(color=MUTED, size=13))
        style_fig(fig, height=300)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    node_labels, node_colors = [], []
    for year in years:
        node_labels += [f"{year} · Winners", f"{year} · Non-winners"]
        node_colors += [WIN, LOSE]
    fig.add_trace(go.Sankey(
        arrangement="snap",
        textfont=dict(color=TEXT, size=12, family=FONT),
        node=dict(label=node_labels, color=node_colors, pad=28, thickness=22,
                  line=dict(color=BG, width=1),
                  hovertemplate="%{label}<br>%{value} candidates<extra></extra>"),
        link=dict(source=sources, target=targets, value=values, color=colors,
                  customdata=np.column_stack([link_names, link_info]),
                  hovertemplate=("<b>%{customdata[0]}</b><br>%{source.label} → %{target.label}<br>"
                                 "%{value} candidates<br>%{customdata[1]}<extra></extra>")),
    ))
    style_fig(fig, height=460)
    fig.update_layout(title="Winners and losers: who changed sides?", margin=dict(t=70, l=24, r=24, b=24))
    return fig

def outcome_sankey_note(stages):
    parts = []
    for from_year, to_year, c in stages:
        w_total = c[(True, True)] + c[(True, False)]
        l_total = c[(False, True)] + c[(False, False)]
        parts.append(
            f"{from_year} → {to_year}: of {w_total} {from_year} winners who contested again, "
            f"{c[(True, True)]} won again and {c[(True, False)]} lost; of {l_total} non-winners who "
            f"returned, {c[(False, True)]} won and {c[(False, False)]} lost again."
        )
    tail = ""
    if len(stages) > 1:
        tail = (" Each interval counts every candidate who contested in both of its elections, so the 2019 boxes "
                "can differ in height on their left and right sides.")
    return " ".join(parts) + tail

# ----------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------
KPI_CLASS = "kpi-value"
KPI_HIGHLIGHT_CLASS = "kpi-value kpi-highlight"

def register_state_callback(prefix, d14, d19, d24, highlight_period=None):
    """highlight_period: interval in which the winners' median-change card is spotlighted."""
    @app.callback(
        Output(f"{prefix}-winners-median-pchange", "children"),
        Output(f"{prefix}-winners-median-change", "children"),
        Output(f"{prefix}-losers-median-pchange", "children"),
        Output(f"{prefix}-losers-median-change", "children"),
        Output(f"{prefix}-asset-graph", "figure"),
        Output(f"{prefix}-asset-histogram", "figure"),
        Output(f"{prefix}-asset-graph-constituency", "figure"),
        Output(f"{prefix}-winners-median-change", "className"),
        Input(f"{prefix}-period", "value"),
    )
    def update(period):
        css = KPI_HIGHLIGHT_CLASS if period == highlight_period else KPI_CLASS
        if period == "2014-2019":
            return (*calculate_statistics(calculate_asset_changes(d14, d19)), css)

        if period == "2019-2024":
            return (*calculate_statistics(calculate_asset_changes(d19, d24)), css)

        records_14_19 = calculate_asset_changes(d14, d19)
        records_19_24 = calculate_asset_changes(d19, d24)
        records_14_19["Period"] = "2014-2019"
        records_19_24["Period"] = "2019-2024"
        combined = pd.concat([records_14_19, records_19_24], ignore_index=True)
        return (*calculate_statistics(combined), css)

    return update

register_state_callback("ap", andhra_14, andhra_19, andhra_24, highlight_period="2019-2024")
register_state_callback("br", bihar_14, bihar_19, bihar_24)
register_state_callback("gj", gujarat_14, gujarat_19, gujarat_24)

def register_sankey_callback(prefix, d14, d19, d24):
    @app.callback(
        Output(f"{prefix}-outcome-sankey", "figure"),
        Output(f"{prefix}-outcome-sankey-note", "children"),
        Input(f"{prefix}-period", "value"),
    )
    def update_sankey(period):
        stages = []
        if period in ("2014-2019", "Both"):
            stages.append((2014, 2019, calculate_outcome_flows(d14, d19)))
        if period in ("2019-2024", "Both"):
            stages.append((2019, 2024, calculate_outcome_flows(d19, d24)))
        return create_outcome_sankey(stages), outcome_sankey_note(stages)

    return update_sankey

register_sankey_callback("ap", andhra_14, andhra_19, andhra_24)
register_sankey_callback("br", bihar_14, bihar_19, bihar_24)
register_sankey_callback("gj", gujarat_14, gujarat_19, gujarat_24)

@app.callback(
    Output("ap-candidate-count", "children"),
    Output("br-candidate-count", "children"),
    Output("gj-candidate-count", "children"),
    Output("ap-total-winner-assets", "children"),
    Output("ap-total-loser-assets", "children"),
    Output("br-total-winner-assets", "children"),
    Output("br-total-loser-assets", "children"),
    Output("gj-total-winner-assets", "children"),
    Output("gj-total-loser-assets", "children"),
    Output("median-assets-comparison-graph", "figure"),
    Output("ap-asset-parties-graph", "figure"),
    Output("br-asset-parties-graph", "figure"),
    Output("gj-asset-parties-graph", "figure"),
    Output("ap-total-winner-assets", "className"),
    Input("assets-period", "value")
)
def update_party_assets_dashboard(year):
    datasets = {
        "2014": (andhra_14, bihar_14, gujarat_14),
        "2019": (andhra_19, bihar_19, gujarat_19),
        "2024": (andhra_24, bihar_24, gujarat_24),
    }
    ap_data, br_data, gj_data = datasets.get(year, datasets["2024"])

    ap_summary = summarize_election_assets(ap_data)
    br_summary = summarize_election_assets(br_data)
    gj_summary = summarize_election_assets(gj_data)

    return (
        f"{len(ap_data):,}",
        f"{len(br_data):,}",
        f"{len(gj_data):,}",
        ap_summary["total_winners"],
        ap_summary["total_non_winners"],
        br_summary["total_winners"],
        br_summary["total_non_winners"],
        gj_summary["total_winners"],
        gj_summary["total_non_winners"],
        create_median_assets_comparison(ap_data, br_data, gj_data, year),
        create_party_assets_pie(ap_data, "Andhra Pradesh", year),
        create_party_assets_pie(br_data, "Bihar", year),
        create_party_assets_pie(gj_data, "Gujarat", year),
        # spotlight the Andhra Pradesh winners' total (₹7698.2 crore) in 2024
        KPI_HIGHLIGHT_CLASS if year == "2024" else KPI_CLASS,
    )

# ----------------------------------------------------------------------------
# Cleaning check
# ----------------------------------------------------------------------------
# light -> mid -> dark so the three stages separate by lightness alone (box traces take no hatching)
STAGE_COLORS = {"Source CSV": "#E2E8F0", "OCR enriched": "#F59E0B", "Missing rows dropped": "#3B82F6"}
STAGES = list(STAGE_COLORS)

def clean_selection(mode, scope):
    """Turn (mode, scope) into states/years. Invalid scopes (mid-switch) fall back to 'all'."""
    if mode == "year":
        states = [scope] if scope in STATES else list(STATES)
        state_label = states[0] if len(states) == 1 else "all three states pooled"
        return states, list(YEARS), "Year", [str(y) for y in YEARS], f"Comparing years, {state_label}"
    years = [int(scope)] if str(scope) in {str(y) for y in YEARS} else list(YEARS)
    year_label = f"the {years[0]} election" if len(years) == 1 else "all three elections pooled"
    return list(STATES), years, "State", list(STATES), f"Comparing states, {year_label}"

def create_cleaning_figure(states, years, dim, groups, context):
    frames = []
    for st in states:
        for year in years:
            raw_assets = RAW[(st, year)].set_index("Sno.")["Total Assets"].map(rupees)
            ocr_assets = OCR[(st, year)].set_index("Sno.")["Total Assets"].map(rupees).reindex(raw_assets.index)
            final_assets = pd.to_numeric(DROPPED[(st, year)]["Total Assets"], errors="coerce")
            for stage, vals in zip(STAGES, (raw_assets, ocr_assets, final_assets)):
                vals = vals.dropna()
                frames.append(pd.DataFrame({"State": st, "Year": str(year), "Stage": stage, "Assets": vals.to_numpy()}))
    clean = pd.concat(frames, ignore_index=True)
    clean = clean[clean["Assets"] > 0]

    fig = px.box(
        clean, x=dim, y="Assets", color="Stage", points=False,
        color_discrete_map=STAGE_COLORS, category_orders={dim: groups, "Stage": STAGES},
        title=("Asset values across the recovery and cleaning stages"
               f"<br><span style='font-size:12px;color:{MUTED}'>{context}; log scale; "
               "only parseable positive values shown</span>"),
    )
    fig.update_traces(hovertemplate="%{x}<br>%{fullData.name}<br>₹%{y:,.0f}<extra></extra>")
    tv, tt = log_ticks(clean["Assets"])
    style_fig(fig, height=480)
    fig.update_layout(margin=dict(t=90, l=60, r=24, b=80),
                      legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5))
    fig.update_xaxes(title="Election year" if dim == "Year" else "", showgrid=False)
    fig.update_yaxes(title="Declared assets (log scale)", type="log", tickvals=tv, ticktext=tt)
    return fig

@app.callback(
    Output("clean-scope", "options"),
    Output("clean-scope", "value"),
    Output("clean-scope-label", "children"),
    Input("clean-mode", "value"),
)
def update_clean_scope(mode):
    return clean_scope_options(mode), "all", "State sample" if mode == "year" else "Election year"

@app.callback(
    Output("clean-source", "children"),
    Output("clean-parsed", "children"),
    Output("clean-recovered", "children"),
    Output("clean-missing", "children"),
    Output("clean-dropped", "children"),
    Output("clean-final", "children"),
    Output("clean-plot", "figure"),
    Output("clean-note", "children"),
    Input("clean-mode", "value"),
    Input("clean-scope", "value"),
)
def update_cleaning(mode, scope):
    states, years, dim, groups, context = clean_selection(mode, scope)
    keys = [(st, y) for st in states for y in years]

    source_n = sum(len(RAW[k]) for k in keys)
    final_n = sum(len(DROPPED[k]) for k in keys)
    dropped_n = sum(len(CLEANED_RAW[k]) for k in keys) - final_n
    missing_rows = int(sum(CLEANED_RAW[k][MISSING_COLUMNS].eq(-1).any(axis=1).sum() for k in keys))
    missing_assets = int(sum(CLEANED_RAW[k]["Total Assets"].eq(-1).sum() for k in keys))

    raw_parsed = ocr_recovered = 0
    for k in keys:
        raw_values = RAW[k].set_index("Sno.")["Total Assets"].map(rupees)
        ocr_values = OCR[k].set_index("Sno.")["Total Assets"].map(rupees).reindex(raw_values.index)
        raw_parsed += int(raw_values.notna().sum())
        ocr_recovered += int((raw_values.isna() & ocr_values.notna()).sum())

    duplicates = dropped_n - missing_rows
    note = (f"Among {source_n:,} source rows, {raw_parsed:,} asset strings parse directly and OCR recovers "
            f"{ocr_recovered:,} more. After parsing, {missing_rows:,} rows still lack age, assets or liabilities "
            f"({missing_assets:,} of them assets) and are dropped"
            + (f", along with {duplicates:,} duplicate row(s)" if duplicates > 0 else "")
            + f", leaving {final_n:,} rows; no values are imputed. "
            "Stage distributions include only parseable positive values.")

    return (
        f"{source_n:,}", f"{raw_parsed:,}", f"{ocr_recovered:,}",
        f"{missing_rows:,}", f"{dropped_n:,}", f"{final_n:,}",
        create_cleaning_figure(states, years, dim, groups, context),
        note,
    )

# ----------------------------------------------------------------------------
# Constituency map
# ----------------------------------------------------------------------------
def _crore(value, signed=False):
    if pd.isna(value):
        return "no data"
    return f"₹{value:+,.2f} crore" if signed else f"₹{value:,.2f} crore"

def _short_change(value, measure):
    if pd.isna(value):
        return "–"
    if measure == "pct":
        return f"{value:+,.0f}%"
    return f"{value:+,.1f}" if abs(value) < 100 else f"{value:+,.0f}"

def _asinh_scale(measure):
    """Softening constant for the colour scale: 1 crore or 10 %."""
    return 10.0 if measure == "pct" else 1.0

def _colorbar(measure, zmax):
    scale = _asinh_scale(measure)
    raw = [-100, -50, 0, 100, 1000, 10000] if measure == "pct" else [-100, -10, -1, 0, 1, 10, 100]
    ticks = [v for v in raw if abs(np.arcsinh(v / scale)) <= zmax * 1.001]
    label = "%" if measure == "pct" else " cr"
    return {
        "title": {"text": "Change<br>(%)" if measure == "pct" else "Change<br>(₹ crore)", "font": {"color": MUTED}},
        "tickvals": [float(np.arcsinh(v / scale)) for v in ticks],
        "ticktext": [f"{v:+,g}{label}" if v else "0" for v in ticks],
        "tickfont": {"color": MUTED}, "thickness": 12, "len": 0.8, "outlinewidth": 0,
    }

def _map_hover(row, period):
    lines = [f"<b>{row['Constituency']}</b> ({row['State']}) · {period}", f"Winner: {row['Winner']} ({row['Party']})"]
    if pd.notna(row["Change (₹ crore)"]):
        lines += [f"Start: {_crore(row['Start assets'])}", f"End: {_crore(row['End assets'])}",
                  f"Change: {_crore(row['Change (₹ crore)'], True)}"
                  + (f" ({row['Change (%)']:+,.0f}%)" if pd.notna(row["Change (%)"]) else " (% undefined)")]
    lines.append(f"Status: {row['Status']}")
    return "<br>".join(lines)

def _tile(value, label, kind="", small=False):
    return html.Div([html.Div(value, className="q-value" + (" q-value-small" if small else "")),
                     html.Div(label, className="q-label")],
                    className="q-tile" + (f" q-{kind}" if kind else ""))

@app.callback(
    Output("map-definition", "children"),
    Output("constituency-map", "figure"),
    Output("constituency-map-note", "children"),
    Output("quality-tiles", "children"),
    Input("map-state", "value"),
    Input("map-period", "value"),
    Input("map-measure", "value"),
)
def update_constituency_map(map_state, period, measure):
    basis = "start"  # winner = won the seat in the period's first election
    start, end = MAP_PERIODS[period]
    seat_year = start if basis == "start" else end
    value_col = "Change (%)" if measure == "pct" else "Change (₹ crore)"
    scale = _asinh_scale(measure)
    unit = "% of first-year assets" if measure == "pct" else "₹ crore"

    current = constituency_changes(period, basis).copy()
    current["Value"] = current[value_col]
    state_rows = current[current["State"] == map_state].copy()
    state_rows["id"] = map_state + "|" + state_rows["key"]
    # one colour range for all three states, so switching state keeps colours comparable
    finite = current["Value"].dropna()
    zmax = float(np.arcsinh(finite.abs().max() / scale)) if len(finite) and finite.abs().max() > 0 else 1.0

    definition = [
        html.Strong("Definition. "),
        f"Asset change = declared total assets at the {end} election − at the {start} election, "
        f"for the candidate who won each constituency in {seat_year}"
        + (" (the sitting MP), followed forward" if basis == "start" else ", followed back")
        + f" to {end if basis == 'start' else start} through Recontester links. Shown in {unit}. ",
        html.Span(
            f"Seats are attributed to the constituency won in {seat_year}, even if the candidate contested "
            "a different seat in the other election. Only declared asset values are used; candidates with "
            "missing declarations were dropped during cleaning.",
            className="callout-detail",
        ),
    ]

    geo_unmatched = []
    fig = go.Figure()
    if GEOMETRY and GEOMETRY.get(map_state, {}).get("features"):
        collection = GEOMETRY[map_state]
        geo_ids = {f["id"] for f in collection["features"]}
        geo_names = {f["id"]: f["properties"]["display_name"] for f in collection["features"]}
        matched = state_rows[state_rows["id"].isin(geo_ids)]
        geo_unmatched = sorted(state_rows.loc[~state_rows["id"].isin(geo_ids), "Constituency"])
        boundary_only = sorted(geo_names[i] for i in geo_ids - set(state_rows["id"]))

        no_value = matched[matched["Value"].isna()]
        fig.add_trace(go.Choropleth(
            geojson=collection, featureidkey="id", locations=no_value["id"], z=[0] * len(no_value),
            colorscale=[[0, CARD], [1, CARD]], showscale=False, name="No computable change (unfilled)",
            marker={"line": {"color": MUTED, "width": 1.2}}, showlegend=True,
            text=[_map_hover(r, period) for _, r in no_value.iterrows()], hovertemplate="%{text}<extra></extra>",
        ))
        with_value = matched[matched["Value"].notna()]
        fig.add_trace(go.Choropleth(
            geojson=collection, featureidkey="id", locations=with_value["id"],
            z=np.arcsinh(with_value["Value"] / scale), zmin=-zmax, zmax=zmax, colorscale=DIVERGING,
            colorbar=_colorbar(measure, zmax), marker={"line": {"color": BG, "width": 1}}, name="Asset change",
            text=[_map_hover(r, period) for _, r in with_value.iterrows()], hovertemplate="%{text}<extra></extra>",
        ))
        fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)", projection_type="mercator",
                        domain={"x": [0, 0.9], "y": [0.06, 1]})
        fig.update_layout(title=f"{map_state}: winners' asset change, {period.replace('–', ' → ')}")
        style_fig(fig, height=560)
        fig.update_layout(margin={"l": 8, "r": 8, "t": 52, "b": 8},
                          legend={"y": 0, "yanchor": "bottom", "x": 0, "xanchor": "left"})
        note = (
            f"{len(with_value)} of {len(state_rows)} seats in {map_state} are drawn with a value; unfilled outlined "
            f"seats have no linked value and {len(geo_unmatched)} seat name(s) could not be matched to a boundary. "
            "Colour uses a compressed (asinh) scale so small and large changes both show; hover for exact amounts. "
            + (f" Boundary names not in the election data: {', '.join(boundary_only)}." if boundary_only else "")
        )
    else:
        fig.add_shape(type="rect", xref="paper", yref="paper", x0=0.02, y0=0.04, x1=0.98, y1=0.94,
                      line={"color": BORDER, "width": 1.5, "dash": "dash"})
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.55, showarrow=False, align="center",
            text=(f"<b>Map placeholder · {map_state}</b><br><br>No constituency boundary file is connected.<br>"
                  "Add <i>geo/india_pc_2019_simplified.geojson</i> (DataMeet, 2008 delimitation)<br>"
                  "and restart the app; constituencies join on normalised names."),
            font={"size": 13, "color": MUTED},
        )
        fig.update_layout(title=f"{map_state}: winners' asset change, {period} (boundaries not connected)")
        style_fig(fig, height=560)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        computed = state_rows["Value"].notna()
        top = state_rows[computed].sort_values("Value", ascending=False)
        describe = lambda r: f"{r['Constituency']} ({_short_change(r['Value'], measure)}{'' if measure == 'pct' else ' cr'})"
        note = (f"Placeholder: no boundaries are drawn because no geometry file is present. "
                f"{int(computed.sum())} of {len(state_rows)} {map_state} seats have a computable change"
                + (f"; highest {describe(top.iloc[0])}, lowest {describe(top.iloc[-1])}." if len(top) else "."))

    # Data quality tiles for the state / period / definition on screen
    counts = state_rows["Status"].value_counts()
    linked = int(counts.filter(like="Linked").sum())
    tiles = [
        _tile(f"{len(state_rows)}", f"official seats in {map_state}"),
        _tile(f"{linked}", "with a linked asset change", "good"),
    ]
    for status, n in counts.items():
        # "Winner not found in <year> candidate file" tiles are deliberately not shown
        if not status.startswith(("Linked", "Winner not found")):
            tiles.append(_tile(f"{n}", status, "warn"))
    if measure == "pct":
        undefined = int((state_rows["Change (₹ crore)"].notna() & state_rows["Change (%)"].isna()).sum())
        if undefined:
            tiles.append(_tile(f"{undefined}", "zero first-year assets (% undefined)", "warn"))
    tiles.append(_tile(
        "Real boundaries" if GEOMETRY else "No geometry",
        f"{len(geo_unmatched)} seat name(s) unmatched to boundaries" if GEOMETRY else "map shown as a labelled placeholder",
        "" if GEOMETRY and not geo_unmatched else "warn", small=True,
    ))
    return definition, fig, note, tiles

# ----------------------------------------------------------------------------
# Relationships (bivariate analysis)
# ----------------------------------------------------------------------------
REL_STATE_OPTIONS = [{"label": "All states", "value": "all"}] + [{"label": s, "value": s} for s in STATES]
REL_YEAR_OPTIONS = [{"label": "All elections", "value": "all"}] + [{"label": str(y), "value": str(y)} for y in YEARS]
CASE_BANDS = ["0", "1", "2–4", "5+"]
EDU_ORDER = ["Illiterate", "Literate", "5th Pass", "8th Pass", "10th Pass", "12th Pass", "Graduate",
             "Graduate Professional", "Post Graduate", "Doctorate", "Others", "Not Given"]


def relationships_page():
    return html.Div([
        page_header(
            "ANALYTICS / RELATIONSHIPS", "How Assets Relate to Other Traits",
            "Bivariate views of the cleaned affidavits: declared assets against age, criminal cases, "
            "liabilities, education and the candidate's own earlier declaration.", STATUS,
        ),
        filters(
            filter_group("State", segmented("rel-state", REL_STATE_OPTIONS, "all")),
            filter_group("Election year", segmented("rel-year", REL_YEAR_OPTIONS, "all")),
        ),

        section("Rank correlation with declared assets",
                "Spearman's ρ: +1 means the two always rise together, 0 means no monotonic link."),
        dbc.Row([
            dbc.Col(kpi("rel-rho-past", "Earlier vs later assets", "win", "Returning candidates"), md=6, xl=3),
            dbc.Col(kpi("rel-rho-liab", "Assets vs liabilities", "neutral", "All candidates"), md=6, xl=3),
            dbc.Col(kpi("rel-rho-age", "Assets vs age", "neutral", "All candidates"), md=6, xl=3),
            dbc.Col(kpi("rel-rho-crime", "Assets vs criminal cases", "lose", "All candidates"), md=6, xl=3),
        ], className="g-3 mb-4"),

        dbc.Row([
            dbc.Col(chart(
                "rel-past-scatter",
                "Each dot is a candidate who contested again: earlier declared assets across, later assets up. "
                "Dots above the dashed line got richer, dots below got poorer. Gold diamonds won the earlier "
                "election, blue circles did not. Both axes are logarithmic.",
            ), xl=12),
        ], className="g-3 mb-4"),

        dbc.Row([
            dbc.Col(chart(
                "rel-crime-box",
                "Declared assets for candidates grouped by how many criminal cases they declared, split into "
                "winners (gold) and non-winners (blue). The line inside each box is the median.",
            ), xl=6),
            dbc.Col(chart(
                "rel-edu-bar",
                "Median declared assets for each education level, winners (gold) next to non-winners (blue, "
                "hatched). The scale is logarithmic; n is the number of candidates at that level and how many "
                "of them won. Groups with fewer than three candidates are left blank.",
            ), xl=6),
        ], className="g-3 mb-4"),

        dbc.Row([
            dbc.Col(chart(
                "rel-edu-crime-bars",
                "Candidates are grouped by education, from least to most educated, and within each group by how "
                "many criminal cases they declared. Each bar is the group's median declared assets, written on "
                "top. Bars rising to the right mean more educated candidates declare more; deeper-coloured bars "
                "standing taller mean candidates with more cases declare more. The axis is a normal (linear) "
                "rupee scale, so small medians in lakhs appear as short bars. Groups with fewer than three "
                "candidates are left out.",
            ), xl=12),
        ], className="g-3"),

        html.Div("Caveat: these are associations in self-declared affidavits, not causal effects. Correlations "
                 "use ranks, so a handful of very rich candidates cannot dominate them.", className="note-panel"),
    ], className="page")


def relationship_frame(state, year):
    states = list(STATES) if state == "all" else [state]
    years = YEARS if year == "all" else [int(year)]
    frames = []
    for st in states:
        for y in years:
            d = CLEANED[(st, y)].copy()
            d["State"], d["Year"] = st, y
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["Won"] = df["winner"].map(_as_bool).astype(int)
    df["Result"] = np.where(df["Won"] == 1, "Winners", "Non-winners")
    return df[df["Total Assets"] > 0]


def recontest_pairs(state, year):
    states = list(STATES) if state == "all" else [state]
    # pairs end in the chosen year; 2014 is the first election, so it has none
    targets = [2019, 2024] if year == "all" else [] if year == "2014" else [int(year)]
    frames = []
    for st in states:
        for y in targets:
            prev = YEARS[YEARS.index(y) - 1]
            ch = calculate_asset_changes(CLEANED[(st, prev)], CLEANED[(st, y)])
            if ch.empty:
                continue
            ch["Later_assets"] = ch["Previous_assets"] + ch["Asset Change"]
            ch["State"], ch["Interval"] = st, f"{prev}→{y}"
            frames.append(ch)
    if not frames:
        return pd.DataFrame(columns=["Previous_assets", "Later_assets", "Winner", "Candidate", "State", "Interval"])
    pairs = pd.concat(frames, ignore_index=True)
    pairs["Winner"] = pairs["Winner"].map(_as_bool)
    return pairs[(pairs["Previous_assets"] > 0) & (pairs["Later_assets"] > 0)]


def rho(a, b):
    if len(a) < 3:
        return "—"
    value = pd.Series(a).corr(pd.Series(b), method="spearman")
    return "—" if pd.isna(value) else f"{value:+.2f}"


RESULT_STYLE = {  # drawn losers first so winners sit on top
    "Non-winners": dict(color=LOSE, symbol="circle", opacity=0.45),
    "Winners": dict(color=WIN, symbol="diamond", opacity=0.9),
}


def scatter_by_result(fig, df, x, y, hover):
    for result, style in RESULT_STYLE.items():
        part = df[df["Result"] == result]
        fig.add_trace(go.Scattergl(
            x=x(part), y=y(part), mode="markers", name=f"{result} (n={len(part)})",
            marker=dict(color=style["color"], symbol=style["symbol"], size=7, opacity=style["opacity"],
                        line=dict(color=BG, width=0.5)),
            customdata=hover(part), hovertemplate="%{customdata}<extra></extra>",
        ))


def empty_figure(message, height=440):
    fig = go.Figure()
    fig.add_annotation(xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, text=message,
                       font=dict(color=MUTED, size=13))
    style_fig(fig, height=height)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def create_past_scatter(pairs, year="all"):
    if year == "2014":
        fig = empty_figure("2014 is the first election in the data, so there is no earlier declaration "
                           "to compare with.<br>Choose 2019, 2024 or All elections to see returning candidates.",
                           height=460)
        fig.update_layout(title="Earlier vs later declared assets (returning candidates)")
        return fig
    if pairs.empty:
        return empty_figure("No returning candidates could be linked for this selection")
    df = pairs.assign(Result=np.where(pairs["Winner"], "Winners", "Non-winners"))
    fig = go.Figure()
    scatter_by_result(
        fig, df, lambda d: d["Previous_assets"], lambda d: d["Later_assets"],
        lambda d: [f"<b>{r.Candidate}</b> ({r.State}, {r.Interval})<br>Earlier: {money_short(r.Previous_assets)}"
                   f"<br>Later: {money_short(r.Later_assets)}" for r in d.itertuples()],
    )
    lo = min(df["Previous_assets"].min(), df["Later_assets"].min())
    hi = max(df["Previous_assets"].max(), df["Later_assets"].max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="No change",
                             line=dict(color=MUTED, dash="dash", width=1.2), hoverinfo="skip"))
    tv, tt = log_ticks(np.r_[df["Previous_assets"], df["Later_assets"]])
    style_fig(fig, height=460)
    fig.update_layout(title="Earlier vs later declared assets (returning candidates)")
    fig.update_xaxes(type="log", title="Assets at earlier election (log scale)", tickvals=tv, ticktext=tt)
    fig.update_yaxes(type="log", title="Assets at later election (log scale)", tickvals=tv, ticktext=tt)
    return fig




def create_crime_box(df):
    cases = df["Criminal Cases"]
    df = df.assign(Band=case_band(cases))
    fig = go.Figure()
    for result, color, fill in (("Winners", WIN, "rgba(252,211,77,0.35)"), ("Non-winners", LOSE, "rgba(59,130,246,0.35)")):
        part = df[df["Result"] == result]
        fig.add_trace(go.Box(
            x=part["Band"], y=part["Total Assets"], name=result, marker_color=color,
            line=dict(color=color), fillcolor=fill, boxpoints=False,
            hovertemplate="%{x} case(s)<br>₹%{y:,.0f}<extra></extra>",
        ))
    counts = df["Band"].value_counts()
    tv, tt = log_ticks(df["Total Assets"])
    style_fig(fig, height=440)
    fig.update_layout(boxmode="group", title="Declared assets by number of criminal cases")
    fig.update_xaxes(title="Criminal cases declared", categoryorder="array", categoryarray=CASE_BANDS,
                     tickvals=CASE_BANDS, ticktext=[f"{b}<br><span style='font-size:10px'>n={counts.get(b, 0)}</span>"
                                                    for b in CASE_BANDS], showgrid=False)
    fig.update_yaxes(type="log", title="Declared assets (log scale)", tickvals=tv, ticktext=tt)
    return fig



def case_band(cases):
    return np.select([cases == 0, cases == 1, cases <= 4], CASE_BANDS[:3], CASE_BANDS[3])


EDU_BANDS = {  # ordered least -> most educated; Others / Not Given are left out
    "Below 10th": ["Illiterate", "Literate", "5th Pass", "8th Pass"],
    "10th–12th": ["10th Pass", "12th Pass"],
    "Graduate": ["Graduate", "Graduate Professional"],
    "Post-graduate & above": ["Post Graduate", "Doctorate"],
}
EDU_BAND_OF = {level: band for band, levels in EDU_BANDS.items() for level in levels}
# light -> deep as criminal cases rise; hatch on the two darker bars keeps them apart in greyscale
CASE_COLORS = {"0": "#A7F3D0", "1": "#38BDF8", "2–4": "#3B82F6", "5+": "#7C3AED"}
CASE_PATTERNS = {"0": "", "1": "", "2–4": "/", "5+": "x"}


def create_edu_crime_bars(df):
    """Median declared assets by education band, one bar per criminal-case band."""
    df = df.assign(Band=case_band(df["Criminal Cases"]), EduBand=df["Education"].map(EDU_BAND_OF))
    df = df[df["EduBand"].notna()]
    if df.empty:
        return empty_figure("No candidates with a known education level for this selection")
    stats = df.groupby(["EduBand", "Band"])["Total Assets"].agg(["median", "size"])
    edu_order = list(EDU_BANDS)
    fig = go.Figure()
    for band in CASE_BANDS:
        rows = stats.xs(band, level="Band") if band in stats.index.get_level_values("Band") else stats.iloc[0:0]
        rows = rows.reindex(edu_order)
        sizes = rows["size"].fillna(0).astype(int)
        values = rows["median"].where(sizes >= 3)  # a median of one or two people is noise
        label = f"{band} case{'' if band == '1' else 's'}"
        fig.add_trace(go.Bar(
            x=edu_order, y=values, name=f"{label} (n={int(sizes.sum())})",
            marker=dict(color=CASE_COLORS[band], pattern=dict(shape=CASE_PATTERNS[band])),
            text=[money_short(v).replace(" Crores", " Cr").replace(" Lakhs", " L") if pd.notna(v) else ""
                  for v in values],
            textposition="outside", textfont=dict(size=11, color=TEXT_2), cliponaxis=False,
            customdata=np.column_stack([[money_short(v) for v in values], sizes]),
            hovertemplate=f"<b>%{{x}}</b> · {label}<br>Median: %{{customdata[0]}}<br>"
                          "Candidates: %{customdata[1]}<extra></extra>",
        ))
    shown = stats["median"].where(stats["size"] >= 3).dropna()
    tv, tt = linear_ticks(np.r_[0.0, shown.to_numpy(dtype=float)], target=6)
    counts = df["EduBand"].value_counts()
    style_fig(fig, height=480)
    fig.update_layout(barmode="group", bargap=0.22, bargroupgap=0.06,
                      title="Median declared assets by education and criminal cases",
                      legend_title_text="Criminal cases:")
    fig.update_xaxes(title="Education (least → most)", showgrid=False, categoryorder="array",
                     categoryarray=edu_order, tickvals=edu_order,
                     ticktext=[f"{e}<br><span style='font-size:10px'>n={counts.get(e, 0)}</span>" for e in edu_order])
    # linear axis from ₹0, with headroom for the value labels above the tallest bar
    fig.update_yaxes(title="Median declared assets", tickvals=tv, ticktext=tt,
                     range=[0, float(shown.max()) * 1.12])
    return fig


def create_edu_bar(df):
    levels = [e for e in EDU_ORDER if e in set(df["Education"])]
    grouped = df.groupby(["Education", "Result"])["Total Assets"].median().unstack()
    n_by = df.groupby(["Education", "Result"]).size().unstack(fill_value=0)
    # a median of one or two people is noise: hide groups with fewer than 3 candidates
    grouped = grouped.where(n_by.reindex_like(grouped).fillna(0) >= 3)
    counts = df["Education"].value_counts()
    winners_n = n_by.get("Winners", pd.Series(0, index=n_by.index))
    fig = go.Figure()
    for result, color, pattern in (("Winners", WIN, ""), ("Non-winners", LOSE, LOSE_PATTERN)):
        values = grouped.reindex(levels).get(result, pd.Series(index=levels, dtype=float))
        fig.add_trace(go.Bar(
            y=[f"{e} (n={counts[e]}, {winners_n.get(e, 0)} won)" for e in levels], x=values, orientation="h", name=result,
            marker=dict(color=color, pattern=dict(shape=pattern)),
            customdata=[money_short(v) for v in values],
            hovertemplate="%{y}<br>" + result + ": %{customdata}<extra></extra>",
        ))
    tv, tt = log_ticks(grouped.to_numpy().ravel())
    style_fig(fig, height=440)
    fig.update_layout(barmode="group", title="Median declared assets by education", bargap=0.25,
                      margin=dict(t=70, l=220, r=24, b=50))
    fig.update_xaxes(type="log", title="Median assets (log scale)", tickvals=tv, ticktext=tt)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11, color=MUTED))
    return fig


@app.callback(
    Output("rel-rho-past", "children"),
    Output("rel-rho-liab", "children"),
    Output("rel-rho-age", "children"),
    Output("rel-rho-crime", "children"),
    Output("rel-past-scatter", "figure"),
    Output("rel-crime-box", "figure"),
    Output("rel-edu-bar", "figure"),
    Output("rel-edu-crime-bars", "figure"),
    Input("rel-state", "value"),
    Input("rel-year", "value"),
)
def update_relationships(state, year):
    df = relationship_frame(state, year)
    pairs = recontest_pairs(state, year)
    return (
        rho(pairs["Previous_assets"], pairs["Later_assets"]),
        rho(df["Total Assets"], df["Liabilities"]),
        rho(df["Total Assets"], df["Age"]),
        rho(df["Total Assets"], df["Criminal Cases"]),
        create_past_scatter(pairs, year),
        create_crime_box(df),
        create_edu_bar(df),
        create_edu_crime_bars(df),
    )


if __name__ == "__main__":
    app.run(debug=False, port=8051)