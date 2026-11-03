# Import required libraries
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re

def normalize_constituency(c):
    return re.sub(r"\s*\((SC|ST)\)\s*$", "", str(c).strip())

def load(path):
    df = pd.read_csv(path)
    df["constituency"] = df["constituency"].map(normalize_constituency)
    return df

# State keys -> display name / file prefix
STATES = {"ap": "Andhra Pradesh", "br": "Bihar", "gj": "Gujarat"}
FILE_PREFIX = {"ap": "andhra", "br": "bihar", "gj": "gujarat"}
YEARS = ("2014", "2019", "2024")

# Load dataset files: DATA["ap"]["2014"] -> Andhra Pradesh 2014 dataframe
DATA = {
    key: {year: load(f"clean/{FILE_PREFIX[key]}_{year}.csv") for year in YEARS}
    for key in STATES
}

# ----------------------------------------------------------------------------
# Layout helpers
# ----------------------------------------------------------------------------
# Text colours
TEAL = "#2DD4BF"     # winners
INDIGO = "#818CF8"   # losers / non-winners
SKY = "#38BDF8"      # neutral (counts)
MUTED = "#8B93A7"    # labels / captions
ACCENTS = {"win": TEAL, "lose": INDIGO, "neutral": SKY}

# Plot colours
PERIOD_COLORS = {"2014-2019": "#F59E0B", "2019-2024": SKY}
PIE_COLORS = ["#2DD4BF", "#818CF8", "#F59E0B", "#38BDF8", "#F472B6", "#A3E635", "#FB923C"]
OTHERS_COLOR = "#64748B"

def kpi(card_id, title, kind="neutral"):
    accent = ACCENTS[kind]
    return dbc.Card(
        dbc.CardBody([
            html.P(
                title,
                className="mb-2",
                style={
                    "color": MUTED,
                    "fontSize": "0.78rem",
                    "fontWeight": "600",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.06em",
                }
            ),
            html.H3(
                id=card_id,
                children="-",
                className="fw-bold mb-0",
                style={"color": accent}
            )
        ]),
        style={
            "backgroundColor": "#000000",
            "border": "1px solid #555555",
            "borderLeft": f"3px solid {accent}"
        }
    )

def create_graph(graph_id):
    return dcc.Graph(id=graph_id)

def section_title(text, tag=html.H2):
    if tag is html.H2:
        return tag(text, className="mb-4 mt-4", style={"fontWeight": "700",
        "backgroundImage": f"linear-gradient(90deg, {TEAL}, {INDIGO})",
        "WebkitBackgroundClip": "text",
        "backgroundClip": "text",
        "color": "transparent",
        "display": "inline-block"})
    return tag(text, className="mb-3 mt-4", style={
        "color": MUTED, "fontSize": "0.85rem", "fontWeight": "600",
        "textTransform": "uppercase", "letterSpacing": "0.08em",
    })

PERIOD_OPTIONS = [
    {"label": "2014 → 2019", "value": "2014-2019"},
    {"label": "2019 → 2024", "value": "2019-2024"},
    {"label": "Both", "value": "Both"},
]
YEAR_OPTIONS = [{"label": y, "value": y} for y in YEARS]

def period_dropdown(dropdown_id, options, value, style=None):
    return dbc.Row([
        dbc.Col([
            html.Label("Select a period:", style={"fontWeight": "bold", "color": "#FFFFFF"}),
            dcc.Dropdown(
                id=dropdown_id,
                options=options,
                value=value,
                clearable=False,
                style=style
            )
        ], md=4)
    ], className="mb-4")

def kpi_row(prefix, specs, md):
    """specs: list of (id_suffix, title, kind) -> one row of KPI cards with ids '<prefix>-<suffix>'."""
    return dbc.Row(
        [dbc.Col(kpi(f"{prefix}-{suffix}", title, kind), md=md) for suffix, title, kind in specs],
        className="mb-4"
    )

# Main dashboard background
background_style = {
    "backgroundColor": "#000000",
    "minHeight": "100vh",
    "padding": "20px"
}

# Unselected tabs
tab_style = {
    "backgroundColor": "#000000",
    "color": "#FFFFFF",
    "padding": "12px",
    "fontWeight": "600",
    "border": "none"
}

# Selected tab
selected_tab_style = {
    "backgroundColor": "#FFFFFF",
    "color": "#000000",
    "padding": "12px",
    "fontWeight": "600",
    "border": "none"
}

def make_tab(label, value, children):
    return dcc.Tab(
        label=label,
        value=value,
        style=tab_style,
        selected_style=selected_tab_style,
        children=[dbc.Container(children, fluid=True)]
    )

STATE_KPI_SPECS = [
    ("winners-median-pchange", "Median % Asset Change of Winners", "win"),
    ("winners-median-change", "Median Asset Change of Winners", "win"),
    ("losers-median-pchange", "Median % Asset Change of Losers", "lose"),
    ("losers-median-change", "Median Asset Change of Losers", "lose"),
]

def state_tab(prefix, name, value):
    return make_tab(name, value, [
        section_title(f"Winners vs Losers: {name}"),

        period_dropdown(
            f"{prefix}-period", PERIOD_OPTIONS, "2014-2019",
            style={"backgroundColor": "#FFFFFF", "color": "#000000"}
        ),

        kpi_row(prefix, STATE_KPI_SPECS, md=3),

        dbc.Row([
            dbc.Col(create_graph(f"{prefix}-asset-graph"), md=5),
            dbc.Col(create_graph(f"{prefix}-asset-histogram"), md=7),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(create_graph(f"{prefix}-asset-graph-constituency"), md=12),
        ], className="mb-4"),
    ])

def parties_tab(value):
    return make_tab("Assets of parties", value, [
        section_title("Analysis of party assets in each state"),

        period_dropdown("assets-period", YEAR_OPTIONS, "2014"),

        section_title("Candidates Who Appeared", html.H4),
        dbc.Row([
            dbc.Col(kpi(f"{k}-candidate-count", f"{name}: Number of Candidates", "neutral"), md=4)
            for k, name in STATES.items()
        ], className="mb-4"),

        section_title("Total Candidate Assets by Election Result", html.H4),
        dbc.Row(
            [dbc.Col(kpi(f"{k}-total-winner-assets", f"{name}: Total Assets of Winners", "win"), md=4)
             for k, name in STATES.items()] +
            [dbc.Col(kpi(f"{k}-total-loser-assets", f"{name}: Total Assets of Non-winners", "lose"), md=4)
             for k, name in STATES.items()],
            className="mb-4"
        ),

        dbc.Row([
            dbc.Col(create_graph("median-assets-comparison-graph"), md=12),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(create_graph("ap-asset-parties-graph"), md=6),
            dbc.Col(create_graph("gj-asset-parties-graph"), md=6),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(create_graph("br-asset-parties-graph"), md=6),
        ], className="mb-4"),
    ])

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Candidate Asset Trends Dashboard", className="mb-4 mt-4", style={
        "fontWeight": "700",
        "backgroundImage": f"linear-gradient(90deg, {TEAL}, {INDIGO})",
        "WebkitBackgroundClip": "text",
        "backgroundClip": "text",
        "color": "transparent",
        "display": "inline-block",
    }),

    dcc.Tabs(id="tabs", value="tab-1", colors={"border": "#FFFFFF", "primary": "#000000", "background": "#000000"},
             parent_style={"marginBottom": "20px"}, children=[
        *[state_tab(key, name, f"tab-{i}") for i, (key, name) in enumerate(STATES.items(), start=1)],
        parties_tab("tab-4"),
    ])
], fluid=True, style=background_style)

# ----------------------------------------------------------------------------
# Data helpers
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

def linear_ticks(lo, hi, target=8):
    """Round-number ticks between lo and hi, labelled in ₹ K / L / Cr."""
    span = hi - lo
    if span <= 0:
        return [0], ["₹0"]
    mag = 10.0 ** np.floor(np.log10(span / target))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= span / target)
    vals = np.arange(np.floor(lo / step) * step, hi + step, step)
    vals = [0.0 if abs(v) < step * 1e-9 else float(v) for v in vals]
    return vals, [tick_label(v) if v != 0 else "₹0" for v in vals]

# ----------------------------------------------------------------------------
# Figure helpers
# ----------------------------------------------------------------------------
def apply_dark_theme(fig, sort_categories=True):
    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font_color="#FFFFFF",
        title_font_color="#FFFFFF",
        legend_font_color=MUTED,
        xaxis=dict(
            gridcolor="#333333",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="#FFFFFF",
            tickfont=dict(color=MUTED),
            title_font=dict(color=MUTED)
        ),
        yaxis=dict(
            **({"categoryorder": "total descending"} if sort_categories else {}),
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="#FFFFFF",
            gridcolor="#333333",
            tickfont=dict(color=MUTED),
            title_font=dict(color=MUTED)
        )
    )
    return fig

CENTRAL_BINS = 60

def variable_bin_edges(values, central_bins=CENTRAL_BINS, lo_pct=5, hi_pct=95):
    """Narrow equal-width bins around the bulk of the data, widening geometrically in the tails."""
    values = np.asarray(values, dtype=float)
    vmin, vmax = values.min(), values.max()
    lo, hi = np.percentile(values, [lo_pct, hi_pct])
    if hi <= lo:
        lo, hi = vmin, vmax
    if hi <= lo:
        return np.array([vmin - 0.5, vmax + 0.5]), lo, hi
    width = (hi - lo) / central_bins
    edges = list(np.linspace(lo, hi, central_bins + 1))

    w, right = width, hi
    while right < vmax:
        w *= 2
        right += w
        edges.append(right)
    w, left = width, lo
    while left > vmin:
        w *= 2
        left -= w
        edges.insert(0, left)
    return np.array(edges), lo, hi

def create_asset_histogram(changes):
    """Distribution of asset changes (winners vs losers) with narrow bins in the middle, wide in the tails."""
    fig = go.Figure()
    values = changes["Asset Change"].to_numpy(dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return apply_dark_theme(fig, sort_categories=False)

    edges, lo, hi = variable_bin_edges(values)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    central_w = (hi - lo) / CENTRAL_BINS if hi > lo else 1.0
    # density unit: a round rupee amount close to the central bin width
    unit = 10.0 ** np.round(np.log10(central_w)) if central_w > 0 else 1.0

    groups = [
        ("Winners", True, TEAL, "rgba(45,212,191,0.28)"),
        ("Losers", False, INDIGO, "rgba(129,140,248,0.28)"),
    ]
    for label, flag, color, fill in groups:
        group = changes.loc[changes["Winner"] == flag, "Asset Change"].dropna().to_numpy(dtype=float)
        if group.size == 0:
            continue
        counts, _ = np.histogram(group, bins=edges)
        share = counts / group.size
        density = share / (widths / unit)   # comparable across different bin widths

        # step outline with soft fill
        xs = np.repeat(edges, 2)
        ys = np.concatenate([[0], np.repeat(density, 2), [0]])
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=label, legendgroup=label,
            line=dict(color=color, width=2.4), fill="tozeroy", fillcolor=fill,
            hoverinfo="skip",
        ))
        # invisible bars only to provide clean hover tooltips
        fig.add_trace(go.Bar(
            x=centers, y=density, width=widths, legendgroup=label, showlegend=False,
            marker_color="rgba(0,0,0,0)",
            customdata=[
                [money_short(a), money_short(b), float(sh)]
                for a, b, sh in zip(edges[:-1], edges[1:], share)
            ],
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Range: %{customdata[0]} to %{customdata[1]}<br>"
                "Share of candidates in bin: %{customdata[2]:.1%}"
                "<extra></extra>"
            ),
        ))

    view = [lo - 4 * central_w, hi + 4 * central_w]
    tickvals, ticktext = linear_ticks(view[0], view[1])

    fig.update_layout(
        barmode="overlay",
        title="Distribution of Asset Changes: Winners vs Losers",
        hovermode="closest",
        margin=dict(t=60, b=45, l=60, r=25),
    )
    apply_dark_theme(fig, sort_categories=False)
    fig.update_xaxes(title="Asset change", range=view, tickvals=tickvals,
                     ticktext=ticktext, zeroline=False)
    fig.update_yaxes(title=f"Share of candidates per {tick_label(unit)}",
                     tickformat=".0%", zeroline=False, rangemode="tozero")
    if view[0] < 0 < view[1]:
        fig.add_vline(x=0, line=dict(color="#FFFFFF", width=1, dash="dot"))
    return fig

def calculate_statistics(changes: pd.DataFrame):
    winners = changes[changes["Winner"] == True]
    denom = np.where(winners["Previous_assets"] > 0, winners["Previous_assets"], 1)
    winners_percen = (winners["Asset Change"] / denom) * 100

    losers = changes[changes["Winner"] == False]
    denom_losers = np.where(losers["Previous_assets"] > 0, losers["Previous_assets"], 1)
    losers_percen = (losers["Asset Change"] / denom_losers) * 100

    period_colors = PERIOD_COLORS

    if "Period" in changes.columns:
        df1 = changes.groupby(
            ["Period", "Winner"]
        )["Asset Change"].mean().reset_index()

        df1["Winner"] = df1["Winner"].replace({
            True: "Winners",
            False: "Losers"
        })

        df1["Scaled Asset Change"] = symlog_transform(df1["Asset Change"])
        df1['Formatted Asset Change'] = df1['Asset Change'].apply(money_short)

        bar_graph1 = px.bar(
            df1,
            x="Winner",
            y="Scaled Asset Change",
            color="Period",
            barmode="group",
            custom_data=["Formatted Asset Change"],
            category_orders={
                "Winner": ["Winners", "Losers"],
                "Period": ["2014-2019", "2019-2024"]
            },
            color_discrete_map=period_colors,
            title="Mean Asset Change: Winners vs Losers Across Both Periods"
        )

        bar_graph1.update_traces(
            hovertemplate=(
                "Category: %{x}<br>"
                "Period: %{fullData.name}<br>"
                "Mean Asset Change: %{customdata[0]}"
                "<extra></extra>"
            )
        )
        bar_graph1.update_layout(
            xaxis_title="Election Outcome"
        )

        df2 = winners.groupby(
            ["Constituency", "Period"]
        )["Asset Change"].mean().reset_index()

        df2["Scaled Asset Change"] = symlog_transform(df2["Asset Change"])

        # Keep both election periods aligned for every constituency.
        period_order = ["2014-2019", "2019-2024"]
        constituency_data = df2.pivot(
            index="Constituency",
            columns="Period",
            values="Asset Change"
        ).reindex(columns=period_order)

        bar_graph2 = go.Figure()

        for period in period_order:
            if period in constituency_data.columns:
                actual_values = constituency_data[period]

                bar_graph2.add_trace(go.Bar(
                    x=symlog_transform(actual_values),
                    y=constituency_data.index,
                    orientation="h",
                    name=period,
                    marker_color=period_colors[period],
                    customdata=actual_values.apply(money_short),
                    hovertemplate=(
                        "Constituency: %{y}<br>"
                        f"Period: {period}<br>"
                        "Actual Asset Change: %{customdata}"
                        "<extra></extra>"
                    )
                ))

        bar_graph2.update_layout(
            barmode="group",
            title="Constituency-wise Asset Change of Winners Across Both Periods",
            xaxis_title="Scaled Asset Change",
            yaxis_title="Constituency",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font_color="#172554",
            legend_title="Election Period",
            height=800
        )
    else:
        df1 = changes.groupby("Winner")["Asset Change"].mean().reset_index()
        df1["Winner"] = df1["Winner"].replace({
            True: "Winners",
            False: "Losers"
        })

        df1["Scaled Asset Change"] = symlog_transform(df1["Asset Change"])
        df1['Formatted Asset Change'] = df1['Asset Change'].apply(money_short)
        bar_graph1 = px.bar(
            df1,
            x="Winner",
            y="Scaled Asset Change",
            color="Winner",
            custom_data=["Formatted Asset Change"],
            color_discrete_map={
                "Losers": INDIGO,
                "Winners": TEAL
            },
            title="Mean Asset Change: Winners vs Losers"
        )

        bar_graph1.update_traces(
            hovertemplate=(
                "Category: %{x}<br>"
                "Mean Asset Change: %{customdata[0]}"
                "<extra></extra>"
            )
        )
        bar_graph1.update_layout(
            xaxis_title="Election Outcome"
        )

        df2 = winners.groupby(
            "Constituency"
        )["Asset Change"].mean().reset_index()

        df2["Scaled Asset Change"] = symlog_transform(df2["Asset Change"])
        df2['Transformed Asset Change'] = df2['Asset Change'].apply(money_short)
        bar_graph2 = px.bar(
            df2,
            y="Constituency",
            x="Scaled Asset Change",
            orientation='h',
            custom_data=["Transformed Asset Change"],
            color_discrete_sequence=[TEAL],
            title="Asset Changes Across Constituencies (Its Winner's)"
        )
        bar_graph2.update_layout(autosize=True, showlegend=False)
        bar_graph2.update_traces(
            hovertemplate=(
                "Constituency: %{y}<br>"
                "Actual Asset Change: %{customdata[0]}"
                "<extra></extra>"
            )
        )

    for fig in [bar_graph1, bar_graph2]:
        apply_dark_theme(fig)

    bar_graph1.update_xaxes(
        categoryorder="array",
        categoryarray=["Winners", "Losers"]
    )

    histogram_graph = create_asset_histogram(changes)

    return (
        f"{winners_percen.median():,.2f}%" if not winners.empty else "N/A",
        money_short(winners['Asset Change'].median()) if not winners.empty else "N/A",
        f"{losers_percen.median():,.2f}%" if not losers.empty else "N/A",
        money_short(losers['Asset Change'].median()) if not losers.empty else "N/A",
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
            {
                "State": state,
                "Candidate Result": "Winners",
                "Median Assets": winners.median() if not winners.empty else 0,
            },
            {
                "State": state,
                "Candidate Result": "Non-winners",
                "Median Assets": non_winners.median() if not non_winners.empty else 0,
            },
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
        color_discrete_map={
            "Winners": TEAL,
            "Non-winners": INDIGO,
        },
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "%{fullData.name}: %{customdata[0]}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font_color="#FFFFFF",
        title_font_color="#FFFFFF",
        legend=dict(
            font=dict(color=MUTED)
        ),
        xaxis=dict(
            title="State",
            color=MUTED,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title="Median Assets (₹)",
            type="log",
            tickmode="linear",
            dtick=1,
            tickprefix="₹",
            tickformat=".0s",
            color=MUTED,
            showgrid=True,
            gridcolor="#444444",
            gridwidth=1,
            minor=dict(showgrid=False),
            zeroline=False
        ),
        margin=dict(t=60, b=40, l=80, r=30),
    )

    return fig


def create_party_assets_pie(data, state, year, top_n=7):

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
    top_parties['Formatted'] = top_parties['Total Assets'].apply(money_short)
    color_map = {
        party: (OTHERS_COLOR if party == "Others" else PIE_COLORS[i % len(PIE_COLORS)])
        for i, party in enumerate(top_parties["Party"])
    }
    fig = px.pie(
        data_frame=top_parties,
        names="Party",
        values="Total Assets",
        title=f"Distribution of Candidate Assets by Party - {state} ({year})",
        hole=0.35,
        color="Party",
        color_discrete_map=color_map,
        category_orders={
            "names": list(top_parties.index)
        }
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent",
        textfont=dict(color="#0B1020"),
        marker=dict(line=dict(color="#000000", width=2)),
        hovertemplate="<b>%{label}</b><br>Assets: %{customdata}<br>Percentage: %{percent}<extra></extra>",
        customdata=top_parties['Formatted'],
        sort=False
    )

    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font_color="#FFFFFF",
        legend=dict(
            traceorder="normal",
            font=dict(color=MUTED)
        ),
        title_font_color="#FFFFFF"
    )
    return fig

# ----------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------
def register_state_callback(prefix):
    d = DATA[prefix]

    @app.callback(
        *[Output(f"{prefix}-{suffix}", "children") for suffix, _, _ in STATE_KPI_SPECS],
        Output(f"{prefix}-asset-graph", "figure"),
        Output(f"{prefix}-asset-histogram", "figure"),
        Output(f"{prefix}-asset-graph-constituency", "figure"),
        Input(f"{prefix}-period", "value"),
    )
    def update_state_dashboard(period):
        if period == "2014-2019":
            records = calculate_asset_changes(d["2014"], d["2019"])
            return calculate_statistics(records)

        if period == "2019-2024":
            records = calculate_asset_changes(d["2019"], d["2024"])
            return calculate_statistics(records)

        if period == "Both":
            records_14_19 = calculate_asset_changes(d["2014"], d["2019"])
            records_19_24 = calculate_asset_changes(d["2019"], d["2024"])
            records_14_19["Period"] = "2014-2019"
            records_19_24["Period"] = "2019-2024"
            combined_records = pd.concat([records_14_19, records_19_24], ignore_index=True)

            return calculate_statistics(combined_records)

    return update_state_dashboard

for _key in STATES:
    register_state_callback(_key)

@app.callback(
    *[Output(f"{k}-candidate-count", "children") for k in STATES],
    *[Output(f"{k}-total-{kind}-assets", "children") for k in STATES for kind in ("winner", "loser")],
    Output("median-assets-comparison-graph", "figure"),
    *[Output(f"{k}-asset-parties-graph", "figure") for k in STATES],
    Input("assets-period", "value")
)
def update_party_assets_dashboard(year):
    year = year if year in YEARS else "2024"
    data = {k: DATA[k][year] for k in STATES}

    # Summaries use winner status in the selected election year.
    summaries = {k: summarize_election_assets(data[k]) for k in STATES}

    counts = [f"{len(data[k]):,}" for k in STATES]
    totals = []
    for k in STATES:
        totals.extend([summaries[k]["total_winners"], summaries[k]["total_non_winners"]])

    median_comparison_fig = create_median_assets_comparison(
        data["ap"], data["br"], data["gj"], year
    )
    pies = [create_party_assets_pie(data[k], STATES[k], year) for k in STATES]

    return (*counts, *totals, median_comparison_fig, *pies)

if __name__ == "__main__":
    app.run(debug=False)