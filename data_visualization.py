# Import required libraries
from dash import Dash, html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# Load dataset files
from pathlib import Path

# Directory containing this Python script
BASE_DIR = Path(__file__).resolve().parent

# Directory containing the cleaned datasets
DATA_DIR = BASE_DIR / "cleaned_data"

# Load datasets
andhra_14 = pd.read_parquet(DATA_DIR / "andhra_2014.parquet")
andhra_19 = pd.read_parquet(DATA_DIR / "andhra_2019.parquet")
andhra_24 = pd.read_parquet(DATA_DIR / "andhra_2024.parquet")

bihar_14 = pd.read_parquet(DATA_DIR / "bihar_2014.parquet")
bihar_19 = pd.read_parquet(DATA_DIR / "bihar_2019.parquet")
bihar_24 = pd.read_parquet(DATA_DIR / "bihar_2024.parquet")

gujarat_14 = pd.read_parquet(DATA_DIR / "gujarat_2014.parquet")
gujarat_19 = pd.read_parquet(DATA_DIR / "gujarat_2019.parquet")
gujarat_24 = pd.read_parquet(DATA_DIR / "gujarat_2024.parquet")

def kpi(card_id, title):
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted mb-2"),
            html.H3(
                id=card_id,
                children="-",
                className="fw-bold mb-0"
            )
        ])
    )

def create_graph(graph_id):
    return dcc.Graph(id=graph_id)

# Main dashboard background
background_style = {
    "backgroundColor": "#172554",
    "minHeight": "100vh",
    "padding": "20px"
}

# Unselected tabs
tab_style = {
    "backgroundColor": "#DBEAFE",
    "color": "#172554",
    "padding": "12px",
    "fontWeight": "600",
    "border": "none"
}

# Selected tab
selected_tab_style = {
    "backgroundColor": "#1E40AF",
    "color": "#FFFFFF",
    "padding": "12px",
    "fontWeight": "600",
    "border": "none"
}

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Candidate Asset Trends Dashboard", className="mb-4 mt-4", style={"color": "#FFFFFF"}),

    dcc.Tabs(id="tabs", value="tab-1", colors={"border": "#CBD5E1", "primary": "#1E40AF", "background": "#DBEAFE"},
             parent_style={"marginBottom": "20px"} ,children=[

        # Andhra Pradesh
        dcc.Tab(label="Andhra Pradesh", value="tab-1", style=tab_style, selected_style=selected_tab_style ,children=[
            dbc.Container([
                html.H2("Winners vs Losers: Andhra Pradesh", className="mb-4 mt-4", style={"color": "#FFFFFF"}),

                dbc.Row([
                    dbc.Col([
                        html.Label("Select a period:", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="ap-period",
                            options=[
                                {"label": "2014 → 2019", "value": "2014-2019"},
                                {"label": "2019 → 2024", "value": "2019-2024"},
                                {"label": "Both", "value": "Both"},
                            ],
                            value="2014-2019",
                            clearable=False
                        )
                    ], md=4)
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col(kpi("ap-winners-median-pchange", "Median % Asset Change of Winners"), md=3),
                    dbc.Col(kpi("ap-winners-median-change", "Median Asset Change of Winners"), md=3),
                    dbc.Col(kpi("ap-losers-median-pchange", "Median % Asset Change of Losers"), md=3),
                    dbc.Col(kpi("ap-losers-median-change", "Median Asset Change of Losers"), md=3)
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col(create_graph("ap-asset-graph"), md=4),
                    dbc.Col(create_graph("ap-asset-graph-constituency"), md=8),
                ], className="mb-4"),

            ], fluid=True)
        ]),

        # Bihar
        dcc.Tab(label="Bihar", value="tab-2", style=tab_style, selected_style=selected_tab_style, children=[
            dbc.Container([
                html.H2("Winners vs Losers: Bihar", className="mb-4 mt-4", style={"color": "#FFFFFF"}),

                dbc.Row([
                    dbc.Col([
                        html.Label("Select a period:", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="br-period",
                            options=[
                                {"label": "2014 → 2019", "value": "2014-2019"},
                                {"label": "2019 → 2024", "value": "2019-2024"},
                                {"label": "Both", "value": "Both"},
                            ],
                            value="2014-2019",
                            clearable=False
                        )
                    ], md=4)
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col(kpi("br-winners-median-pchange", "Median % Asset Change of Winners"), md=3),
                    dbc.Col(kpi("br-winners-median-change", "Median Asset Change of Winners"), md=3),
                    dbc.Col(kpi("br-losers-median-pchange", "Median % Asset Change of Losers"), md=3),
                    dbc.Col(kpi("br-losers-median-change", "Median Asset Change of Losers"), md=3)
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col(create_graph("br-asset-graph"), md=4),
                    dbc.Col(create_graph("br-asset-graph-constituency"), md=8),
                ], className="mb-4"),

            ], fluid=True)
        ]),

        # Gujarat
        dcc.Tab(label="Gujarat", value="tab-3", style=tab_style, selected_style=selected_tab_style, children=[
            dbc.Container([
                html.H2("Winners vs Losers: Gujarat", className="mb-4 mt-4", style={"color": "#FFFFFF"}),

                dbc.Row([
                    dbc.Col([
                        html.Label("Select a period:", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="gj-period",
                            options=[
                                {"label": "2014 → 2019", "value": "2014-2019"},
                                {"label": "2019 → 2024", "value": "2019-2024"},
                                {"label": "Both", "value": "Both"},
                            ],
                            value="2014-2019",
                            clearable=False
                        )
                    ], md=4)
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col(kpi("gj-winners-median-pchange", "Median % Asset Change of Winners"), md=3),
                    dbc.Col(kpi("gj-winners-median-change", "Median Asset Change of Winners"), md=3),
                    dbc.Col(kpi("gj-losers-median-pchange", "Median % Asset Change of Losers"), md=3),
                    dbc.Col(kpi("gj-losers-median-change", "Median Asset Change of Losers"), md=3)
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col(create_graph("gj-asset-graph"), md=4),
                    dbc.Col(create_graph("gj-asset-graph-constituency"), md=8),
                ], className="mb-4"),

            ], fluid=True)
        ]),

        # Assets of parties
        dcc.Tab(label="Assets of parties", value="tab-4", style=tab_style, selected_style=selected_tab_style, children=[
            dbc.Container([
                html.H2("Analysis of party assets in each state", className="mb-4 mt-4", style={"color": "#FFFFFF"}),

                dbc.Row([
                    dbc.Col([
                        html.Label("Select a period:", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="assets-period",
                            options=[
                                {"label": "2014", "value": "2014"},
                                {"label": "2019", "value": "2019"},
                                {"label": "2024", "value": "2024"},
                            ],
                            value="2014",
                            clearable=False
                        )
                    ], md=4)
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col(create_graph("ap-asset-parties-graph"), md=6),
                    dbc.Col(create_graph("gj-asset-parties-graph"), md=6),
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col(create_graph("br-asset-parties-graph"), md=6),
                ], className="mb-4"),

            ], fluid=True)
        ]),
    ])
], fluid=True, style=background_style)

def calculate_asset_changes(previous_df: pd.DataFrame, current_df: pd.DataFrame):
    previous_df = previous_df.copy().set_index("Sno.")
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


def calculate_statistics(changes: pd.DataFrame):
    winners = changes[changes["Winner"] == True]
    denom = np.where(winners["Previous_assets"] > 0, winners["Previous_assets"], 1)
    winners_percen = (winners["Asset Change"] / denom) * 100

    losers = changes[changes["Winner"] == False]
    denom_losers = np.where(losers["Previous_assets"] > 0, losers["Previous_assets"], 1)
    losers_percen = (losers["Asset Change"] / denom_losers) * 100

    if "Period" in changes.columns:
        df1 = changes.groupby(
            ["Period", "Winner"]
        )["Asset Change"].mean().reset_index()

        df1["Winner"] = df1["Winner"].replace({
            True: "Winners",
            False: "Losers"
        })

        bar_graph1 = px.bar(
            df1,
            x="Winner",
            y="Asset Change",
            color="Period",
            barmode="group",
            category_orders={
                "Winner": ["Winners", "Losers"],
                "Period": ["2014-2019", "2019-2024"]
            },
            color_discrete_map={
                "2014-2019": "#DC2626",
                "2019-2024": "#2563EB"
            },
            title="Average Asset Change: Winners vs Losers Across Both Periods"
        )

        df2 = winners.groupby(
            ["Constituency", "Period"]
        )["Asset Change"].mean().reset_index()

        # Keep both election periods aligned for every constituency.
        period_order = ["2014-2019", "2019-2024"]
        constituency_data = df2.pivot(
            index="Constituency",
            columns="Period",
            values="Asset Change"
        ).reindex(columns=period_order)

        bar_graph2 = go.Figure()

        period_colors = {
            "2014-2019": "#DC2626",
            "2019-2024": "#2563EB"
        }

        for period in period_order:
            if period in constituency_data.columns:
                bar_graph2.add_trace(go.Bar(
                    x=constituency_data.index,
                    y=constituency_data[period],
                    name=period,
                    marker_color=period_colors[period]
                ))

        bar_graph2.update_layout(
            barmode="group",
            title="Constituency-wise Asset Change Across Both Periods",
            xaxis_title="Constituency",
            yaxis_title="Average Asset Change (₹)",
            xaxis_tickangle=-45,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font_color="#172554",
            legend_title="Election Period",
            height=600
        )
    else:
        df1 = changes.groupby("Winner")["Asset Change"].mean().reset_index()
        df1["Winner"] = df1["Winner"].replace({
            True: "Winners",
            False: "Losers"
        })

        bar_graph1 = px.bar(
            df1,
            x="Winner",
            y="Asset Change",
            color="Winner",
            color_discrete_map={
                "Winners": "#0D9488",
                "Losers": "#3B82F6"
            },
            title="Average Asset Change: Winners vs Losers"
        )

        df2 = winners.groupby(
            "Constituency"
        )["Asset Change"].mean().reset_index()

        bar_graph2 = px.bar(
            df2,
            x="Constituency",
            y="Asset Change",
            color="Constituency",
            title="Asset Change in Each Constituency"
        )

    return (
        f"{winners_percen.median():,.2f}%" if not winners.empty else "N/A",
        f"₹{winners['Asset Change'].median():,.2f}" if not winners.empty else "N/A",
        f"{losers_percen.median():,.2f}%" if not losers.empty else "N/A",
        f"₹{losers['Asset Change'].median():,.2f}" if not losers.empty else "N/A",
        bar_graph1,
        bar_graph2
    )

def create_party_assets_pie(data, state, year, top_n=7):

    party_assets = (
        data.groupby("Party")["Total Assets"]
        .sum()
        .sort_values(ascending=False)
    )

    independent_assets = party_assets.get("IND", 0)
    party_assets = party_assets.drop("IND", errors="ignore")
    top_parties = party_assets.head(top_n)
    other_assets = party_assets.iloc[top_n:].sum() + independent_assets

    if other_assets > 0:
        top_parties.loc["Others"] = other_assets

    fig = px.pie(
        names=top_parties.index,
        values=top_parties.values,
        title=f"Distribution of Candidate Assets by Party - {state} ({year})",
        hole=0.35,
        category_orders={
            "names": list(top_parties.index)
        }
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent",
        sort=False
    )

    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        font_color="#6A54AC",
        legend=dict(traceorder="normal")
    )

    return fig

@app.callback(
    Output("ap-winners-median-pchange", "children"),
    Output("ap-winners-median-change", "children"),
    Output("ap-losers-median-pchange", "children"),
    Output("ap-losers-median-change", "children"),
    Output("ap-asset-graph", "figure"),
    Output("ap-asset-graph-constituency", "figure"),
    Input("ap-period", "value"),
)
def update_ap_dashboard(period):

    if period == "2014-2019":
        records = calculate_asset_changes(andhra_14, andhra_19)
        return calculate_statistics(records)

    if period == "2019-2024":
        records = calculate_asset_changes(andhra_19, andhra_24)
        return calculate_statistics(records)

    if period == "Both":
        records_14_19 = calculate_asset_changes(andhra_14, andhra_19)
        records_19_24 = calculate_asset_changes(andhra_19, andhra_24)
        records_14_19["Period"] = "2014-2019"
        records_19_24["Period"] = "2019-2024"
        combined_records = pd.concat([records_14_19, records_19_24], ignore_index=True)
        
        return calculate_statistics(combined_records)

@app.callback(
    Output("br-winners-median-pchange", "children"),
    Output("br-winners-median-change", "children"),
    Output("br-losers-median-pchange", "children"),
    Output("br-losers-median-change", "children"),
    Output("br-asset-graph", "figure"),
    Output("br-asset-graph-constituency", "figure"),
    Input("br-period", "value"),
)
def update_ap_dashboard(period):

    if period == "2014-2019":
        records = calculate_asset_changes(bihar_14, bihar_19)
        return calculate_statistics(records)

    if period == "2019-2024":
        records = calculate_asset_changes(bihar_19, bihar_24)
        return calculate_statistics(records)

    if period == "Both":
        records_14_19 = calculate_asset_changes(bihar_14, bihar_19)
        records_19_24 = calculate_asset_changes(bihar_19, bihar_24)
        records_14_19["Period"] = "2014-2019"
        records_19_24["Period"] = "2019-2024"
        combined_records = pd.concat([records_14_19, records_19_24], ignore_index=True)
        
        return calculate_statistics(combined_records)

@app.callback(
    Output("gj-winners-median-pchange", "children"),
    Output("gj-winners-median-change", "children"),
    Output("gj-losers-median-pchange", "children"),
    Output("gj-losers-median-change", "children"),
    Output("gj-asset-graph", "figure"),
    Output("gj-asset-graph-constituency", "figure"),
    Input("gj-period", "value"),
)
def update_ap_dashboard(period):

    if period == "2014-2019":
        records = calculate_asset_changes(gujarat_14, gujarat_19)
        return calculate_statistics(records)

    if period == "2019-2024":
        records = calculate_asset_changes(gujarat_19, gujarat_24)
        return calculate_statistics(records)

    if period == "Both":
        records_14_19 = calculate_asset_changes(gujarat_14, gujarat_19)
        records_19_24 = calculate_asset_changes(gujarat_19, gujarat_24)
        records_14_19["Period"] = "2014-2019"
        records_19_24["Period"] = "2019-2024"
        combined_records = pd.concat([records_14_19, records_19_24], ignore_index=True)
        
        return calculate_statistics(combined_records)

@app.callback(
    Output("ap-asset-parties-graph", "figure"),
    Output("gj-asset-parties-graph", "figure"),
    Output("br-asset-parties-graph", "figure"),
    Input("assets-period", "value")
)
def update_party_assets_dashboard(year):

    if year == "2014":
        ap_data = andhra_14
        gj_data = gujarat_14
        br_data = bihar_14

    elif year == "2019":
        ap_data = andhra_19
        gj_data = gujarat_19
        br_data = bihar_19

    else:
        ap_data = andhra_24
        gj_data = gujarat_24
        br_data = bihar_24

    ap_fig = create_party_assets_pie(ap_data, "Andhra Pradesh", year)
    gj_fig = create_party_assets_pie(gj_data, "Gujarat", year)
    br_fig = create_party_assets_pie(br_data, "Bihar", year)

    return ap_fig, gj_fig, br_fig

if __name__ == "__main__":
    app.run(debug=False)