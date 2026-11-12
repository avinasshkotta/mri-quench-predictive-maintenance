#!/usr/bin/env python
"""Build the **interactive** HTML report for Case 1 (not a notebook export).

    python scripts/build_report.py

Produces a single self-contained file:

    reports/quenchguard_report.html

Design goals (deliberately unlike a static notebook page):

* interactive Plotly charts - hover tooltips, zoom/pan, range slider, PNG download
* a **threshold explorer**: drag a slider and watch false pages, event recall and net
  saving update live against the validation sweep
* a **sensor-story explorer**: dropdown per magnet, four synchronised channels with
  alert episodes and the 96 h / 72 h markers
* sortable model-comparison and per-event tables
* sticky navigation, KPI cards, collapsible methodology sections

Inputs (all produced by ``scripts/run_pipeline.py``):
    reports/metrics.json, reports/threshold_sweep.csv, reports/training_history.csv,
    models/lstm_detector.pt, data/gold/features.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import torch
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from quenchguard.baselines import baseline_scores, train_baselines
from quenchguard.config import REPORTS_DIR, default_config
from quenchguard.data_gen import load_quench_events
from quenchguard.evaluate import (alert_episodes, false_alerts, match_events,
                                  make_score_frame)
from quenchguard.features import FEATURE_GROUPS, ensure_features
from quenchguard.lstm import load_model, predict_scores
from quenchguard.sequences import prepare_data

OUT_PATH = REPORTS_DIR / "quenchguard_report.html"

PLOT_CONFIG = {"displaylogo": False, "responsive": True,
               "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

COLORS = {
    "lstm": "#6366f1", "logreg": "#94a3b8", "hist_gbdt": "#f59e0b",
    "pressure": "#2563eb", "cold": "#f97316", "vib": "#16a34a", "rf": "#a855f7",
    "good": "#16a34a", "bad": "#dc2626", "neutral": "#64748b", "ink": "#0f172a",
}
GROUP_COLOR = {"helium_pressure": COLORS["pressure"], "cold_head_temp": COLORS["cold"],
               "vibration": COLORS["vib"], "rf_power": COLORS["rf"]}


def div(fig: go.Figure, div_id: str | None = None) -> str:
    kwargs = {"div_id": div_id} if div_id else {}
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config=PLOT_CONFIG, **kwargs)


def iso(ts) -> str:
    return pd.Timestamp(ts).isoformat()


# ======================================================================================
# data assembly
# ======================================================================================
class ReportData:
    """Everything the page needs, computed once from the pipeline artifacts."""

    def __init__(self) -> None:
        cfg = self.cfg = default_config()

        self.metrics = json.loads(cfg.metrics_path.read_text(encoding="utf-8"))
        self.sweep_all = pd.read_csv(REPORTS_DIR / "threshold_sweep.csv")
        hist_path = REPORTS_DIR / "training_history.csv"
        self.history = pd.read_csv(hist_path) if hist_path.exists() else None

        # rebuild the test scores from the saved checkpoint (no retraining)
        self.splits, _scaler, self.labelled = prepare_data(cfg, verbose=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(cfg.model_path, self.device)
        self.baselines = train_baselines(cfg, self.splits["train"].X,
                                         self.splits["train"].y,
                                         self.splits["val"].X, self.splits["val"].y,
                                         verbose=False)
        self.scores = {"lstm": predict_scores(self.model, self.splits["test"].X,
                                              self.device),
                       **baseline_scores(self.baselines, self.splits["test"].X)}

        self.events = load_quench_events(cfg)
        test0 = make_score_frame(self.splits["test"], self.scores["lstm"], "test")
        self.test_events = self.events[
            (self.events.event_time >= test0.t.min())
            & (self.events.event_time <= test0.t.max())]

        self.thresholds = {k: float(v) for k, v in
                           self.metrics["config"]["thresholds"].items()}
        self.frames = {n: make_score_frame(self.splits["test"], s, "test")
                       for n, s in self.scores.items()}
        self.episodes = alert_episodes(self.frames["lstm"], self.thresholds["lstm"])
        self.false_eps = false_alerts(self.episodes, self.test_events, cfg)
        self.outcomes = match_events(self.test_events, self.episodes, cfg)
        self.cost = self.metrics["results"]["lstm"]["cost"]

    @property
    def n_events(self) -> int:
        return len(self.test_events)


# ======================================================================================
# figure builders
# ======================================================================================
def kpi_cards(d: ReportData) -> str:
    m = d.metrics["results"]["lstm"]
    ev, al, cost = m["events"], m["alerts"], m["cost"]
    items = [
        ("Event recall &ge; 72 h",
         f"{ev['event_recall_in_time']:.0%}",
         f"{ev['detected_in_time']}/{m['n_quench_events']} test quenches in time",
         "good" if ev["event_recall_in_time"] >= 1 else "neutral"),
        ("Lead time (median)",
         f"{ev['lead_hours_median']:.0f} h" if ev["lead_hours_median"] else "-",
         f"required {d.cfg.window.min_lead_hours} h &middot; "
         f"horizon {d.cfg.window.horizon_hours} h", "good"),
        ("False pages / magnet-month",
         f"{al['false_per_magnet_month']:.2f}",
         f"budget {d.cfg.threshold.false_alert_budget_per_magnet_month:g} &middot; "
         f"{al['false_alerts']} false alerts",
         "good" if al["false_per_magnet_month"] <= 1 else "bad"),
        ("Net saving (test period)",
         f"${cost['net_savings_usd']:,.0f}",
         f"reactive ${cost['reactive_total_usd']:,.0f} &rarr; "
         f"predictive ${cost['predictive_total_usd']:,.0f}", "good"),
        ("Test PR-AUC",
         f"{m['classification']['pr_auc']:.3f}",
         f"ROC-AUC {m['classification']['roc_auc']:.3f}", "good"),
    ]
    return "\n".join(
        f'<div class="kpi kpi-{tone}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>'
        for label, value, sub, tone in items)


def make_story_figure(g: pd.DataFrame, evs: pd.DataFrame, eps: pd.DataFrame,
                      cfg) -> go.Figure:
    t = [iso(v) for v in g.window_end]
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=("Helium pressure (MPa) — failure signal below 2.0",
                        "Cold head (K) — failure signal above 5.5",
                        "Vibration RMS (g) — 100-200 Hz fault band",
                        "RF dropouts per 15-min window"),
    )
    series = [
        ("p_mean", "pressure", COLORS["pressure"], 1),
        ("t_mean", "cold head", COLORS["cold"], 2),
        ("v_rms", "vibration", COLORS["vib"], 3),
        ("dropout_count", "dropouts", COLORS["rf"], 4),
    ]
    for col, name, color, row in series:
        fig.add_trace(
            go.Scatter(x=t, y=g[col].round(4).tolist(), name=name, mode="lines",
                       line={"width": 1.4, "color": color},
                       hovertemplate="%{x}<br>" + name + " %{y:.3f}<extra></extra>"),
            row=row, col=1)

    shapes, anns = [], []
    for ev in evs.itertuples():
        shapes.append(dict(type="rect",
                           x0=iso(ev.event_time - pd.Timedelta(hours=cfg.window.horizon_hours)),
                           x1=iso(ev.event_time), yref="paper", y0=0, y1=1,
                           fillcolor="orange", opacity=0.10, line_width=0,
                           layer="below"))
        shapes.append(dict(type="line", x0=iso(ev.event_time), x1=iso(ev.event_time),
                           yref="paper", y0=0, y1=1,
                           line=dict(color="crimson", width=1.6)))
        anns.append(dict(x=iso(ev.event_time), xref="x", yref="paper", y=1.075,
                         text=f"quench {ev.event_time:%m-%d %H:%M}", showarrow=False,
                         font=dict(color="crimson", size=11)))
    for ep in eps.itertuples():
        shapes.append(dict(type="rect", x0=iso(ep.alert_time), x1=iso(ep.last_time),
                           yref="paper", y0=0, y1=1, fillcolor="#22c55e",
                           opacity=0.13, line_width=0))
        anns.append(dict(x=iso(ep.alert_time), xref="x", yref="paper", y=1.075,
                         text="alert", showarrow=False,
                         font=dict(color="#15803d", size=10)))

    # update_layout(annotations=...) REPLACES existing annotations - keep the
    # subplot titles created by make_subplots and append ours after them
    title_anns = list(fig.layout.annotations or [])
    fig.update_layout(shapes=shapes, annotations=title_anns + anns, height=780,
                      margin=dict(l=60, r=20, t=74, b=30),
                      hovermode="x unified", showlegend=False, plot_bgcolor="white",
                      font=dict(size=11))
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=4, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#eef2f7")
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f7")
    return fig


def sensor_story_divs(d: ReportData) -> tuple[str, dict, list[str], dict]:
    """Story pane containers + per-magnet figure JSON (rendered lazily by JS)."""
    cfg = d.cfg
    test_start = pd.Timestamp(d.frames["lstm"].t.min())
    df = d.labelled[d.labelled.window_end >= test_start]

    magnets = sorted(df.magnet_id.unique())
    meta: dict[str, dict] = {}
    figs: dict[str, dict] = {}
    panes = []

    for magnet in magnets:
        g = df[df.magnet_id == magnet].sort_values("window_end").iloc[::2]
        evs = d.test_events[d.test_events.magnet_id == magnet]
        eps = d.episodes[d.episodes.magnet_id == magnet]
        falses = d.false_eps[(d.false_eps.magnet_id == magnet)
                             & (d.false_eps.is_false)]

        panes.append(f'<div class="story-pane" id="story-{magnet}" style="display:none">'
                     f'<div id="plot-{magnet}"></div></div>')
        payload = json.loads(pio.to_json(make_story_figure(g, evs, eps, cfg)))
        payload["config"] = PLOT_CONFIG
        figs[magnet] = payload

        caps = []
        for ev in evs.itertuples():
            row = d.outcomes[d.outcomes.event_time == ev.event_time]
            if len(row) and bool(row.iloc[0].detected):
                caps.append(f"quench {ev.event_time:%Y-%m-%d %H:%M} &mdash; first alert "
                            f"{row.iloc[0].lead_hours_unclamped:.1f} h ahead "
                            f"&rarr; <b class='ok'>in time</b>")
            else:
                caps.append(f"quench {ev.event_time:%Y-%m-%d %H:%M} &mdash; "
                            f"<b class='bad'>missed</b>")
        if not len(evs):
            caps.append("no quench in the test period")
        caps.append(f"{len(eps)} alert episodes, {len(falses)} of them false pages")
        meta[magnet] = {"caption": " &middot; ".join(caps)}

    return "\n".join(panes), meta, magnets, figs


def threshold_section(d: ReportData) -> tuple[str, str, int, int]:
    """Returns (chart div, sweep JSON, chosen slider index, number of points)."""
    sw = (d.sweep_all[d.sweep_all.model == "lstm"]
          .sort_values("threshold").reset_index(drop=True))
    x = sw.threshold.tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=sw.false_per_magnet_month.tolist(),
        name="false pages / magnet-month", mode="lines+markers",
        line=dict(color=COLORS["bad"], width=2),
        hovertemplate="thr %{x:.3f}<br>%{y:.2f} false/mo<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=x, y=sw.event_recall_in_time.tolist(),
        name="event recall ≥ 72 h", mode="lines+markers",
        line=dict(color=COLORS["good"], width=2), yaxis="y2",
        hovertemplate="thr %{x:.3f}<br>%{y:.0%} recall<extra></extra>"))
    # live markers driven by the slider (trace indices 2 and 3)
    fig.add_trace(go.Scatter(
        x=[0], y=[0], name="selected point", mode="markers",
        marker=dict(size=14, color=COLORS["bad"], line=dict(width=2, color="white")),
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[0], y=[0], name="selected point", mode="markers", yaxis="y2",
        marker=dict(size=14, color=COLORS["good"], line=dict(width=2, color="white")),
        hoverinfo="skip"))

    chosen = d.thresholds["lstm"]
    idx = int((sw.threshold - chosen).abs().idxmin())
    budget = d.cfg.threshold.false_alert_budget_per_magnet_month
    fig.add_vline(x=chosen, line_dash="dot", line_color=COLORS["ink"],
                  annotation_text=f"chosen {chosen:.3f}", annotation_position="top")
    fig.add_hline(y=budget, line_dash="dash", line_color=COLORS["bad"],
                  annotation_text=f"budget {budget:g} false/mo",
                  annotation_position="bottom right")
    fig.add_hline(y=budget * d.cfg.threshold.selection_safety_factor,
                  line_dash="dash", line_color="#fca5a5",
                  annotation_text="selection budget ×0.6",
                  annotation_position="top left")

    fig.update_layout(
        height=440, margin=dict(l=55, r=60, t=50, b=35), hovermode="x unified",
        legend=dict(orientation="h", y=1.14, x=0), plot_bgcolor="white",
        xaxis=dict(title="threshold", rangeslider=dict(visible=False),
                   gridcolor="#eef2f7"),
        yaxis=dict(title="false pages / magnet-month", domain=[0, 0.94],
                   range=[-0.05, None], gridcolor="#eef2f7"),
        yaxis2=dict(title="event recall", domain=[0, 0.94], range=[0, 1.05],
                    overlaying="y", side="right", gridcolor="#eef2f7"),
        font=dict(size=11))

    rows = sw[["threshold", "false_per_magnet_month", "event_recall_in_time",
               "false_alerts", "n_alerts", "precision", "recall", "f1"]].round(6).to_dict("records")
    cost = d.cfg.cost
    payload = {"rows": rows, "n_events": d.n_events,
               "emergency": cost.emergency_total, "planned": cost.planned_total,
               "false_usd": cost.false_alarm_usd, "chosen": chosen}
    return div(fig, div_id="thrChart"), json.dumps(payload), idx, len(sw)


def curves_figure(d: ReportData) -> str:
    y = d.frames["lstm"]["y"].to_numpy()
    fig = make_subplots(cols=2, rows=1,
                        subplot_titles=("Precision – Recall", "ROC curve"))
    for name in ["lstm", "hist_gbdt", "logreg"]:
        s = d.frames[name]["score"].to_numpy()
        p, r, _ = precision_recall_curve(y, s)
        fig.add_trace(go.Scatter(
            x=r, y=p, name=f"{name} (AP {average_precision_score(y, s):.3f})",
            mode="lines", line=dict(color=COLORS[name], width=2.2),
            hovertemplate="recall %{x:.3f}<br>precision %{y:.3f}<extra>" + name + "</extra>"),
            row=1, col=1)
        fpr, tpr, _ = roc_curve(y, s)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", showlegend=False,
            line=dict(color=COLORS[name], width=2.2),
            hovertemplate="FPR %{x:.4f}<br>TPR %{y:.3f}<extra>" + name + "</extra>"),
            row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                             line=dict(dash="dot", color="#cbd5e1"),
                             hoverinfo="skip"), row=1, col=2)
    fig.update_layout(height=400, margin=dict(l=55, r=20, t=50, b=35),
                      hovermode="closest", plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.16, x=0), font=dict(size=11))
    fig.update_xaxes(title="recall", range=[0, 1], gridcolor="#eef2f7", row=1, col=1)
    fig.update_yaxes(title="precision", range=[0, 1.02], gridcolor="#eef2f7", row=1, col=1)
    fig.update_xaxes(title="false positive rate", range=[0, 1], gridcolor="#eef2f7",
                     row=1, col=2)
    fig.update_yaxes(title="true positive rate", range=[0, 1.02], gridcolor="#eef2f7",
                     row=1, col=2)
    return div(fig)


def lead_figure(d: ReportData) -> str:
    out = d.outcomes.copy()
    out["lead"] = out.lead_hours_unclamped.fillna(0.0)
    labels = [f"{r.magnet_id}<br>{r.event_time:%m-%d %H:%M}" for r in out.itertuples()]
    colors = [COLORS["good"] if r.detected_in_time else COLORS["bad"]
              for r in out.itertuples()]
    custom = [[iso(r.alert_time) if r.alert_time else "no alert",
               f"{r.lead_hours_unclamped:.1f} h" if pd.notna(r.lead_hours_unclamped)
               else "-"] for r in out.itertuples()]
    fig = go.Figure(go.Bar(
        x=labels, y=out.lead.tolist(), marker_color=colors,
        text=[f"{v:.1f} h" if v else "missed" for v in out.lead],
        textposition="outside", customdata=custom,
        hovertemplate="%{x}<br>first alert %{customdata[0]}<br>lead %{customdata[1]}"
                      "<extra></extra>"))
    fig.add_hline(y=d.cfg.window.min_lead_hours, line_dash="dash", line_color="crimson",
                  annotation_text=f"required {d.cfg.window.min_lead_hours} h",
                  annotation_position="bottom right")
    fig.add_hline(y=d.cfg.window.horizon_hours, line_dash="dot", line_color="orange",
                  annotation_text=f"label horizon {d.cfg.window.horizon_hours} h",
                  annotation_position="top right")
    fig.update_layout(height=370, margin=dict(l=55, r=20, t=40, b=60),
                      plot_bgcolor="white", showlegend=False,
                      yaxis=dict(title="lead time before quench (h)",
                                 range=[0, 132], gridcolor="#eef2f7"),
                      xaxis=dict(title="test-period quench event"),
                      font=dict(size=11))
    return div(fig)


def importance_figure(d: ReportData) -> tuple[str, str]:
    rows = d.metrics["importance"][:15][::-1]
    group = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    fig = go.Figure(go.Bar(
        x=[r["importance"] for r in rows], y=[r["feature"] for r in rows],
        orientation="h",
        marker_color=[GROUP_COLOR.get(group.get(r["feature"]), "#94a3b8")
                      for r in rows],
        hovertemplate="%{y}<br>importance %{x:.5f}<extra></extra>"))
    fig.update_layout(height=440, margin=dict(l=160, r=20, t=25, b=45),
                      plot_bgcolor="white", showlegend=False,
                      xaxis=dict(title="mean |gradient × input| "
                                       "(standardised units)", gridcolor="#eef2f7"),
                      yaxis=dict(gridcolor="#eef2f7"), font=dict(size=11))
    legend = "".join(
        f'<span class="lg"><span class="sw" style="background:{c}"></span>'
        f'{g.replace("_", " ")}</span>' for g, c in GROUP_COLOR.items())
    return div(fig), legend


def cost_figure(d: ReportData) -> str:
    c = d.cost
    planned = c["detected_in_time"] * c["cost_per_planned_repair_usd"]
    missed = c["missed"] * c["cost_per_quench_usd"]
    false = c["false_alerts"] * c["cost_per_false_alarm_usd"]
    cats = ["reactive status quo", "predictive (QuenchGuard)"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="emergency repair + 5 d downtime", x=cats,
        y=[c["reactive_total_usd"], None], marker_color="#ef4444",
        hovertemplate="emergency: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(
        name="planned repair (predicted)", x=cats, y=[None, planned],
        marker_color="#22c55e",
        hovertemplate="planned repair: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(
        name="missed events (emergency)", x=cats, y=[None, missed],
        marker_color="#f97316",
        hovertemplate="missed events: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(
        name="false alarms", x=cats, y=[None, false], marker_color="#a855f7",
        hovertemplate="false alarms: $%{y:,.0f}<extra></extra>"))
    fig.update_layout(
        barmode="stack", height=410, margin=dict(l=75, r=20, t=45, b=35),
        plot_bgcolor="white", legend=dict(orientation="h", y=1.18, x=0),
        yaxis=dict(title="test-period cost (USD)", gridcolor="#eef2f7"),
        annotations=[dict(x=1, y=c["predictive_total_usd"] * 1.05,
                          text=f"net saving <b>${c['net_savings_usd']:,.0f}</b>",
                          showarrow=False,
                          font=dict(size=14, color="#16a34a"))],
        font=dict(size=11))
    return div(fig)


def history_figure(d: ReportData) -> str | None:
    if d.history is None or not len(d.history):
        return None
    h = d.history
    fig = make_subplots(specs=[[{"secondary_y": True}]],
                        subplot_titles=("Training run — loss vs. validation scores "
                                        "(early stopping on PR-AUC)",))
    fig.add_trace(go.Scatter(x=h.epoch, y=h.train_loss, name="train BCE loss",
                             mode="lines+markers",
                             line=dict(color=COLORS["neutral"], width=2),
                             hovertemplate="epoch %{x}<br>loss %{y:.4f}<extra></extra>"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=h.epoch, y=h.val_ap, name="val PR-AUC",
                             mode="lines+markers",
                             line=dict(color=COLORS["lstm"], width=2.5),
                             hovertemplate="epoch %{x}<br>PR-AUC %{y:.4f}<extra></extra>"),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=h.epoch, y=h.val_roc, name="val ROC-AUC",
                             mode="lines+markers",
                             line=dict(color=COLORS["hist_gbdt"], width=2),
                             hovertemplate="epoch %{x}<br>ROC-AUC %{y:.4f}<extra></extra>"),
                  secondary_y=True)
    fig.update_layout(height=350, margin=dict(l=55, r=60, t=80, b=35),
                      hovermode="x unified", plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.30, x=0), font=dict(size=11))
    fig.update_annotations(yshift=16)
    fig.update_xaxes(title="epoch", gridcolor="#eef2f7")
    fig.update_yaxes(title="loss", gridcolor="#eef2f7", secondary_y=False)
    fig.update_yaxes(title="score", range=[0, 1.02], gridcolor="#eef2f7",
                     secondary_y=True)
    return div(fig)


# ======================================================================================
# tables
# ======================================================================================
def model_table(d: ReportData) -> str:
    rows = []
    for name in ("lstm", "hist_gbdt", "logreg"):
        r = d.metrics["results"][name]
        cl, ev, al, co = r["classification"], r["events"], r["alerts"], r["cost"]
        rows.append({
            "model": "<b>lstm</b>" if name == "lstm" else name,
            "threshold": d.thresholds[name], "pr_auc": cl["pr_auc"],
            "roc_auc": cl["roc_auc"], "precision": cl["precision"],
            "recall": cl["recall"], "event_recall": ev["event_recall_in_time"],
            "lead": ev["lead_hours_median"] or 0.0,
            "false_mo": al["false_per_magnet_month"], "saving": co["net_savings_usd"],
        })
    cols = [("model", "Model", "text"), ("threshold", "Threshold", "num"),
            ("pr_auc", "PR-AUC", "num"), ("roc_auc", "ROC-AUC", "num"),
            ("precision", "Precision", "num"), ("recall", "Recall", "num"),
            ("event_recall", "Event recall &ge;72 h", "num"),
            ("lead", "Lead median", "num"), ("false_mo", "False/mo", "num"),
            ("saving", "Net saving", "num")]

    head = "".join(
        f'<th class="sortable" onclick="sortTable(\'tblModels\',{i},\'{t}\')">'
        f'{lab} <span class="arrow">&#8597;</span></th>'
        for i, (_, lab, t) in enumerate(cols))
    body = []
    for r in rows:
        cells = []
        for k, _, _t in cols:
            v = r[k]
            if k == "model":
                cells.append(f"<td>{v}</td>")
            elif k == "threshold":
                cells.append(f'<td data-v="{v}">{v:.4f}</td>')
            elif k in ("pr_auc", "roc_auc", "precision", "recall"):
                cells.append(f'<td data-v="{v}">{v:.3f}</td>')
            elif k == "event_recall":
                cls = "ok" if v >= 1 else ("warn" if v >= 0.5 else "bad")
                cells.append(f'<td data-v="{v}"><span class="{cls}">{v:.0%}</span></td>')
            elif k == "lead":
                cells.append(f'<td data-v="{v}">{v:.1f} h</td>')
            elif k == "false_mo":
                cls = "ok" if v <= 1 else "bad"
                cells.append(f'<td data-v="{v}"><span class="{cls}">{v:.2f}</span></td>')
            else:
                cells.append(f'<td data-v="{v}">${v:,.0f}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<table class="tbl" id="tblModels"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def event_table(d: ReportData) -> str:
    head = ["Magnet", "Quench time", "First alert", "Lead time", "&ge; 72 h?"]
    types = ["text", "text", "text", "num", "text"]
    th = "".join(
        f'<th class="sortable" onclick="sortTable(\'tblEvents\',{i},\'{t}\')">'
        f'{h} <span class="arrow">&#8597;</span></th>'
        for i, (h, t) in enumerate(zip(head, types)))
    body = []
    for r in d.outcomes.itertuples():
        lead = r.lead_hours_unclamped
        lead_txt = f"{lead:.2f} h" if pd.notna(lead) else "missed"
        lead_v = float(lead) if pd.notna(lead) else 0.0
        verdict = ('<span class="ok">&#10003; in time</span>' if r.detected_in_time
                   else '<span class="bad">&#10007; too late</span>')
        alert = iso(r.alert_time) if r.alert_time else "&mdash;"
        body.append(
            f'<tr><td>{r.magnet_id}</td>'
            f'<td data-v="{iso(r.event_time)}">{iso(r.event_time).replace("T", " ")}</td>'
            f'<td data-v="{alert}">{alert.replace("T", " ")}</td>'
            f'<td data-v="{lead_v}">{lead_txt}</td><td>{verdict}</td></tr>')
    return (f'<table class="tbl" id="tblEvents"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


# ======================================================================================
# page templates
# ======================================================================================
HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>QuenchGuard &mdash; Interactive Report: LSTM Predictive Maintenance for MRI Magnet Quench</title>
<style>
:root{--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--bg:#f6f8fb;--card:#fff;
      --good:#16a34a;--bad:#dc2626;--accent:#6366f1}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:60px}
body{margin:0;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
     color:var(--ink);background:var(--bg)}
header.hero{background:linear-gradient(135deg,#0f172a 0%,#1e293b 55%,#312e81 100%);color:#e2e8f0;
            padding:34px 28px 26px}
header.hero h1{margin:0 0 6px;font-size:27px;color:#fff;letter-spacing:-.4px}
header.hero p{margin:4px 0;max-width:1000px;color:#cbd5e1}
.badge{display:inline-block;background:rgba(99,102,241,.25);border:1px solid #818cf8;
       color:#c7d2fe;border-radius:999px;padding:2px 11px;font-size:12.5px;margin:4px 8px 0 0}
nav{position:sticky;top:0;z-index:50;background:#ffffffee;backdrop-filter:blur(6px);
    border-bottom:1px solid var(--line);display:flex;gap:2px;overflow-x:auto;
    box-shadow:0 1px 6px rgba(15,23,42,.05)}
nav a{padding:12px 13px;font-size:13.5px;color:var(--muted);text-decoration:none;
      white-space:nowrap;border-bottom:2px solid transparent}
nav a:hover{color:var(--accent);border-bottom-color:var(--accent)}
main{max-width:1180px;margin:0 auto;padding:22px 20px 60px}
section{background:var(--card);border:1px solid var(--line);border-radius:14px;
        padding:22px 24px;margin:20px 0;box-shadow:0 1px 3px rgba(15,23,42,.04)}
section h2{margin:0 0 6px;font-size:20px;letter-spacing:-.2px}
section h3{margin:24px 0 8px;font-size:15.5px}
section .lede{color:var(--muted);margin:0 0 14px;max-width:980px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(196px,1fr));gap:14px;margin:18px 0 4px}
.kpi{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--muted);
     border-radius:12px;padding:13px 16px}
.kpi-good{border-left-color:var(--good)}.kpi-bad{border-left-color:var(--bad)}
.kpi-neutral{border-left-color:var(--accent)}
.kpi-label{font-size:11.5px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted)}
.kpi-value{font-size:29px;font-weight:700;margin:3px 0 1px}
.kpi-sub{font-size:12.5px;color:var(--muted)}
.ctrl{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:8px 0 14px}
select{padding:7px 11px;border:1px solid var(--line);border-radius:8px;background:#fff;font:inherit}
.tbl{border-collapse:collapse;width:100%;font-size:13.6px;margin-top:8px}
.tbl th{background:#f1f5f9;text-align:left;padding:9px 11px;border-bottom:2px solid var(--line);
        cursor:pointer;user-select:none}
.tbl th:hover{background:#e8edf5}
.tbl td{padding:8px 11px;border-bottom:1px solid var(--line)}
.tbl tr:hover td{background:#f8fafc}
.arrow{color:#94a3b8;font-size:11px}
.ok{color:var(--good);font-weight:600}.bad{color:var(--bad);font-weight:600}
.warn{color:#d97706;font-weight:600}
.lg{font-size:12.5px;color:var(--muted);margin-right:16px}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;transform:translateY(1px)}
details{border:1px solid var(--line);border-radius:10px;margin:10px 0;background:#fbfcfe;overflow:hidden}
details summary{cursor:pointer;padding:11px 15px;font-weight:600;list-style:none}
details summary::before{content:"\\25B8 ";color:var(--accent)}
details[open] summary::before{content:"\\25BE "}
details .body{padding:2px 16px 14px;color:#334155}
details .body li{margin:6px 0}
.thr-card{background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:14px 18px;
          display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:12px}
.thr-card .v{font-size:23px;font-weight:700}
.thr-card .l{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.slider-row{display:flex;align-items:center;gap:14px;margin-top:6px}
.slider-row input{flex:1;accent-color:var(--accent);height:26px}
.slider-row .thr{font-variant-numeric:tabular-nums;font-weight:700;min-width:74px;font-size:17px;color:var(--accent)}
#storyCaption{color:var(--muted);font-size:13.5px}
footer{color:var(--muted);font-size:12.5px;text-align:center;padding:24px 10px 8px}
code{background:#eef2f7;border:1px solid var(--line);border-radius:5px;padding:1px 5px;font-size:13px}
pre.cmd{background:#0f172a;color:#e2e8f0;padding:12px 14px;border-radius:8px;overflow:auto;font-size:13px}
@media (max-width:760px){.kpi-value{font-size:24px}nav a{padding:11px 8px;font-size:12.5px}}
</style>
__PLOTLY_JS__
</head>
<body>
<header class="hero">
  <h1>QuenchGuard &mdash; early warning of an MRI magnet quench</h1>
  <p>Predictive maintenance case study: an LSTM over multi-rate sensor time-series warns
     &ge; 72 h before a quench ($120k repair + 5 days downtime) while paging clinical staff
     &le; once per magnet per month.</p>
  <p style="margin-top:10px">
    <span class="badge">$120k + 5 d downtime per quench</span>
    <span class="badge">72 h required lead time</span>
    <span class="badge">&le; 1 false page / magnet-month</span>
    <span class="badge">12 h look-back &middot; 36 features &middot; 4 sensors</span>
  </p>
</header>
<nav>
  <a href="#kpis">Results</a>
  <a href="#sensors">Sensor explorer</a>
  <a href="#threshold">Threshold explorer</a>
  <a href="#models">Model comparison</a>
  <a href="#events">Events &amp; lead times</a>
  <a href="#why">Why it alerts</a>
  <a href="#cost">Cost impact</a>
  <a href="#method">Methodology</a>
</nav>
<main>
"""

TAIL = """
<footer>
  QuenchGuard &middot; Case 1 &mdash; LSTM predictive maintenance &middot;
  generated by <code>scripts/build_report.py</code> &middot; synthetic data (seed 42) &middot; MIT licence
</footer>
</main>
<script>
function sortTable(id, idx, type){
  const tb = document.querySelector('#'+id+' tbody');
  const rows = Array.from(tb.rows);
  const dir = tb.dataset.dir === 'asc' ? -1 : 1;
  rows.sort(function(a,b){
    const ca = a.cells[idx], cb = b.cells[idx];
    const x = ca.dataset.v !== undefined ? ca.dataset.v : ca.textContent;
    const y = cb.dataset.v !== undefined ? cb.dataset.v : cb.textContent;
    if(type === 'num'){ return dir * (parseFloat(x) - parseFloat(y)); }
    return dir * String(x).localeCompare(String(y));
  });
  tb.dataset.dir = dir === 1 ? 'asc' : 'desc';
  rows.forEach(function(r){ tb.appendChild(r); });
}
__THRESHOLD_JS__
__STORY_JS__
</script>
</body>
</html>
"""

THRESHOLD_JS = """
const SWEEP = __SWEEP__;
const slider = document.getElementById('thrSlider');
function updateThreshold(){
  const r = SWEEP.rows[parseInt(slider.value, 10)];
  document.getElementById('thrVal').textContent = r.threshold.toFixed(3);
  document.getElementById('thrFalse').textContent = r.false_per_magnet_month.toFixed(2);
  document.getElementById('thrRecall').textContent =
      Math.round(r.event_recall_in_time * 100) + '%';
  const detected = Math.round(r.event_recall_in_time * SWEEP.n_events);
  const missed = SWEEP.n_events - detected;
  const saving = SWEEP.n_events * SWEEP.emergency
               - (missed * SWEEP.emergency + detected * SWEEP.planned
                  + r.false_alerts * SWEEP.false_usd);
  document.getElementById('thrSave').textContent =
      (saving < 0 ? '-$' : '$') + Math.abs(Math.round(saving)).toLocaleString();
  document.getElementById('thrDetected').textContent = detected + '/' + SWEEP.n_events;
  document.getElementById('thrPages').textContent = r.false_alerts;
  Plotly.restyle('thrChart', {x:[r.threshold], y:[r.false_per_magnet_month]}, [2]);
  Plotly.restyle('thrChart', {x:[r.threshold], y:[r.event_recall_in_time]}, [3]);
}
slider.addEventListener('input', updateThreshold);
updateThreshold();
"""

STORY_JS = """
const STORY_META = __STORY_META__;
const STORY_FIGS = __STORY_FIGS__;
function showStory(magnet){
  document.querySelectorAll('.story-pane').forEach(function(p){ p.style.display = 'none'; });
  const el = document.getElementById('story-' + magnet);
  if(el){
    el.style.display = 'block';
    const gd = document.getElementById('plot-' + magnet);
    const f = STORY_FIGS[magnet];
    if(gd && f){
      if(gd.dataset.rendered !== '1'){
        Plotly.newPlot(gd, f.data, f.layout, f.config || {});
        gd.dataset.rendered = '1';
      } else {
        Plotly.Plots.resize(gd);
      }
    }
  }
  document.getElementById('storyCaption').innerHTML = STORY_META[magnet].caption;
}
showStory(document.getElementById('magnetSel').value);
"""


# ======================================================================================
# page assembly
# ======================================================================================
def build(d: ReportData) -> Path:
    story_panes, story_meta, magnets, story_figs = sensor_story_divs(d)
    thr_div, sweep_json, thr_idx, thr_n = threshold_section(d)
    imp_div, imp_legend = importance_figure(d)
    hist_div = history_figure(d)
    n_ev = d.n_events

    body = f"""
<section id="kpis">
  <h2>Headline results &mdash; held-out test period</h2>
  <p class="lede">Thresholds were selected on <b>validation only</b> under a tightened
     false-page budget (1 &times; 0.6 per magnet-month) and then frozen; the test split is
     scored exactly once. {n_ev} quench events fall inside the test period.</p>
  <div class="kpis">{kpi_cards(d)}</div>
</section>

<section id="sensors">
  <h2>Sensor explorer &mdash; when did the alert arrive?</h2>
  <p class="lede">Four synchronised channels for one magnet over the test period.
     <span style="color:#ea580c">&#9644;</span> orange band = 96 h labelling horizon before a
     quench &middot; <span style="color:#dc2626">&#9474;</span> crimson line = failure &middot;
     <span style="color:#16a34a">&#9644;</span> green band = model alert episode (a green band
     with no crimson line nearby is a <b>false page</b>). Drag the range slider, zoom with the
     box-select tool, hover for exact values.</p>
  <div class="ctrl">
    <label for="magnetSel"><b>Magnet:</b></label>
    <select id="magnetSel" onchange="showStory(this.value)">
      {"".join('<option value="' + m + '">' + m + "</option>" for m in magnets)}
    </select>
    <span id="storyCaption"></span>
  </div>
  {story_panes}
</section>

<section id="threshold">
  <h2>Threshold explorer &mdash; the false-page trade-off</h2>
  <p class="lede">Drag the slider to move the operating point along the <b>validation</b>
     sweep and watch the numbers update live &mdash; net saving is recomputed from the cost
     model at every threshold. The dotted vertical line is the operating point actually
     chosen and frozen for the test results shown above.</p>
  <div class="slider-row">
    <span style="font-size:13px;color:var(--muted)">threshold</span>
    <span class="thr" id="thrVal">0.000</span>
    <input type="range" id="thrSlider" min="0" max="{thr_n - 1}" value="{thr_idx}"
           step="1"/>
  </div>
  <div class="thr-card">
    <div><div class="l">False pages / magnet-month</div><div class="v" id="thrFalse">&ndash;</div></div>
    <div><div class="l">Event recall &ge; 72 h</div><div class="v" id="thrRecall">&ndash;</div></div>
    <div><div class="l">Events detected</div><div class="v" id="thrDetected">&ndash;</div></div>
    <div><div class="l">False alerts (count)</div><div class="v" id="thrPages">&ndash;</div></div>
    <div><div class="l">Net saving</div><div class="v" id="thrSave">&ndash;</div></div>
  </div>
  <p style="font-size:12.5px; color:var(--muted); margin:8px 0 0">
    All five numbers above are <b>validation-split</b> figures &mdash; the held-out
    <b>test</b> numbers are in the headline cards. At the chosen threshold
    ({d.thresholds["lstm"]:.3f}) validation and test figures differ slightly, which is
    exactly why selection carries a 0.6&times; safety factor on the budget.
  </p>
  {thr_div}
</section>

<section id="models">
  <h2>Model comparison</h2>
  <p class="lede">The baselines see only the <b>last window</b> (a snapshot); the LSTM also
     sees the full 12 h history. Click a column header to sort &mdash; each model runs at its
     own validation-selected threshold.</p>
  {model_table(d)}
  <h3>Discrimination curves (test split, hover for values)</h3>
  {curves_figure(d)}
  <h3>Training run</h3>
  {hist_div or "<p class='lede'>training_history.csv not found &mdash; re-run the pipeline.</p>"}
</section>

<section id="events">
  <h2>Events &amp; lead times</h2>
  <p class="lede">Per-quench first-warning lead time (unclamped) against the 72 h acceptance
     line and the 96 h labelling horizon.</p>
  {lead_figure(d)}
  <h3>Event detail (LSTM, sortable)</h3>
  {event_table(d)}
</section>

<section id="why">
  <h2>Why did it alert? &mdash; gradient attribution</h2>
  <p class="lede">Mean |gradient &times; input| over the test set (inputs are standardised, so
     values are directly comparable). A useful model leans on the <b>slow physics channels</b>
     &mdash; that is the difference between a three-day warning and a 27-hour one.</p>
  <div style="margin-bottom:6px">{imp_legend}</div>
  {imp_div}
</section>

<section id="cost">
  <h2>Cost impact</h2>
  <p class="lede">Reactive quench = $120,000 repair + 5 &times; $15,000 downtime = $195,000.
     Planned repair = 55 % of repair + one scheduled day = $81,000. False alarm = $2,000
     wasted service visit. Hover the bars for the component breakdown.</p>
  {cost_figure(d)}
</section>

<section id="method">
  <h2>Methodology &mdash; the parts reviewers challenge</h2>
  <details open><summary>Sequence &amp; label geometry</summary><div class="body"><ul>
    <li>15-minute feature windows, 48 steps &rarr; <b>12 h</b> look-back per prediction.</li>
    <li>Label = quench inside the next <b>96 h</b>; acceptance = first alert &ge; <b>72 h</b>
        before the event. The 24 h slack lets the model act on the early trend.</li>
    <li>Seven days after every event are embargoed (magnet offline for repair).</li>
  </ul></div></details>
  <details><summary>Leakage guards</summary><div class="body"><ul>
    <li>Rolling statistics and diffs look strictly backwards; the 7-day trend is fit on
        <b>complete calendar days</b> and joined with a 1-day lag.</li>
    <li>Splits are <b>global time cuts</b> (60/20/20), never shuffled; sequences must be
        contiguous and entirely inside one split.</li>
    <li>Scaler fit on training rows only; NaNs forward-filled within a magnet, never
        back-filled.</li>
    <li>Test set scored exactly once, at a threshold chosen on validation.</li>
  </ul></div></details>
  <details><summary>Threshold policy</summary><div class="body"><ul>
    <li>Sweep candidate thresholds on validation and convert false alerts into
        <b>false pages per magnet-month</b>.</li>
    <li>Keep operating points &le; 1 &times; 0.6 = 0.6/magnet-month (the safety factor absorbs
        the val&rarr;test shift), then pick the best event recall &ge; 72 h among them.</li>
  </ul></div></details>
  <details><summary>Honest limitations</summary><div class="body"><ul>
    <li>The data is <b>synthetic</b> &mdash; the physics ramps are documented assumptions in
        <code>config.py</code>; real telemetry slots into the same four parquet schemas.</li>
    <li>{n_ev} test events is a small sample: read the lead-time <i>distribution</i>, not just
        the mean.</li>
    <li>Gradient attribution is a first-order explanation, not causal proof.</li>
    <li>Results are for seed 42; another seed may shift figures by a few points.</li>
  </ul></div></details>
  <details><summary>Reproduce</summary><div class="body"><pre class="cmd">pip install -r requirements.txt
python scripts/run_pipeline.py        # data -&gt; model -&gt; metrics.json / sweep / history
python scripts/build_report.py        # this interactive report
python scripts/build_notebook.py      # walkthrough notebook + static HTML export
python -m pytest                      # 25 tests</pre></div></details>
</section>
"""

    html = (HEAD.replace("__PLOTLY_JS__", "<script>" + get_plotlyjs() + "</script>")
            + body
            + TAIL.replace("__THRESHOLD_JS__",
                           THRESHOLD_JS.replace("__SWEEP__", sweep_json))
                   .replace("__STORY_JS__",
                            STORY_JS.replace("__STORY_META__", json.dumps(story_meta))
                                    .replace("__STORY_FIGS__", json.dumps(story_figs))))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    return OUT_PATH


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print("loading pipeline artifacts ...")
    d = ReportData()
    print(f"  {d.n_events} test events | {len(d.episodes)} alert episodes | "
          f"lstm threshold {d.thresholds['lstm']:.3f} | device {d.device}")
    print("rendering interactive report ...")
    path = build(d)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
