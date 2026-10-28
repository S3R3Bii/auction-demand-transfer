"""Model-selection evidence, maturity transfer and size stress in ratio units."""

import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prepare import ROOT

OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
summary = json.loads((ROOT / "results/test_summary.json").read_text())
selection = json.loads((ROOT / "results/selection.json").read_text())
search = pd.read_csv(ROOT / "results/search.csv")
metrics = summary["metrics"]
NAVY = "#17324d"
TEAL = "#007d7b"
GREY = "#8393a3"
ORANGE = "#bc5d28"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": NAVY,
        "text.color": NAVY,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    }
)


def save(fig, name, title, subtitle, footnote):
    fig.suptitle(title, x=0.07, y=0.965, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.07, 0.90, subtitle, fontsize=10, color=TEAL)
    fig.text(0.07, 0.035, footnote, fontsize=8, color="#4a5a68", linespacing=1.5)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=180)
    plt.close(fig)


names = {
    "baseline_prior6_mean": "Previous-six mean",
    "baseline_prior6_median": "Previous-six median",
    "baseline_prior1": "Previous auction",
}
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.subplots_adjust(left=0.08, right=0.97, top=0.79, bottom=0.24, wspace=0.38)
x = np.arange(2)
cohorts = ["held10year", "non10year"]
for j, (name, color) in enumerate(zip(names, [TEAL, GREY, ORANGE])):
    rows = [
        next(
            r
            for r in metrics
            if r["cohort"] == c and r["model"] == name and "stratum" not in r
        )
        for c in cohorts
    ]
    bars = axes[0].bar(
        x + (j - 1) * 0.24,
        [r["rmse"] for r in rows],
        width=0.22,
        color=color,
        label=names[name],
    )
    axes[0].bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
axes[0].set(
    xticks=x,
    xticklabels=["10-year holdout\n45 auctions", "Other terms\n270 auctions"],
    ylabel="RMSE (bid-to-cover ratio units)",
    ylim=(0, 0.185),
    title="Simple baselines remain competitive",
)
axes[0].legend(frameon=False, fontsize=8)
axes[0].grid(axis="y", alpha=0.15)
axes[0].set_axisbelow(True)
for i, interval in enumerate(summary["block_intervals"]):
    lo, hi = interval["percentile95"]
    axes[1].errorbar(
        interval["mean"],
        i,
        xerr=[[interval["mean"] - lo], [hi - interval["mean"]]],
        fmt="o",
        color=TEAL,
        capsize=5,
    )
axes[1].axvline(0, color=ORANGE, lw=1)
axes[1].set(
    yticks=[0, 1],
    yticklabels=["10-year holdout", "Other terms"],
    xlabel="Mean minus median MSE (ratio units²)",
    ylim=(-0.6, 1.6),
    title="A modest, cohort-dependent difference",
)
axes[1].invert_yaxis()
axes[1].grid(axis="x", alpha=0.15)
save(
    fig,
    "decision-summary",
    "The finite search chose a simple history rule",
    "RETROSPECTIVE ANNOUNCEMENT-FIELD EVALUATION  |  2023–2026 Q3; no trading-return claim",
    "Selected: previous-six same-term mean. No10-year rows fit models or select parameters; earlier10-year outcomes remain available as lags.\nIntervals:2,000 calendar-quarter block bootstrap draws over15 quarters. Other terms include20-year auctions absent from fitting.",
)

ordered = search.sort_values("mse")
labels = []
for _, row in ordered.iterrows():
    name = row["name"]
    labels.append(
        names.get(
            name,
            name.replace("forest_depth", "Forest depth ")
            .replace("None", "unlimited")
            .replace("ridge_", "Ridge alpha ")
            .replace("boost_depth", "Boost depth ")
            .replace("_lr", " / LR "),
        )
    )
fig, ax = plt.subplots(figsize=(12, 7))
fig.subplots_adjust(left=0.28, right=0.96, top=0.80, bottom=0.21)
ax.barh(
    np.arange(len(ordered)),
    ordered.mse,
    color=[TEAL if n == selection["winner"] else GREY for n in ordered.name],
)
ax.set_yticks(np.arange(len(ordered)), labels)
ax.invert_yaxis()
ax.set(
    xlabel="2020–2022 validation MSE (ratio units²)",
    title="Every candidate retained; models do not win by default",
)
ax.grid(axis="x", alpha=0.15)
ax.set_axisbelow(True)
save(
    fig,
    "optimization-evidence",
    "Fourteen candidates, one declared selection rule",
    "599 FIT AUCTIONS / 207 TUNING AUCTIONS  |  Ten-year originals excluded from both",
    "Ridge scaling uses fitting rows only; every validation prediction and fitted ridge parameter is available for independent checking.\nOne forward validation period, not repeated rolling folds. A finite grid does not prove that no useful pooled demand model exists.",
)

rows = [
    r
    for r in metrics
    if r.get("stratum") == "offering_ratio"
    and r["model"] in [selection["winner"], selection["baseline"]]
]
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.subplots_adjust(left=0.08, right=0.97, top=0.79, bottom=0.24, wspace=0.3)
for ax, cohort in zip(axes, cohorts):
    bands = ["below0.8", "0.8to1.2", "above1.2"]
    x = np.arange(3)
    for j, (name, color) in enumerate(
        [(selection["baseline"], GREY), (selection["winner"], TEAL)]
    ):
        lookup = {
            r["value"]: r for r in rows if r["cohort"] == cohort and r["model"] == name
        }
        for i, band in enumerate(bands):
            if band not in lookup:
                if j == 0:
                    ax.text(i, 0.012, "No\nobservations", ha="center", fontsize=8)
                continue
            r = lookup[band]
            ax.bar(
                i + (j - 0.5) * 0.3,
                r["rmse"],
                width=0.28,
                color=color,
                label=names[name] if i == 1 else None,
            )
            ax.text(
                i + (j - 0.5) * 0.3,
                r["rmse"] + 0.005,
                str(r["auctions"]),
                ha="center",
                fontsize=9,
            )
    ax.set(
        xticks=x,
        xticklabels=["Below 0.8", "0.8–1.2", "Above 1.2"],
        ylabel="RMSE (ratio units)",
        xlabel="Offering / previous-six mean offering",
        title="10-year holdout" if cohort == "held10year" else "Other terms",
        ylim=(0, 0.23),
        xlim=(-0.5, 2.5),
    )
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
axes[1].legend(frameon=False, fontsize=8)
save(
    fig,
    "size-stress",
    "Issuance-size strata reveal where evidence is sparse",
    "RETROSPECTIVE / DESCRIPTIVE STRATA  |  Fixed0.8 and1.2 boundaries, not economic stress definitions",
    "Numbers above bars are auction counts; a blank stratum is not a zero error. Current source amounts may contain revisions.\nThe design evaluates demand-ratio forecast error, not a causal issuance effect, tail-yield prediction or an investment strategy.",
)
print(json.dumps({"figures": 3, "formats": ["png", "pdf"]}))
