"""Announcement-field covariates with explicit earlier-auction lineage."""

from pathlib import Path
import calendar
import hashlib
import json
import math
import statistics
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TERMS = {f"{x}-Year": x for x in [2, 3, 5, 7, 10, 20, 30]}
FEATURES = [
    "term_years",
    "log_offering_billions",
    "reopening_flag",
    "announcement_month",
    "announcement_lead_days",
    "auction_weekday",
    "days_to_month_end",
    "prior6_mean",
    "prior6_median",
    "prior1",
    "prior6_offering_billions",
    "offering_ratio",
]


def source_rows():
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    rows = []
    for item in manifest["files"]:
        blob = (ROOT / item["file"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == item["sha256"]
        rows.extend(json.loads(blob)["data"])
    assert len(rows) == manifest["rows"]
    keys = [(r["cusip"], r["auction_date"]) for r in rows]
    assert len(keys) == len(set(keys))
    return rows


def nominal(rows, end="2026-09-30"):
    eligible = []
    excluded = []
    for record in rows:
        if record["auction_date"] > end:
            continue
        key = record["cusip"] + "_" + record["auction_date"]
        reason = None
        if (
            record["security_type"] not in ["Note", "Bond"]
            or record["original_security_term"] not in TERMS
        ):
            reason = "outside declared instrument/term family"
        elif (
            record["floating_rate"] != "No"
            or record["inflation_index_security"] != "No"
        ):
            reason = "floating/inflation flag or unverified flag"
        elif any(
            record[k] in ["null", "", None]
            for k in ["bid_to_cover_ratio", "offering_amt", "announcemt_date"]
        ):
            reason = "missing outcome/announced amount/date"
        if reason:
            excluded.append(dict(key=key, reason=reason))
            continue
        r = dict(record)
        r["key"] = key
        r["term_years"] = TERMS[r["original_security_term"]]
        r["target"] = float(r["bid_to_cover_ratio"])
        r["offering_billions"] = float(r["offering_amt"]) / 1e9
        assert r["target"] > 0 and r["offering_billions"] > 0
        assert (
            r["announcemt_date"]
            <= r["auction_date"]
            <= r["issue_date"]
            < r["maturity_date"]
        )
        assert r["reopening"] in ["Yes", "No"]
        eligible.append(r)
    return sorted(eligible, key=lambda r: (r["auction_date"], r["cusip"])), excluded


def build(rows, end="2026-09-30"):
    eligible, excluded = nominal(rows, end)
    output = []
    lineage = []
    for r in eligible:
        prior = [
            p
            for p in eligible
            if p["term_years"] == r["term_years"]
            and p["auction_date"] < r["announcemt_date"]
        ][-6:]
        if len(prior) < 6:
            excluded.append(
                dict(
                    key=r["key"],
                    reason="fewer than six earlier same-term auction results",
                )
            )
            continue
        announce = pd.Timestamp(r["announcemt_date"])
        auction = pd.Timestamp(r["auction_date"])
        amount = statistics.mean(p["offering_billions"] for p in prior)
        values = [p["target"] for p in prior]
        cohort = (
            "warmup"
            if r["auction_date"] < "2010-01-01"
            else (
                "fit"
                if r["auction_date"] < "2020-01-01"
                else ("tune" if r["auction_date"] < "2023-01-01" else "test")
            )
        )
        if r["term_years"] == 10 and cohort in ["fit", "tune"]:
            cohort = "held_maturity_history_only"
        output.append(
            dict(
                key=r["key"],
                auction_date=r["auction_date"],
                announcement_date=r["announcemt_date"],
                cohort=cohort,
                term_years=r["term_years"],
                target=r["target"],
                log_offering_billions=math.log(r["offering_billions"]),
                reopening_flag=int(r["reopening"] == "Yes"),
                announcement_month=announce.month,
                announcement_lead_days=(auction - announce).days,
                auction_weekday=auction.dayofweek,
                days_to_month_end=calendar.monthrange(auction.year, auction.month)[1]
                - auction.day,
                prior6_mean=statistics.mean(values),
                prior6_median=statistics.median(values),
                prior1=values[-1],
                prior6_offering_billions=amount,
                offering_ratio=r["offering_billions"] / amount,
            )
        )
        for position, p in enumerate(prior):
            lineage.append(
                dict(
                    key=r["key"],
                    position=position + 1,
                    source_key=p["key"],
                    source_auction_date=p["auction_date"],
                    source_target=p["target"],
                    source_offering_billions=p["offering_billions"],
                )
            )
    frame = pd.DataFrame(output)
    assert frame[FEATURES].notna().all().all()
    assert not set(FEATURES) & {
        "target",
        "bid_to_cover_ratio",
        "total_tendered",
        "total_accepted",
        "high_yield",
    }
    return frame, pd.DataFrame(lineage), pd.DataFrame(excluded)


def fingerprint(frame):
    return hashlib.sha256(
        frame.sort_values("key")
        .to_csv(index=False, float_format="%.14g", lineterminator="\n")
        .encode()
    ).hexdigest()


if __name__ == "__main__":
    frame, lineage, excluded = build(source_rows(), end="2022-12-31")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    frame.to_csv(out / "development_features.csv", index=False)
    lineage.to_csv(out / "development_lineage.csv", index=False)
    excluded.to_csv(out / "development_exclusions.csv", index=False)
    summary = {
        "rows_by_cohort": frame.cohort.value_counts().to_dict(),
        "fit_term_counts": frame[frame.cohort == "fit"]
        .term_years.value_counts()
        .sort_index()
        .to_dict(),
        "tuning_term_counts": frame[frame.cohort == "tune"]
        .term_years.value_counts()
        .sort_index()
        .to_dict(),
        "development_input_sha256": fingerprint(frame),
        "test_metrics_not_read": True,
    }
    (out / "source_quality.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
