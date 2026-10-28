"""SQL lineage and scalar metric checks independent of preparation/model scoring."""

from pathlib import Path
from collections import defaultdict
from datetime import date
import argparse
import calendar
import csv
import hashlib
import json
import math
import sqlite3
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def read(name):
    with open(OUT / name, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-10), (a, b)


def group(rows, fields):
    result = defaultdict(list)
    for r in rows:
        result[tuple(r[k] for k in fields)].append(r)
    return result


def source():
    values = []
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for item in manifest["files"]:
        blob = (ROOT / item["file"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == item["sha256"]
        values.extend(json.loads(blob)["data"])
    assert len(values) == manifest["rows"] and len(
        {(r["cusip"], r["auction_date"]) for r in values}
    ) == len(values)
    return values


def verify_features(prefix):
    feature = read(prefix + "_features.csv")
    actual = {r["key"]: r for r in feature}
    raw = source()
    lookup = {}
    eligible = []
    for r in raw:
        if (
            r["security_type"] not in ["Note", "Bond"]
            or r["original_security_term"]
            not in [f"{x}-Year" for x in [2, 3, 5, 7, 10, 20, 30]]
            or r["floating_rate"] != "No"
            or r["inflation_index_security"] != "No"
        ):
            continue
        if any(
            r[k] in ["null", "", None]
            for k in ["bid_to_cover_ratio", "offering_amt", "announcemt_date"]
        ):
            continue
        key = r["cusip"] + "_" + r["auction_date"]
        term = int(r["original_security_term"].split("-")[0])
        target = float(r["bid_to_cover_ratio"])
        offering = float(r["offering_amt"]) / 1e9
        lookup[key] = r
        eligible.append(
            (
                key,
                r["auction_date"],
                r["announcemt_date"],
                term,
                target,
                offering,
                r["reopening"],
            )
        )
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute(
        "CREATE TABLE auctions(key TEXT PRIMARY KEY,auction_date TEXT,announcement_date TEXT,term_years INTEGER,target REAL,offering_billions REAL,reopening TEXT)"
    )
    db.executemany("INSERT INTO auctions VALUES(?,?,?,?,?,?,?)", eligible)
    db.execute("CREATE TABLE scored(key TEXT PRIMARY KEY)")
    db.executemany("INSERT INTO scored VALUES(?)", [(k,) for k in actual])
    db.execute("CREATE INDEX history ON auctions(term_years,auction_date)")
    query = (ROOT / "sql/eligible_history.sql").read_text()
    lineage = group([dict(r) for r in db.execute(query)], ["key"])
    exported = group(read(prefix + "_lineage.csv"), ["key"])
    for key, row in actual.items():
        original = lookup[key]
        history = lineage[(key,)]
        provided = sorted(exported[(key,)], key=lambda r: int(r["position"]))
        assert len(history) == len(provided) == 6
        for expected, published in zip(history, provided):
            assert (
                expected["source_key"] == published["source_key"]
                and expected["source_auction_date"] < row["announcement_date"]
            )
            close(expected["source_target"], published["source_target"])
            close(
                expected["source_offering_billions"],
                published["source_offering_billions"],
            )
        targets = [r["source_target"] for r in history]
        amount = st.mean(r["source_offering_billions"] for r in history)
        auction = date.fromisoformat(original["auction_date"])
        announce = date.fromisoformat(original["announcemt_date"])
        expected = {
            "target": float(original["bid_to_cover_ratio"]),
            "term_years": int(original["original_security_term"].split("-")[0]),
            "prior6_mean": st.mean(targets),
            "prior6_median": st.median(targets),
            "prior1": targets[-1],
            "prior6_offering_billions": amount,
            "offering_ratio": float(original["offering_amt"]) / 1e9 / amount,
            "log_offering_billions": math.log(float(original["offering_amt"]) / 1e9),
            "reopening_flag": int(original["reopening"] == "Yes"),
            "announcement_month": announce.month,
            "announcement_lead_days": (auction - announce).days,
            "auction_weekday": auction.weekday(),
            "days_to_month_end": calendar.monthrange(auction.year, auction.month)[1]
            - auction.day,
        }
        for field, value in expected.items():
            close(value, row[field])
        if row["cohort"] in ["fit", "tune"]:
            assert int(row["term_years"]) != 10
        if row["cohort"] == "fit":
            assert "2010-01-01" <= row["auction_date"] < "2020-01-01"
        elif row["cohort"] == "tune":
            assert "2020-01-01" <= row["auction_date"] < "2023-01-01"
        elif row["cohort"] == "test":
            assert "2023-01-01" <= row["auction_date"] <= "2026-09-30"
    return feature


def scores(records):
    errors = [float(r["prediction"]) - float(r["target"]) for r in records]
    return dict(
        auctions=len(records),
        mse=st.mean(x * x for x in errors),
        rmse=math.sqrt(st.mean(x * x for x in errors)),
        mae=st.mean(abs(x) for x in errors),
    )


def verify_validation():
    features = verify_features("development")
    lookup = {r["key"]: r for r in features}
    trials = read("search.csv")
    predictions = group(read("validation_predictions.csv"), ["model"])
    selection = json.loads((OUT / "selection.json").read_text())
    losses = {}
    fit = [r for r in features if r["cohort"] == "fit"]
    tune = [r for r in features if r["cohort"] == "tune"]
    assert len(fit) == selection["fit_rows"] and len(tune) == selection["tuning_rows"]
    expected_keys = {r["key"] for r in tune}
    for trial in trials:
        rows = predictions[(trial["name"],)]
        assert {r["key"] for r in rows} == expected_keys
        metric = scores(rows)
        losses[trial["name"]] = metric["mse"]
        for name, value in metric.items():
            close(value, trial[name])
        for row in rows:
            close(row["target"], lookup[row["key"]]["target"])
        if trial["kind"] == "baseline":
            for row in rows:
                close(row["prediction"], lookup[row["key"]][trial["feature"]])
        if trial["kind"] == "ridge":
            params = json.loads(
                (OUT / (trial["name"] + "_parameters.json")).read_text()
            )
            for index, field in enumerate(params["features"]):
                values = [float(r[field]) for r in fit]
                close(st.mean(values), params["mean"][index])
                close(st.pstdev(values) or 1, params["scale"][index])
            for row in rows:
                value = params["intercept"] + sum(
                    coef * (float(lookup[row["key"]][field]) - mean) / scale
                    for field, coef, mean, scale in zip(
                        params["features"],
                        params["coefficient"],
                        params["mean"],
                        params["scale"],
                    )
                )
                close(max(0, value), row["prediction"])
    assert (
        len(trials) == len(predictions) == selection["candidates"] == 14
        and min(losses, key=lambda k: (losses[k], k)) == selection["winner"]
    )
    return {
        "status": "PASS",
        "candidate_metrics_recomputed": 14,
        "development_feature_rows_checked": len(features),
        "lineage": "Every exported earlier-result identity and feature checked with independent SQL/scalar reconstruction",
        "ridge": "All train-only means/scales and validation predictions independently reconstructed",
    }


def verify_test():
    features = verify_features("test")
    lookup = {r["key"]: r for r in features}
    summary = json.loads((OUT / "test_summary.json").read_text())
    predictions = read("test_predictions.csv")
    selection = json.loads((OUT / "selection.json").read_text())
    for expected in summary["metrics"]:
        rows = [
            r
            for r in predictions
            if r["model"] == expected["model"]
            and r["test_cohort"] == expected["cohort"]
        ]
        if expected.get("stratum") == "term":
            rows = [r for r in rows if int(r["term_years"]) == expected["value"]]
        elif expected.get("stratum") == "offering_ratio":
            value = expected["value"]
            rows = [
                r
                for r in rows
                if (
                    float(r["offering_ratio"]) < 0.8
                    if value == "below0.8"
                    else (
                        0.8 <= float(r["offering_ratio"]) <= 1.2
                        if value == "0.8to1.2"
                        else float(r["offering_ratio"]) > 1.2
                    )
                )
            ]
        for field, value in scores(rows).items():
            close(value, expected[field])
        for row in rows:
            close(row["target"], lookup[row["key"]]["target"])
            assert (int(row["term_years"]) == 10) == (
                row["test_cohort"] == "held10year"
            )
    blocks = group(read("quarter_blocks.csv"), ["cohort"])
    draws = group(read("bootstrap_draws.csv"), ["cohort"])
    for interval in summary["block_intervals"]:
        cohort = interval["cohort"]
        items = sorted(blocks[(cohort,)], key=lambda r: r["quarter"])
        assert len(items) == interval["quarters"]
        a = {
            r["key"]: r
            for r in predictions
            if r["model"] == selection["winner"] and r["test_cohort"] == cohort
        }
        b = {
            r["key"]: r
            for r in predictions
            if r["model"] == selection["baseline"] and r["test_cohort"] == cohort
        }
        assert a.keys() == b.keys()
        quarter = defaultdict(list)
        for key, r in a.items():
            year, month = map(int, r["auction_date"].split("-")[:2])
            q = f"{year}Q{(month-1)//3+1}"
            quarter[q].append(
                (float(r["prediction"]) - float(r["target"])) ** 2
                - (float(b[key]["prediction"]) - float(r["target"])) ** 2
            )
        for item in items:
            assert len(quarter[item["quarter"]]) == int(item["auctions"])
            close(sum(quarter[item["quarter"]]), item["error_sum"])
        close(
            st.mean(x for values in quarter.values() for x in values), interval["mean"]
        )
        replicates = []
        for draw in draws[(cohort,)]:
            indices = [int(x) for x in draw["quarter_indices"].split()]
            assert len(indices) == len(items)
            value = sum(float(items[i]["error_sum"]) for i in indices) / sum(
                int(items[i]["auctions"]) for i in indices
            )
            close(value, draw["mse_difference"])
            replicates.append(value)
        assert len(replicates) == interval["draws"] == 2000
        replicates.sort()
        for p, published in zip([0.025, 0.975], interval["percentile95"]):
            location = (len(replicates) - 1) * p
            lo = math.floor(location)
            hi = math.ceil(location)
            close(
                replicates[lo] + (location - lo) * (replicates[hi] - replicates[lo]),
                published,
            )
    return {
        "status": "PASS",
        "test_feature_rows_checked": len(features),
        "metrics_checked": len(summary["metrics"]),
        "bootstrap_draws_checked": sum(len(x) for x in draws.values()),
        "scope": "Full feature lineage, targets, all metric/stratum exports and calendar-quarter blocks independently reconstructed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["validation", "test"])
    phase = parser.parse_args().phase
    result = verify_validation() if phase == "validation" else verify_test()
    (OUT / f"verification_{phase}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
