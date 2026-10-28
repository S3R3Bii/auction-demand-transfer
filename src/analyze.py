"""Finite demand-model selection followed by separate time and maturity tests."""

from datetime import datetime, timezone
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from prepare import ROOT, FEATURES, build, source_rows, fingerprint

OUT = ROOT / "results"
CACHE = ROOT / "data/cache"
SEED = 20261028


def write(name, obj):
    (OUT / name).write_text(
        json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def candidates():
    result = [
        dict(name="baseline_" + x, kind="baseline", feature=x)
        for x in ["prior6_median", "prior6_mean", "prior1"]
    ]
    result += [
        dict(name="ridge_" + str(a), kind="ridge", alpha=a) for a in [0.1, 1, 10, 100]
    ]
    result += [
        dict(
            name="forest_depth" + str(d),
            kind="forest",
            max_depth=d,
            n_estimators=200,
            min_samples_leaf=10,
        )
        for d in [4, 8, None]
    ]
    result += [
        dict(
            name=f"boost_depth{d}_lr{rate}",
            kind="boost",
            max_depth=d,
            learning_rate=rate,
            max_iter=100,
            min_samples_leaf=20,
            l2_regularization=1,
        )
        for d in [2, 3]
        for rate in [0.05, 0.1]
    ]
    return result


def fit(config, frame):
    if config["kind"] == "baseline":
        return config["feature"]
    params = {k: v for k, v in config.items() if k not in ["name", "kind"]}
    if config["kind"] == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(**params, solver="svd"))
    elif config["kind"] == "forest":
        model = RandomForestRegressor(
            **params, max_features="sqrt", bootstrap=True, random_state=SEED, n_jobs=2
        )
    else:
        model = HistGradientBoostingRegressor(
            **params, early_stopping=False, random_state=SEED
        )
    with threadpool_limits(limits=2):
        model.fit(frame[FEATURES], frame.target)
    return model


def predict(model, frame):
    if isinstance(model, str):
        return frame[model].to_numpy()
    with threadpool_limits(limits=2):
        return np.maximum(0, model.predict(frame[FEATURES]))


def table(model, frame, name):
    output = frame[
        [
            "key",
            "auction_date",
            "announcement_date",
            "term_years",
            "target",
            "offering_ratio",
        ]
    ].copy()
    output["model"] = name
    output["prediction"] = predict(model, frame)
    return output


def metrics(frame):
    error = frame.prediction - frame.target
    return dict(
        auctions=int(len(frame)),
        mse=float(np.mean(error**2)),
        rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.mean(abs(error))),
    )


def validate():
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    frame, _, _ = build(source_rows(), end="2022-12-31")
    fitting = frame[frame.cohort == "fit"]
    tuning = frame[frame.cohort == "tune"]
    trials = []
    predictions = []
    assert not (fitting.term_years == 10).any() and not (tuning.term_years == 10).any()
    for config in candidates():
        start = time.perf_counter()
        model = fit(config, fitting)
        prediction = table(model, tuning, config["name"])
        score = metrics(prediction)
        predictions.append(prediction)
        trials.append(
            dict(
                **config,
                **score,
                elapsed_seconds=time.perf_counter() - start,
                status="success",
            )
        )
        joblib.dump(model, CACHE / (config["name"] + ".joblib"))
        print(
            json.dumps({"candidate": config["name"], "validation_mse": score["mse"]}),
            flush=True,
        )
        if config["kind"] == "ridge":
            scaler, regressor = model.steps[0][1], model.steps[1][1]
            write(
                config["name"] + "_parameters.json",
                dict(
                    features=FEATURES,
                    mean=scaler.mean_.tolist(),
                    scale=scaler.scale_.tolist(),
                    coefficient=regressor.coef_.tolist(),
                    intercept=float(regressor.intercept_),
                    alpha=config["alpha"],
                ),
            )
    pd.DataFrame(trials).to_csv(OUT / "search.csv", index=False)
    pd.concat(predictions).to_csv(OUT / "validation_predictions.csv", index=False)
    winner = min(trials, key=lambda r: (r["mse"], r["name"]))["name"]
    write(
        "selection.json",
        dict(
            frozen_utc=datetime.now(timezone.utc).isoformat(),
            winner=winner,
            baseline="baseline_prior6_median",
            criterion="Auction-level validation MSE, then name",
            candidates=len(trials),
            features=FEATURES,
            development_input_sha256=fingerprint(frame),
            fit_rows=len(fitting),
            tuning_rows=len(tuning),
            fit_key_sha256=fingerprint(fitting[["key"]]),
            tune_key_sha256=fingerprint(tuning[["key"]]),
            seed=SEED,
            test_not_evaluated=True,
        ),
    )


def test():
    selection = json.loads((OUT / "selection.json").read_text())
    frame, lineage, excluded = build(source_rows())
    test_frame = frame[frame.cohort == "test"].copy()
    test_frame["test_cohort"] = np.where(
        test_frame.term_years == 10, "held10year", "non10year"
    )
    test_frame.to_csv(OUT / "test_features.csv", index=False)
    lineage[lineage.key.isin(test_frame.key)].to_csv(
        OUT / "test_lineage.csv", index=False
    )
    excluded.to_csv(OUT / "all_exclusions.csv", index=False)
    predictions = []
    summary = []
    blocks = []
    draws_export = []
    intervals = []
    rng = np.random.default_rng(SEED)
    names = list(
        dict.fromkeys(
            [
                selection["winner"],
                "baseline_prior6_median",
                "baseline_prior6_mean",
                "baseline_prior1",
            ]
        )
    )
    for cohort, group in test_frame.groupby("test_cohort", sort=True):
        for name in names:
            model = joblib.load(CACHE / (name + ".joblib"))
            result = table(model, group, name)
            result["test_cohort"] = cohort
            predictions.append(result)
            summary.append(dict(cohort=cohort, model=name, **metrics(result)))
            for term, subset in result.groupby("term_years"):
                summary.append(
                    dict(
                        cohort=cohort,
                        model=name,
                        stratum="term",
                        value=int(term),
                        **metrics(subset),
                    )
                )
            for label, condition in [
                ("below0.8", result.offering_ratio < 0.8),
                ("0.8to1.2", result.offering_ratio.between(0.8, 1.2)),
                ("above1.2", result.offering_ratio > 1.2),
            ]:
                if condition.any():
                    summary.append(
                        dict(
                            cohort=cohort,
                            model=name,
                            stratum="offering_ratio",
                            value=label,
                            **metrics(result[condition]),
                        )
                    )
        a = next(
            r
            for r in predictions
            if r.model.iloc[0] == selection["winner"]
            and r.test_cohort.iloc[0] == cohort
        )
        b = next(
            r
            for r in predictions
            if r.model.iloc[0] == selection["baseline"]
            and r.test_cohort.iloc[0] == cohort
        )
        assert a.key.tolist() == b.key.tolist()
        error = (a.prediction - a.target) ** 2 - (b.prediction - b.target) ** 2
        part = pd.DataFrame(
            {
                "quarter": pd.to_datetime(a.auction_date).dt.to_period("Q").astype(str),
                "difference": error,
            }
        )
        aggregate = part.groupby("quarter").difference.agg(["sum", "count"])
        quarters = aggregate.index.tolist()
        blocks += [
            dict(
                cohort=cohort,
                quarter=q,
                error_sum=float(r["sum"]),
                auctions=int(r["count"]),
            )
            for q, r in aggregate.iterrows()
        ]
        if len(quarters) < 8:
            intervals.append(
                dict(
                    cohort=cohort,
                    state="insufficient quarter blocks",
                    quarters=len(quarters),
                )
            )
            continue
        indices = rng.integers(0, len(quarters), size=(2000, len(quarters)))
        replicates = []
        for i, row in enumerate(indices):
            value = float(
                aggregate["sum"].to_numpy()[row].sum()
                / aggregate["count"].to_numpy()[row].sum()
            )
            replicates.append(value)
            draws_export.append(
                dict(
                    cohort=cohort,
                    draw=i,
                    quarter_indices=" ".join(str(x) for x in row),
                    mse_difference=value,
                )
            )
        intervals.append(
            dict(
                cohort=cohort,
                quantity="Selected minus six-median MSE",
                quarters=len(quarters),
                draws=2000,
                mean=float(error.mean()),
                percentile95=np.quantile(replicates, [0.025, 0.975]).tolist(),
                scope="Calendar-quarter block bootstrap conditional on observed quarters; no trading-return or causal claim",
            )
        )
    pd.concat(predictions).to_csv(OUT / "test_predictions.csv", index=False)
    pd.DataFrame(blocks).to_csv(OUT / "quarter_blocks.csv", index=False)
    pd.DataFrame(draws_export).to_csv(OUT / "bootstrap_draws.csv", index=False)
    write(
        "test_summary.json",
        dict(
            evaluated_utc=datetime.now(timezone.utc).isoformat(),
            scope="Retrospective announcement-field evaluation; original publication vintages unverified",
            test_input_sha256=fingerprint(test_frame),
            metrics=summary,
            block_intervals=intervals,
        ),
    )
    print(
        json.dumps(
            {
                "test_auctions": len(test_frame),
                "test_cohorts": test_frame.test_cohort.value_counts().to_dict(),
                "model_names": names,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["validate", "test"])
    args = parser.parse_args()
    globals()[args.phase]()
