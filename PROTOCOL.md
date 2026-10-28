# Auction demand across size and maturity changes

Declared28October2026 after checking official source definitions and API field metadata, before downloading the analysis cohort or inspecting its results.

## Decision and scope

Can a pooled announcement-field model improve a simple maturity-specific historical demand baseline, and does the improvement survive a maturity excluded from model fitting? The outcome is the Treasury's reported bid-to-cover ratio. It is not an investment return, an auction recommendation or a causal effect of issuance size. The current historical source is not a first-release archive: this is explicitly a **retrospective announcement-field evaluation**, with revision and release-time limitations.

Use U.S. Treasury Fiscal Data aggregate auction records from2009-01-01 through2026-09-30. Analyze nominal fixed-rate notes and bonds with original terms2,3,5,7,10,20or30years; exclude inflation-indexed and floating-rate instruments.2009 supplies prior history only. Fit2010–2019 excluding10-year originals; tune2020–2022 excluding10-year originals; protected tests2023–2026Q3 split into seen maturities and10-year parameter-transfer. Never use10-year rows to fit or select parameters, but earlier10-year auction outcomes may form date-available lag features. This is **parameter transfer**, not a claim that an entirely new maturity has no prior history.

## Source and information boundary

Validate exact source fields, missingness, unique(CUSIP,auction_date), date order, types, flags and units. Use the official bid-to-cover ratio rather than incorrectly dividing totals that include SOMA. Official regulation defines the ratio excluding bids and awards for the Federal Reserve's own account. Retain source URLs, query parameters, response hashes, retrieval timestamp, reported source units and all exclusions.

Allowed current-auction features: original term in years; log(announced offering billions); reopening flag; announcement month; days from announcement to auction; auction weekday; month-end distance. Current accepted/tendered amounts, yields, bidder shares, prices and target are never current features. Lag features use only auction results with auction_date strictly earlier than the current announcement date, avoiding unresolved intraday order. Source current-history values may have revisions; no first-release guarantee is asserted. When a maturity has fewer than six eligible earlier results, exclude that auction from scoring and record it.

Add maturity-specific mean and median of the previous six eligible bid-to-cover outcomes, previous-one value, previous-six mean announced offering size, and current/previous-six offering ratio. No future interpolation, centered window, full-history scaling or test-derived cutoff. Fit preprocessing on fitting rows only.

Source records are federal aggregate auction statistics, with no participant records. Do not redistribute the raw API response while a dataset-specific license statement is unresolved; retain it locally with exact hashes and provide reproducible official retrieval. Publish original code and derived analysis only. Do not copy source papers or unpublished manuscript drafts into GitHub.

## Search, selection and uncertainty

Fourteen candidates: three declared baselines (previous-six maturity median, previous-six maturity mean, previous-one result); four standardized ridge models(alpha0.1,1,10,100); three random forests(200trees,max_depth4/8/unlimited,minleaf10,sqrtfeatures,bootstrap,two workers); four histogram-boosting models(depth2/3×learning rate0.05/0.1,100iterations,minleaf20,L2=1,early stoppingoff). Seed20261028. No pretrained fine-tuning. Ridge uses SVD for a transparent deterministic solve; scale and center using fitting rows only. Tree features remain unscaled.

Select by unweighted auction-level MSE on2020–2022 seen maturities, then name. Retain every candidate's parameters, status, runtime and per-auction validation predictions. No test-driven reoptimization. Commit selected model and input/cohort hashes before protected test evaluation. Baselines remain individually visible even if one wins.

Report MSE, RMSE and MAE by test cohort and original term, and descriptive error by offering ratio<0.8,0.8–1.2,>1.2. These ratio bands are fixed engineering strata, not definitions of economic stress. Compare selected-minus-six-median squared errors with2,000 calendar-quarter block bootstrap draws; intervals are conditional on observed quarters and not independent-auction or causal guarantees. If fewer than eight observed test quarters, report insufficient blocks rather than an interval.

## Independent checks and release

Independently reconstruct all lag-feature source dates and values, targets, splits, every candidate's validation MSE and selection, test MSE/MAE/strata and quarter-block intervals. Test same-day and future-result exclusion, current-result-field exclusion, maturity partitioning, source-key uniqueness and train-only standardization. Inspect decision figures and private LaTeX/PDF. A full clean replay must preserve selection and required metrics at1e-7absolute/1e-9relative tolerance before a release is complete. Keep blocked source, portability or evidence gates visible; do not count configurations or protocols as completed studies.
