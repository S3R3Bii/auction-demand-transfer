from pathlib import Path
from datetime import date, timedelta
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare import build, FEATURES


def fixture(term="2-Year"):
    result = []
    for i in range(12):
        d = date(2018, 1, 15) + timedelta(days=31 * i)
        result.append(
            dict(
                cusip=f"TEST{i:05}",
                auction_date=d.isoformat(),
                announcemt_date=(d - timedelta(days=7)).isoformat(),
                issue_date=(d + timedelta(days=3)).isoformat(),
                maturity_date=(d + timedelta(days=733)).isoformat(),
                security_type="Note",
                original_security_term=term,
                floating_rate="No",
                inflation_index_security="No",
                bid_to_cover_ratio=str(2 + i / 100),
                offering_amt="20000000000",
                reopening="No",
            )
        )
    return result


class Contracts(unittest.TestCase):
    def test_same_day_result_is_not_available(self):
        rows = fixture()
        target = rows[-1]
        extra = dict(
            target,
            cusip="SAME-DAY",
            auction_date=target["announcemt_date"],
            bid_to_cover_ratio="999",
        )
        a, _, _ = build(rows)
        b, _, _ = build(rows + [extra])
        key = target["cusip"] + "_" + target["auction_date"]
        np.testing.assert_array_equal(
            a[a.key == key][FEATURES].to_numpy(), b[b.key == key][FEATURES].to_numpy()
        )

    def test_current_and_future_results_cannot_change_current_features(self):
        rows = fixture()
        changed = [dict(r) for r in rows]
        changed[-1]["bid_to_cover_ratio"] = "900"
        changed[-1]["total_accepted"] = "99999999999999"
        a, _, _ = build(rows)
        b, _, _ = build(changed)
        np.testing.assert_array_equal(a[FEATURES].to_numpy(), b[FEATURES].to_numpy())

    def test_lag_history_requires_six_results(self):
        frame, lineage, excluded = build(fixture())
        self.assertEqual(len(frame), 6)
        self.assertEqual(len(lineage), 36)
        self.assertEqual(len(excluded), 6)

    def test_ten_year_history_does_not_enter_fitting(self):
        frame, _, _ = build(fixture("10-Year"))
        self.assertTrue((frame.cohort == "held_maturity_history_only").all())

    def test_soma_and_bidder_outcomes_are_not_features(self):
        self.assertFalse(
            set(FEATURES)
            & {
                "target",
                "bid_to_cover_ratio",
                "total_tendered",
                "total_accepted",
                "soma_accepted",
                "high_yield",
                "indirect_bidder_accepted",
            }
        )


if __name__ == "__main__":
    unittest.main()
