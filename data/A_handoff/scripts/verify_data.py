"""Boundary tests plus independent checks of actual output against original BTS rows."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
import zipfile
from datetime import datetime, timezone
import pandas as pd

from prepare_data import derive, hhmm_to_minutes, assign_split

ROOT = Path(__file__).resolve().parents[1]


class BoundaryTests(unittest.TestCase):
    def test_hhmm_boundaries(self):
        values = pd.Series([0, 5, 930, 2359, 2400, 2360, 1261, -1, 2500, 930.5, None])
        expected = pd.Series([0, 5, 570, 1439, 0, None, None, None, None, None, None], dtype="Int64")
        pd.testing.assert_series_equal(hhmm_to_minutes(values), expected)

    def fixture(self):
        return pd.DataFrame({"FlightDate": ["2025-01-01"] * 5, "Year": [2025]*5,
            "Month": [1]*5, "DayofMonth": [1]*5, "DayOfWeek": [3]*5,
            "Reporting_Airline": ["AA"]*5, "Origin": ["DFW"]*5, "Dest": ["ATL"]*5,
            "Flight_Number_Reporting_Airline": ["1","2","3","4","5"],
            "CRSDepTime": [930]*5, "DepTime": [925,944,945,946,None],
            "DepDelay": [-5,14,15,16,None], "DepDel15": [0,0,1,1,None],
            "Cancelled": [0,0,0,0,1], "Diverted": [0]*5,
            "Distance": [700]*5, "CRSElapsedTime": [120]*5})

    def test_label_threshold_and_cancelled(self):
        d = derive(self.fixture())
        self.assertEqual(d.TargetDelayed15.iloc[:4].tolist(), [0,0,1,1])
        self.assertTrue(pd.isna(d.TargetDelayed15.iloc[4]))

    def test_label_contradiction_fails(self):
        d = self.fixture()
        d.loc[2, "DepDel15"] = 0
        with self.assertRaisesRegex(ValueError, "disagrees"):
            derive(d)

    def test_calendar_contradiction_fails(self):
        d = self.fixture()
        d.loc[0, "DayOfWeek"] = 4
        with self.assertRaisesRegex(ValueError, "calendar"):
            derive(d)

    def test_split_boundaries(self):
        dates = pd.Series(pd.to_datetime(["2025-02-14", "2025-02-15", "2025-02-28", "2025-03-01"]))
        self.assertEqual(assign_split(dates, {"train_end":"2025-02-14", "validation_end":"2025-02-28"}).tolist(),
                         ["train","validation","validation","test"])


class ActualDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = ROOT / "data/processed"
        cls.d = pd.read_csv(cls.out / "cleaned_flights.csv", dtype={"FlightNumber":"string"})
        cls.model = pd.read_csv(cls.out / "model_data.csv")
        cls.manifest = json.loads((ROOT / "reports/manifest.json").read_text())

    def test_source_reconciliation_all_selected_records(self):
        # Independently reload original archive rows and compare every retained source field.
        numeric = ["Year","Month","DayofMonth","DayOfWeek","CRSDepTime","DepTime","DepDelay",
                   "DepDel15","Cancelled","Diverted","Distance","CRSElapsedTime"]
        rename = {"Reporting_Airline":"Airline", "Flight_Number_Reporting_Airline":"FlightNumber"}
        columns = numeric + ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest"]
        for source in self.manifest["sources"]:
            name = source["url"].rsplit("/",1)[-1]
            with zipfile.ZipFile(ROOT / "data/raw" / name) as z:
                with z.open(source["csv_members"][0]) as f:
                    raw = pd.read_csv(f, usecols=columns, dtype="string")
            chosen = raw.loc[raw.Origin.isin(self.manifest["config"]["origins"])].copy()
            self.assertEqual(len(raw), source["total_rows"])
            self.assertEqual(len(chosen), source["selected_rows"])
            actual = self.d.loc[self.d.SourceArchive.eq(name)].set_index("SourceRow").sort_index()
            chosen.index = chosen.index + 1
            chosen.index.name = "SourceRow"
            chosen = chosen.rename(columns=rename).sort_index()
            self.assertEqual(len(actual), len(chosen))
            for field in chosen:
                left, right = actual[field].copy(), chosen[field].copy()
                if field in numeric:
                    left, right = pd.to_numeric(left).astype("Float64"), pd.to_numeric(right).astype("Float64")
                else:
                    left, right = left.astype("string"), right.astype("string")
                pd.testing.assert_series_equal(left, right, check_dtype=False)

    def test_labels_cancelled_and_diversion(self):
        d = self.d
        self.assertTrue(d.loc[d.Cancelled.eq(1), "TargetDelayed15"].isna().all())
        labelled = d.TargetDelayed15.notna()
        self.assertTrue(d.loc[labelled, "TargetDelayed15"].eq(d.loc[labelled, "DepDelay"].ge(15).astype(int)).all())
        self.assertTrue(d.loc[d.Cancelled.eq(1), "ModelEligible"].eq(False).all())
        # Valid diverted departures remain eligible; diversion itself is not an exclusion.
        diverted = d.Diverted.eq(1) & d.Cancelled.eq(0) & d.TargetDelayed15.notna()
        self.assertTrue(d.loc[diverted, "ModelEligible"].all())

    def test_scope_ids_and_chronology(self):
        d = self.d
        self.assertTrue(d.record_id.is_unique)
        self.assertEqual(set(d.Origin), set(self.manifest["config"]["origins"]))
        self.assertTrue(pd.to_datetime(d.FlightDate).between("2025-01-01", "2025-03-31").all())
        m = self.model
        self.assertLess(m.loc[m.split.eq("train"),"FlightDate"].max(), m.loc[m.split.eq("validation"),"FlightDate"].min())
        self.assertLess(m.loc[m.split.eq("validation"),"FlightDate"].max(), m.loc[m.split.eq("test"),"FlightDate"].min())
        self.assertTrue(m.groupby("FlightDate").split.nunique().eq(1).all())

    def test_model_inputs_and_row_alignment(self):
        expected = ["Airline","Origin","Dest","DayOfWeek","ScheduledHour","ScheduledMinuteOfDay","IsWeekend","Distance","CRSElapsedTime"]
        combined = []
        for split in ("train","validation","test"):
            x = pd.read_csv(self.out / f"X_{split}.csv")
            y = pd.read_csv(self.out / f"y_{split}.csv")
            meta = pd.read_csv(self.out / f"metadata_{split}.csv")
            part = self.model.loc[self.model.split.eq(split)].reset_index(drop=True)
            self.assertEqual(x.columns.tolist(), expected)
            self.assertEqual(len(x), len(y))
            self.assertEqual(len(y), len(meta))
            self.assertFalse(x.isna().any().any())
            pd.testing.assert_frame_equal(x, part[expected])
            pd.testing.assert_series_equal(y.TargetDelayed15, part.TargetDelayed15)
            pd.testing.assert_series_equal(meta.record_id, part.record_id)
            combined.extend(meta.record_id.tolist())
        self.assertEqual(len(combined), len(set(combined)))
        self.assertEqual(set(combined), set(self.d.loc[self.d.ModelEligible,"record_id"]))

    def test_audit_and_hashes(self):
        audit = pd.read_csv(ROOT / "reports/cleaning_audit.csv")
        self.assertTrue((audit.before-audit.excluded).eq(audit.after).all())
        self.assertEqual(audit.after.iloc[-1], len(self.model))
        for relative, expected in self.manifest["output_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "tests_run":result.testsRun,
              "failures":len(result.failures), "errors":len(result.errors), "successful": result.wasSuccessful(),
              "details":[{"test": str(t), "traceback": detail} for t,detail in result.failures + result.errors]}
    (ROOT / "reports/verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    sys.exit(0 if result.wasSuccessful() else 1)
