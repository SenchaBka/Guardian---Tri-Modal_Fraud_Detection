import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from NPL.training.utils import (
    LABEL_COLUMN,
    TEXT_COLUMN,
    TRANSACTION_ID_COLUMN,
    SplitConfig,
    clean_dataset,
    compute_metrics,
    find_best_threshold,
    format_metrics_block,
    load_dataset,
    save_dataframe,
    save_metrics,
    save_model,
    save_text_report,
    stratified_split,
    stratified_split_dataframe,
)


class SplitConfigTests(unittest.TestCase):
    def test_validate_accepts_default_ratios(self):
        SplitConfig().validate()

    def test_validate_rejects_invalid_total(self):
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            SplitConfig(train_size=0.6, val_size=0.3, test_size=0.3).validate()

    def test_validate_rejects_zero_or_out_of_range_values(self):
        with self.assertRaisesRegex(ValueError, "must be between 0 and 1"):
            SplitConfig(train_size=0.0, val_size=0.5, test_size=0.5).validate()


class DatasetTests(unittest.TestCase):
    def test_load_dataset_reads_csv_with_optional_nrows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.csv"
            path.write_text("text,label\nA,0\nB,1\n", encoding="utf-8")

            df = load_dataset(path, nrows=1)

        self.assertEqual(len(df), 1)
        self.assertEqual(list(df.columns), ["text", "label"])

    def test_clean_dataset_keeps_required_columns_and_normalizes_values(self):
        df = pd.DataFrame(
            {
                TEXT_COLUMN: ["  hello  ", "world", None, ""],
                LABEL_COLUMN: [0, 1, 1, 0],
                TRANSACTION_ID_COLUMN: [101, 102, 103, 104],
            }
        )

        cleaned = clean_dataset(df)

        self.assertEqual(cleaned[TEXT_COLUMN].tolist(), ["hello", "world"])
        self.assertEqual(cleaned[LABEL_COLUMN].tolist(), [0, 1])
        self.assertEqual(cleaned[TRANSACTION_ID_COLUMN].tolist(), ["101", "102"])

    def test_clean_dataset_raises_for_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            clean_dataset(pd.DataFrame({TEXT_COLUMN: ["hello"]}))

    def test_clean_dataset_raises_if_only_one_label_class_remains(self):
        df = pd.DataFrame({TEXT_COLUMN: ["hello", "world"], LABEL_COLUMN: [1, 1]})

        with self.assertRaisesRegex(ValueError, "at least two label classes"):
            clean_dataset(df)


class SplitTests(unittest.TestCase):
    def test_stratified_split_returns_expected_partition_sizes(self):
        X = pd.Series([f"text-{i}" for i in range(20)])
        y = pd.Series([0, 1] * 10)

        X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)

        self.assertEqual((len(X_train), len(X_val), len(X_test)), (14, 3, 3))
        self.assertEqual((len(y_train), len(y_val), len(y_test)), (14, 3, 3))
        self.assertEqual(y_train.sum(), 7)

    def test_stratified_split_dataframe_preserves_all_rows(self):
        df = pd.DataFrame(
            {
                TEXT_COLUMN: [f"text-{i}" for i in range(20)],
                LABEL_COLUMN: [0, 1] * 10,
            }
        )

        train_df, val_df, test_df = stratified_split_dataframe(df)

        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(df))
        self.assertEqual(train_df[LABEL_COLUMN].sum(), 7)


class MetricsTests(unittest.TestCase):
    def test_find_best_threshold_returns_default_when_no_thresholds_exist(self):
        with mock.patch(
            "NPL.training.utils.precision_recall_curve",
            return_value=([1.0], [1.0], []),
        ):
            threshold, f1 = find_best_threshold(pd.Series([1]), [0.9])

        self.assertEqual(threshold, 0.5)
        self.assertEqual(f1, 0.0)

    def test_find_best_threshold_selects_highest_f1(self):
        with mock.patch(
            "NPL.training.utils.precision_recall_curve",
            return_value=([1.0, 0.5, 0.1], [0.25, 1.0, 1.0], [0.2, 0.8]),
        ):
            threshold, f1 = find_best_threshold(pd.Series([0, 1]), [0.2, 0.8])

        self.assertEqual(threshold, 0.8)
        self.assertAlmostEqual(f1, 2 * 0.5 * 1.0 / 1.5)

    def test_compute_metrics_returns_expected_values(self):
        metrics = compute_metrics(
            y_true=[0, 1, 1, 0],
            y_pred=[0, 1, 0, 0],
            y_prob=[0.1, 0.9, 0.4, 0.2],
        )

        self.assertEqual(metrics["support"], 4)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 0.5)

    def test_format_metrics_block_renders_report_lines(self):
        metrics = {
            "precision": 1.0,
            "recall": 0.5,
            "f1": 0.6667,
            "roc_auc": 0.9,
            "pr_auc": 0.8,
            "support": 4,
            "fraud_rate": 0.25,
            "confusion_matrix": [[2, 0], [1, 1]],
        }

        block = format_metrics_block("Validation", metrics)

        self.assertIn("Validation", block)
        self.assertIn("precision: 1.0000", block)
        self.assertIn("support:   4", block)


class PersistenceTests(unittest.TestCase):
    def test_save_metrics_creates_parent_directory(self):
        metrics = {"precision": 1.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reports" / "metrics.json"
            save_metrics(metrics, path)

            written = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(written, metrics)

    def test_save_text_report_writes_contents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reports" / "summary.txt"
            save_text_report("hello", path)

            self.assertEqual(path.read_text(encoding="utf-8"), "hello")

    def test_save_model_delegates_to_joblib_dump(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "models" / "artifact.joblib"
            with mock.patch("NPL.training.utils.joblib.dump") as dump_mock:
                save_model({"model": "x"}, path)

        dump_mock.assert_called_once()
        self.assertEqual(dump_mock.call_args[0][1], path)

    def test_save_dataframe_writes_csv(self):
        df = pd.DataFrame({TEXT_COLUMN: ["a"], LABEL_COLUMN: [1]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "exports" / "data.csv"
            save_dataframe(df, path)

            written = pd.read_csv(path)

        self.assertEqual(written.to_dict(orient="records"), df.to_dict(orient="records"))
