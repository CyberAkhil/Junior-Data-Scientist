from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = PROJECT_DIR / "data" / "food_inspection_dataset.zip"
LABEL_COLUMN = "Inspection_Results"
INSPECTION_CATEGORICAL_COLUMNS = ("Type", "Reason", "RiskLevel")
INSPECTION_DATE_FORMAT = "%d-%m-%Y"


def read_workbooks(archive_path: Path):
    with ZipFile(archive_path) as archive:
        train = pd.read_excel(archive.open("Data_Train.xlsx"), engine="openpyxl")
        test = pd.read_excel(archive.open("Data_Test.xlsx"), engine="openpyxl")
        sample = pd.read_excel(
            archive.open("Sample_Submission.xlsx"), engine="openpyxl"
        )

    if LABEL_COLUMN not in train.columns:
        raise ValueError(f"Data_Train.xlsx is missing {LABEL_COLUMN!r}")
    if LABEL_COLUMN in test.columns:
        raise ValueError(f"Data_Test.xlsx must not contain {LABEL_COLUMN!r}")
    if len(test) != len(sample):
        raise ValueError("Test and sample-submission row counts do not match")
    if set(train.columns) - {LABEL_COLUMN} != set(test.columns):
        raise ValueError("Train and test predictor columns do not match")
    return train, test


def normalize_category(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    cleaned = cleaned.mask(cleaned.eq(""))
    return cleaned.str.upper()


def profile_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        values = frame[column]
        rows.append(
            {
                "dataset": name,
                "column": column,
                "dtype": str(values.dtype),
                "rows": len(values),
                "missing_count": int(values.isna().sum()),
                "missing_percent": (
                    round(float(values.isna().mean() * 100), 4)
                    if len(values)
                    else 0.0
                ),
                "unique_non_missing": int(values.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def clean_inspection_frame(
    source: pd.DataFrame, dataset_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = source.copy()
    cleaned.insert(0, "_source_row", np.arange(2, len(cleaned) + 2))
    issue_flags: dict[str, pd.Series] = {}

    if "Date" in cleaned.columns:
        parsed_dates = pd.to_datetime(
            cleaned["Date"],
            format=INSPECTION_DATE_FORMAT,
            errors="coerce",
        )
        cleaned["Date_parsed"] = parsed_dates
        invalid_date = cleaned["Date"].notna() & parsed_dates.isna()
        issue_flags["invalid_date"] = invalid_date

    for column in INSPECTION_CATEGORICAL_COLUMNS:
        if column in cleaned.columns:
            normalized = normalize_category(cleaned[column])
            cleaned[f"{column}_clean"] = normalized
            issue_flags[f"missing_{column.lower()}"] = normalized.isna()

    duplicate_id = pd.Series(False, index=cleaned.index)
    if "ID" in cleaned.columns:
        duplicate_id = cleaned["ID"].duplicated(keep=False)
        issue_flags["duplicate_inspection_id"] = duplicate_id

    duplicate_row = source.duplicated(keep=False)
    issue_flags["exact_duplicate_row"] = duplicate_row

    flag_names = list(issue_flags)
    cleaned["quality_flags"] = [
        "|".join(name for name, flags in issue_flags.items() if bool(flags.iloc[index]))
        for index in range(len(cleaned))
    ]
    cleaned["duplicate_inspection_id"] = duplicate_id.to_numpy()
    cleaned["exact_duplicate_row"] = duplicate_row.to_numpy()
    cleaned["dataset_split"] = dataset_name

    has_issue = cleaned["quality_flags"].ne("")
    exceptions = cleaned.loc[has_issue].copy()
    return cleaned, exceptions


def run_inspection_cleaning(archive_path: Path, output_dir: Path) -> None:
    train_raw, test_raw = read_workbooks(archive_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = pd.concat(
        [
            profile_frame(train_raw, "train"),
            profile_frame(test_raw, "test"),
        ],
        ignore_index=True,
    )

    cleaned_frames = {}
    exception_frames = []
    for name, raw_frame in (("train", train_raw), ("test", test_raw)):
        cleaned, exceptions = clean_inspection_frame(raw_frame, name)
        cleaned_frames[name] = cleaned
        if not exceptions.empty:
            exception_frames.append(exceptions)
        cleaned.to_csv(output_dir / f"inspection_{name}_clean.csv", index=False)

    profile.to_csv(output_dir / "inspection_quality_profile.csv", index=False)
    exceptions = (
        pd.concat(exception_frames, ignore_index=True, sort=False)
        if exception_frames
        else pd.DataFrame()
    )
    exceptions.to_csv(output_dir / "inspection_exceptions.csv", index=False)

    summary = {
        "source_archive": str(archive_path.resolve()),
        "output_directory": str(output_dir.resolve()),
        "cleaning_policy": (
            "Source values are preserved. Missing SectionViolations are not set to "
            "zero. Invalid dates and duplicate records are flagged, not dropped. "
            "Category text is trimmed, internal whitespace collapsed, and case "
            "standardized in separate *_clean columns."
        ),
        "train_rows": len(train_raw),
        "test_rows": len(test_raw),
        "train_profile_missing": train_raw.isna().sum().astype(int).to_dict(),
        "test_profile_missing": test_raw.isna().sum().astype(int).to_dict(),
        "train_duplicate_ids": (
            int(train_raw["ID"].duplicated(keep=False).sum())
            if "ID" in train_raw.columns
            else None
        ),
        "test_duplicate_ids": (
            int(test_raw["ID"].duplicated(keep=False).sum())
            if "ID" in test_raw.columns
            else None
        ),
        "train_exact_duplicate_rows": int(train_raw.duplicated(keep=False).sum()),
        "test_exact_duplicate_rows": int(test_raw.duplicated(keep=False).sum()),
        "train_invalid_dates": int(
            cleaned_frames["train"].get(
                "quality_flags", pd.Series(dtype="string")
            ).str.contains("invalid_date", regex=False).sum()
        ),
        "test_invalid_dates": int(
            cleaned_frames["test"].get(
                "quality_flags", pd.Series(dtype="string")
            ).str.contains("invalid_date", regex=False).sum()
        ),
        "flagged_exception_rows": len(exceptions),
        "rows_removed": 0,
    }
    (output_dir / "inspection_preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, default=str))
    print(f"Output written to: {output_dir.resolve()}")


def robust_outlier_mask(
    values: pd.Series,
    group_keys: pd.DataFrame | None = None,
    threshold: float = 6.0,
) -> pd.Series:
    if threshold <= 0:
        raise ValueError("MAD threshold must be greater than zero")

    numeric = pd.to_numeric(values, errors="coerce")
    mask = pd.Series(False, index=values.index)
    if group_keys is None or group_keys.empty:
        groups = [(None, numeric.index)]
    else:
        groups = (
            (key, group.index)
            for key, group in group_keys.groupby(
                list(group_keys.columns),
                dropna=False,
                sort=False,
            )
        )

    for _, indices in groups:
        group_values = numeric.loc[indices].dropna()
        if len(group_values) < 5:
            continue
        median = group_values.median()
        mad = (group_values - median).abs().median()
        if not np.isfinite(mad) or mad == 0:
            continue
        modified_z = 0.67448975 * (group_values - median).abs() / mad
        mask.loc[group_values.index] = modified_z > threshold
    return mask


def read_limits(path: Path | None) -> dict[str, tuple[float, float]]:
    if path is None:
        return {}
    parsed = json.loads(path.read_text(encoding="utf-8"))
    limits: dict[str, tuple[float, float]] = {}
    for column, bounds in parsed.items():
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or not all(isinstance(value, (int, float)) for value in bounds)
            or bounds[0] >= bounds[1]
        ):
            raise ValueError(
                f"Limits for {column!r} must be [minimum, maximum] numbers"
            )
        limits[column] = (float(bounds[0]), float(bounds[1]))
    return limits


def run_sensor_cleaning(
    input_path: Path,
    output_dir: Path,
    timestamp_column: str,
    sensor_columns: list[str],
    group_columns: list[str],
    limits: dict[str, tuple[float, float]],
    mad_threshold: float,
) -> None:
    if input_path.resolve() == (output_dir / "sensor_clean.csv").resolve():
        raise ValueError("Output path must not overwrite the input file")

    source = pd.read_csv(input_path)
    required = {timestamp_column, *sensor_columns, *group_columns}
    missing_columns = sorted(required - set(source.columns))
    if missing_columns:
        raise ValueError(f"Sensor CSV is missing columns: {missing_columns}")

    unknown_limits = sorted(set(limits) - set(sensor_columns))
    if unknown_limits:
        raise ValueError(
            "Every limits.json column must be listed in --sensors: "
            f"{unknown_limits}"
        )

    cleaned = source.copy()
    cleaned.insert(0, "_source_row", np.arange(2, len(source) + 2))
    flags: dict[str, pd.Series] = {}
    raw_timestamps = source[timestamp_column]
    parsed_timestamps = pd.to_datetime(
        raw_timestamps,
        errors="coerce",
        utc=True,
        format="mixed",
    )
    cleaned[f"{timestamp_column}_utc"] = parsed_timestamps
    flags["invalid_timestamp"] = raw_timestamps.notna() & parsed_timestamps.isna()

    duplicate_key_columns = group_columns + [f"{timestamp_column}_utc"]
    if duplicate_key_columns:
        duplicate_timestamps = cleaned.duplicated(
            subset=duplicate_key_columns,
            keep=False,
        ) & parsed_timestamps.notna()
        flags["duplicate_group_timestamp"] = duplicate_timestamps
    flags["exact_duplicate_row"] = source.duplicated(keep=False)

    quality_rows = []
    group_keys = source[group_columns] if group_columns else None
    for column in sensor_columns:
        numeric = pd.to_numeric(source[column], errors="coerce")
        invalid_numeric = source[column].notna() & numeric.isna()
        missing = numeric.isna()
        outlier = robust_outlier_mask(
            numeric,
            group_keys=group_keys,
            threshold=mad_threshold,
        )
        flags[f"invalid_numeric_{column}"] = invalid_numeric
        flags[f"missing_{column}"] = missing
        flags[f"statistical_outlier_{column}"] = outlier
        cleaned[f"{column}_numeric"] = numeric
        cleaned[f"{column}_missing"] = missing
        cleaned[f"{column}_statistical_outlier"] = outlier

        outside_limits = pd.Series(False, index=source.index)
        if column in limits:
            lower, upper = limits[column]
            outside_limits = numeric.notna() & (
                (numeric < lower) | (numeric > upper)
            )
            cleaned[f"{column}_outside_engineering_limits"] = outside_limits
            flags[f"outside_engineering_limits_{column}"] = outside_limits

        quality_rows.append(
            {
                "column": column,
                "rows": len(source),
                "missing_count": int(missing.sum()),
                "invalid_numeric_count": int(invalid_numeric.sum()),
                "statistical_outlier_count": int(outlier.sum()),
                "engineering_limit_flag_count": int(outside_limits.sum()),
                "minimum": numeric.min(),
                "median": numeric.median(),
                "maximum": numeric.max(),
                "limit_minimum": limits.get(column, (None, None))[0],
                "limit_maximum": limits.get(column, (None, None))[1],
            }
        )

    timestamp_order = parsed_timestamps.sort_values(kind="stable").index
    cleaned = cleaned.loc[timestamp_order].copy()
    for name, mask in flags.items():
        cleaned[name] = mask.loc[timestamp_order].to_numpy()
    cleaned["quality_flags"] = [
        "|".join(name for name, mask in flags.items() if bool(mask.loc[index]))
        for index in timestamp_order
    ]

    profile_rows = [
        {
            "column": column,
            "dtype": str(source[column].dtype),
            "rows": len(source),
            "missing_count": int(source[column].isna().sum()),
            "unique_non_missing": int(source[column].nunique(dropna=True)),
        }
        for column in source.columns
    ]
    timestamp_gaps = parsed_timestamps.sort_values().diff().dropna()
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_dir / "sensor_clean.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(
        output_dir / "sensor_reading_quality_profile.csv",
        index=False,
    )
    pd.DataFrame(profile_rows).to_csv(
        output_dir / "sensor_source_profile.csv",
        index=False,
    )
    exceptions = cleaned.loc[cleaned["quality_flags"].ne("")]
    exceptions.to_csv(output_dir / "sensor_exceptions.csv", index=False)

    summary = {
        "source_file": str(input_path.resolve()),
        "output_directory": str(output_dir.resolve()),
        "rows_read": len(source),
        "rows_written": len(cleaned),
        "rows_removed": 0,
        "timestamp_column": timestamp_column,
        "sensor_columns": sensor_columns,
        "group_columns": group_columns,
        "timezone_assumption": (
            "Parsed with utc=True; offset-free source timestamps are interpreted "
            "as UTC and must be converted only if their source timezone is known."
        ),
        "outliers": (
            "MAD-screened observations are flagged, not deleted or capped. "
            "Groups smaller than five observations and groups with zero MAD "
            "are not statistically flagged."
        ),
        "missing_values": "Retained as missing; no readings are imputed.",
        "timestamp_gaps": {
            "gap_count": int(len(timestamp_gaps)),
            "minimum_seconds": (
                float(timestamp_gaps.dt.total_seconds().min())
                if len(timestamp_gaps)
                else None
            ),
            "median_seconds": (
                float(timestamp_gaps.dt.total_seconds().median())
                if len(timestamp_gaps)
                else None
            ),
            "maximum_seconds": (
                float(timestamp_gaps.dt.total_seconds().max())
                if len(timestamp_gaps)
                else None
            ),
        },
        "flagged_exception_rows": len(exceptions),
        "rows": len(quality_rows),
    }
    (output_dir / "sensor_preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))
    print(f"Output written to: {output_dir.resolve()}")


def make_model_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scaling: str = "robust",
) -> ColumnTransformer:
    """Build fold-fitted preprocessing; fit only after splitting training data."""
    scalers = {
        "standard": StandardScaler,
        "robust": RobustScaler,
        "none": None,
    }
    if scaling not in scalers:
        raise ValueError("scaling must be standard, robust, or none")

    numeric_steps = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True))
    ]
    if scalers[scaling] is not None:
        numeric_steps.append(("scale", scalers[scaling]()))

    transformers = [
        (
            "numeric",
            Pipeline(steps=numeric_steps),
            numeric_features,
        )
    ]
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "impute",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="Missing",
                            ),
                        ),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=10,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean the supplied inspection data or profile a sensor CSV."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspection = commands.add_parser(
        "inspection",
        help="Profile and prepare the inspection workbooks in the supplied ZIP.",
    )
    inspection.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    inspection.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "data" / "processed" / "inspection",
    )

    sensor = commands.add_parser(
        "sensor",
        help="Profile sensor readings in a wide-format CSV without imputing values.",
    )
    sensor.add_argument("input", type=Path, help="Input sensor CSV file.")
    sensor.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "data" / "processed" / "sensor",
    )
    sensor.add_argument("--timestamp-col", required=True)
    sensor.add_argument(
        "--sensors",
        required=True,
        help="Comma-separated numeric sensor columns.",
    )
    sensor.add_argument(
        "--group-cols",
        default="",
        help="Optional comma-separated line, sensor, product, or process-state columns.",
    )
    sensor.add_argument(
        "--limits-json",
        type=Path,
        help="Optional JSON object of approved engineering ranges by sensor.",
    )
    sensor.add_argument("--mad-threshold", type=float, default=6.0)

    args = parser.parse_args()
    if args.command == "inspection":
        if not args.archive.is_file():
            raise FileNotFoundError(f"Dataset archive not found: {args.archive}")
        run_inspection_cleaning(args.archive, args.output_dir)
    else:
        if not args.input.is_file():
            raise FileNotFoundError(f"Sensor CSV not found: {args.input}")
        sensors = [item.strip() for item in args.sensors.split(",") if item.strip()]
        groups = [item.strip() for item in args.group_cols.split(",") if item.strip()]
        if not sensors:
            raise ValueError("Provide at least one sensor column with --sensors")
        run_sensor_cleaning(
            args.input,
            args.output_dir,
            args.timestamp_col,
            sensors,
            groups,
            read_limits(args.limits_json),
            args.mad_threshold,
        )


if __name__ == "__main__":
    main()
