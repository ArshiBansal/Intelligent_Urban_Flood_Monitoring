from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

PRECIP_FORECASTS = [
    "fc_precipitation_gfs_seamless",
    "fc_precipitation_ecmwf_ifs025",
    "fc_precipitation_icon_seamless",
    "fc_precipitation_gem_seamless",
]
BASE_COLUMNS = [
    "loc_id",
    "lat",
    "lon",
    "elevation_m",
    "admin1",
    "valid_time",
    "lead_hours",
    "truth_precipitation",
    *PRECIP_FORECASTS,
]


def list_parquet_files(data_dir: str | Path) -> list[Path]:
    files = sorted(Path(data_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {data_dir}")
    return files


def load_rows(
    data_dir: str | Path,
    columns: Iterable[str] = BASE_COLUMNS,
    max_rows: int | None = None,
    seed: int = 26085,
) -> pd.DataFrame:
    """Read selected columns, sampling across all files deterministically so all locations are represented."""
    files = list_parquet_files(data_dir)
    cols = list(columns)
    frames: list[pd.DataFrame] = []
    
    if max_rows is not None and max_rows > 0:
        per_file = max(10, max_rows // len(files))
        for path in files:
            # Read parquet efficiently with PyArrow
            table = pq.read_table(path, columns=cols)
            frame = table.to_pandas()
            if len(frame) > per_file:
                frame = frame.sample(per_file, random_state=seed)
            frames.append(frame)
    else:
        for path in files:
            table = pq.read_table(path, columns=cols)
            frames.append(table.to_pandas())

    if not frames:
        raise ValueError("The requested row limit produced no rows")
    result = pd.concat(frames, ignore_index=True)
    result["valid_time"] = pd.to_datetime(result["valid_time"], utc=True)
    return result


def validate_precipitation_frame(frame: pd.DataFrame) -> None:
    """Fail clearly on invalid rows before a model or demonstration uses them."""
    required = set(BASE_COLUMNS)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required precipitation columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Precipitation frame is empty")
    if frame["truth_precipitation"].isna().any():
        raise ValueError("Precipitation target contains missing values")
    numeric = frame[["truth_precipitation", *PRECIP_FORECASTS]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().all(axis=1).any():
        raise ValueError("Precipitation row has no usable forecast values")
    if (numeric["truth_precipitation"] < 0).any():
        raise ValueError("Precipitation target contains negative values")
    if frame["lead_hours"].isna().any() or (frame["lead_hours"] < 0).any():
        raise ValueError("Lead hours must be present and non-negative")


def sample_frame(frame: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    """Sample only after a leakage-safe partition has already been defined."""
    if max_rows is None or len(frame) <= max_rows:
        return frame.copy()
    return frame.sample(max_rows, random_state=seed).sort_values(["valid_time", "loc_id", "lead_hours"]).reset_index(drop=True)
