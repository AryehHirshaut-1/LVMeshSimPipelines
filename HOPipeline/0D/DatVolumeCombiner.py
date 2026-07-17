"""
combine_dat_and_volume.py

Combines:
  - a .dat file containing time + pressure (with a header line to skip)
  - a .csv file containing volume (from your volume extractor)

into a single output file with columns:
  time    pressure    volume

Run this FIRST, then feed the resulting file into
build_calibration_json_from_combined.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path

folderpath = Path("/users/alanh/Documents/CBLResearch-github/HOPipeline/3D/VolumeResults810")
filecount = sum(1 for file in folderpath.iterdir() if file.is_file())


for file_num in range(0, filecount):
    # ============================================================
    # CONFIG -- edit these to match your actual files
    # ============================================================

    GeneralFile = "HOPipeline"

    # --- .dat file (time + pressure), has a header line ---
    DAT_PATH = f"{GeneralFile}/3D/lv_pressure.dat"

    # Number of header lines to skip at the top of the .dat file.
    DAT_HEADER_LINES = 1

    # Whitespace-delimited is the svMultiPhysics/svZeroDSolver default.
    # Set to "," if your .dat is actually comma-separated.
    DAT_DELIMITER = None  # None = any whitespace (spaces or tabs)

    # Column order in the .dat file (0-indexed), AFTER the header is skipped.
    DAT_TIME_COL = 0
    DAT_PRESSURE_COL = 1

    # --- .csv file (volume) from your volume extractor ---
    CSV_PATH = f"{GeneralFile}/3D/VolumeResults810/lv_volumes_{file_num}.csv"

    # If the CSV has a header row with column names, set this to the exact
    # column name. If there's no header, set to None and use CSV_VOLUME_COL_INDEX.
    CSV_HAS_HEADER = False
    #   # e.g. "volume" or "Volume (mL)"
    CSV_VOLUME_COL_INDEX = 0      # only used if CSV_VOLUME_COL_NAME is None

    # If the CSV also has its own time column, set this so volume can be
    # interpolated onto the .dat time grid (recommended -- the two files
    # are rarely sampled at exactly the same timepoints).
    CSV_TIME_COL_NAME = None         # e.g. "time"; set None if no time col
    CSV_TIME_COL_INDEX = 0           # only used if CSV_TIME_COL_NAME is None and CSV_HAS_HEADER is False
    INTERPOLATE_VOLUME_ONTO_DAT_TIME = False

    # --- Output ---
    OUTPUT_PATH = f"{GeneralFile}/0D/InputFiles/TPV_{file_num}.csv"

    # Whether to write a header line in the combined output file.
    WRITE_OUTPUT_HEADER = False

    # ============================================================
    # LOAD TIME + PRESSURE FROM .dat (header stripped)
    # ============================================================

    dat_data = np.loadtxt(DAT_PATH, delimiter=DAT_DELIMITER, skiprows=DAT_HEADER_LINES)

    time = dat_data[:, DAT_TIME_COL]
    pressure = dat_data[:, DAT_PRESSURE_COL]

    print(f"Loaded .dat file: {DAT_PATH} (skipped {DAT_HEADER_LINES} header line(s))")
    print(f"  N points   = {len(time)}")
    print(f"  time range = [{time[0]:.6f}, {time[-1]:.6f}]")
    print(f"  pressure range = [{pressure.min():.6e}, {pressure.max():.6e}]")
    print()

    # ============================================================
    # LOAD VOLUME FROM .csv
    # ============================================================

    csv_df = pd.read_csv(CSV_PATH)

    csv_df = pd.read_csv(CSV_PATH, header=0 if CSV_HAS_HEADER else None)

    volume_raw = csv_df.iloc[:, CSV_VOLUME_COL_INDEX].to_numpy(dtype=float)

    if CSV_TIME_COL_NAME is not None:
        csv_time_raw = csv_df[CSV_TIME_COL_NAME].to_numpy(dtype=float)
    elif not CSV_HAS_HEADER:
        csv_time_raw = csv_df.iloc[:, CSV_TIME_COL_INDEX].to_numpy(dtype=float)
    else:
        csv_time_raw = None

    print(f"Loaded .csv file: {CSV_PATH}")
    print(f"  N points     = {len(volume_raw)}")
    print(f"  volume range = [{volume_raw.min():.6e}, {volume_raw.max():.6e}]")
    print()

    # ============================================================
    # ALIGN VOLUME TO THE .dat TIME GRID
    # ============================================================

    if INTERPOLATE_VOLUME_ONTO_DAT_TIME:
        if csv_time_raw is None:
            raise ValueError(
                "INTERPOLATE_VOLUME_ONTO_DAT_TIME=True but no CSV time column "
                "was found. Set CSV_TIME_COL_NAME (or CSV_TIME_COL_INDEX with "
                "CSV_HAS_HEADER=False)."
            )
        volume = np.interp(time, csv_time_raw, volume_raw)
        print("Interpolated volume onto .dat time grid.")
    else:
        if len(volume_raw) != len(time):
            raise ValueError(
                f"Volume array length ({len(volume_raw)}) does not match "
                f".dat time array length ({len(time)}), and interpolation is "
                f"disabled. Either enable INTERPOLATE_VOLUME_ONTO_DAT_TIME, or "
                f"make sure the two files share the same time samples."
            )
        volume = volume_raw
        print("Used volume array directly (no interpolation).")

    print(f"  Final aligned N points = {len(time)}")
    print()

    # ============================================================
    # WRITE COMBINED FILE: time, pressure, volume
    # ============================================================

    combined = np.column_stack([time, pressure, volume])

    if WRITE_OUTPUT_HEADER:
        np.savetxt(
            OUTPUT_PATH,
            combined,
            fmt="%.10e",
            header="time pressure volume",
            comments="",  # no leading '#' on the header line
        )
    else:
        np.savetxt(OUTPUT_PATH, combined, fmt="%.10e", delimiter=",")

    print(f"Wrote combined file: {OUTPUT_PATH}")
    # print("Preview of first 3 rows (time, pressure, volume):")
    # for i in range(min(3, len(time))):
    #     print(f"  {time[i]:.6e}  {pressure[i]:.6e}  {volume[i]:.6e}")