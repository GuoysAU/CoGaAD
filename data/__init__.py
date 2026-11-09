"""The series of the paper: one folder per dataset under data/ (see README.md; ``python data/fetch.py`` downloads them).

``load(name)`` returns the values and labels of a dataset; ``DATASETS`` maps each name to its file.
"""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

DATASETS = {
    "MBA805":   "MBA_ECG805_data_12.out",
    "MBA14046": "MBA_ECG14046_data_1.out",
    "MSL":      "T-9.test.out",
    "SensorSc": "stb-19.test.out",
    "IOPS":     "KPI-4d2af31a-9916-3d9f-8a8e-8a268a48c095.train.out",
}
PAPER = ["MBA805", "MBA14046", "MSL", "SensorSc", "IOPS"]   # the five datasets of the paper, in the order of its tables


def load(name):
    """Values and labels of one dataset."""
    frame = pd.read_csv(HERE / name / DATASETS[name], header=None).dropna()
    return frame.iloc[:, 0].to_numpy().astype(float), frame.iloc[:, 1].to_numpy().astype(int)


def raw_size(values):
    """Size in bytes of the series itself: its values as 32-bit floats after gzip compression."""
    return len(gzip.compress(np.asarray(values, dtype="<f4").tobytes(), compresslevel=9))
