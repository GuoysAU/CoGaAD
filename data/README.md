# Data

One folder per dataset, each with one series file of two columns (value, label) in the TSB-UAD format. `python data/fetch.py` downloads
them from the public TSB-UAD data archive.

| Folder | File | Source |
|---|---|---|
| `MBA805/` | `MBA_ECG805_data_12.out` | TSB-UAD, ECG collection (MIT-BIH Supraventricular Arrhythmia Database, record 805) |
| `MBA14046/` | `MBA_ECG14046_data_1.out` | TSB-UAD, ECG collection (record 14046, piece 1 of 47) |
| `MSL/` | `T-9.test.out` | TSB-UAD, NASA-MSL collection, channel T-9 |
| `SensorSc/` | `stb-19.test.out` | TSB-UAD, SensorScope collection |
| `IOPS/` | `KPI-4d2af31a-9916-3d9f-8a8e-8a268a48c095.train.out` | TSB-UAD, IOPS collection |

The series files are not tracked; `fetch.py` writes them into these folders.
