"""Download the series of the paper from the public TSB-UAD data archive into data/<dataset>/.

    python data/fetch.py            # the five series of the paper

The archive (https://www.thedatum.org/datasets/TSB-UAD-Public.zip, about 240 MB) is downloaded once into data/_archive/
and the needed files are extracted from it.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE_URL = "https://www.thedatum.org/datasets/TSB-UAD-Public.zip"
FILES = {
    "MBA805/MBA_ECG805_data_12.out": "TSB-UAD-Public/ECG/MBA_ECG805_data_12.out",
    "MBA14046/MBA_ECG14046_data_1.out": "TSB-UAD-Public/ECG/MBA_ECG14046_data_1.out",
    "MSL/T-9.test.out": "TSB-UAD-Public/NASA-MSL/T-9.test.out",
    "SensorSc/stb-19.test.out": "TSB-UAD-Public/SensorScope/stb-19.test.out",
    "IOPS/KPI-4d2af31a-9916-3d9f-8a8e-8a268a48c095.train.out": "TSB-UAD-Public/IOPS/KPI-4d2af31a-9916-3d9f-8a8e-8a268a48c095.train.out",
}


def main():
    archive = HERE / "_archive" / "TSB-UAD-Public.zip"
    if not archive.exists():
        archive.parent.mkdir(exist_ok=True)
        print(f"downloading {ARCHIVE_URL} ...", flush=True)
        urllib.request.urlretrieve(ARCHIVE_URL, archive)
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        for target, member in FILES.items():
            path = HERE / target
            if path.exists():
                continue
            if member not in names:
                sys.exit(f"{member} is not in the archive; see data/README.md for the sources")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read(member))
            print("wrote", path.relative_to(HERE))
    print("done")


if __name__ == "__main__":
    main()
