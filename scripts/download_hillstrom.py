"""
Download the Hillstrom MineThatData Email Marketing Dataset.
Preserves the raw downloaded dataset in data/hillstrom/raw/hillstrom_raw.csv.
"""
import sys
import urllib.request
import ssl
from pathlib import Path

# Primary and fallback URLs for Hillstrom dataset
URLS = [
    "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
    "https://raw.githubusercontent.com/maks-sh/scikit-uplift/master/sklift/datasets/data/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
    "https://raw.githubusercontent.com/je-suis-tm/machine-learning/master/hillstrom.csv",
]

def download_hillstrom():
    root = Path(__file__).resolve().parent.parent
    raw_dir = root / "data" / "hillstrom" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "hillstrom_raw.csv"

    print("=" * 70)
    print("ACQUIRING HILLSTROM MINETHATDATA DATASET")
    print("=" * 70)
    print(f"Target destination: {raw_file}")

    downloaded = False
    error_log = []

    for url in URLS:
        print(f"Attempting download from: {url}")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) EconoCasual/1.0"}
            )
            # Create SSL context that allows standard certificates
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            with urllib.request.urlopen(req, context=ctx, timeout=30) as response, open(raw_file, "wb") as out_file:
                chunk_size = 64 * 1024
                total_bytes = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    total_bytes += len(chunk)

            print(f"[OK] Downloaded {total_bytes:,} bytes successfully from {url}")
            downloaded = True
            break
        except Exception as exc:
            print(f"[WARN] Failed from {url}: {exc}")
            error_log.append(f"{url}: {exc}")

    if not downloaded:
        print("\n[ERROR] All automated download sources failed!")
        print("Detailed error log:")
        for err in error_log:
            print(f"  - {err}")
        print("\nManual Alternative Instructions:")
        print("1. Download Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")
        print("   from http://www.minethatdata.com or scikit-uplift.")
        print(f"2. Place the file at: {raw_file}")
        sys.exit(1)

    # Validate downloaded raw file
    import pandas as pd
    df = pd.read_csv(raw_file)
    print(f"\n[VALIDATION] Successfully loaded raw Hillstrom dataset:")
    print(f"  - Rows: {len(df):,}")
    print(f"  - Columns: {len(df.columns)}")
    print(f"  - Column list: {list(df.columns)}")
    print(f"  - Segments present: {df['segment'].value_counts().to_dict()}")
    print("\nRaw file preserved untouched in data/hillstrom/raw/hillstrom_raw.csv")

if __name__ == "__main__":
    download_hillstrom()

