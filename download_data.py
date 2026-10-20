import urllib.request
from pathlib import Path
import sys

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "Train_Beneficiarydata.csv": "https://raw.githubusercontent.com/Ali-Tharwat/Healthcare-Provider-Fraud-Detection/main/data/Train_Beneficiarydata.csv",
    "Train_Inpatientdata.csv": "https://raw.githubusercontent.com/Ali-Tharwat/Healthcare-Provider-Fraud-Detection/main/data/Train_Inpatientdata.csv",
    "Train_Outpatientdata.csv": "https://raw.githubusercontent.com/Ali-Tharwat/Healthcare-Provider-Fraud-Detection/main/data/Train_Outpatientdata.csv",
    "Train.csv": "https://raw.githubusercontent.com/Ali-Tharwat/Healthcare-Provider-Fraud-Detection/main/data/Train_labels.csv",
}

def download():
    for filename, url in FILES.items():
        dest = DATA_DIR / filename
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"[SKIP] {filename} already exists ({dest.stat().st_size} bytes)")
            continue
        print(f"[DOWNLOADING] {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"[OK] Downloaded {filename} ({dest.stat().st_size} bytes)")
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}", file=sys.stderr)
            raise

if __name__ == "__main__":
    download()
