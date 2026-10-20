import urllib.request
import zipfile
from pathlib import Path
import os
import sys

TOOLS_DIR = Path("C:/Users/DT/tools")
TOOLS_DIR.mkdir(parents=True, exist_ok=True)

JDK_URL = "https://api.adoptium.net/v3/binary/latest/17/ga/windows/x64/jdk/hotspot/normal/eclipse"
MAVEN_URL = "https://archive.apache.org/dist/maven/maven-3/3.9.6/binaries/apache-maven-3.9.6-bin.zip"

def download_and_extract(url, zip_name, extract_to):
    zip_path = TOOLS_DIR / zip_name
    if not zip_path.exists():
        print(f"[DOWNLOADING] {zip_name} from {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(zip_path, 'wb') as out_f:
            total = int(resp.headers.get('content-length', 0))
            downloaded = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = (downloaded / total) * 100
                    print(f"  {downloaded / 1024 / 1024:.1f}MB / {total / 1024 / 1024:.1f}MB ({pct:.1f}%)", end='\r')
            print()
        print(f"[OK] Downloaded {zip_name} ({zip_path.stat().st_size} bytes)")
    
    print(f"[EXTRACTING] {zip_name} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[OK] Extracted {zip_name}")

if __name__ == "__main__":
    download_and_extract(JDK_URL, "openjdk17.zip", TOOLS_DIR / "jdk17")
    download_and_extract(MAVEN_URL, "maven.zip", TOOLS_DIR / "maven")
    print("\n[SUCCESS] JDK 17 and Maven downloaded and extracted successfully!")
