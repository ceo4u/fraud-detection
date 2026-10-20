import os
import re
from pathlib import Path
import urllib.request
import json

print("=" * 60)
print("FRAUDGUARD SYSTEM & DATA INTEGRITY VERIFICATION")
print("=" * 60)

# Check 1: PRV5 in backend
backend_hits = []
for p in [Path("fraudguard-java/backend/api.py"), Path("fraudguard-java/backend/model_service.py")]:
    lines = p.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, 1):
        if "PRV5" in line and not line.strip().startswith("#"):
            backend_hits.append((str(p), i, line.strip()))
print(f"Check 1 - PRV5 in backend code: {len(backend_hits)} hits (expected 0)")

# Check 2: Random in backend
rand_hits = []
pat_rand = re.compile(r"np\.random|random\.", re.IGNORECASE)
for p in Path("fraudguard-java/backend").glob("*.py"):
    lines = p.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, 1):
        if pat_rand.search(line):
            rand_hits.append((str(p), i, line.strip()))
print(f"Check 2 - np.random / random in backend: {len(rand_hits)} hits (expected 0)")

# Check 3: Hardcoded metrics in frontend
lit_hits = []
pat_lit = re.compile(r'"0\.(9|78)"')
for p in Path("fraudguard-java/frontend/src/main/resources").rglob("*"):
    if p.suffix in [".html", ".js"]:
        lines = p.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, 1):
            if pat_lit.search(line):
                lit_hits.append((str(p), i, line.strip()))
print(f"Check 3 - Literal metric strings in templates/js: {len(lit_hits)} hits (expected 0)")

# Check 4: Backend API checks
print("\n--- Backend REST Endpoints (Port 8000) ---")
endpoints = [
    ("/health", ["status", "model", "pr_auc"]),
    ("/providers?limit=3", ["Provider", "y_proba", "risk_tier"]),
    ("/providers/count", ["total", "high_risk"]),
    ("/metrics", ["pr_auc", "roc_auc", "recall", "precision", "f1"]),
    ("/explain/PRV52340", ["provider", "shap_values", "top_drivers"]),
    ("/demo-provider", ["provider", "prob"]),
]
for path, required_keys in endpoints:
    url = f"http://localhost:8000{path}"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode())
        if isinstance(data, list):
            sample = data[0]
        else:
            sample = data
        all_keys = all(k in sample for k in required_keys)
        status = "PASS" if resp.status == 200 and all_keys else "FAIL"
        print(f"  [{status}] {path} -> Status {resp.status}")

# Check 5: Frontend Page checks (Port 8081)
print("\n--- Frontend UI & Proxy Endpoints (Port 8081) ---")
pages = ["/", "/explain", "/metrics", "/about", "/api/providers?limit=3", "/api/providers/count", "/api/metrics"]
for path in pages:
    url = f"http://localhost:8081{path}"
    with urllib.request.urlopen(url) as resp:
        body = resp.read()
        status = "PASS" if resp.status == 200 and len(body) > 0 else "FAIL"
        print(f"  [{status}] {path} -> Status {resp.status} ({len(body)} bytes)")

print("\n" + "=" * 60)
print("ALL VERIFICATION CHECKS COMPLETE")
print("=" * 60)
