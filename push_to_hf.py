import os
import sys
from huggingface_hub import HfApi

sys.stdout.reconfigure(encoding='utf-8')

TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN")
REPO_ID = "Youcif/AlphaShariaBot_Daily"

api = HfApi(token=TOKEN)

# Instead of uploading the whole folder with 3000 files, we just upload 
# exactly what's needed to run the app. This avoids the 500 server error.
files_to_upload = [
    "app.py",
    "requirements.txt",
    "README.md",
    "data/halal_stocks.csv",
    "data/fundamentals.csv",
]

# Scripts to upload
scripts = [
    "scripts/alpha_intraday.py",
    "scripts/intraday_features.py",
    "scripts/realtime_news.py",
    "scripts/sentiment_scorer.py",
    "scripts/risk_manager.py",
]

print(f"Uploading files to {REPO_ID} in smaller batches to avoid server timeouts...")

try:
    # 1. Upload root files
    for f in files_to_upload:
        if os.path.exists(f):
            print(f"Uploading {f}...")
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f,
                repo_id=REPO_ID,
                repo_type="space"
            )
            
    # 2. Upload scripts
    for f in scripts:
        if os.path.exists(f):
            print(f"Uploading {f}...")
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f,
                repo_id=REPO_ID,
                repo_type="space"
            )
            
    # Also upload the model just in case any script tries to load it
    model_path = "models/ranker_model.txt"
    if os.path.exists(model_path):
        print(f"Uploading {model_path}...")
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=model_path,
            repo_id=REPO_ID,
            repo_type="space"
        )
        
    print(f"✅ Successfully uploaded! View your space here: https://huggingface.co/spaces/{REPO_ID}")
except Exception as e:
    print(f"❌ Upload failed: {e}")
