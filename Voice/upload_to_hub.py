from pathlib import Path
from huggingface_hub import HfApi, create_repo
from config import (
    HF_REPO_ID,
    MODEL_PATH,
    THRESHOLD_PATH,
    MODEL_FILENAME,
    THRESHOLD_FILENAME,
)

def upload_to_hub():
    """Upload model files to Hugging Face Hub."""
    
    api = HfApi()
    
    # Check files exist locally
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Threshold not found: {THRESHOLD_PATH}")
    
    print(f"📤 Uploading to {HF_REPO_ID}...")
    
    try:
        # Create repo if it doesn't exist
        print(f"  Creating/checking repo...")
        create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True, private=False)
    except Exception as e:
        print(f"  ⚠️  Repo creation note: {e}")
    
    # Upload classifier model
    print(f"  Uploading {MODEL_FILENAME}...")
    api.upload_file(
        path_or_fileobj=str(MODEL_PATH),
        path_in_repo=MODEL_FILENAME,
        repo_id=HF_REPO_ID,
        repo_type="model",
    )
    print(f"    ✓ Uploaded {MODEL_FILENAME}")
    
    # Upload threshold
    print(f"  Uploading {THRESHOLD_FILENAME}...")
    api.upload_file(
        path_or_fileobj=str(THRESHOLD_PATH),
        path_in_repo=THRESHOLD_FILENAME,
        repo_id=HF_REPO_ID,
        repo_type="model",
    )
    print(f"    ✓ Uploaded {THRESHOLD_FILENAME}")
    
    print(f"\n✅ Success! Models available at: https://huggingface.co/{HF_REPO_ID}")

if __name__ == "__main__":
    upload_to_hub()
