import os
from kaggle.api.kaggle_api_extended import KaggleApi

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
META_PATH = os.path.join(DATA_DIR, "dataset-metadata.json")

api = KaggleApi()
api.authenticate()

try:
    print("🚀 Attempting to publish new dataset...")
    api.dataset_create_new(folder=DATA_DIR, public=True)
    print("✅ Dataset published successfully!")
except Exception as e:
    print("⚠️ Dataset may already exist. Trying to create a new version...")
    try:
        api.dataset_create_version(
            folder=DATA_DIR,
            version_notes="Updated posture images and dataset",
            delete_old_versions=False
        )
        print("✅ Dataset version updated successfully!")
    except Exception as ve:
        print(f"❌ Failed to update dataset: {ve}")
