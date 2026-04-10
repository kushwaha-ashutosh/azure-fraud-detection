import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

STORAGE_CONN = os.environ.get("STORAGE_CONNECTION_STRING", "")
SOURCE_CONTAINER = "scored-transactions"
OUTPUT_CONTAINER = "etl-output"
COSMOS_ENDPOINT = os.environ.get("COSMOS_ENDPOINT", "")
COSMOS_KEY = os.environ.get("COSMOS_KEY", "")
COSMOS_DATABASE = "frauddetection"
COSMOS_CONTAINER = "features"

def get_pipeline_status():
    from azure.storage.blob import BlobServiceClient
    client = BlobServiceClient.from_connection_string(STORAGE_CONN)
    container = client.get_container_client(OUTPUT_CONTAINER)
    blobs = list(container.list_blobs(name_starts_with="processed/"))
    return {
        "processed_files": len([b for b in blobs if b.name.endswith(".json")]),
        "last_run": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    status = get_pipeline_status()
    print(json.dumps(status, indent=2))