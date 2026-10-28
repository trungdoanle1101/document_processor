from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
# from document_processor.retrieval.huggingface_embedder import hash_model_files

api = HfApi()
revision = api.model_info("BAAI/bge-m3").sha        # the current commit on main
print("revision:", revision)
for name in api.list_repo_files("BAAI/bge-m3", revision=revision):
    print(name)

folder = Path("hf_models/BAAI/bge-m3")
folder.mkdir(exist_ok=True, parents=True)
