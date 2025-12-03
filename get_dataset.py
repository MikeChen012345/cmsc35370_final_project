from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files, HfFolder
from pathlib import Path
from typing import List, Optional
import warnings

### Note: after downloading files, you need to move them out of the cache folder and 
# into your project folder to use them.
def download_subfolder(repo_id: str, subfolder: str) -> List[str]:
    """Download all files under `subfolder/` from the HF repo and return local paths.

    Returns an empty list if the repo or subfolder can't be accessed.
    """
    # Try listing files as a dataset repo first, then fall back to model repo.
    all_files = None
    for repo_type in ('dataset', 'model'):
        try:
            all_files = list_repo_files(repo_id, repo_type=repo_type)
            repo_type_used = repo_type
            break
        except Exception as e:
            # try next repo_type
            all_files = None
            repo_type_used = repo_type
            last_exc = e

    if all_files is None:
        warnings.warn(f"Could not list files for repo '{repo_id}': {last_exc}")
        return []

    files_to_download = [f for f in all_files if f.startswith(subfolder.rstrip('/') + '/')]
    if not files_to_download:
        warnings.warn(f"No files found under '{subfolder}/' in repo '{repo_id}'")
        return []

    local_paths: List[str] = []
    for file in files_to_download:
        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=file, repo_type=repo_type_used)
            local_paths.append(local_path)
        except Exception as e:
            warnings.warn(f"Failed to download {file} from {repo_id} (repo_type={repo_type_used}): {e}")
    return local_paths


def _infer_format_from_paths(paths: List[str]) -> Optional[str]:
    """Infer the dataset loader format from file extensions.

    Returns one of 'json', 'csv', 'parquet' or None if unknown/mixed.
    """
    exts = {Path(p).suffix.lower() for p in paths}
    if not exts:
        return None
    if exts <= {'.json', '.jsonl'}:
        return 'json'
    if exts <= {'.csv'}:
        return 'csv'
    if exts <= {'.parquet'}:
        return 'parquet'
    return None


def get_dataset(dataset_name: str, data_files: Optional[str] = None):
    """Load a dataset. If `data_files` is a repo subfolder pattern like 'roi/*',
    attempt to download files from the HF repo and load them locally.
    Otherwise delegate to datasets.load_dataset as usual.
    """
    # If user passed a simple subfolder pattern like 'roi/*', treat it as a
    # request to fetch the files under that subfolder from the HF repo.
    if isinstance(data_files, str) and data_files.endswith('/*'):
        subfolder = data_files[:-2]
        print(f"Attempting to download files from repo '{dataset_name}' subfolder '{subfolder}/' ...")
        local_paths = download_subfolder(dataset_name, subfolder)
        if not local_paths:
            print("No local files found/downloaded; falling back to loading remote dataset config (may fail if repo missing)")
            return load_dataset(dataset_name)

        fmt = _infer_format_from_paths(local_paths)
        if fmt is None:
            # Unknown file types: try JSON loader first, then CSV
            try:
                return load_dataset('json', data_files=local_paths)
            except Exception:
                return load_dataset('csv', data_files=local_paths)

        return load_dataset(fmt, data_files=local_paths)

    # otherwise behave as before
    if data_files:
        return load_dataset(dataset_name, data_files=data_files)
    return load_dataset(dataset_name)


if __name__ == "__main__":
    data_files = "visual-genome/*"
    dataset = get_dataset("BoltzmachineQ/brain-instruction-tuning", data_files=data_files)
    print(dataset)
