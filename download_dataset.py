from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yinongh/automate_task_2",
    repo_type="dataset",
    local_dir="./automate_task_2",
    local_dir_use_symlinks=False,
)

print("Downloaded to ./automate_task_2")