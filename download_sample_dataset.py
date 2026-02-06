"""
下载示例数据集 DailyTalkContiguous (约14GB)
运行: python download_sample_dataset.py
"""
from huggingface_hub import snapshot_download

print("开始下载示例数据集...")
local_dir = snapshot_download(
    "kyutai/DailyTalkContiguous",
    repo_type="dataset",
    local_dir="./daily-talk-contiguous"
)
print(f"数据集已下载到: {local_dir}")
print("\n现在修改 example/moshi_7B.yaml 中的 train_data 为:")
print(f"  train_data: \"{local_dir}/train.jsonl\"")