"""
下载示例数据集 DailyTalkContiguous (支持部分下载)
运行示例:
  python download_sample_dataset.py --max-size 1GB  # 下载约1GB数据
  python download_sample_dataset.py --max-files 10  # 下载前10个文件
  python download_sample_dataset.py                 # 下载全部数据
"""
import argparse
from huggingface_hub import snapshot_download
from pathlib import Path


def parse_size(size_str):
    """解析大小字符串，如 '1GB', '500MB', '2.5GB'"""
    size_str = size_str.upper().strip()
    units = {'KB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}
    
    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            try:
                number = float(size_str[:-len(unit)].strip())
                return int(number * multiplier)
            except ValueError:
                raise ValueError(f"无法解析大小: {size_str}")
    
    raise ValueError(f"无效的大小格式: {size_str}。请使用如 '1GB', '500MB' 等格式")


def main():
    parser = argparse.ArgumentParser(
        description='下载 DailyTalkContiguous 数据集（支持部分下载）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --max-size 1GB        # 下载约1GB的数据
  %(prog)s --max-size 500MB      # 下载约500MB的数据
  %(prog)s --max-files 5         # 只下载前5个文件
  %(prog)s                       # 下载完整数据集
        """
    )
    
    parser.add_argument(
        '--max-size',
        type=str,
        help='最大下载大小（如: 1GB, 500MB, 2.5GB）'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        help='最大下载文件数量'
    )
    
    parser.add_argument(
        '--local-dir',
        type=str,
        default='./daily-talk-contiguous',
        help='本地保存目录（默认: ./daily-talk-contiguous）'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        help='只下载匹配的文件模式（如: "*.jsonl" 或 "train/*"）'
    )
    
    args = parser.parse_args()
    
    # 构建下载参数
    download_kwargs = {
        "repo_id": "kyutai/DailyTalkContiguous",
        "repo_type": "dataset",
        "local_dir": args.local_dir,
    }
    
    # 设置文件过滤
    max_size_bytes = None
    if args.max_size:
        max_size_bytes = parse_size(args.max_size)
        print(f"📦 将下载约 {args.max_size} 的数据")
    
    if args.max_files:
        print(f"📁 将下载前 {args.max_files} 个文件")
    
    if args.pattern:
        download_kwargs["allow_patterns"] = args.pattern
        print(f"🔍 只下载匹配 '{args.pattern}' 的文件")
    
    # 自定义过滤函数
    if max_size_bytes or args.max_files:
        downloaded_size = 0
        downloaded_count = 0
        
        def file_filter(filename):
            nonlocal downloaded_size, downloaded_count
            
            # 检查文件数量限制
            if args.max_files and downloaded_count >= args.max_files:
                return False
            
            # 检查大小限制（粗略估计）
            if max_size_bytes and downloaded_size >= max_size_bytes:
                return False
            
            # 假设平均文件大小（可根据实际情况调整）
            estimated_file_size = 100 * 1024 * 1024  # 假设每个文件约100MB
            downloaded_size += estimated_file_size
            downloaded_count += 1
            
            return True
        
        download_kwargs["allow_patterns"] = file_filter if not args.pattern else args.pattern
    
    print("🚀 开始下载数据集...")
    print(f"📂 保存位置: {args.local_dir}")
    
    try:
        local_dir = snapshot_download(**download_kwargs)
        
        print(f"\n✅ 数据集已下载到: {local_dir}")
        
        # 显示下载的文件信息
        data_path = Path(local_dir)
        files = list(data_path.rglob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        
        print(f"📊 下载统计:")
        print(f"   - 文件数量: {len([f for f in files if f.is_file()])} 个")
        print(f"   - 总大小: {total_size / (1024**3):.2f} GB")
        
        # 查找主要数据文件
        jsonl_files = list(data_path.rglob("*.jsonl"))
        if jsonl_files:
            print(f"\n💡 找到数据文件:")
            for jsonl_file in jsonl_files[:5]:  # 只显示前5个
                rel_path = jsonl_file.relative_to(data_path)
                size_mb = jsonl_file.stat().st_size / (1024**2)
                print(f"   - {rel_path} ({size_mb:.2f} MB)")
            
            if len(jsonl_files) > 5:
                print(f"   ... 还有 {len(jsonl_files) - 5} 个文件")
            
            main_data = jsonl_files[0]
            print(f"\n📝 现在修改 example/moshi_7B.yaml 中的 train_data 为:")
            print(f'  train_data: "{main_data}"')
        
    except KeyboardInterrupt:
        print("\n⚠️  下载被用户中断")
    except Exception as e:
        print(f"\n❌ 下载出错: {e}")
        raise


if __name__ == "__main__":
    main()