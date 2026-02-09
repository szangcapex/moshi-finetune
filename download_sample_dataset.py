"""
下载示例数据集 DailyTalkContiguous (支持部分下载)
运行示例:
  python download_sample_dataset.py --max-files 10  # 下载前10个文件
  python download_sample_dataset.py --pattern "*.jsonl"  # 只下载jsonl文件
  python download_sample_dataset.py                 # 下载全部数据
"""
import argparse
from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path
import os


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


def download_files(repo_id, files_to_download, local_dir, repo_type="dataset"):
    """下载指定的文件列表"""
    downloaded_files = []
    total_files = len(files_to_download)
    
    for idx, filename in enumerate(files_to_download, 1):
        try:
            print(f"📥 [{idx}/{total_files}] 下载: {filename}")
            
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            downloaded_files.append(local_path)
            
        except KeyboardInterrupt:
            print("\n⚠️  下载被用户中断")
            return downloaded_files
        except Exception as e:
            print(f"❌ 下载 {filename} 失败: {e}")
            continue
    
    return downloaded_files


def main():
    parser = argparse.ArgumentParser(
        description='下载 DailyTalkContiguous 数据集（支持部分下载）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --max-files 5         # 只下载前5个文件
  %(prog)s --pattern "*.jsonl"   # 只下载jsonl文件
  %(prog)s --pattern "train/*"   # 只下载train目录下的文件
  %(prog)s                       # 列出所有文件（不下载）
        """
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
    
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='只列出文件，不下载'
    )
    
    args = parser.parse_args()
    
    repo_id = "kyutai/DailyTalkContiguous"
    repo_type = "dataset"
    
    print("🔍 获取数据集文件列表...")
    
    try:
        # 获取所有文件列表
        all_files = list_repo_files(repo_id=repo_id, repo_type=repo_type)
        
        # 过滤掉 .gitattributes 等元数据文件
        data_files = [f for f in all_files if not f.startswith('.')]
        
        print(f"📋 数据集共有 {len(data_files)} 个文件")
        
        # 应用模式过滤
        if args.pattern:
            import fnmatch
            filtered_files = [f for f in data_files if fnmatch.fnmatch(f, args.pattern)]
            print(f"🔍 匹配 '{args.pattern}' 的文件: {len(filtered_files)} 个")
            data_files = filtered_files
        
        # 限制文件数量
        if args.max_files:
            data_files = data_files[:args.max_files]
            print(f"📁 将下载前 {len(data_files)} 个文件")
        
        # 显示文件列表
        print(f"\n📄 文件列表:")
        for idx, f in enumerate(data_files[:20], 1):  # 显示前20个
            print(f"   {idx}. {f}")
        
        if len(data_files) > 20:
            print(f"   ... 还有 {len(data_files) - 20} 个文件")
        
        # 如果只是列出文件，到此结束
        if args.list_only:
            print(f"\n💡 使用 --max-files N 来下载前N个文件")
            return
        
        # 确认下载
        if not args.max_files and not args.pattern:
            print(f"\n⚠️  警告: 将下载全部 {len(data_files)} 个文件")
            response = input("是否继续? (y/N): ")
            if response.lower() != 'y':
                print("已取消下载")
                return
        
        print(f"\n🚀 开始下载到: {args.local_dir}")
        
        # 下载文件
        downloaded_files = download_files(
            repo_id=repo_id,
            files_to_download=data_files,
            local_dir=args.local_dir,
            repo_type=repo_type
        )
        
        if not downloaded_files:
            print("\n❌ 没有成功下载任何文件")
            return
        
        print(f"\n✅ 成功下载 {len(downloaded_files)} 个文件到: {args.local_dir}")
        
        # 显示下载统计
        data_path = Path(args.local_dir)
        if data_path.exists():
            files = list(data_path.rglob("*"))
            file_list = [f for f in files if f.is_file()]
            total_size = sum(f.stat().st_size for f in file_list)
            
            print(f"📊 下载统计:")
            print(f"   - 文件数量: {len(file_list)} 个")
            print(f"   - 总大小: {total_size / (1024**3):.2f} GB")
            
            # 查找主要数据文件
            jsonl_files = list(data_path.rglob("*.jsonl"))
            if jsonl_files:
                print(f"\n💡 找到 {len(jsonl_files)} 个数据文件:")
                for jsonl_file in jsonl_files[:5]:
                    rel_path = jsonl_file.relative_to(data_path)
                    size_mb = jsonl_file.stat().st_size / (1024**2)
                    print(f"   - {rel_path} ({size_mb:.2f} MB)")
                
                if len(jsonl_files) > 5:
                    print(f"   ... 还有 {len(jsonl_files) - 5} 个文件")
                
                main_data = jsonl_files[0]
                print(f"\n📝 修改 example/moshi_7B.yaml 中的 train_data 为:")
                print(f'  train_data: "{main_data}"')
        
    except KeyboardInterrupt:
        print("\n⚠️  操作被用户中断")
    except Exception as e:
        print(f"\n❌ 出错: {e}")
        raise


if __name__ == "__main__":
    main()