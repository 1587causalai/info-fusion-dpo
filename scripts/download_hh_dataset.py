"""下载 Anthropic HH 数据集的工具脚本

这个脚本使用 HuggingFace Hub 直接下载 Anthropic HH-RLHF 数据集。
数据将被下载到 data/hh-rlhf/ 目录下。

使用方法：
```bash
python scripts/download_hh_dataset.py
```
"""

import os
import json
import requests
from tqdm import tqdm

def download_file(url, filename):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def download_dataset():
    """下载 Anthropic HH 数据集"""
    print("开始下载 Anthropic HH-RLHF 数据集...")
    
    # 创建输出目录
    output_dir = 'data/hh-rlhf'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 数据集文件 URLs
        files = {
            'train': 'https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/harmless-base/train.jsonl',
            'test': 'https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/harmless-base/test.jsonl'
        }
        
        # 下载并处理每个文件
        for split, url in files.items():
            print(f"\n下载 {split} 集...")
            jsonl_file = os.path.join(output_dir, f'{split}.jsonl')
            download_file(url, jsonl_file)
            
            # 读取 JSONL 文件并转换格式
            data = []
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    # 从 chosen 回复中提取 prompt
                    chosen = item['chosen']
                    assistant_idx = chosen.find('\n\nAssistant:')
                    if assistant_idx == -1:
                        print(f"警告：跳过格式不正确的数据项: {chosen[:100]}...")
                        continue
                        
                    prompt = chosen[:assistant_idx + len('\n\nAssistant:')]
                    
                    data.append({
                        'prompt': prompt,
                        'chosen': chosen,
                        'rejected': item['rejected']
                    })
            
            # 保存为 JSON 格式
            output_file = os.path.join(output_dir, f'{split}.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"{split} 集处理完成，共 {len(data)} 条记录")
            
            # 打印一条示例数据
            if len(data) > 0:
                print(f"\n{split} 集示例数据:")
                print(json.dumps(data[0], indent=2, ensure_ascii=False))
            
            # 删除原始 JSONL 文件
            os.remove(jsonl_file)
        
        return True
        
    except Exception as e:
        print(f"数据集下载失败: {str(e)}")
        return False

def main():
    if download_dataset():
        print("""
数据集下载完成！你现在可以使用以下命令开始训练：

python -u train.py \\
    model=internlm2 \\
    use_local_data=true \\
    local_data_path="data/hh-rlhf/train.json" \\
    # ... 其他参数保持不变
        """)
    else:
        print("数据集下载失败，请检查错误信息并重试。")

if __name__ == "__main__":
    main() 

    