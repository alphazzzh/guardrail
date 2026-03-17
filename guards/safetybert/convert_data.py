import json
import os

source_dir = "/data/liyc/LLaMA-Factory/data/DNA_lmft"
source_files = ["DNA_en_lmft.json", "DNA_zh_en_lmft.json", "DNA_zh_zh_lmft.json"]
output_file = "/data/liyc/SafetyBERT/data/converted_data.jsonl"

# Label mapping based on SafetyBERT constants
UNSAFE_LABEL_MAP = {
    "包含违反社会主义核心价值观的内容": "违反社会主义核心价值观",
    "包含歧视性内容": "歧视性内容",
    "商业违法违规": "商业违法违规",
    "侵犯他人合法权益": "侵犯他人合法权益",
    "无法满足特定服务类型的安全需求": "无法满足特定服务类型的安全需求",
}

def convert():
    all_converted = []
    for file_name in source_files:
        file_path = os.path.join(source_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Skipping {file_path}, not found.")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for item in data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output_text = item.get("output", "")
            
            # Construct text
            text = instruction + "\n" + input_text if instruction else input_text
            
            # Parse output for labels
            parts = output_text.split("\n类别：")
            safety_label = parts[0].strip()
            
            unsafe_labels = []
            if len(parts) > 1:
                category = parts[1].strip()
                if category != "None" and category in UNSAFE_LABEL_MAP:
                    unsafe_labels = [UNSAFE_LABEL_MAP[category]]
                # '其他' remains an empty list
            
            converted_item = {
                "text": text,
                "safety_label": safety_label,
                "unsafe_labels": unsafe_labels
            }
            all_converted.append(converted_item)

    # Write to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Successfully converted {len(all_converted)} items to {output_file}")

if __name__ == "__main__":
    convert()
