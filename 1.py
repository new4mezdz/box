import os

LABEL_DIR = r'F:\qq\yolo_project\box_missing_detector\labels\val'  # 替换成你的标签目录路径

for filename in os.listdir(LABEL_DIR):
    if filename.endswith(".txt"):
        file_path = os.path.join(LABEL_DIR, filename)
        new_lines = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0] == '15':
                    parts[0] = '0'
                elif parts and parts[0] == '16':
                    parts[0] = '1'
                new_line = ' '.join(parts)
                new_lines.append(new_line)

        # 覆盖写入原文件
        with open(file_path, 'w') as f:
            f.write('\n'.join(new_lines))

print("✅ 所有标签文件已成功更新（15→0，16→1）")
