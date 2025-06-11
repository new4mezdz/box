import os

label_dir = r'F:\qq\yolo_project\box_missing_detector\labels\val'

for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        path = os.path.join(label_dir, file)
        with open(path, 'r') as f:
            lines = f.readlines()
        with open(path, 'w') as f:
            for line in lines:
                parts = line.strip().split()
                parts[0] = '0'  # 把 class id 改为 0（或 1，视你的用途）
                f.write(' '.join(parts) + '\n')
