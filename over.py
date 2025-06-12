import os
import shutil

# === 配置 ===
IMAGE_DIR = r'F:\qq\yolo_project\box_missing_detector\images\train'
LABEL_DIR = r'F:\qq\yolo_project\box_missing_detector\labels\train'
OVERSAMPLE_TIMES = 20 # 复制次数

def has_fbox(label_path):
    with open(label_path, 'r') as f:
        for line in f:
            if line.strip().startswith('1 '):  # 类别为 1，即 fbox
                return True
    return False

def copy_sample(img_file, label_file, base_name):
    for i in range(1, OVERSAMPLE_TIMES + 1):
        new_img = os.path.join(IMAGE_DIR, f"{base_name}_aug{i}.jpg")
        new_lbl = os.path.join(LABEL_DIR, f"{base_name}_aug{i}.txt")
        shutil.copy(img_file, new_img)
        shutil.copy(label_file, new_lbl)

# === 遍历标签文件 ===
count = 0
for label_filename in os.listdir(LABEL_DIR):
    if not label_filename.endswith('.txt'):
        continue

    label_path = os.path.join(LABEL_DIR, label_filename)
    base_name = os.path.splitext(label_filename)[0]
    img_path_jpg = os.path.join(IMAGE_DIR, base_name + '.jpg')
    img_path_png = os.path.join(IMAGE_DIR, base_name + '.png')

    # 跳过无图像的标签
    if os.path.exists(img_path_jpg):
        img_path = img_path_jpg
    elif os.path.exists(img_path_png):
        img_path = img_path_png
    else:
        print(f"⚠️ 找不到图像文件：{base_name}")
        continue

    if has_fbox(label_path):
        copy_sample(img_path, label_path, base_name)
        count += 1

print(f"\n✅ 已扩增 {count} 个包含 fbox 的样本 × {OVERSAMPLE_TIMES} 次")
