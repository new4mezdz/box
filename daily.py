import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from datetime import datetime, timedelta

# === 配置 ===
NUM_ROWS = 5
NUM_COLUMNS = 5
EXPECTED_PER_CELL = 1
BASE_DIR = r'F:\qq\yolo_project'
MODEL_PATH = r'F:\qq\yolo_project\box_missing_detector\runs\detect\train_optimized3\weights\best.pt'
OUTPUT_SUFFIX = "识别"

model = YOLO(MODEL_PATH)
class_map = {0: 'box', 1: 'fbox'}

def sort_kmeans_labels(centers):
    sorted_indices = np.argsort(centers.reshape(-1))
    return {orig: new for new, orig in enumerate(sorted_indices)}

def analyze_and_save(image_path, save_dir, pos_txt_writer):
    results = model(image_path, conf=0.4, iou=0.5)[0]
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return False  # 不含 fbox

    x_centers = boxes.xywh[:, 0].cpu().numpy() * results.orig_shape[1]
    y_centers = boxes.xywh[:, 1].cpu().numpy() * results.orig_shape[0]
    cls_indices = boxes.cls.cpu().numpy().astype(int)
    coords = np.stack([x_centers, y_centers], axis=1)

    kmeans_x = KMeans(n_clusters=NUM_COLUMNS, random_state=0).fit(x_centers.reshape(-1, 1))
    kmeans_y = KMeans(n_clusters=NUM_ROWS, random_state=0).fit(y_centers.reshape(-1, 1))
    x_label_map = sort_kmeans_labels(kmeans_x.cluster_centers_)
    y_label_map = sort_kmeans_labels(kmeans_y.cluster_centers_)

    cols = [x_label_map[label] for label in kmeans_x.labels_]
    rows = [y_label_map[label] for label in kmeans_y.labels_]

    grid = [[{'box': 0, 'fbox': 0} for _ in range(NUM_COLUMNS)] for _ in range(NUM_ROWS)]
    fbox_positions = []

    for i in range(len(coords)):
        row = rows[i]
        col = cols[i]
        label = class_map.get(cls_indices[i], 'unknown')
        if 0 <= row < NUM_ROWS and 0 <= col < NUM_COLUMNS and label in grid[row][col]:
            grid[row][col][label] += 1
            if label == 'fbox':
                fbox_positions.append((row + 1, col + 1))

    # 保存图像
    annotated = results.plot()
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    save_img_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(save_img_path, annotated_bgr)

    # 输出 fbox 位置（如果有）
    if fbox_positions:
        fbox_str = ', '.join([f"{r}行{c}列" for r, c in fbox_positions])
        pos_txt_writer.write(f"{os.path.basename(image_path)}: {fbox_str}\n")
        return True
    return False

# === 主流程 ===
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
input_folder = os.path.join(BASE_DIR, yesterday)
output_folder = input_folder + OUTPUT_SUFFIX
os.makedirs(output_folder, exist_ok=True)

fbox_log_path = os.path.join(output_folder, "含fbox图像列表.txt")
with open(fbox_log_path, 'w', encoding='utf-8') as fbox_log:
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(input_folder, file)
            has_fbox = analyze_and_save(img_path, output_folder, fbox_log)

print("✅ 所有图片处理完成，检测图像已保存，fbox图像列表已生成。")
