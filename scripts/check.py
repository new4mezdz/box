import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans  # ✅ 添加这行

# === 配置 ===
NUM_ROWS = 5
NUM_COLUMNS = 5
EXPECTED_PER_CELL = 1
IMAGE_DIR = r'F:\qq\yolo_project\验证集'
MODEL_PATH = r'F:\qq\yolo_project\box_missing_detector\runs\detect\train_optimized\weights\best.pt'

model = YOLO(MODEL_PATH)
class_map = {0: 'box', 1: 'fbox'}


def analyze_image(image_path):
    results = model(image_path, conf=0.8, iou=0.5)[0]

    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        print(f"{os.path.basename(image_path)} ❌ 无检测结果")
        return

    # 获取中心点和类别
    x_centers = boxes.xywh[:, 0].cpu().numpy() * results.orig_shape[1]
    y_centers = boxes.xywh[:, 1].cpu().numpy() * results.orig_shape[0]
    cls_indices = boxes.cls.cpu().numpy().astype(int)

    coords = np.stack([x_centers, y_centers], axis=1)
    if len(coords) < NUM_ROWS * NUM_COLUMNS:
        print(f"{os.path.basename(image_path)} ⚠️ 检测框数不足，仅有 {len(coords)} 个")

    # KMeans 聚类用于分配行列
    kmeans_x = KMeans(n_clusters=NUM_COLUMNS, random_state=0).fit(x_centers.reshape(-1, 1))
    kmeans_y = KMeans(n_clusters=NUM_ROWS, random_state=0).fit(y_centers.reshape(-1, 1))

    cols = kmeans_x.predict(x_centers.reshape(-1, 1))
    rows = kmeans_y.predict(y_centers.reshape(-1, 1))

    # 初始化网格
    grid = [[{'box': 0, 'fbox': 0} for _ in range(NUM_COLUMNS)] for _ in range(NUM_ROWS)]
    for i in range(len(coords)):
        row = rows[i]
        col = cols[i]
        label = class_map.get(cls_indices[i], 'unknown')
        if 0 <= row < NUM_ROWS and 0 <= col < NUM_COLUMNS and label in grid[row][col]:
            grid[row][col][label] += 1

    # === 输出文本结果 ===
    print(f"\n🖼️ {os.path.basename(image_path)} 检测结果（仅缺失）：")
    missing = []
    for r in range(NUM_ROWS):
        for c in range(NUM_COLUMNS):
            cell = grid[r][c]
            if cell['box'] == 0:
                print(f"❌ 缺失 - 第{r+1}行第{c+1}列（box: {cell['box']}, fbox: {cell['fbox']}）")
                missing.append((r + 1, c + 1))
    if not missing:
        print("✅ 本图格子均正常")

    # === 显示图像 ===
    annotated = results.plot()
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imshow("Detection", annotated_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === 遍历目录 ===
for fn in os.listdir(IMAGE_DIR):
    if fn.lower().endswith(('.jpg', '.png')):
        analyze_image(os.path.join(IMAGE_DIR, fn))




