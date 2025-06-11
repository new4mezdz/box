
import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

# === 配置 ===
NUM_ROWS = 5
NUM_COLUMNS = 5
EXPECTED_PER_CELL = 1
IMAGE_DIR = r'F:\qq\yolo_project\6.10测试'
MODEL_PATH = r'F:\qq\yolo_project\box_missing_detector\runs\detect\train9\weights\best.pt'

model = YOLO(MODEL_PATH)
class_map = {0: 'box', 1: 'fbox'}

