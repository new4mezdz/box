import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

# ==== 配置 ====
NUM_ROWS, NUM_COLUMNS = 5, 5
IMAGE_DIR = r'F:\qq\yolo_project\20250619'
MODEL_PATH = r'F:\qq\yolo_project\box_missing_detector\runs\detect\train_optimized3\weights\best.pt'

model = YOLO(MODEL_PATH)
class_map = {0: 'box', 1: 'fbox'}

def sort_kmeans_labels(centers):
    return {orig: new for new, orig in enumerate(np.argsort(centers.reshape(-1)))}

def has_fbox(image_path):
    res = model(image_path, conf=0.4, iou=0.5)[0]
    if res.boxes is None:
        return False, None
    if 1 not in res.boxes.cls.cpu().numpy().astype(int):
        return False, res.plot()  # 没有 fbox
    return True, res.plot()

# ==== 预扫描文件，记录哪些含 fbox 并缓存可视化 ====
file_infos = []   # [(filename, is_fbox, annotated_numpy), ...]
for fn in sorted(os.listdir(IMAGE_DIR)):
    if fn.lower().endswith(('.jpg', '.png')):
        full = os.path.join(IMAGE_DIR, fn)
        fbox_flag, vis_rgb = has_fbox(full)
        file_infos.append((fn, fbox_flag, vis_rgb))

# ==== Tkinter GUI ====
root = Tk()
root.title("Box / fBox 检测浏览器")

# 左侧列表框
listbox = Listbox(root, width=40, height=25, font=("Consolas", 10))
listbox.pack(side=LEFT, fill=Y, padx=5, pady=5)

# 右侧图片显示
img_label = Label(root)
img_label.pack(side=RIGHT, padx=5, pady=5)

# 往列表框插入条目，并标红含 fbox 的
for idx, (fn, is_fbox, _) in enumerate(file_infos):
    listbox.insert(idx, fn)
    if is_fbox:
        listbox.itemconfig(idx, {'fg': 'red'})

def on_select(event):
    w = event.widget
    if not w.curselection():
        return
    idx = int(w.curselection()[0])
    _, _, vis_rgb = file_infos[idx]
    if vis_rgb is None: return
    # 转为 Tk 图像并显示
    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
    vis_pil = Image.fromarray(vis_bgr)
    vis_pil.thumbnail((640, 480))
    tk_img = ImageTk.PhotoImage(vis_pil)
    img_label.configure(image=tk_img)
    img_label.image = tk_img  # keep reference

listbox.bind('<<ListboxSelect>>', on_select)
root.mainloop()
