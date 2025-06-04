import os
from ultralytics import YOLO

EXPECTED_COUNT = 25 # 每张图应有盒子数量
model = YOLO(r'F:\qq\yolo_project\box_missing_detector\runs\detect\train10\weights\best.pt')



# 替换成你训练集图片所在的文件夹路径
image_dir = r'F:\qq\yolo_project\训练集loss'


missing_files = []
total_images = 0
missing_count = 0

def check_missing(image_path):
    global missing_count
    total = EXPECTED_COUNT
    results = model(image_path)
    boxes = results[0].boxes
    detected = len(boxes)
    filename = os.path.basename(image_path)

    print(f"{filename}: 检测到 {detected}/{total} 个盒子")
    if detected < total:
        print(f"⚠️ 缺失 {total - detected} 个盒子")
        missing_files.append(filename)
        missing_count += 1
    else:
        print("✅ 正常")

# 遍历检测所有图片
for file in os.listdir(image_dir):
    if file.endswith('.jpg') or file.endswith('.png'):
        total_images += 1
        check_missing(os.path.join(image_dir, file))

# 输出统计信息
print("\n📊 检测统计结果：")
print(f"总图片数：{total_images}")
print(f"有缺失的图片数：{missing_count}")
print(f"无缺失的图片数：{total_images - missing_count}")

# 计算准确率（判断是否缺失的二分类）
accuracy = (total_images - missing_count) / total_images if total_images > 0 else 0
print(f"✅ 判断准确率（未缺失图片占比）：{accuracy:.2%}")

# 输出缺失图片编号
if missing_files:
    print("\n📂 缺失图片编号列表：")
    for name in missing_files:
        print(" -", name)
else:
    print("\n🎉 所有图片均正常，无缺失！")
