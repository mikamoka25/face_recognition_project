import json
import os
import glob

# 路径配置
TEST_IMAGES_DIR = r"D:\face_recognition_project\dataset\test\images"
OUTPUT_FILE = r"D:\face_recognition_project\dataset\annotations.jsonl"

results = []

# 遍历所有json文件
json_files = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.json"))
print(f"找到 {len(json_files)} 个JSON文件")

for json_path in sorted(json_files):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_filename = os.path.basename(json_path).replace(".json", ".jpg")
    img_relative_path = f"test/images/{img_filename}"

    faces = []
    for shape in data.get("shapes", []):
        # 只处理矩形框，跳过关键点
        if shape["shape_type"] != "rectangle":
            continue

        label = shape["label"]

        # 跳过关键点label
        skip_labels = ["left_eye", "right_eye", "nost_tip", "left_mouth_corner", "right_mouth_corner"]
        if label in skip_labels:
            continue

        # 4个点的矩形，取所有x和y的最小最大值
        points = shape["points"]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1 = int(min(xs))
        y1 = int(min(ys))
        x2 = int(max(xs))
        y2 = int(max(ys))

        width = x2 - x1
        height = y2 - y1

        faces.append({
            "identity_id": label,
            "bbox": [x1, y1, width, height]
        })

    if not faces:
        print(f"警告：{img_filename} 没有找到人脸框，跳过")
        continue

    image_type = "single" if len(faces) == 1 else "multi"

    record = {
        "image": img_relative_path,
        "image_type": image_type,
        "faces": faces
    }
    results.append(record)

# 写入jsonl文件
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for record in results:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\n完成！共处理 {len(results)} 张图片")
print(f"输出文件：{OUTPUT_FILE}")

# 打印预览
print("\n前3条预览：")
for r in results[:3]:
    print(json.dumps(r, ensure_ascii=False))