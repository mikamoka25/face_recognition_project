import insightface
import numpy as np
import os
import cv2
from insightface.app import FaceAnalysis

# 初始化InsightFace
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 路径配置
REGISTERED_DIR = r"D:\face_recognition_project\dataset\registered"
OUTPUT_FILE = r"D:\face_recognition_project\database.npy"

database = {}

# 遍历注册集
for person_id in sorted(os.listdir(REGISTERED_DIR)):
    person_dir = os.path.join(REGISTERED_DIR, person_id)
    if not os.path.isdir(person_dir):
        continue

    embeddings = []
    img_files = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    print(f"处理 {person_id}，共 {len(img_files)} 张图片")

    for img_file in img_files:
        img_path = os.path.join(person_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  警告：无法读取 {img_file}，跳过")
            continue

        faces = app.get(img)
        if not faces:
            print(f"  警告：{img_file} 未检测到人脸，跳过")
            continue

        # 取置信度最高的人脸
        face = max(faces, key=lambda x: x.det_score)
        embeddings.append(face.embedding)
        print(f"  ✓ {img_file} 提取成功")

    if embeddings:
        # 取所有图片embedding的均值作为该人物的特征
        database[person_id] = np.mean(embeddings, axis=0)
        print(f"  {person_id} 建库完成，使用了 {len(embeddings)} 张图片\n")
    else:
        print(f"  警告：{person_id} 没有成功提取到任何特征，跳过\n")

# 保存数据库
np.save(OUTPUT_FILE, database)
print(f"数据库保存成功：{OUTPUT_FILE}")
print(f"共建库 {len(database)} 个人物：{list(database.keys())}")
