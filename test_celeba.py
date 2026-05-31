import numpy as np
import os
import cv2
from insightface.app import FaceAnalysis

# 路径配置
CELEBA_DIR = r"D:\face_recognition_project\celeba_100_identities_3reg_3test"
REGISTER_DIR = os.path.join(CELEBA_DIR, "register")
TEST_DIR = os.path.join(CELEBA_DIR, "test")

# 初始化InsightFace
print("加载模型...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def get_embedding(img_path):
    """提取图片中最大人脸的embedding"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    face = max(faces, key=lambda x: x.det_score)
    return face.embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Step 1: 建立100类身份库
print("\n建立CelebA身份库...")
database = {}
identity_dirs = sorted(os.listdir(REGISTER_DIR))

for identity in identity_dirs:
    identity_path = os.path.join(REGISTER_DIR, identity)
    if not os.path.isdir(identity_path):
        continue
    
    embeddings = []
    for img_file in os.listdir(identity_path):
        if not img_file.endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(identity_path, img_file)
        emb = get_embedding(img_path)
        if emb is not None:
            embeddings.append(emb)
    
    if embeddings:
        database[identity] = np.mean(embeddings, axis=0)

print(f"建库完成，共 {len(database)} 个身份")

# Step 2: 测试集推理
print("\n开始测试...")
correct = 0
total = 0
failed_cases = []
success_cases = []

for identity in sorted(os.listdir(TEST_DIR)):
    identity_path = os.path.join(TEST_DIR, identity)
    if not os.path.isdir(identity_path):
        continue

    for img_file in os.listdir(identity_path):
        if not img_file.endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(identity_path, img_file)
        emb = get_embedding(img_path)

        if emb is None:
            print(f"  警告：{identity}/{img_file} 未检测到人脸")
            total += 1
            failed_cases.append({'true': identity, 'pred': 'NO_FACE', 'file': img_file})
            continue

        # 和库中每个人比对
        best_id = None
        best_score = -1
        for db_id, db_emb in database.items():
            score = cosine_similarity(emb, db_emb)
            if score > best_score:
                best_score = score
                best_id = db_id

        total += 1
        if best_id == identity:
            correct += 1
            if len(success_cases) < 5:
                success_cases.append({
                    'true': identity,
                    'pred': best_id,
                    'score': round(float(best_score), 3),
                    'file': img_file
                })
        else:
            failed_cases.append({
                'true': identity,
                'pred': best_id,
                'score': round(float(best_score), 3),
                'file': img_file
            })

# 输出结果
accuracy = correct / total * 100
print(f"\n========== 测试结果 ==========")
print(f"总测试图片：{total}")
print(f"正确识别：{correct}")
print(f"Top-1 准确率：{accuracy:.2f}%")

print(f"\n成功样例（前5个）：")
for c in success_cases[:5]:
    print(f"  ✓ {c['file']} | 真实:{c['true']} 预测:{c['pred']} 相似度:{c['score']}")

print(f"\n失败样例（前5个）：")
for c in failed_cases[:5]:
    print(f"  ✗ {c['file']} | 真实:{c['true']} 预测:{c['pred']} 相似度:{c.get('score','N/A')}")
