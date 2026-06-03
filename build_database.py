import numpy as np
import os
import cv2
from insightface.app import FaceAnalysis

# ============ 配置 ============
REGISTERED_DIR = r"D:\face_recognition_project\dataset\registered"
OUTPUT_FILE = r"D:\face_recognition_project\database.npy"
USE_TTA = True           # 是否使用水平翻转 TTA
DET_SIZE = (640, 640)    # 注册图通常是清晰单人照，640 够用
MIN_DET_SCORE = 0.5      # 注册时过滤低质量检测
# ===============================


def l2norm(x):
    return x / (np.linalg.norm(x) + 1e-9)


def extract_embedding_with_tta(app, img, use_tta=True):
    """
    对一张图提取一个 embedding（取最大人脸）。
    若启用 TTA，则用对齐后的 112x112 人脸水平翻转再提一次特征并融合。
    返回 None 表示未检测到人脸或质量不达标。
    """
    faces = app.get(img)
    faces = [f for f in faces if f.det_score >= MIN_DET_SCORE]
    if not faces:
        return None
    face = max(faces, key=lambda x: x.det_score)
    emb = face.normed_embedding if hasattr(face, "normed_embedding") else face.embedding
    emb = l2norm(emb)

    if use_tta:
        try:
            from insightface.utils import face_align
            rec_model = app.models.get('recognition')
            if rec_model is not None and face.kps is not None:
                input_size = rec_model.input_size[0] if hasattr(rec_model, 'input_size') else 112
                aimg = face_align.norm_crop(img, landmark=face.kps, image_size=input_size)
                aimg_flip = cv2.flip(aimg, 1)
                emb_flip = l2norm(rec_model.get_feat(aimg_flip).flatten())
                emb = l2norm(emb + emb_flip)
        except Exception as e:
            print(f"    TTA 失败（不致命）：{e}")

    return emb


def main():
    print("加载 InsightFace 模型...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=DET_SIZE)

    # database: {person_id: ndarray(K, 512)}  K 个模板，每张注册图一个
    database = {}

    for person_id in sorted(os.listdir(REGISTERED_DIR)):
        person_dir = os.path.join(REGISTERED_DIR, person_id)
        if not os.path.isdir(person_dir):
            continue

        img_files = [f for f in os.listdir(person_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\n处理 {person_id}（{len(img_files)} 张）")

        embeddings = []
        for img_file in img_files:
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"  ✗ 无法读取：{img_file}")
                continue
            emb = extract_embedding_with_tta(app, img, use_tta=USE_TTA)
            if emb is None:
                print(f"  ✗ 未检测到合格人脸：{img_file}")
                continue
            embeddings.append(emb)
            print(f"  ✓ {img_file}")

        if not embeddings:
            print(f"  警告：{person_id} 没有可用模板，跳过！")
            continue

        # 关键改动：保留每张图的 embedding，而不是只存均值
        # 推理时用 max-pooling 比对，对姿态/年龄差异更鲁棒
        database[person_id] = np.stack(embeddings, axis=0).astype(np.float32)
        print(f"  → {person_id} 入库：{len(embeddings)} 个模板")

    np.save(OUTPUT_FILE, database)
    print(f"\n========== 建库完成 ==========")
    print(f"输出：{OUTPUT_FILE}")
    print(f"人物数：{len(database)}")
    total_tpls = sum(v.shape[0] for v in database.values())
    print(f"模板总数：{total_tpls}（平均每人 {total_tpls/len(database):.1f} 个）")


if __name__ == "__main__":
    main()
