import numpy as np
import os
import cv2
from insightface.app import FaceAnalysis

# ============ 配置 ============
CELEBA_DIR = r"D:\face_recognition_project\celeba_100_identities_3reg_3test"
REGISTER_DIR = os.path.join(CELEBA_DIR, "register")
TEST_DIR = os.path.join(CELEBA_DIR, "test")
USE_TTA = True
MATCH_STRATEGY = "max"   # "max" 或 "mean"
# ===============================


def l2norm(x):
    return x / (np.linalg.norm(x) + 1e-9)


def extract_embedding(app, img_path, use_tta=True):
    img = cv2.imread(img_path)
    if img is None:
        return None
    faces = app.get(img)
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
        except Exception:
            pass
    return emb


def main():
    print("加载模型...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Step 1: 建立 100 类身份库（多模板）
    print("\n建立 CelebA 身份库...")
    templates = []   # (N, 512)
    owners = []      # 每行对应哪个 identity
    identity_dirs = sorted(os.listdir(REGISTER_DIR))

    for identity in identity_dirs:
        identity_path = os.path.join(REGISTER_DIR, identity)
        if not os.path.isdir(identity_path):
            continue
        for img_file in os.listdir(identity_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            emb = extract_embedding(app, os.path.join(identity_path, img_file), USE_TTA)
            if emb is not None:
                templates.append(emb)
                owners.append(identity)

    templates = np.stack(templates, axis=0).astype(np.float32)
    owners = np.array(owners)
    unique_ids = sorted(set(owners.tolist()))
    print(f"建库完成：{len(unique_ids)} 个身份，{len(templates)} 个模板")

    # 预计算：每个身份对应的模板行索引（加速）
    id_to_rows = {pid: np.where(owners == pid)[0] for pid in unique_ids}

    # Step 2: 测试
    print("\n开始测试...")
    correct = 0
    total = 0
    failed_cases = []
    success_cases = []
    all_records = []  # 用于阈值扫描

    for identity in sorted(os.listdir(TEST_DIR)):
        identity_path = os.path.join(TEST_DIR, identity)
        if not os.path.isdir(identity_path):
            continue
        for img_file in os.listdir(identity_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(identity_path, img_file)
            emb = extract_embedding(app, img_path, USE_TTA)
            total += 1
            if emb is None:
                failed_cases.append({'true': identity, 'pred': 'NO_FACE', 'file': img_file})
                continue

            # 矩阵化一次性算所有相似度
            sims = templates @ emb  # (N,)
            # 聚合到身份级别
            id_scores = {}
            for pid, rows in id_to_rows.items():
                s = sims[rows]
                id_scores[pid] = float(s.max() if MATCH_STRATEGY == "max" else s.mean())

            ranked = sorted(id_scores.items(), key=lambda kv: kv[1], reverse=True)
            best_id, best_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else -1.0

            all_records.append({
                'true': identity, 'pred': best_id,
                'score': best_score, 'margin': best_score - second_score,
                'file': img_file
            })

            if best_id == identity:
                correct += 1
                if len(success_cases) < 5:
                    success_cases.append({'true': identity, 'pred': best_id,
                                          'score': round(best_score, 3), 'file': img_file})
            else:
                failed_cases.append({'true': identity, 'pred': best_id,
                                     'score': round(best_score, 3), 'file': img_file})

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n========== CelebA 100 类测试结果 ==========")
    print(f"总测试图片：{total}")
    print(f"正确识别：{correct}")
    print(f"Top-1 准确率：{accuracy:.2f}%")
    print(f"TTA: {USE_TTA}，匹配策略: {MATCH_STRATEGY}")

    print(f"\n成功样例（前 5）：")
    for c in success_cases[:5]:
        print(f"  ✓ {c['file']} | 真:{c['true']} 预:{c['pred']} sim={c['score']}")

    print(f"\n失败样例（前 5）：")
    for c in failed_cases[:5]:
        print(f"  ✗ {c['file']} | 真:{c['true']} 预:{c['pred']} sim={c.get('score','N/A')}")

    # 闭集评测不需要阈值，但仍打印 score 分布，方便调阈值（用于自收集集）
    if all_records:
        correct_scores = [r['score'] for r in all_records if r['true'] == r['pred']]
        wrong_scores = [r['score'] for r in all_records if r['true'] != r['pred']]
        print(f"\n相似度分布：")
        if correct_scores:
            print(f"  正确样本 score: 均值={np.mean(correct_scores):.3f}, "
                  f"min={np.min(correct_scores):.3f}, p10={np.percentile(correct_scores,10):.3f}")
        if wrong_scores:
            print(f"  错误样本 score: 均值={np.mean(wrong_scores):.3f}, "
                  f"max={np.max(wrong_scores):.3f}, p90={np.percentile(wrong_scores,90):.3f}")


if __name__ == "__main__":
    main()
