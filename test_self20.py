import json
import numpy as np
import os
import cv2
from insightface.app import FaceAnalysis

# ============ 配置 ============
DATASET_DIR = r"D:\face_recognition_project\dataset"
DATABASE_PATH = r"D:\face_recognition_project\database.npy"
ANNOTATIONS = os.path.join(DATASET_DIR, "annotations.jsonl")

USE_TTA = True
MATCH_STRATEGY = "max"
MIN_DET_SCORE = 0.5
MIN_FACE_SIZE = 20
IOU_MATCH_THRESHOLD = 0.3
# 阈值扫描候选（用来找最优阈值）
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
DEFAULT_THRESHOLD = 0.40
DEFAULT_MARGIN = 0.04
# ===============================


def l2norm(x):
    return x / (np.linalg.norm(x) + 1e-9)


def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    inter = max(0, xb-xa) * max(0, yb-ya)
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0


def extract_face_embedding(app, img, face, use_tta=True):
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
    # 加载模型
    print("加载模型...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])

    # 加载特征库并堆成矩阵
    raw_db = np.load(DATABASE_PATH, allow_pickle=True).item()
    templates = []
    owners = []
    for pid in sorted(raw_db.keys()):
        embs = raw_db[pid]
        if embs.ndim == 1:
            embs = embs[None, :]
        for e in embs:
            templates.append(l2norm(e))
            owners.append(pid)
    templates = np.stack(templates, axis=0).astype(np.float32)
    owners = np.array(owners)
    unique_ids = sorted(set(owners.tolist()))
    id_to_rows = {pid: np.where(owners == pid)[0] for pid in unique_ids}
    print(f"特征库：{len(unique_ids)} 人，{len(templates)} 模板")

    # 读取标注
    records = []
    with open(ANNOTATIONS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"测试图：{len(records)} 张\n")

    # 收集每个 GT 人脸的 (best_score, best_pid, margin, true_id)
    # 然后用不同阈值来评估，省去重复推理
    eval_items = []   # list of dict
    current_det_size = (640, 640)
    app.prepare(ctx_id=0, det_size=current_det_size)

    for record in records:
        img_path = os.path.join(DATASET_DIR, record["image"])
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取 {record['image']}")
            continue

        # 自适应检测分辨率
        h, w = img.shape[:2]
        target = (960, 960) if max(h, w) > 1000 else (640, 640)
        if target != current_det_size:
            app.prepare(ctx_id=0, det_size=target)
            current_det_size = target

        faces = app.get(img)
        detected = []
        for face in faces:
            if face.det_score < MIN_DET_SCORE:
                continue
            x1, y1, x2, y2 = face.bbox.astype(int)
            fw, fh = x2 - x1, y2 - y1
            if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
                continue
            emb = extract_face_embedding(app, img, face, USE_TTA)
            sims = templates @ emb
            id_scores = {}
            for pid, rows in id_to_rows.items():
                s = sims[rows]
                id_scores[pid] = float(s.max() if MATCH_STRATEGY == "max" else s.mean())
            ranked = sorted(id_scores.items(), key=lambda kv: kv[1], reverse=True)
            best_pid, best_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else -1.0
            detected.append({
                "best_pid": best_pid,
                "best_score": best_score,
                "margin": best_score - second_score,
                "bbox": [int(x1), int(y1), int(fw), int(fh)],
            })

        # 与标注做 IoU 匹配（贪心一对一）
        used = set()
        for gt_face in record["faces"]:
            gt_id = gt_face["identity_id"]
            gt_bbox = gt_face["bbox"]
            best_iou = 0
            best_idx = -1
            for i, det in enumerate(detected):
                if i in used:
                    continue
                v = iou(gt_bbox, det["bbox"])
                if v > best_iou:
                    best_iou = v
                    best_idx = i
            if best_idx >= 0 and best_iou >= IOU_MATCH_THRESHOLD:
                used.add(best_idx)
                d = detected[best_idx]
                eval_items.append({
                    "image": record["image"],
                    "true_id": gt_id,
                    "best_pid": d["best_pid"],
                    "best_score": d["best_score"],
                    "margin": d["margin"],
                    "detected": True,
                })
            else:
                eval_items.append({
                    "image": record["image"],
                    "true_id": gt_id,
                    "best_pid": None,
                    "best_score": -1,
                    "margin": 0,
                    "detected": False,
                })

    # 评估函数
    def evaluate(thr, margin_thr):
        c = 0
        for item in eval_items:
            if not item["detected"]:
                pred = "NO_DETECT"
            elif item["best_score"] < thr or item["margin"] < margin_thr:
                pred = "unknown"
            else:
                pred = item["best_pid"]
            if pred == item["true_id"]:
                c += 1
        return c, len(eval_items)

    # 阈值扫描（固定 margin），找最优阈值
    print("========== 阈值扫描 ==========")
    print(f"{'threshold':>10} {'margin':>8} {'acc':>8} {'correct':>10}")
    best = (0, 0, DEFAULT_THRESHOLD, DEFAULT_MARGIN)
    for thr in THRESHOLDS:
        for m in [0.0, 0.04, 0.06]:
            c, n = evaluate(thr, m)
            acc = c / n * 100 if n > 0 else 0
            print(f"{thr:>10.2f} {m:>8.2f} {acc:>7.2f}% {c:>4}/{n}")
            if c > best[0]:
                best = (c, n, thr, m)
    print(f"\n最优组合：threshold={best[2]}, margin={best[3]}，准确率={best[0]/best[1]*100:.2f}%")

    # 用默认阈值打印详细结果
    print(f"\n========== 默认配置详细结果 (thr={DEFAULT_THRESHOLD}, margin={DEFAULT_MARGIN}) ==========")
    success, failed = [], []
    for item in eval_items:
        if not item["detected"]:
            pred = "NO_DETECT"
        elif item["best_score"] < DEFAULT_THRESHOLD or item["margin"] < DEFAULT_MARGIN:
            pred = "unknown"
        else:
            pred = item["best_pid"]
        rec = (item["image"], item["true_id"], pred,
               round(item["best_score"], 3) if item["best_score"] >= 0 else "N/A")
        if pred == item["true_id"]:
            success.append(rec)
        else:
            failed.append(rec)

    total = len(eval_items)
    correct = len(success)
    print(f"总人脸：{total}，正确：{correct}，Top-1：{correct/total*100:.2f}%")
    print(f"\n成功样例（前 5）：")
    for s in success[:5]:
        print(f"  ✓ {s[0]} | 真:{s[1]} 预:{s[2]} sim={s[3]}")
    print(f"\n失败样例（前 10）：")
    for s in failed[:10]:
        print(f"  ✗ {s[0]} | 真:{s[1]} 预:{s[2]} sim={s[3]}")


if __name__ == "__main__":
    main()
