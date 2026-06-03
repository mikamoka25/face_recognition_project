import insightface
import numpy as np
import cv2
from insightface.app import FaceAnalysis


class FaceEngine:
    def __init__(self, database_path, threshold=0.40,
                 min_det_score=0.5, min_face_size=20,
                 use_tta=True, match_strategy="max",
                 margin_threshold=0.04):
        """
        人脸识别引擎（性能优化版）

        参数：
          database_path:   特征库 .npy 路径
          threshold:       余弦相似度阈值，低于则判定为 unknown
          min_det_score:   检测置信度阈值，过滤误检
          min_face_size:   最小人脸边长（像素），过滤过小人脸
          use_tta:         是否启用水平翻转 TTA（推理时也用）
          match_strategy:  "max"（多模板取最大）或 "mean"（取均值）
          margin_threshold: top-1 和 top-2 差距小于此值时判为 unknown
                            （置信度过低，怕认错；设为 0 可关闭）
        """
        # InsightFace 模型（RetinaFace 检测 + ArcFace 识别）
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        # 默认 det_size 先准备好，实际识别时再根据图片大小自适应
        self.default_det_size = (640, 640)
        self.large_det_size = (960, 960)
        self.app.prepare(ctx_id=0, det_size=self.default_det_size)
        self._current_det_size = self.default_det_size

        # 加载特征库
        raw_db = np.load(database_path, allow_pickle=True).item()
        # raw_db: {person_id: ndarray(K, 512)}  K 个模板
        self.person_ids = sorted(raw_db.keys())
        # 把所有模板堆成一个大矩阵，便于一次性矩阵乘法
        templates = []                # 每行是一个已归一化的 embedding
        self.tpl_owner = []           # 每行对应哪个 person_id
        for pid in self.person_ids:
            embs = raw_db[pid]
            if embs.ndim == 1:
                embs = embs[None, :]  # 兼容旧版只存一个均值的 .npy
            for e in embs:
                e = e / (np.linalg.norm(e) + 1e-9)
                templates.append(e)
                self.tpl_owner.append(pid)
        self.templates = np.stack(templates, axis=0).astype(np.float32)  # (N, 512)
        self.tpl_owner = np.array(self.tpl_owner)

        self.threshold = threshold
        self.min_det_score = min_det_score
        self.min_face_size = min_face_size
        self.use_tta = use_tta
        self.match_strategy = match_strategy
        self.margin_threshold = margin_threshold

        print(f"特征库加载成功：{len(self.person_ids)} 个人物，共 {len(self.templates)} 个模板")

    # ---------- 工具方法 ----------

    @staticmethod
    def _l2norm(x):
        return x / (np.linalg.norm(x) + 1e-9)

    def _ensure_det_size(self, img):
        """根据图片尺寸自适应选择检测分辨率（多人合影更友好）。"""
        h, w = img.shape[:2]
        target = self.large_det_size if max(h, w) > 1000 else self.default_det_size
        if target != self._current_det_size:
            self.app.prepare(ctx_id=0, det_size=target)
            self._current_det_size = target

    def _extract_face_embedding(self, face, img):
        """对单张检测到的人脸提取 embedding，可选 TTA（水平翻转）。"""
        emb = face.normed_embedding if hasattr(face, "normed_embedding") else face.embedding
        emb = self._l2norm(emb)
        if not self.use_tta:
            return emb

        # TTA：先按关键点把人脸对齐到 112x112，再水平翻转，重新提取特征后融合
        # 这是 ArcFace 论文里的标准 TTA 做法，能稳定提升 1-3% 准确率
        try:
            from insightface.utils import face_align
            rec_model = self.app.models.get('recognition')
            if rec_model is None or face.kps is None:
                return emb
            input_size = rec_model.input_size[0] if hasattr(rec_model, 'input_size') else 112
            aimg = face_align.norm_crop(img, landmark=face.kps, image_size=input_size)
            aimg_flip = cv2.flip(aimg, 1)
            emb_flip = rec_model.get_feat(aimg_flip).flatten()
            emb_flip = self._l2norm(emb_flip)
            emb = self._l2norm(emb + emb_flip)
        except Exception as e:
            # TTA 失败回退到原 embedding，不影响正常识别
            pass
        return emb

    def _match(self, query_emb):
        """
        和特征库比对，返回 (best_pid, best_score, margin)
        margin = best_score - second_best_score（不同身份之间）
        """
        # 一次矩阵乘法算出和所有模板的相似度
        sims = self.templates @ query_emb.astype(np.float32)  # (N,)

        # 按 person_id 聚合
        unique_pids = self.person_ids
        scores_per_person = {}
        for pid in unique_pids:
            mask = (self.tpl_owner == pid)
            person_sims = sims[mask]
            if self.match_strategy == "max":
                scores_per_person[pid] = float(person_sims.max())
            else:
                scores_per_person[pid] = float(person_sims.mean())

        # 排序找 top-1 和 top-2
        ranked = sorted(scores_per_person.items(), key=lambda kv: kv[1], reverse=True)
        best_pid, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = best_score - second_score
        return best_pid, best_score, margin

    # ---------- 对外接口 ----------

    def recognize(self, img_path):
        """识别图片中所有人脸。返回 [{'name','bbox','score','margin'}, ...]"""
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图片 {img_path}")
            return []

        self._ensure_det_size(img)
        faces = self.app.get(img)
        if not faces:
            return []

        results = []
        for face in faces:
            # 过滤低置信度检测
            if face.det_score < self.min_det_score:
                continue
            x1, y1, x2, y2 = face.bbox.astype(int)
            w, h = x2 - x1, y2 - y1
            # 过滤过小人脸
            if w < self.min_face_size or h < self.min_face_size:
                continue

            emb = self._extract_face_embedding(face, img)
            best_pid, best_score, margin = self._match(emb)

            # 判定 unknown：分数低 或 top-1/top-2 差距过小（系统不确定）
            if best_score < self.threshold:
                pred = "unknown"
            elif margin < self.margin_threshold:
                pred = "unknown"
            else:
                pred = best_pid

            results.append({
                'name': pred,
                'bbox': [int(x1), int(y1), int(w), int(h)],
                'score': round(float(best_score), 3),
                'margin': round(float(margin), 3),
            })
        return results

    def recognize_and_draw(self, img_path, output_path=None):
        img = cv2.imread(img_path)
        if img is None:
            return None, []
        results = self.recognize(img_path)
        for r in results:
            x, y, w, h = r['bbox']
            color = (0, 255, 0) if r['name'] != "unknown" else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            label = f"{r['name']} {r['score']:.2f}"
            cv2.putText(img, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if output_path:
            cv2.imwrite(output_path, img)
        return img, results


if __name__ == "__main__":
    engine = FaceEngine(
        database_path=r"D:\face_recognition_project\database.npy",
        threshold=0.40,
        use_tta=True,
        match_strategy="max",
    )
    test_img = r"D:\face_recognition_project\dataset\test\images\p01_t01.jpg"
    results = engine.recognize(test_img)
    for r in results:
        print(r)
