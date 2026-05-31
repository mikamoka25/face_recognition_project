import insightface
import numpy as np
import cv2
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self, database_path, threshold=0.45):
        """
        初始化人脸识别引擎
        database_path: 特征库路径（.npy文件）
        threshold: 余弦相似度阈值，低于此值判定为unknown
        """
        # 加载InsightFace模型
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # 加载特征库
        self.database = np.load(database_path, allow_pickle=True).item()
        self.threshold = threshold
        print(f"特征库加载成功，共 {len(self.database)} 个人物：{list(self.database.keys())}")

    def cosine_similarity(self, a, b):
        """计算两个向量的余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def recognize(self, img_path):
        """
        识别图片中的所有人脸
        返回列表：[{'name': 'p01', 'bbox': [x, y, w, h], 'score': 0.85}, ...]
        """
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图片 {img_path}")
            return []

        # 检测人脸
        faces = self.app.get(img)
        if not faces:
            print(f"警告：未检测到人脸")
            return []

        results = []
        for face in faces:
            # 提取当前人脸的embedding
            embedding = face.embedding

            # 和特征库中每个人比对
            best_name = "unknown"
            best_score = -1

            for person_id, db_embedding in self.database.items():
                score = self.cosine_similarity(embedding, db_embedding)
                if score > best_score:
                    best_score = score
                    best_name = person_id

            # 低于阈值判定为unknown
            if best_score < self.threshold:
                best_name = "unknown"

            # 转换bbox格式为 [x, y, width, height]
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box
            bbox = [x1, y1, x2 - x1, y2 - y1]

            results.append({
                'name': best_name,
                'bbox': bbox,
                'score': round(float(best_score), 3)
            })

        return results

    def recognize_and_draw(self, img_path, output_path=None):
        """
        识别并在图片上画框，可选保存结果
        """
        img = cv2.imread(img_path)
        if img is None:
            return None

        results = self.recognize(img_path)

        for r in results:
            x, y, w, h = r['bbox']
            name = r['name']
            score = r['score']

            # 画框
            color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

            # 写名字和分数
            label = f"{name} {score:.2f}"
            cv2.putText(img, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if output_path:
            cv2.imwrite(output_path, img)
            print(f"结果已保存到：{output_path}")

        return img, results


# 测试代码
if __name__ == "__main__":
    engine = FaceEngine(
        database_path=r"D:\face_recognition_project\database.npy",
        threshold=0.45
    )

    # 测试一张图片
    test_img = r"D:\face_recognition_project\dataset\test\images\p01_t01.jpg"
    results = engine.recognize(test_img)

    print(f"\n识别结果：")
    for r in results:
        print(f"  身份：{r['name']}，相似度：{r['score']}，位置：{r['bbox']}")
