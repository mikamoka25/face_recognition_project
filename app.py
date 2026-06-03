import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys

sys.path.append(r"D:\face_recognition_project")
from face_engine import FaceEngine

st.set_page_config(
    page_title="人脸识别系统",
    page_icon="🎭",
    layout="wide"
)

IDENTITY_NAMES = {
    "p01": "Jackie Chan", "p02": "Zhang Ziyi", "p03": "Andy Lau",
    "p04": "Jay Chou", "p05": "G.E.M.", "p06": "Yao Ming",
    "p07": "Eileen Gu", "p08": "Lang Ping", "p09": "Jack Ma",
    "p10": "Tu Youyou", "p11": "Barack Obama", "p12": "Angela Merkel",
    "p13": "Narendra Modi", "p14": "Jacinda Ardern", "p15": "Taylor Swift",
    "p16": "Beyonce", "p17": "Dwayne Johnson", "p18": "Emma Watson",
    "p19": "Lionel Messi", "p20": "Serena Williams",
    "unknown": "Unknown"
}


@st.cache_resource
def load_engine():
    return FaceEngine(
        database_path=r"D:\face_recognition_project\database.npy",
        threshold=0.40,
        use_tta=True,
        match_strategy="max",
        margin_threshold=0.04,
    )


def check_image_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def draw_results(img, results):
    for r in results:
        x, y, w, h = r['bbox']
        name = r['name']
        score = r['score']
        display_name = IDENTITY_NAMES.get(name, name)
        color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = f"{display_name} ({score:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x, y - th - 10), (x + tw + 4, y), color, -1)
        cv2.putText(img, label, (x + 2, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


st.title("🎭 Face Recognition System")
st.markdown("基于 InsightFace (RetinaFace + ArcFace) 的人脸检测与识别")
st.caption("优化项：TTA 水平翻转 · 多模板 max-pooling · 自适应检测分辨率 · margin 判定")

with st.sidebar:
    st.header("⚙️ 系统设置")
    threshold = st.slider("识别阈值 (threshold)", 0.20, 0.80, 0.40, 0.01,
                          help="余弦相似度低于此值判定为 Unknown")
    margin_thr = st.slider("差距阈值 (margin)", 0.0, 0.20, 0.04, 0.01,
                           help="top-1 与 top-2 差距小于此值也判为 Unknown，防止误识")
    use_tta = st.checkbox("启用 TTA（水平翻转）", value=True,
                          help="略增加推理时间，通常提升 1-3% 准确率")
    st.markdown("---")
    st.markdown("**支持的身份：**")
    for pid, name in IDENTITY_NAMES.items():
        if pid != "unknown":
            st.markdown(f"- {pid}: {name}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 上传图片")
    uploaded_file = st.file_uploader("选择图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption="原始图片", use_container_width=True)

    quality = check_image_quality(img)
    if quality < 50:
        st.warning(f"⚠️ Warning: Low Image Quality (清晰度: {quality:.1f})，识别结果可能不准确")

    with st.spinner("识别中..."):
        engine = load_engine()
        # 应用当前 UI 参数
        engine.threshold = threshold
        engine.margin_threshold = margin_thr
        engine.use_tta = use_tta

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, img)

        results = engine.recognize(tmp_path)
        os.unlink(tmp_path)

    with col2:
        st.subheader("🎯 识别结果")

        if not results:
            st.error("未检测到人脸，请换一张图片")
        else:
            result_img = img.copy()
            result_img = draw_results(result_img, results)
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                     caption="识别结果", use_container_width=True)

            st.markdown(f"**检测到 {len(results)} 张人脸：**")
            for i, r in enumerate(results):
                name = r['name']
                display_name = IDENTITY_NAMES.get(name, name)
                score = r['score']
                margin = r.get('margin', 0)
                if name != "unknown":
                    st.success(
                        f"人脸 {i+1}：**{display_name}** ({name})，"
                        f"相似度：{score:.3f}，margin：{margin:.3f}"
                    )
                else:
                    st.error(
                        f"人脸 {i+1}：**Unknown**，"
                        f"最高相似度：{score:.3f}，margin：{margin:.3f}"
                    )
