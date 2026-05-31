# 人脸识别系统 Face Recognition System

基于 InsightFace (RetinaFace + ArcFace) 的人脸检测与识别系统。

## 环境要求

- Python 3.8+
- Windows / macOS / Linux

## 安装依赖

```bash
pip install insightface onnxruntime numpy opencv-python streamlit
```

## 项目结构

```
face_recognition_project/
├── scripts/
│   ├── app.py                  # Streamlit 前端界面
│   ├── build_database.py       # 建立人脸特征库
│   ├── face_engine.py          # 人脸识别引擎
│   ├── test_celeba.py          # CelebA 100类测试脚本
│   └── convert_annotations.py # 标注文件格式转换脚本
├── dataset/
│   ├── identities.csv          # 身份ID与姓名对照表
│   ├── registered/             # 注册集（每人2张以上）
│   │   ├── p01/
│   │   └── ...
│   └── test/
│       ├── images/             # 测试集图片
│       └── annotations.jsonl  # 测试集标注文件
├── database.npy                # 人脸特征库（运行build_database.py后生成）
└── README.md
```

## 模型说明

本项目使用 InsightFace 的 `buffalo_l` 模型，大小约 300MB，超过 GitHub 限制，**不包含在仓库中**。

首次运行 `build_database.py` 时会自动从网络下载，下载路径为：
`C:\Users\用户名\.insightface\models\buffalo_l\`

## 运行步骤

### Step 1：建立人脸特征库

确保 `dataset/registered/` 中已放好注册集图片，然后运行：

```bash
python build_database.py
```

运行完成后会在项目根目录生成 `database.npy`。

### Step 2：启动前端界面

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`，上传图片即可进行人脸识别。

### Step 3：运行 CelebA 100类测试（可选）

```bash
python test_celeba.py
```

需要将 `celeba_100_identities_3reg_3test/` 文件夹放在项目根目录下。

## 支持的身份

| ID | 姓名 |
|---|---|
| p01 | 成龙 |
| p02 | 章子怡 |
| p03 | 刘德华 |
| p04 | 周杰伦 |
| p05 | 邓紫棋 |
| p06 | 姚明 |
| p07 | 谷爱凌 |
| p08 | 郎平 |
| p09 | 马云 |
| p10 | 屠呦呦 |
| p11 | Barack Obama |
| p12 | Angela Merkel |
| p13 | Narendra Modi |
| p14 | Jacinda Ardern |
| p15 | Taylor Swift |
| p16 | Beyoncé |
| p17 | Dwayne Johnson |
| p18 | Emma Watson |
| p19 | Lionel Messi |
| p20 | Serena Williams |

## 注意事项

- `database.npy`、`dataset/`、模型权重文件不包含在Git仓库中
- 识别阈值默认为 0.45，可在前端界面侧边栏调整
- 纯 CPU 运行即可，无需 GPU
