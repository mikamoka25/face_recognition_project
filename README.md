# 人脸识别系统（Face Recognition System）

基于 **InsightFace (RetinaFace + ArcFace)** 的本地人脸识别系统，支持对输入图片进行人脸检测，并将检测到的人脸识别为 20 类指定身份或 `unknown`。前端使用 Streamlit 构建，所有推理在 CPU 本地完成，不调用任何云端 API。

---

## 一、运行环境

| 组件 | 版本 / 说明 |
|---|---|
| 操作系统 | Windows 10 / 11 |
| Python | 3.10+（推荐 conda 虚拟环境） |
| 推理设备 | CPU（CPUExecutionProvider） |
| 项目根目录 | `D:\face_recognition_project\` |

### 依赖安装

推荐使用 conda 创建独立环境：

```bash
conda create -n face_project python=3.10 -y
conda activate face_project

pip install insightface==0.7.3
pip install onnxruntime
pip install opencv-python
pip install numpy
pip install streamlit
```

> 首次运行 InsightFace 时会自动下载 `buffalo_l` 模型包（约 280 MB）到 `~/.insightface/models/`，无需手动操作。如果下载失败，可手动从 InsightFace 官方仓库下载后解压到该目录。

---

## 二、项目结构

```
D:\face_recognition_project\
├── scripts\                         # 代码文件夹
│   ├── face_engine.py               # 核心识别引擎（FaceEngine 类）
│   ├── build_database.py            # 注册集 → 特征库构建脚本
│   ├── convert_annotations.py       # labelme JSON → annotations.jsonl
│   ├── test_celeba.py               # CelebA 100 类评测脚本
│   ├── test_self20.py               # 自收集 20 类评测脚本（含阈值扫描）
│   └── app.py                       # Streamlit 前端
│
├── dataset\                         # 自收集数据集
│   ├── identities.csv               # 20 类身份 ID 与姓名映射
│   ├── registered\                  # 注册集（每人 ≥2 张）
│   │   ├── p01\
│   │   │   ├── p01_r01.jpg
│   │   │   └── p01_r02.jpg
│   │   ├── p02\
│   │   └── ...\p20\
│   ├── test\
│   │   └── images\                  # 测试集图片 + labelme JSON
│   │       ├── p01_t01.jpg
│   │       ├── p01_t01.json
│   │       └── ...
│   └── annotations.jsonl            # 测试集标注（脚本自动生成）
│
├── celeba_100_identities_3reg_3test\  # CelebA 100 类评测数据（作业提供）
│   ├── register\
│   └── test\
│
├── database.npy                     # 特征库文件（脚本自动生成）
└── README.md                        # 本文档
```

---

## 三、快速开始（5 分钟跑通）

按顺序执行以下 4 步即可完整复现：

### 步骤 1：生成测试集标注

将 labelme 标注的 JSON 转换为统一的 JSONL 格式。

```bash
cd D:\face_recognition_project\scripts
python convert_annotations.py
```

**输出**：`D:\face_recognition_project\dataset\annotations.jsonl`

### 步骤 2：构建特征库

从 `dataset\registered\` 提取所有注册图的 embedding，并保存为特征库。

```bash
python build_database.py
```

**输出**：`D:\face_recognition_project\database.npy`

正常输出示例：
```
========== 建库完成 ==========
人物数：20
模板总数：60（平均每人 3.0 个）
```

> ⚠️ 必须确认人物数为 **20**，否则部分身份不可识别。

### 步骤 3：运行评测脚本

**(a) CelebA 100 类评测：**
```bash
python test_celeba.py
```

**(b) 自收集 20 类评测（含阈值扫描）：**
```bash
python test_self20.py
```

脚本会自动扫描多组阈值并打印最优组合。

### 步骤 4：启动前端

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`，上传图片即可看到识别结果。

---

## 四、代码文件说明

### `face_engine.py` — 核心引擎

封装 `FaceEngine` 类，对外提供统一的识别接口。

**关键功能：**
- 加载 InsightFace 的 RetinaFace 检测模型 + ArcFace 识别模型
- 加载特征库 `database.npy` 并堆成矩阵，支持矩阵化批量比对
- 支持 TTA（水平翻转融合）提升 embedding 质量
- 支持多模板 max-pooling 匹配（每个身份保留多张注册图的独立特征）
- 自适应检测分辨率（大图自动用 960×960，小图用 640×640）
- 基于 threshold + margin 双阈值的 unknown 判定

**主要参数：**
| 参数 | 默认值 | 说明 |
|---|---|---|
| `threshold` | 0.40 | 余弦相似度阈值，低于则判 unknown |
| `margin_threshold` | 0.04 | top-1 与 top-2 差距阈值，过小判 unknown |
| `min_det_score` | 0.5 | 检测置信度阈值 |
| `min_face_size` | 20 | 最小人脸边长（像素） |
| `use_tta` | True | 是否启用水平翻转 TTA |
| `match_strategy` | "max" | 多模板聚合方式（max / mean） |

### `build_database.py` — 特征库构建

遍历 `dataset\registered\` 中每个身份的所有注册图：

1. RetinaFace 检测最大人脸
2. ArcFace 提取 512 维 embedding
3. 启用 TTA：使用 `face_align.norm_crop` 把人脸对齐到 112×112，水平翻转后再提取一次特征，与原特征 L2 归一化融合
4. 每个身份保留**所有注册图的独立特征**（不取均值），形状 `(K, 512)`

**输出格式**：`database.npy` 是一个 `dict`，键为身份 ID（如 `"p01"`），值为 `(K, 512)` 的 numpy 矩阵。

### `convert_annotations.py` — 标注格式转换

将 labelme 工具标注的 JSON 文件转换为统一的 `annotations.jsonl`：

- 只保留矩形框（`shape_type == "rectangle"`），跳过五官关键点
- 坐标从 `[[x1,y1],[x2,y2]]` 转为 `[x, y, width, height]` 整数像素格式
- 自动判断 `image_type`：1 个人脸为 `single`，多个为 `multi`

### `test_celeba.py` — CelebA 100 类闭集评测

使用 `celeba_100_identities_3reg_3test/register/` 构建 100 类身份库，对 `test/` 进行 Top-1 识别。**闭集评测无需阈值**（测试图的身份保证在库中）。

输出准确率、相似度分布、成功 / 失败样例。

### `test_self20.py` — 自收集 20 类开集评测

读取 `annotations.jsonl`，对每张测试图：

1. 检测所有人脸 → 提取 embedding
2. 与特征库矩阵化比对（一次矩阵乘法得到所有相似度）
3. 按身份聚合（max-pooling）
4. 检测框与 GT 框做 IoU 贪心匹配（IoU ≥ 0.3）
5. 应用 threshold + margin 双阈值判定 unknown

**额外功能**：自动扫描 6 组 threshold × 3 组 margin 的组合，打印准确率表，帮助找到最优参数。

### `app.py` — Streamlit 前端

**功能：**
- 上传图片（支持 jpg/jpeg/png）
- 显示原图 + 识别结果对比
- 检测框上叠加身份名称和相似度
- 已知身份用绿色框，unknown 用红色框
- 侧边栏可实时调节 threshold、margin、TTA 开关
- 图片模糊度（Laplacian 方差）检测，自动给出低质量提示

**运行方式：**
```bash
streamlit run app.py
```

---

## 五、关键技术与优化

| 优化项 | 收益 |
|---|---|
| 多模板存储 + max-pooling | 注册集姿态/年龄差异大时显著提升识别率 |
| TTA 水平翻转（对齐后翻转 112×112 人脸） | 稳定 +1~3% 准确率 |
| L2 归一化 + 矩阵化批量比对 | 多人合影推理加速 3~10 倍 |
| 自适应检测分辨率 | 大图小脸检测更稳定 |
| 双阈值（threshold + margin） | 降低相似身份之间的误识别 |

---

## 六、参考性能

| 评测集 | Top-1 准确率 |
|---|---|
| CelebA 100 类（闭集） | 见 `test_celeba.py` 输出 |
| 自收集 20 类（开集） | 96.20% (177/184) |

> 准确率会因注册集 / 测试集差异略有波动。

---

## 七、注意事项

1. **绝对路径**：本项目所有路径均使用 `D:\face_recognition_project\` 作为根目录。如需迁移到其他位置，请修改各脚本顶部的路径配置。

2. **模型自动下载**：首次运行任意脚本时，InsightFace 会自动下载 `buffalo_l` 模型包（约 280 MB）到用户目录。请确保首次运行时网络可用。

3. **运行顺序**：必须先 `convert_annotations.py` 和 `build_database.py`，才能运行 `test_self20.py` 或 `app.py`。

4. **CPU 推理时间**：每张图约 0.5~1.5 秒（取决于人脸数量和图片分辨率）。

5. **数据集隐私**：人脸图像属于敏感信息，本项目数据仅用于课程作业，不得用于其他用途。

---

## 八、组员

| 姓名 | 学号 | 分工 |
|---|---|---|
| 组员 1 | | 数据收集 / 模型实现 |
| 组员 2 | | 标注 / 测试评估 |
| 组员 3 | | 前端开发 |
| 组员 4 | | 报告撰写 |

> 请根据实际情况填写。

---

## 九、参考链接

- [InsightFace 官方仓库](https://github.com/deepinsight/insightface)
- [ArcFace 论文](https://arxiv.org/abs/1801.07698)
- [CelebA 数据集](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [Streamlit 文档](https://docs.streamlit.io/)
