# 表情对数据集：方法全文

本文是 `expverify/` + `pipelines/` 这套代码的完整技术说明。所有数字都来自本仓库里已经跑完的
产物文件（`out/calibration/*.json`、`out/demo1_liveportrait/`、`out/demo2_cremad/`、`pairs/*/metrics.json`）
或直接来自源码，引用格式为 `文件:行号`。凡是无法在代码或产物里核实的说法，正文里会显式标注
"未证实"，并在第 10 节汇总。

---

## 1. 核心思路：为什么"构造"打得过"检索"

### 1.1 命题

需求是：两段视频，**逐帧**表情相同，其余一切（身份、背景、头部姿态、光照、镜头）都不同。
"大致相似"明确不接受。

对这个需求有两种做法：

- **检索路线**：抽特征 → 算相似度 → 取最近的若干对。表情相等是一个*测量结论*。
- **构造路线**：让两段视频的表情来自同一个张量。表情相等是*生成过程的数学后果*，
  验证器只负责把不满足的拒掉。

本仓库选第二条，`expverify/__init__.py:1-8` 把这句话写成了包的设计前提：
"expression equality must be *constructed* in a shared latent space, never discovered by search…
This package therefore only ever *rejects* candidate pairs."

### 1.2 为什么检索路线到不了"严格相同"——用校准文件里的数字说

检索路线的上限由**测量噪声地板**决定，而这个地板已经被本仓库量出来了。

`pipelines/calibrate.py` 支持两种正样本源（`pipelines/calibrate.py:120-125`），产出两份校准文件：

| 正样本源 | 含义 | 产物 |
| --- | --- | --- |
| `--positives augment` | 同一张脸 + 仅改外观的增广孪生体。表情按定义逐帧完全相同，测的是**验证器自身的噪声地板** | `out/calibration/augment.json` |
| `--positives liveportrait` | Demo 1 输出：不同的人 + 同一个表情张量。测的是**跨身份地板** | `out/calibration/liveportrait.json` |

两份文件里 `metrics.<m>.pos_median`（正样本距离中位数）对照如下：

| 指标 | 噪声地板（同一张脸，augment.json） | 跨身份地板（不同人同表情，liveportrait.json） | 比值 |
| --- | ---: | ---: | ---: |
| `d_bs` | 0.004445 | 0.016729 | 3.76× |
| `d_deform` | 0.020901 | 0.020419 | **0.98×** |
| `d_gaze` | 0.055228 | 0.150246 | 2.72× |
| `d_region` | 0.986765 | 1.090101 | 1.10× |
| `d_au` | 0.102868 | 0.142682 | 1.39× |

`d_deform` 那一行是整件事的关键：**M2 形变场在同一张脸上重复测两遍的差异（0.0209），
已经等于两个不同的人做同一个表情时的全部差异（0.0204）**。也就是说 M2 的"预算"全部被噪声吃掉，
它没有任何剩余分辨力去区分"表情不同"。

再看这个地板相对于"差不多"的距离。硬负样本定义为**同一对正样本平移 k 帧**
（`expverify/calibrate.py:141-160`），即身份、光照、背景、外观变换、噪声机制全部相同，
只差 k 帧肌肉运动。`liveportrait.json` 里 k=3 时：

| 指标 | 正样本中位数 | 硬负样本(+3f)中位数 | AUC(+3f) | AUC(+5f) | AUC(+10f) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `d_bs` | 0.016729 | 0.016991 | 0.505 | 0.510 | 0.524 |
| `d_deform` | 0.020419 | 0.023679 | 0.553 | 0.590 | 0.666 |
| `d_gaze` | 0.150246 | 0.206112 | 0.645 | 0.730 | 0.909 |
| `d_region` | 1.090101 | 1.152182 | 0.574 | 0.631 | 0.743 |
| `d_au` | 0.142682 | 0.147478 | 0.507 | 0.514 | 0.502 |

**在 3 帧尺度上，五个指标的 AUC 是 0.505–0.645。** `d_bs` 和 `d_au` 基本等于抛硬币。
`d_gaze` 要把负样本拉到 10 帧才升到 0.909。

这一张表就是检索路线的判决书：如果单帧距离分不清"表情相同"和"差 3 帧"，那么"抽特征算相似度取最近"
产出的东西，在 3 帧粒度上没有任何保证。你可以把阈值调得很紧，但紧到低于正样本中位数就等于全拒；
`expverify/calibrate.py:297-343` 的 `fit_threshold` 专门处理了这个退化，并且它的记录显示
**本次校准中五个指标全部触发了退化分支**（见 4.4 节）。

### 1.3 但 pair 级聚合是可分的

同一份校准文件里，`report.pair_level`（`out/calibration/liveportrait.json:212-233`）是把
**真实的 `verify_pair` 跑在对齐 pair 和故意平移 ±5/±10 帧的 pair 上**得到的：

- 对齐组：n=6，逐帧通过率中位数 0.0789，q05 = 0.0453
- 平移组：n=24，逐帧通过率中位数 0.0000，q95 = 0.0262
- AUC = 0.9931，precision = 1.000，recall = 0.833，`separated: true`（两组分布完全不重叠）

结论必须说得精确：**"严格逐帧一致"目前只能在整段统计意义上被证明，不能逐帧证明。**
单帧证据弱（AUC≈0.5），50 个弱证据聚合成决定性判决（AUC 0.993）。
`expverify/calibrate.py:264-272` 的注释就是这个意思："Per-frame the rank-1 test is noisy;
per pair it is not… which is why the accept/reject decision is made at the pair level and never frame by frame."

### 1.4 构造路线为什么能绕过这一切

Demo 1 里两段输出的表情张量是**同一个数组的同一份拷贝**（第 5 节给出上游源码逐行证明）。
表情相等不依赖任何指标的分辨力，指标只用来检查"构造没坏掉"以及"其余因素确实不同"。
噪声地板从"决定数据质量"降级成"决定验证器能给出多强的证据"——后者可以诚实报出来，前者不能。

> 附带说明：`expverify/__init__.py:4-7` 引用了 FEC 基准上的数字（AU 距离类指标 40.7–47.1%、
> 情绪 embedding 53.3%、人类中位数 87.5%），用来论证检索路线的天花板。这些数字来自外部文献，
> **本仓库没有复现代码，也没有产物文件支撑**，属于引用而非本仓库实测。

---

## 2. 开源组件清单

版本一栏：`requirements.txt` 有钉版本的写钉的版本；实际装的版本从
`.venv/lib/python3.11/site-packages/*.dist-info` 读出。权重大小是 `ls -l` 的字节数。

### 2.1 生成侧（只有 Demo 1 用）

| 组件 | 版本 / 文件 | 大小 | 具体干什么 | 许可 |
| --- | --- | ---: | --- | --- |
| LivePortrait 代码 | vendored 于 `third_party/LivePortrait/` | — | 唯一的生成器。video-to-video 隐式关键点重定向 | **MIT**（`third_party/LivePortrait/LICENSE:1`） |
| `spade_generator.pth` | KwaiVGI/LivePortrait HF | 221,813,590 B | 解码器 G，出最终像素 | 同上（权重随 repo 发布） |
| `warping_module.pth` | 同上 | 182,180,086 B | 3-D 形变场 W，把 source 外观特征按新关键点扭曲 | 同上 |
| `landmark.onnx` | 同上 | 114,666,491 B | 203 点 landmark，供 crop / lip-eye ratio 计算 | 同上 |
| `motion_extractor.pth` | 同上 | 112,545,506 B | 运动提取器 M，出 21×3 的 `exp`、`R`、`scale`、`t` | 同上 |
| `appearance_feature_extractor.pth` | 同上 | 3,387,959 B | 外观特征 F（3-D 体特征） | 同上 |
| `stitching_retargeting_module.pth` | 同上 | 2,393,098 B | stitching 模块 S，把驱动后的关键点贴回 source 躯干 | 同上 |
| InsightFace `det_10g.onnx` | LivePortrait 附带 | 16,923,827 B | **仅**用于 LivePortrait 内部人脸检测（裁剪） | **非商业**（InsightFace 权重） |
| InsightFace `2d106det.onnx` | 同上 | 5,030,888 B | 同上，106 点 landmark | 同上 |
| `pykalman`（间接） | LivePortrait 依赖 | — | `third_party/LivePortrait/src/utils/filter.py:8-19` 的 `smooth()`，对驱动表情序列做 RTS Kalman 平滑 | BSD |

`pipelines/fetch_liveportrait.py:14-23` 只拉这 8 个文件（约 660 MB），显式跳过 animals 模型
（约 1.4 GB）和完整 buffalo_l 包——video-to-video 人类重定向用不到。
实测 `du -sh`：`liveportrait/` 625 MB + `insightface/` 22 MB。

### 2.2 验证侧

| 组件 | 版本 / 文件 | 大小 | 具体干什么 | 许可 |
| --- | --- | ---: | --- | --- |
| MediaPipe FaceLandmarker | `mediapipe==0.10.21`（钉版本，`requirements.txt:19`）；`models/face_landmarker.task` | 3,758,596 B | **M1 + M2 + 头部姿态的唯一来源**：478 landmark、52 blendshape、4×4 facial transformation matrix。float16/1 版本，URL 见 `expverify/landmarks.py:20-23` | Apache-2.0（README 记载） |
| OpenFace 3.0 | `openface-test==0.1.26`；`models/MTL_backbone.pth` | 101,710,914 B | **只**做 M3：8 通道 AU 激活。EfficientNet-B0 + graph AU head，读像素 | **研究许可**（`expverify/au.py:18-19`），只在离线验证器里用，不进交付物 |
| InsightFace ArcFace | `models/w600k_r50.onnx`（从 `buffalo_l.zip` v0.7 里抽出，`expverify/identity.py:29,48-66`） | 174,383,860 B | **只**做身份门槛：(a) 证明 ref/tgt 是两个人；(b) Demo 1 的双向渗漏检查 | **非商业研究**（`expverify/identity.py:12-15`）。权重缺失时退化到几何代理 `GeometricIdentity`（`identity.py:174-189`） |
| `onnxruntime` | 1.28.0 | — | 只跑 ArcFace，且**强制 CPUExecutionProvider**（`identity.py:110`） | MIT |
| PyTorch | `torch>=2.9`，实装 2.13.0；`torchvision` 0.28.0 | — | 两处：LivePortrait 推理（MPS）、OpenFace MTL 推理（`au.py:57` 自动选 `mps`） | BSD |
| OpenCV | `opencv-python` 4.11.0.86（+ `opencv-contrib-python`） | — | 解码、resize、crop/pad、HSV 直方图、convexHull、warpAffine、JPEG round-trip、`cv2.dnn.blobFromImages` | Apache-2.0 |
| NumPy | `numpy<2`，实装 1.26.4 | — | 全部数值计算 | BSD |
| SciPy | 1.13.0 | — | **只用两个函数**：`scipy.io.wavfile.read` 和 `scipy.signal.stft`（`expverify/audio.py:17-18`）。mel 滤波器组、DTW、Sakoe-Chiba band 全是自己写的 | BSD |
| imageio-ffmpeg | 0.6.0 | — | 自带 ffmpeg 二进制。裁剪定长、CREMA-D 转码、给 LivePortrait 造 PATH 软链（`pipelines/demo1_liveportrait.py:74-86`） | BSD |
| `av` | 18.1.0 | — | LivePortrait 依赖（本仓库代码未直接调用） | BSD |
| matplotlib | 3.10.1 | — | `expverify/report.py:82-151` 逐帧曲线图，Agg 后端 | PSF-like |
| tqdm / Pillow | 4.66.2 / 9.4.0 | — | 进度条 / 图像 IO | MIT / HPND |

### 2.3 数据

| 组件 | 内容 | 许可 |
| --- | --- | --- |
| CREMA-D | 91 演员 × 12 句 × 6 情绪 × 4 强度。`pipelines/fetch_cremad.py` 按单文件走 git-LFS media 端点（每条约 260 KB），不 clone 7.5 GB | **ODbL**：开放、无 EULA、无注册、允许商用（`fetch_cremad.py:6-8`） |
| LivePortrait 自带示例素材 | `assets/examples/driving/d0.mp4`（driver）、`source/s13.mp4`、`s18.mp4`、`s32.mp4` | 随 LivePortrait repo |

本仓库实际落盘的 CREMA-D 子集（`data/cremad/clips.csv`）：**431 条、12 个演员、6 个句子
（IEO / IOM / IWW / MTI / TAI / TIE）**，情绪 ANG/DIS/FEA/HAP/SAD 各 72 条 + NEU 71 条，
强度 XX 371 条 + HI 60 条。
> 注意：这跟 `fetch_cremad.py:91-93` 的默认参数（3 个句子、`--intensities HI`）不一致，
> 说明实际拉取时传了不同参数。落盘数据是权威的，脚本默认值不是。

### 2.4 显式**没有**用的东西（这些是决策，不是遗漏）

- **没有 librosa / python_speech_features**：log-mel 自己写（`audio.py:21-63`）。
- **没有 dtw / dtaidistance / fastdtw**：带 Sakoe-Chiba 约束的 DTW 自己写（`audio.py:66-107`）。
- **没有人脸分割网络**：背景相似度用"外边框环 减 膨胀后的人脸凸包"（`scene.py:25-38`），
  理由写在 `scene.py:9-11`——加一个分割网络换来的精度提升不值得再走一遍许可审查。
- **没有 img2img 扩散做风格化**：理由在 `augment.py:11-16`——扩散没有表情保持约束、没有时序模型，
  会逐帧改动嘴形和眼神，对一份以表情相等为唯一卖点的数据集是破坏性的。
- **没有换脸（inswapper / SimSwap）、没有 FLAME / pytorch3d**：README 第 202-212 行记录了理由。
- **没有任何感知/情绪 embedding 模型**（如 FaRL、SigLIP2 微调的表情度量）：README 把它列为 M4，需要 GPU，本次范围外。

---

## 3. 自创部分

以下每一项都不来自任何库，是本仓库的实际贡献。

### 3.1 rank-1 时序可辨识性（核心严格性判据）

实现：`expverify/verify.py:207-242`（判定）、`expverify/calibrate.py:198-294`（诊断与负控制）。
设计动机写在 `verify.py:1-27`。

**为什么需要它。** 绝对阈值能表达"这两张脸相差小于 ε"，表达不了"这个匹配比视频自身的逐帧表情变化还细"。
后者才是"严格相同"的可执行定义。而且阈值需要一个手调常数，rank-1 不需要。

**精确算法。** 对参考序列的每个帧 `t_ref = tr`，对齐到目标帧 `tt`：

1. **候选集**：`cand = { s ∈ [tt-W, tt+W] : tgt.ok[s] }`，`W = rank1_window = 12`
   （`verify.py:222-223`）。`|cand| < 3` 时放弃 rank-1（`verify.py:229-230`，退化为 `best = tt`，`d_far = ∞`）。
2. **最近邻**：`best = argmin_{s∈cand} d_deform(ref.deform[tr], tgt.deform[s])`（`verify.py:225-226`）。
3. **rank-1 门**：`g_rank1 = (|best - tt| ≤ tol)`，`tol = rank1_tolerance = 1`（`verify.py:239`）。
4. **远处硬负样本**：`d_far = min{ d(ref[tr], tgt[s]) : s ∈ cand, |s - tt| ≥ K }`，`K = rank1_offset_k = 3`
   （`verify.py:227-228`）。
5. **自对比**：`d_self = min( d(ref[tr], ref[tr-K]), d(ref[tr], ref[tr+K]) )`（`verify.py:233-235`）。
   这是**同一段视频自己**跨 K 帧的表情变化量。
6. **比值门**：`ratio = d_deform / d_self`；`beats_self = (ratio ≤ max_ratio)`（仅当 testable），
   `beats_far = (d_deform ≤ max_ratio · d_far)`；`g_ratio = beats_self ∧ beats_far`（`verify.py:238-242`）。
7. **可测性**：`testable = (d_self ≥ min_self_contrast)`，默认 `0.006`（`verify.py:60,237`）。

**`min_self_contrast` 是干什么的。** 第 5 步的 `d_self` 是"这段视频在 ±3 帧内变了多少"。
如果参考帧落在一个静止段里（表情保持不动），`d_self ≈ 0`，那么 `ratio = d/0 → ∞`，
而且第 2 步的 argmin 在一堆彼此相同的候选里落到哪一帧纯属噪声。**这种帧的 rank-1 检验按构造无法成功**，
不是 pair 的问题。`min_self_contrast` 就是"这段视频在这一帧附近确实在动"的门槛，
低于它的帧标为 `testable = False`。

**为什么不可测帧要排除而不是算失败。** `expverify/calibrate.py:249-251` 写得很直接：
"Frames inside a static segment are genuinely equal to their neighbours, so a ranking test cannot
succeed there by construction. Scoring them as failures understates the metric; the testable subset
is the honest number." 具体做法有两层：
- 帧级：`beats_self` 在 `testable = False` 时直接置 `True`（`verify.py:241`），即比值门对不可测帧不生效。
- pair 级：接受条件用的是 `n_pass_testable = (accepted ∧ testable).sum()`（`verify.py:281,287-289`），
  必须 ≥ `min_pass_frames = 8`；同时 `n_testable ≥ min_testable_frames = 5`，
  否则理由是 "clip too static to prove fine-grained agreement"（`verify.py:284-286`）。

**它为什么能操作化"不要差不多的"。** 一对同步视频最难的负样本本来就是时序相邻帧：
身份、姿态、光照、背景全同，只差几十毫秒肌肉运动。要求跨身份匹配赢过它们，
就是"差不多我不要"的可执行定义。

**负控制。** `calibrate.py:198-261` 的 `shift` 参数把同一个检验跑在故意错位的 pair 上，
argmin 直方图必须跟着移动，否则这个排序检验什么也没测。实测（`liveportrait.json:137-182`，
只统计可测帧）：

| 情形 | 可测帧数 | exact (offset=0) | ±1 | ±2 | \|offset\| 中位数 | 平均 offset |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aligned | 261 | 0.234 | 0.475 | 0.582 | 2.0 | −0.36 |
| shift −5 | 251 | 0.040 | 0.100 | 0.155 | 5.0 | **+3.20** |
| shift +5 | 239 | 0.025 | 0.100 | 0.163 | 5.0 | **−4.27** |

平均 offset 的符号和量级跟着人为平移走（−5 → +3.20，+5 → −4.27），负控制成立。
对齐组的 ±1 命中率 0.475 对平移组的 0.100，比值约 4.75×。

**⚠️ 一个必须说清的实情：校准后的 `max_ratio` 把"严格条款"废掉了。**
`verify.py:12-18` 描述的意图是 `ratio` 要小于 1（跨身份匹配要赢过自身时序邻帧），默认值 `0.80`
（`verify.py:59`）。但 `calibrate.py:420-423` 明确说手选 0.8 会全拒，所以改成拟合；
拟合结果是 **`max_ratio = 4.4321`**（`liveportrait.json:8`），正样本 ratio 中位数 1.4717、
平移组 2.0911、AUC 0.6399、`separated: false`、`target_precision_unreachable: true`
（`liveportrait.json:184-195`）。`max_ratio = 4.43 > 1` 意味着 `g_ratio` 已经不再要求"赢过邻帧"，
只拒绝严重错配——它在正样本上的通过率是 0.939（`liveportrait.json:220`），几乎是空门。
**真正提供严格性的是 `g_rank1`（正样本通过率 0.378）和 pair 级聚合，不是 ratio。**
这跟 `verify.py` 顶部的文档描述有出入，文档描述的是设计意图，代码执行的是校准结果。

### 3.2 per-person neutral 减法

实现：`expverify/neutral.py:39-79`（估计）、`neutral.py:100-120`（减法）、`au.py:78-87`（AU 版）。

**为什么每个指标都要相对这个人自己的中性脸算。** `neutral.py:1-13`：不做这一步，
两个描述子都会被**永久性面部结构**主导——静息眉高、眼睑形状、皱纹、面部毛发——
而 AU 和 blendshape 估计器会系统性地把这些读成"肌肉正在活动"。
于是跨身份比较测的是脸型，不是表情。这是整个验证器里单点收益最高的一步。

**怎么估的：两趟中位数**（`neutral.py:39-49`，思想来自 Baltrušaitis et al. FG 2015）：

1. 第一趟：`m1 = median(x, axis=0)`，`x` 是这个人所有有效帧的特征矩阵。
2. 若 `T ≥ 8`：算每帧到 `m1` 的 L2 距离 `d`，取最近的 `k = max(4, round(0.4·T))` 帧
（`keep_frac = 0.4`），在这个子集上再取一次中位数。

第二趟的理由：当某个强表情占了片段的很大一部分时，朴素中位数会被它带偏。

**"中位数是在什么上取的"（三个不同对象，分别取中位数）**（`neutral.py:52-79`）：

- `bs_feat`：M1 特征（30 维），两趟中位数；
- `gaze`：注视特征（4 维），两趟中位数；
- `canon`：**规范化后的 478×3 landmark**，两趟中位数（逐坐标）；
- `pose`：3 维欧拉角，**只做单趟** `np.median`（`neutral.py:76`）；
- `au`：8 维 AU，两趟中位数（`au.py:78-87`），在 `pipelines/common.py:146-149` 按人聚合。

**"一个人"的粒度在两条 track 里不同：**
- Demo 2：一个演员的**所有片段拼起来**（含 NEU 中性片段）估一个 neutral（`common.py:130-150`）。
- Demo 1：**每个 clip 自己是一个 person**（`pipelines/demo1_liveportrait.py:247-248`，
  `estimate_neutral([f], plan, k)`）。理由写在 `demo1_liveportrait.py:245-246`：
  输出继承了 source 的脸型，混用 neutral 会把 neutral 减法本来要去掉的身份偏置重新引进来。

**这带来一个真实的副作用，必须说。** Demo 1 里每段输出的 neutral 是它自己那 78 帧的两趟中位数。
由于所有输出的表情轨迹按构造相同，它们的中位数也近乎相同，于是减法会把**共享的表情基线**
一起减掉，剩下的主要是"同一个表情张量渲染到不同脸上产生的几何差"。
这解释了为什么接受的 pair 的 `med_energy` 只有 0.0088（`pairs/01_.../metrics.json`）——
减完之后大部分帧离"自己的中性"很近。这不是 bug，但它意味着 Demo 1 的 `energy` 数值
不能和 CREMA-D 的 `energy` 直接比（4.5 节会看到这正好出了问题）。

### 3.3 13 点 Umeyama 刚性规范化

实现：`expverify/descriptors.py:179-266`，全局引用装载在 `pipelines/common.py:22-40`。

**修的是什么问题。** 早先的坐标系是 3 点解析构造的 `face_frame`（`descriptors.py:157-176`，
现在只作为 bootstrap 保留）：原点取 13 个刚性点的均值，x 轴取双眼外角连线，
**y 轴由额顶点 `p[10]` 决定**。`descriptors.py:158-162` 的注释说明了它为什么脆：
额顶是一个**无纹理点**，它的定位误差会**旋转整张脸**，然后在离原点最远的下巴处放大成巨大的幻影位移。
也就是说，坐标系估计误差被伪装成了表情差异。

**`fit_rigid_reference` 做什么**（`descriptors.py:200-221`）：对刚性点集做广义 Procrustes 分析。

```
ref ← samples[0]，去均值，按 RMS 归一
重复 4 次：
    ref ← mean over s of ( c·(s @ Rᵀ) + t )，其中 (R, c, t) = umeyama(s, ref)
    ref 去均值、按 RMS 归一
最后：ref ← ref / ‖ref[EYE_OUTER_R] − ref[EYE_OUTER_L]‖   ← 眼距恰好为 1
```

关键点有两个：
- **平均到 13 个点上**，而不是从 2–3 个点解析构造轴，这是消除坐标系估计噪声的原因
  （`descriptors.py:203-205`）。
- 最后一步把**双眼外角距离归一到 1**，于是下游所有距离的单位是"该受试者自己的眼距"，
  可读且跨脸可比。

`RIGID_IDS = [33, 133, 362, 263, 168, 6, 197, 195, 4, 1, 234, 454, 10]`（`landmarks.py:35`），
13 个在表情变化下基本不动的点（眼角 ×4、鼻梁 ×5、鼻尖、颊侧 ×2、额顶）。
落盘引用是 `models/rigid_reference.npy`，实测 shape `(13, 3)`、dtype `float32`。

**`canonicalize` 做什么**（`descriptors.py:241-251`）：

```python
R, c, t = umeyama(p[RIGID_IDS], _REFERENCE)   # 13 点最小二乘相似变换
return c * (p @ R.T) + t                       # 对全部 478 点施加同一个变换
```

即：用 13 个刚性点解出一个相似变换（旋转 + 各向同性缩放 + 平移），把整张脸搬进公共坐标系。
`_REFERENCE is None` 时才退回 3 点老路（`descriptors.py:247-249`）。

**为什么"共享全局引用"是必要的。** `pipelines/common.py:26-28`：
"One reference is used for every clip in every track, which is what puts two different people's
deformation fields into a common frame." 如果每段视频各自拟合一个引用，
两个人的形变场就活在两个不同的坐标系里，`deform_distance` 里会混进坐标系之间的差；
共享引用之后，A 和 B 的形变向量可以逐点相减。

形变场定义（`neutral.py:108`）：`deform[t] = canonicalize(landmarks[t]) − neutral.canon`，
单位是眼距。

**配套的一个小修正**（`landmarks.py:63-67, 174, 194`）：MediaPipe 的 landmark 是归一化坐标
（x 除以宽、y 除以高），在非正方形帧上两个轴单位不同，任何欧氏距离都是**静默各向异性**的。
代码把它们重标到 `(x·W, y·H, z·W)`（z 的量级文档上跟 x 一致），恢复各向同性。

### 3.4 合取 AND 网关

实现：帧级 `verify.py:252-262`，pair 级 `verify.py:272-320`。

**帧级 7（或 8）个门，全部 AND**：

```python
gates = dict(g_bs, g_deform, g_gaze, g_region, g_rank1, g_ratio, g_energy)
if d_au is not None and spec.max_au is not None: gates["g_au"] = ...
accepted = all(gates.values())
```

**pair 级门**（任一不满足就写进 `reasons`，`accepted = not reasons`，`verify.py:351`）：

1. `n_testable ≥ min_testable_frames = 5`
2. `n_pass_testable ≥ min_pass_frames = 8`
3. `pass_rate ≥ min_pass_rate`（校准值 0.0658）
4. `identity_cos ≤ max_identity_cos`
5. `median(max(|Δyaw|,|Δpitch|)) ≥ min_pose_delta_deg`（`reference` spec = 8.0°）
6. `bg_hist ≤ max_bg_hist`（`reference` spec = 0.60）
7. `source_identity_cos ≥ min_source_identity_cos`（Demo 1 设为 0.45）
8. `driver_identity_cos ≤ max_driver_identity_cos`（Demo 1 设为 0.30）

**为什么是 AND 而不是加权分数。** `verify.py:24-26`："Averaging metrics would let one confident
metric mask another's rejection, which is precisely the failure mode a strict verifier exists to prevent."
一个具体场景：嘴形匹配得极好（`d_bs` 很小）而眉毛完全不同（`d_region` 很大），
加权平均会给出一个"还行"的总分并放行；AND 会因为 `g_region` 失败而拒掉。
严格验证器的任务是**拒绝有疑义的 pair，而不是给它打分裁决**。

但 AND 只有在各门真的独立时才有意义（否则是"同一个门按两次"），所以校准会算两两相关，见 3.5 和 4.3。

另外注意：`spec.max_au is not None` 才启用 `g_au`（`verify.py:261-262`），
而 `apply_to_spec` 只在校准里存在 `d_au` 时才设 `max_au`（`calibrate.py:521-522`）——
所以 M3 是"有权重就自动接线，没有就自动摘掉"，不需要改代码。

### 3.5 `d_region` 的相对归一化

实现：`expverify/verify.py:129-148`。

**原来的定义**是"最差分区的形变失配 RMS"，即 `max_r RMS(da[r] − db[r])`。
问题写在 `verify.py:132-135`：嘴唇的运动幅度最大，所以它同时主导全局 RMS（`d_deform`）
和"最差分区"，两个门说的是同一件事。**代码注释记录的相关系数是 r = 0.99**（`verify.py:133`），
**README 第 157 行记录为 0.987**。
> 这个"改造前"的数字只存在于注释和 README 里，**两个已落盘的校准 json 都是改造后的版本，
> 无法从产物文件复现 0.987**。标为文档记载、未在产物中证实。

**改造后的定义**（`verify.py:141-148`）：

```
对每个分区 r ∈ {brow, eyelid, nose, lip, jaw}：
    motion_r = max( RMS(da[r]), RMS(db[r]), min_motion )     # min_motion = 0.010
    score_r  = RMS(da[r] − db[r]) / motion_r
d_region = max_r score_r ,   worst_region = argmax_r score_r
```

其中 `RMS(v) = sqrt( mean_i ‖v_i‖² )`（`verify.py:125-126`）。

问的问题变了：不再是"哪块脸的绝对失配最大"（答案永远是嘴），而是
**"有没有哪块脸在成比例地做错事"**——这才能抓住"嘴对上了但眉毛没对上"。
`min_motion = 0.010` 的地板防止一个几乎不动的分区靠噪声造出一个巨大的比值。

**改造后的实测相关系数**（`out/calibration/liveportrait.json:125-136`，正负样本合池计算）：

| 指标对 | Pearson r |
| --- | ---: |
| `d_bs ~ d_deform` | **+0.482** |
| `d_deform ~ d_gaze` | +0.261 |
| `d_au ~ d_deform` | +0.250 |
| `d_gaze ~ d_region` | +0.222 |
| **`d_deform ~ d_region`** | **+0.194** |
| `d_bs ~ d_gaze` | −0.148 |
| `d_au ~ d_bs` | +0.093 |
| `d_au ~ d_region` | −0.052 |
| `d_au ~ d_gaze` | +0.042 |
| `d_bs ~ d_region` | +0.042 |

`d_deform ~ d_region` 从注释记载的 0.99 降到 **0.194**，而 AUC 反而升了：
`d_region` 改造后 +3f 0.574 / +5f 0.631 / +10f 0.743（`liveportrait.json:98-101`），
高于同表里的 `d_deform`（0.553 / 0.590 / 0.666）。
> README 第 160 行给的"改造前"AUC 是 0.556 → 0.574 和 0.664 → 0.743。改造后的两个数
> （0.574、0.743）与 json 完全一致；改造前的两个数（0.556、0.664）**不在任何产物文件里**。
> 它们跟 `d_deform` 自己的 AUC（0.553、0.666）非常接近，这与"改造前 d_region 与 d_deform 几乎共线"
> 的说法自洽，但这只是推断，不是可复现的测量。

另外注意 `augment.json:125-135` 里同样的相关矩阵是另一组值（`d_bs~d_deform` +0.748、
`d_deform~d_region` +0.423）。相关性依赖正样本源；跨身份正样本上的那一组才是生产网关的依据。

### 3.6 paired-negative 校准

实现：`expverify/calibrate.py:141-160`（成对负样本）、`163-179`（同段负样本，仅作诊断）、
设计理由 `calibrate.py:16-24`。

**三个分布**（`calibrate.py:5-14`）：

| 分布 | 定义 | 函数 |
| --- | --- | --- |
| positives | `d(a_t, b_t)`，`b` 是 `a` 的外观增广孪生体（或同 driver 的另一路输出） | `collect_positives`（`calibrate.py:128-138`） |
| hard negatives | `d(a_t, b_{t+k})`，**同一对 pair 平移 k 帧**，`k ∈ {3,5,10}` | `collect_paired_negatives`（`calibrate.py:141-160`） |
| null | 不同人的随机帧对，4000 个样本 | `collect_null`（`calibrate.py:182-195`） |

**为什么硬负样本必须和正样本在同一个噪声机制里。** 一个看起来等价但错误的做法是：
在**单次抽取**里取 `d(t, t+k)` 当负样本。`calibrate.py:16-24` 解释了为什么不行：
FaceLandmarker 跑在 **VIDEO 模式**（`landmarks.py:158`），它跨帧携带跟踪状态，
所以**同一次抽取内部的 landmark 是被时序平滑过的，噪声相关**；
而正样本比较的是两次**独立**抽取，携带完整的独立噪声。
把这两个分布放在一起比，会系统性地**低估每一个负样本**，让一个本来不错的指标看起来比抛硬币还差。

**这件事在产物文件里是可验证的。** `report.within_run_median` 记录了同段负样本（k=3）的中位数
（`liveportrait.json:17-22`），跟正样本中位数并列：

| 指标 | 同段负样本(+3f)中位数 | 正样本中位数 | 成对硬负样本(+3f)中位数 |
| --- | ---: | ---: | ---: |
| `d_bs` | **0.003532** | 0.016729 | 0.016991 |
| `d_deform` | **0.011588** | 0.020419 | 0.023679 |
| `d_gaze` | **0.022208** | 0.150246 | 0.206112 |
| `d_region` | **0.551271** | 1.090101 | 1.152182 |

四个指标的同段负样本中位数**全部小于正样本中位数**，而且小 2–7 倍。
由于判据是"距离小 ⇒ 判为正"，负样本系统性地比正样本更小，必然导致 **AUC < 0.5**——
这正是当初校准报出低于随机的 AUC 的机制。换成成对硬负样本后，负样本中位数
（0.0170 / 0.0237 / 0.2061 / 1.1522）全部**大于**正样本中位数，方向恢复正常，
AUC 变成 0.505–0.645。这些数字就是这个设计决策的证据。

**成对构造把什么变量控住了。** 对每个正样本对 `(a, b)`，取 `d(a_t, b_{t+k})`：
同一个人、同一个背景、同一个光照、同一个外观变换、同样是两次独立抽取——
唯一剩下的差异就是 k 帧的表情变化，也就是我们要测的那个量。

**阈值拟合按 precision 而不是 accuracy**（`calibrate.py:26-30, 297-343`）：
在 `pos ∪ neg` 的取值网格上（超过 4000 个候选时抽 4000 个分位点）搜索
**满足 `precision ≥ 0.95` 且 `recall ≥ 0.5` 的最大阈值**。
`min_recall` 不是修饰："a threshold with 1.6% recall is not strict, it is broken"（`calibrate.py:303-306`）。
两个分布重叠严重时找不到可用工作点，函数**退回正样本 q90 并在 report 里打上
`target_precision_unreachable: true`**，让失败可见而不是静默清空数据集。

### 3.7 两趟人脸裁剪与 `locate_face` 由粗到细分块搜索

实现：`expverify/landmarks.py:247-282`（`run_face_crop`）、`212-245`（`locate_face`）、
`130-143`（`crop_frames`）、`319-334`（`landmarks_in_original`）。

**两趟裁剪**（`run_face_crop`）：

1. **第一趟（probe）**：在 `frames[::max(1, T//24)]`（约 24 帧的稀疏子集）上跑一次检测。
2. 若 probe 有有效帧，用这些帧的 landmark 算裁剪框；若全无效，退到 `locate_face` 在
   3 个等距帧上做分块搜索（`landmarks.py:262-268`）；仍失败则整帧跑（`landmarks.py:266-267`）。
3. **框是整段一个、取中位数**：
   `cx = median(每帧 landmark x 均值)`、`cy` 同理，
   `half = median( max(x 跨度, y 跨度) ) · margin / 2`，`margin = 1.9`，
   `side = round(half·2)`（`landmarks.py:270-278`）。
4. **第二趟**：把这个固定方框从每帧切出来（越界用 `BORDER_REPLICATE` 补边），
   `INTER_CUBIC` 放大到 `crop_size = 512`，再跑一次完整抽取（`landmarks.py:279-281`）。

**为什么框要固定、要取中位数。** `landmarks.py:250-257`：逐帧重新检测框会引入**框自己的抖动**，
那个抖动会以"表情变化"的形式进入形变场。整段一个框把这个噪声源彻底移除。

**`locate_face` 解决什么。** MediaPipe 发布的是**短距（short-range）BlazeFace** 检测器，
需要人脸占输入的相当比例。全身镜头里头部只占画面高度约 7%，整帧检测**直接返回空**——
不是报错，是安静地给出一整段全无效（`landmarks.py:214-221`）。搜索策略：

- 三个尺度 `levels = (1.0, 0.5, 0.28)`，`side = min(H,W)·frac`，步长 `side//2`（50% 重叠）；
- 每个 tile 放大到 `probe_size = 448` 再送检测；
- tile 按到**先验位置 `(W/2, H/3)`** 的距离排序（竖幅人像里头部通常在上中部），
  通常几个 tile 就命中，省掉大量检测调用（`landmarks.py:226-235`）；
- 命中后把 landmark 从 tile 坐标映回原帧：`x·(side/probe_size) + box_x`（`landmarks.py:240-243`）。

**为什么"测量分辨率"就是可达粒度的天花板。** `landmarks.py:251-256` 说得很具体：
landmark 精度随人脸占的像素数变化，而这个精度是**整个验证器的粒度上限**；
在 480×360 的录播素材上人脸约 150 px，**单是测量噪声就超过好几帧的真实表情变化**。
Demo 1 的 summary 因此把每段的 `face px` 列成表，并声明"一对里较小的那侧是瓶颈"
（`demo1_liveportrait.py:337-341`）。

> **一处需要修正的表述**：summary.md 里这一列写的是 "face height in decoded pixels"，
> 但代码里 `face_px = f.crop_box[2]`（`demo1_liveportrait.py:235, 317-318`）是**裁剪框边长**，
> 而框边长 = 人脸跨度 × `margin(1.9)`。所以 s13 的 "120 px" 对应的真实人脸跨度约
> 120 / 1.9 ≈ **63 px**，driver 的 "526 px" 对应约 277 px。方向和排序没错，绝对值被高估了 1.9 倍。

**坐标空间的往返**（`landmarks.py:319-334`）：所有几何计算在 512×512 裁剪空间里做，
但 ArcFace 对齐和背景比较消费的是**原始解码帧**。`landmarks_in_original` 把 landmark 映回去：
`x·(side/max(W,1)) + x0`。`landmarks.py:322-324` 特别说明：
喂错坐标空间**不会报错，只会静默产出垃圾**。

### 3.8 其余自创小件（各自解决一个具体的静默失败）

| 件 | 位置 | 解决什么 |
| --- | --- | --- |
| blendshape **按名寻址** | `descriptors.py:51-55`、`landmarks.py:3-6` | MediaPipe 在 0 号槽发 `_neutral` 且不发 `tongueOut`，所以每个 ARKit 名字相对标准 ARKit-52 顺序都偏移了。按下标取通道会静默取错 |
| **死通道剔除** | `descriptors.py:25-36` | `noseSneerL/R`、`mouthFrownL/R`、`jawForward`、`cheekSquintL/R`、`cheekPuff` 从不激发（8 个）；`eyeWideL/R`、`mouthDimpleL/R`、`mouthPucker` 不可靠（5 个）。死通道给 L1 距离贡献纯噪声，过激通道制造假差异 |
| **mean + \|L−R\| 编码** | `descriptors.py:114-118, 136-142` | 左右均值携带表情，`\|L−R\|` 携带**不对称性**——这是真实的细粒度线索，对称化摘要会把它抹掉 |
| **眨眼掩蔽 eyeSquint** | `descriptors.py:37-41, 132-134` | `eyeSquint*` 每次眨眼都激发；用 `eyeBlink* > 0.3` 把它置零而不是整通道丢掉，因为眼睛真开着的时候它是有信息的 |
| `similarity_2d` 精确最小二乘 | `identity.py:78-96` | 不用 `cv2.estimateAffinePartial2D`：它的鲁棒估计在随机子集上拟合，只有 5 个点时挑到退化子集的概率很高。轻微跑偏**不抛异常**，只是把所有 embedding 拉向平均脸，抬高不同人之间的余弦，**静默禁用身份门槛** |
| `identity_separation` 自检 | `identity.py:143-171` | 一个门槛只有真的可分才值得有。在**实际素材上**测同人 / 不同人余弦分布，并给出 EER 阈值，而不是照抄论文常数 |
| 无模型背景相似度 | `scene.py:25-65` | 外边框环（margin 0.15）减去膨胀后的人脸凸包，HSV 16×8×8 直方图相关 + dHash。零依赖、零许可问题 |
| 自写 log-mel + 带约束 DTW | `audio.py:21-107` | 见 6.3 |
| `redundancy()` | `au.py:97-108` | 把"M3 独立"从假设变成可测量的断言 |
| `fit_threshold` 退化可见化 | `calibrate.py:297-343` | 见 3.6 末 |
| `calibrate_pair_level` | `calibrate.py:461-510` | 用**真实的 `verify_pair`** 而不是代理指标去拟合 `min_pass_rate`，并给出每个门在正样本上的通过率，从而知道**哪个门是瓶颈**（否则通过率低了无法归因，直觉反应是把所有门一起放松） |
| 保留被拒 pair | `report.py:1-6, 159-164` | 被拒的那一半是让通过率可审计的东西，而且它们本身就是已校准的硬负样本 |

---

## 4. 指标定义

### 4.1 总表

所有阈值取自 `out/calibration/liveportrait.json`（两个 demo 的默认校准，
`demo1_liveportrait.py:159`、`demo2_cremad.py:48` 都指向 `out/calibration/liveportrait`）。
`✔` = 帧级门，`◆` = pair 级门。

| 指标 | 层级 | 测什么 | 精确计算 | 阈值（现行） | 方向 |
| --- | :-: | --- | --- | ---: | --- |
| `d_bs` (M1) | ✔ | 精选 blendshape 通道上的加权 L1 | `Σ_j w_j·\|a_j − b_j\|`，30 维；`w_j ∝ 1/σ_j`，`Σw = 1`（`verify.py:103-111`） | **≤ 0.047431** | 越小越好 |
| `d_deform` (M2) | ✔ | landmark 形变场失配 | `sqrt( mean_{i∈134} ‖da_i − db_i‖² )`，眼距单位（`verify.py:114-116`） | **≤ 0.034109** | 越小越好 |
| `d_region` | ✔ | 最差分区的**相对**失配 | `max_r RMS(da_r−db_r) / max(RMS(da_r), RMS(db_r), 0.010)`，5 个分区（`verify.py:129-148`） | **≤ 1.352156** | 越小越好 |
| `d_gaze` | ✔ | 有符号注视通道差 | `mean\|a − b\|`，4 维（`verify.py:119-122`） | **≤ 0.310878** | 越小越好 |
| `d_au` (M3) | ✔ | OpenFace AU 激活差 | `mean\|a − b\|`，8 维（`au.py:90-94`） | **≤ 0.219957** | 越小越好 |
| `energy` | ✔ | 表现力（离自己中性多远） | `min( sqrt(mean_{i∈134}‖da_i‖²), 同 tgt )`（`descriptors.py:281-284`、`verify.py:244`） | **≥ 0.023650** | 越大越好 |
| `ratio` | ✔ | 跨身份距离 / 自身 ±3 帧距离 | `d_deform / d_self`；另有 `d_deform ≤ max_ratio·d_far`（`verify.py:238-242`） | **≤ 4.432121** | 越小越好 |
| rank-1 | ✔ | 最近邻是否就是对齐帧 | `\|argmin_{s∈[tt−12,tt+12]} d_deform − tt\| ≤ 1`（`verify.py:221-239`） | 容差 **±1** | 布尔 |
| `rank1_rate` | 报告 | 上一行在整段上的命中率 | `mean(g_rank1)`（`verify.py:334`） | **无 pair 级门** | 仅记录 |
| `pass_rate` | ◆ | 逐帧通过率 | `n_pass / n_eval`，`n_eval` = 双侧同时有效的对齐帧数 | **≥ 0.065789** | 越大越好 |
| `n_pass_testable` | ◆ | 通过且可测的帧数 | `(accepted ∧ testable).sum()` | **≥ 8** | 越大越好 |
| `n_testable` | ◆ | 可测帧数 | `(d_self ≥ 0.006).sum()` | **≥ 5** | 越大越好 |
| `identity_cos` | ◆ | 两人是不是同一个人 | 至多 12 帧 ArcFace embedding 取均值再归一，然后余弦（`identity.py:119-134`） | **≤ 0.25** | 越小越好 |
| pose delta | ◆ | 头部运动够不够不同 | `median_t max(\|Δyaw\|, \|Δpitch\|)`，角度环绕取 `min(d, 360−d)`（`scene.py:19-22`、`verify.py:302`） | **≥ 8.0°**（`reference`；`editing` 关闭） | 越大越好 |
| `bg_hist` | ◆ | 背景够不够不同 | 3 对采样帧的 HSV 16×8×8 直方图相关（`HISTCMP_CORREL`）取中位数（`scene.py:54-65`） | **≤ 0.60**（`reference`；`editing` 关闭） | 越小越好 |
| `source_identity_cos` | ◆ | 输出还是不是自己的 source | `min` over 两侧的 `cos(out, own source)` | **≥ 0.45**（Demo 1 硬写，`demo1_liveportrait.py:261`） | 越大越好 |
| `driver_identity_cos` | ◆ | driver 脸型是否渗漏 | `max` over 两侧的 `cos(out, driver)` | **≤ 0.30**（`demo1_liveportrait.py:262`） | 越小越好 |
| `dtw_cost` | ◆ | 音频对齐质量（Demo 2） | DTW 平均代价（`audio.py:106`） | **≤ 0.55**（CLI 默认，`demo2_cremad.py:52`） | 越小越好 |

阈值来源分工（`calibrate.py:513-528` 的 `apply_to_spec`）：
`d_bs / d_deform / d_gaze / d_region / d_au / max_ratio / min_energy / min_pass_rate` 由校准覆盖；
`rank1_window / rank1_offset_k / rank1_tolerance / min_self_contrast / min_pass_frames /
min_testable_frames / max_identity_cos / min_pose_delta_deg / max_bg_hist` **不被校准覆盖**，
是 `DatasetSpec` 里的固定契约（`verify.py:42-77`）。

### 4.2 需要展开的几个

**`d_bs` 的权重是怎么来的。** `feature_scales`（`calibrate.py:106-125`）在语料上采样 20000 个
**跨身份随机帧对**，算每个 M1 特征的平均绝对差 `σ_j`（下限 1e-4），存为 `bs_sigma`
（实测 shape `(30,)`）。然后 `w_j = (1/σ_j) / Σ_k(1/σ_k)`（`verify.py:103-107`）。
目的是让动态范围天然大的通道不主宰 L1，也让阈值跨语料可搬。
**M1 维度 = 30**：52 个 blendshape 去掉 `_neutral`(1) + 死通道(8) + 不可靠通道(5) + `eyeLook*`(8) = 30 个可用通道，
其中 11 组左右对，每组出 `mean` 和 `asym` 两个特征（22），加 8 个单通道，共 30 维。

**134 个"表情点"。** `EXPRESSION_POINTS`（`descriptors.py:48`）是 5 个分区的并集：
brow 20 + eyelid 32 + nose 21 + lip 40 + jaw 21 = **134 个点，无重叠**（实测验证）。
`d_deform` 和 `energy` 都只在这 134 个点上算，不含 478 点里那些对表情无信息的点。

**`d_gaze`。** 每只眼两个有符号通道：水平 `eyeLookOut − eyeLookIn`、垂直 `eyeLookUp − eyeLookDown`
（`descriptors.py:145-154`），两眼共 4 维。注意 `eyeLook*` 被从 M1 里排除（`descriptors.py:88`），
所以注视是**独立的一路**，不会被算进 `d_bs`。

**`energy` 为什么必要，以及它的门为什么装在 `min` 上。** 近中性帧彼此天然匹配，
接受它们会抬高所有数字而不提供任何信息（`verify.py:62-63`）。门装在 `min(ref, tgt)` 上
（`verify.py:244`）：两侧都必须够有表情。`calibrate.py:432-436` 指出这带来的代价：
"a percentile p costs noticeably more than (1 − p) of frames"，
所以它被明确定性为**产量/质量旋钮，不是判别力旋钮**，并在 report 里单独记录。

### 4.3 M1 / M2 / M3 为什么必须互相独立，冗余为什么是缺陷

`expverify/au.py:3-9` 把这件事说得最清楚：**M1 和 M2 同源**——
MediaPipe 的 blendshape head 字面上消费 146 个 2-D landmark 坐标，
而 M2 就是那套 landmark 的几何。所以它们**可以一起错**，
一个由两个高相关指标组成的合取门远不如它看起来那么有力。

M3 是不同架构（EfficientNet-B0 + graph AU head）、不同训练数据、**读像素而不是读 landmark 几何**，
所以它的误差与前两者不相关。`redundancy()`（`au.py:97-108`）把这个断言变成测量。
实测（`liveportrait.json:126-129`）：`d_au` 与三个几何指标的相关系数分别是
`d_deform` +0.250、`d_bs` +0.093、`d_gaze` +0.042、`d_region` −0.052，全部 ≤ 0.25。
**M3 确实是独立的一路。**

反过来，`d_bs ~ d_deform` 的 +0.482 是本仓库里最高的一对，这是同源的直接后果，
也是 `d_region` 必须重定义（3.5 节）的原因：如果 `d_region` 还跟 `d_deform` 0.99 相关，
七个门里实际上只有五个在起作用。

**M3 的一个范围修正**（`au.py:12-16`，值得单独说）：OpenFace 3.0 发布的 multitask head
输出的是 **8 个 AU 通道的 logits**，不是常被误认为的 12 维 0–5 FACS 强度向量。
代码因此把它当成一个**无名的 8 维 AU 激活描述子**——照样做 neutral 减法、照样进门，
但**不能拿去当 FACS 强度交给标注员**。

### 4.4 ⚠️ 必须报的一件事：五个距离阈值全部是退化回退值

`out/calibration/liveportrait.json` 里，`metrics` 下的五个指标**每一个**都带着：

```
"separated": false,
"target_precision_unreachable": true,
"recall": 0.8982683982683982
```

`recall = 0.8983 = 415/462` 对所有五个指标完全相同，说明它们全部走了
`calibrate.py:335-342` 的回退分支：**阈值 = 正样本分布的 q90**。
也就是说 `d_bs ≤ 0.0474`、`d_deform ≤ 0.0341`、`d_gaze ≤ 0.3109`、`d_region ≤ 1.3522`、
`d_au ≤ 0.2200` **不是"能以 95% precision 区分同表情与差 3 帧"的阈值**，
而是"放过 90% 的已知真阳性"的阈值。它们的作用是拦掉粗差，不是提供细粒度判别。

细粒度判别全部来自：`g_rank1`（正样本通过率 0.378）、`g_energy`（0.442）、
以及 pair 级聚合（AUC 0.993）。`augment.json` 的情况完全一样
（五个指标同为 `recall = 0.8996`、`target_precision_unreachable: true`）。

这不是把结论说难听，这是把结论说准确：**这套指标目前的定量能力是"pair 级可证明、帧级不可证明"。**

### 4.5 ⚠️ 第二件事：`min_energy` 是跨语料搬过来的

`calibrate.py:437-441`：`min_energy` = 语料 energy 的 35% 分位数。
但**这里的"语料"永远是 CREMA-D**（`descs = corpus.desc_list()`，语料在
`pipelines/calibrate.py:128-144` 构建，与 `--positives` 选什么无关）。
证据：两份校准文件里 `min_energy` 的值**逐位相同**——
`0.023649911954998968`（`liveportrait.json:9` 与 `augment.json:9`），
`energy.corpus_median` 也相同（0.029216665774583817）。

然后这个来自 CREMA-D 的阈值被套到 LivePortrait 输出上，而后者的 energy 明显更低：
接受的 pair 01 的 `med_energy` 只有 **0.008772**（`pairs/01_.../metrics.json`），
低于阈值 0.02365，于是 `g_energy` 在这对上的通过率只有 **0.423**。
这就是 `g_energy` 成为第二瓶颈门（0.442）的原因，而且这个瓶颈**有一部分是跨语料阈值搬运的产物，
不是数据质量问题**。

---

## 5. 端到端流程 Demo 1（生产路线）

入口：`pipelines/demo1_liveportrait.py`。README 给出的调用是
`.venv/bin/python -m pipelines.demo1_liveportrait --au`。

### 5.1 步骤 0：默认素材与长度锁定

默认 driver 是 `third_party/LivePortrait/assets/examples/driving/d0.mp4`，
默认 3 个 source 是 `assets/examples/source/{s13,s18,s32}.mp4`（`demo1_liveportrait.py:151-156`）。

```python
n = n_frames_of(driver_src)                       # 实测 78
too_short = [s for s in sources_src if n_frames_of(s) < n]
if too_short: raise SystemExit(...)               # demo1:178-180
driver  = trim(driver_src, work/f"D_{stem}.mp4", n)
sources = [trim(s, work/f"S_{stem}.mp4", n, 720) for s in sources_src]
```

`trim`（`demo1_liveportrait.py:89-105`）的实际 ffmpeg 命令：

```
ffmpeg -y -loglevel error -i <src> -frames:v <n> \
       [-vf scale='min(720,iw)':-2] \
       -c:v libx264 -crf 16 -pix_fmt yuv420p -an <dst>
```

然后**回读校验** `n_frames_of(dst) == n`，不等就抛异常（`demo1:102-104`）。

**为什么长度锁定是正确性要求而不是优化。** 上游 `n_frames = min(len(source_rgb_lst), driving_n_frames)`
（`third_party/LivePortrait/src/live_portrait_pipeline.py:147`），
而驱动表情序列会被 **Kalman 平滑**（`live_portrait_pipeline.py:229` 调
`src/utils/filter.py:8-19` 的 `smooth()`，用的是 `pykalman` 的 RTS 平滑器 `kf.smooth`）。
RTS 是**全局**平滑器：序列长度变了，**每一帧**的平滑输出都会变。
所以两个不同长度的 source 会各自拿到一条**不同的**平滑表情序列，
"完全相同"就退化成"几乎相同"。把所有 source 预先裁到与 driver 完全等长，这个退化就不存在。
`demo1_liveportrait.py:19-22` 记录的正是这条。

### 5.2 步骤 1：LivePortrait 调用

`run_liveportrait`（`demo1_liveportrait.py:118-146`）拼出的命令（`cwd = third_party/LivePortrait`）：

```
<python> inference.py \
    -s <abs path source> -d <abs path driver> -o <out_dir> \
    --animation-region exp \
    {--no-flag-relative-motion | --flag-relative-motion} \
    --source-max-dim 720 \
    --flag-pasteback
```

环境上有一个必要的补丁：LivePortrait 的入口在 `inference.py:45-46` 会经
`fast_check_ffmpeg()`（`inference.py:21-26`）跑一次 `ffmpeg -version`，失败就直接 `ImportError`。`child_env()`（`demo1_liveportrait.py:74-86`）
在 `out/work/bin/ffmpeg` 造一个指向 `imageio_ffmpeg.get_ffmpeg_exe()` 的软链并加进 PATH，
于是不需要 Homebrew。（`ffprobe` 只被一个会退化为 False 的音频探测用到，可以缺席。）

各 flag 的含义：

- `--animation-region exp`：只驱动**表情**这一路。上游 `animation_region` 默认是 `"all"`
  （`src/config/argument_config.py:37`），`all` 会连 `R`（头部旋转）、`scale`、`t`（平移）
  一起从 driver 拿——那就把"各自保留自己的头部姿态"破坏了。选 `exp` 之后
  `R_new = R_s`、`scale_new = x_s_info['scale']`、`t_new = x_s_info['t']`
  （`live_portrait_pipeline.py:366, 380, 384`），三者全部来自 source。
- `--no-flag-relative-motion`：即 `flag_relative_motion = False`（上游默认 True，
  `argument_config.py:30`）。这是"absolute 模式"，见 5.3。
- `--flag-pasteback`：把驱动后的人脸贴回 source 自己的整帧。这是**差异化背景得以保留的原因**
  （`demo1_liveportrait.py:131-133`）。
- `--source-max-dim 720`：source 长边上限。

实测速度（README）约 0.9–1.1 s/帧（M3 Pro）。

### 5.3 步骤 2：中心论断的源码证明

**absolute + exp + source 是视频**这条路径上，表情张量的构建（`live_portrait_pipeline.py:226-229`）：

```python
226|            else:
227|                if flag_is_driving_video:
228|                    x_d_exp_lst = [driving_template_dct['motion'][i]['exp'] for i in range(n_frames)]
229|                    x_d_exp_lst_smooth = smooth(x_d_exp_lst, source_template_dct['motion'][0]['exp'].shape, device, inf_cfg.driving_smooth_observation_variance)
```

**第 228 行右侧完全不含 `source_template_dct`。** 它是驱动模板 `exp` 的逐帧直接拷贝。
第 229 行的 `smooth()` 只用 source 的 `.shape`（一个形状元组）来 reshape，不用 source 的数值。
所以只要 `n_frames` 相同（这就是 5.1 长度锁定的作用），
**每个 source 拿到的 `x_d_exp_lst_smooth` 是逐元素相同的同一条序列。**

对照的 relative 分支（`live_portrait_pipeline.py:212-215`）：

```python
212|            if inf_cfg.flag_relative_motion:
213|                if flag_is_driving_video:
214|                    x_d_exp_lst = [source_template_dct['motion'][i]['exp'] + driving_template_dct['motion'][i]['exp'] - driving_template_dct['motion'][0]['exp'] for i in range(n_frames)]
```

第 214 行第一项就是 `source_template_dct['motion'][i]['exp']`：**显式的 source 相关项**。
所以 relative 模式的输出**按构造就不该表情相同**。这也是为什么 relative 是**对照组而不是备选方案**：
它存在的意义是证明验证器的判决跟着构造走，而不是跟着文件名走。

这条序列怎么写进最终关键点（absolute 分支，`live_portrait_pipeline.py:362-384`）：

```python
319|            delta_new = x_s_info['exp'].clone()          # ← 起点是 source 自己的 exp
...
367|                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "exp":
368|                    for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
369|                        delta_new[:, idx, :] = x_d_exp_lst_smooth[i][idx, :] if flag_is_source_video else ...
370|                    delta_new[:, 3:5, 1] = x_d_exp_lst_smooth[i][3:5, 1] if flag_is_source_video else ...
371|                    delta_new[:, 5, 2]   = x_d_exp_lst_smooth[i][5, 2]   if flag_is_source_video else ...
372|                    delta_new[:, 8, 2]   = x_d_exp_lst_smooth[i][8, 2]   if flag_is_source_video else ...
373|                    delta_new[:, 9, 1:]  = x_d_exp_lst_smooth[i][9, 1:]  if flag_is_source_video else ...
```

**必须精确说清覆盖范围，因为"整个表情张量被拷过去"是不准确的。**
`exp` 的形状是 `(1, 21, 3)`，共 63 个数。上面五行覆盖了：

| 写入 | 覆盖的数值个数 |
| --- | ---: |
| 13 个关键点（1,2,6,11–20）的全部 3 个坐标 | 39 |
| 关键点 3、4 的 y | 2 |
| 关键点 5 的 z | 1 |
| 关键点 8 的 z | 1 |
| 关键点 9 的 y、z | 2 |
| **合计来自 driving template** | **45 / 63** |

剩下 **18 / 63** 保留 source 自己的值：关键点 0、7、10 的全部 3 个坐标（9 个），
关键点 3、4 的 x 和 z（4 个），关键点 5 的 x、y（2 个），关键点 8 的 x、y（2 个），关键点 9 的 x（1 个）。
按"完全没被覆盖的关键点"数是 8 个（0,3,4,5,7,8,9,10），这与 README 第 115 行
"21 个关键点里有 8 个不被覆盖，身份不被抹掉"一致。

**所以准确的表述是：驱动的那 45 个表情分量在所有 source 上逐元素相同，另 18 个分量保留身份。**
这仍然完全支持"表情由构造保证相同"这个论断（被驱动的子集是同一个张量的同一份拷贝，
且这个子集是 LivePortrait 定义的"表情区域"），但它不是"整个 exp 张量都一样"。

之后关键点合成（`live_portrait_pipeline.py:386-387`）：

```python
386|            t_new[..., 2].fill_(0)  # zero tz
387|            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new
```

`exp` 模式下 `scale_new / R_new / t_new` 全部来自 source，`x_c_s` 是 source 的规范关键点。
再经过 stitching（`flag_stitching` 默认 True，`argument_config.py:29`）：

```python
411|                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
```

`stitching` 是一个**吃 `x_s` 的学习模块**，所以最终关键点在数值上仍然依赖 source。
最后 `driving_multiplier = 1.0`（`argument_config.py:34`）让第 439 行
`x_d_i_new = x_s + (x_d_i_new - x_s) * 1.0` 成为恒等。

**诚实的边界**：相同的是**表情增量的驱动子集**；最终渲染出的几何和像素当然依赖 source
（这正是我们要的：不同的人）。`pairs/README.md` 第 20-21 行说"表情参数逐帧完全相同"，
在"驱动子集"这个限定下成立，去掉限定就过强了。

顺便确认两个可能干扰的默认值都是安全的：
`flag_normalize_lip` 在 CLI 层默认 **False**（`argument_config.py:25`，而 `InferenceConfig` 是 True，
`inference.py:53` 的 `partial_fields` 用 CLI 的值覆盖），而且它的所有分支都要求
`inf_cfg.flag_relative_motion`（`live_portrait_pipeline.py:259, 288`），absolute 模式下不生效；
`driving_option = "expression-friendly"` 那条修正路径要求 `not flag_is_source_video`
（`live_portrait_pipeline.py:389`），source 是视频时不生效。

### 5.4 步骤 3：特征抽取

`demo1_liveportrait.py:206-237`。参与抽取的 clip 集合：

- `"D"` → 裁剪后的 driver
- `"src:S_s13"` / `"src:S_s18"` / `"src:S_s32"` → 裁剪后的原始 source（只用于身份对照）
- `"absolute:S_s13"` … / `"relative:S_s13"` … → 6 段生成结果

每段：`read_video(path, resize_long=1024)`（`--read-long` 默认 1024，
`demo1_liveportrait.py:166-167`）→ `extractor.run_face_crop(fr, fps, name=key)`（3.7 节）。
少于 5 个有效帧就丢弃并打印（`demo1:230-231`）。`--au` 打开时同步跑 M3
（`au_for_feats`，`pipelines/common.py:43-51`，**从 `crop_box` 重建同一块裁剪**，
保证 M3 和 M1/M2 读的是同一批像素）。

理由（`demo1_liveportrait.py:221-223`）：source 的取景是任意的——s13 是全身镜头、
头部只占画面高度约 7%——所以"解码得大一点再裁"，而不是"解码得小一点然后祈祷"。

### 5.5 步骤 4：刚性引用、通道计划、neutral

```python
242|    ensure_reference(list(feats.values()))                      # 装载/拟合 13 点共享引用
243|    plan = build_channel_plan(feats["D"].bs_names)              # 30 维 M1 布局
247|    descs = {k: describe(f, plan, estimate_neutral([f], plan, k), au=aus.get(k))
248|             for k, f in feats.items()}                         # 每段一个 neutral
```

`ensure_reference`（`common.py:22-40`）优先加载 `models/rigid_reference.npy`，
不存在才用当前 feats 拟合并落盘。**一次拟合、全局复用**是不同人的形变场可比的前提。

### 5.6 步骤 5：装载校准、身份 embedding

```python
253|        calib = Calibration.load(args.calibration)     # out/calibration/liveportrait
254|        spec  = apply_to_spec(spec, calib)             # spec.name → "reference+calibrated"
255|        weights = calib.weights()                      # bs_sigma → 30 维逆尺度权重
261|    spec.min_source_identity_cos = 0.45
262|    spec.max_driver_identity_cos = 0.30
270|    lm_orig = {k: landmarks_in_original(f) for k, f in feats.items()}
273|        emb[k] = arcface.embed_video(frames[k], lm_orig[k], f.ok)
```

注意第 270 行：ArcFace 必须吃**原帧坐标**（3.7 节末）。
校准缺失时打印 "no calibration; using spec defaults" 然后照常跑——**生成本身不依赖阈值**，
这也是 README 第 94-96 行描述的鸡蛋问题的解法（先生成、再校准、再 `--skip-generation` 重判）。

### 5.7 步骤 6：pair 枚举与验证

每个 variant 内部（`demo1_liveportrait.py:290-297`）：

```python
pairs  = [(a, b, "gen-gen")  for a, b in itertools.combinations(keys, 2)]   # C(3,2) = 3
pairs += [("D", k, "real-gen") for k in keys]                               # 3
```

共 6 对 / variant，两个 variant 共 12 行 manifest。
`real-gen` 的意义（`demo1_liveportrait.py:295-296`）：这类 pair 的 ref 侧是**真人实拍**，
免费缓解"两侧都是生成视频"的分布偏置——这就是三元组设计（D / A′ / B′）的具体落地。

每对调 `verify_pair`（`demo1_liveportrait.py:305-319`），传入：
`identity_cos`、`pose_delta`（逐帧 3 轴角度差）、`bg_hist`（3 对采样帧直方图相关的中位数，
`demo1:281-288`）、`source_identity_cos = min(两侧 cos(out, 自己 source))`、
`driver_identity_cos = max(两侧 cos(out, D))`（**只对 gen-gen 计算**，`demo1:312-315`）、
以及 `extra = {variant, kind, face_px}`。

`align` 不传，于是走默认恒等映射 `[(0,0),(1,1),…]`（`verify.py:202-204`）——
Demo 1 的 pair 按构造是帧同步的。

### 5.8 步骤 7：产物

`out/demo1_liveportrait/` 下：`manifest.jsonl`（12 行，含逐帧数组）、
`manifest.schema.md`（`report.py:159-206` 的固定 schema）、`summary.md`、
最多 8 张逐对曲线图（`report.py:82-151`：ref/tgt 抽帧缩略图 +
`d_deform` vs `d_self` 曲线 + `ratio` 曲线 + rank-1 offset 阶梯图，灰色带标出不可测段）。

---

## 6. 端到端流程 Demo 2（评测路线）

入口：`pipelines/demo2_cremad.py`，README 调用 `.venv/bin/python -m pipelines.demo2_cremad --au`。
这条路线**零生成模型**。

### 6.1 子集抓取

`pipelines/fetch_cremad.py`：从 `raw.githubusercontent.com` 拉 `SentenceFilenames.csv` 和
`VideoDemographics.csv`，按性别平衡选演员（`fetch_cremad.py:106-109`），
然后从 `media.githubusercontent.com/media/...`（git-LFS media 端点）**单文件**下载 `.flv`，
8 线程并发，再用 imageio-ffmpeg 转成两份：

```
ffmpeg -i x.flv -c:v libx264 -crf 18 -pix_fmt yuv420p -an  x.mp4     # 视频，无音轨
ffmpeg -i x.flv -vn -ac 1 -ar 16000                        x.wav     # 单声道 16 kHz
```

最后写 `clips.csv`（`file, actor, sentence, emotion, intensity, sex, age, race`）。
`--with-neutral` 默认 True：**NEU_XX 片段也要拉**，因为它们是 per-person neutral 估计的最好素材
（`fetch_cremad.py:94-95`）。

落盘实况：431 条、12 演员、6 句、NEU 71 条（6.4 节复述）。

### 6.2 分组

`demo2_cremad.py:96-107`：

```python
for c in clips:
    if c.meta["emotion"] == "NEU": continue          # 中性不进配对，只进 neutral 估计
    if c.stem in corpus.descs:
        groups[(sentence, emotion, intensity)].append(c.stem)
groups = {k: v for k, v in groups.items() if len(v) >= 2}
keys = sorted(groups); rng.shuffle(keys); keys = keys[:40]      # seed 0
```

一个 `(sentence, emotion, intensity)` 组就是"一批不同的人做同一件被规定的事"。
每组最多取 8 对（`--max-pairs-per-group`，`demo2:163`）。
实测 manifest 里出现 **30 个组、177 对**，另有 **3 对因对齐失败被丢**（`summary.md`）。

### 6.3 音频 DTW 对齐

**为什么用音频而不是用脸。** `expverify/audio.py:6-8`：
"the alignment must be derived from a signal that is *independent of the thing being measured*,
otherwise the aligner can manufacture the expression agreement the verifier is supposed to test."
如果用表情距离去对齐，然后再用表情距离去验证，那验证的是对齐器的搜索能力，不是数据的性质。
音频与被测的几何量无关，所以它是合法的对齐信号。同时两个演员念同一句话速度不同，
帧不是一一对应的，不对齐就是在比不同的音素。

**log-mel**（`audio.py:47-63`，全自写）：`scipy.io.wavfile.read` 读波形 →
若多声道取均值 → 按峰值归一 → `n_fft = 2^ceil(log2(sr·0.025))`（16 kHz 下 = 512）、
`hop = sr·0.010`（= 160 样本 = 10 ms）→ `scipy.signal.stft(padded=False, boundary=None)` →
功率谱 → **自写 mel 滤波器组**（`audio.py:29-44`，40 个三角带，`fmin = 60 Hz`，`fmax = sr/2`，
HTK 式 `2595·log10(1+f/700)`）→ `log(mel + 1e-8)` → **逐维减均值除标准差**。
返回 `(T, 40)` 和 hop 秒数。

**DTW**（`audio.py:66-107`，全自写）：

1. 两侧逐帧 L2 归一，代价矩阵 `C = 1 − A_norm @ B_normᵀ`（**余弦距离**）。
2. **Sakoe-Chiba band**：`band = max(8, round(0.25·max(n, m)))`。
   第 `i` 行只允许 `j ∈ [ (i−1)·m/n − band + 1, (i−1)·m/n + band + 1 ]`
   （`audio.py:79-80`）——沿对角线的斜带，把复杂度和病态路径同时压掉。
3. 标准三向递推 `D[i,j] = C[i-1,j-1] + min(D[i-1,j-1], D[i-1,j], D[i,j-1])`。
4. 从 `(n, m)` 回溯出路径，`cost = D[n,m] / len(path)`（平均代价）。

**音频帧 → 视频帧**（`audio.py:124-133`）：`t = audio_frame · hop`，
`video_frame = clip(int(t · fps), 0, n−1)`；对每个参考视频帧，
把落到它上的所有目标帧**取中位数**，输出 `(N, 2)` 的索引对。

**质量门**：`align.shape[0] < 8 or cost > 0.55` 就丢弃（`demo2_cremad.py:176-178`），
理由是"对齐坏了产出的帧对根本不是对应时刻，没有任何表情指标能补救"（`audio.py:113-116`）。
实测 177 对的 `dtw_cost` 中位数 **0.308**，最小 0.145，最大 0.514；3 对被丢。

### 6.4 验证：`editing` spec

`demo2_cremad.py:49` 默认 `--spec editing`。`EDITING_SPEC`（`verify.py:87-92`）相对
`REFERENCE_SPEC` 的差别只有三处：

```python
max_identity_cos  = 0.25     # 保留：必须是两个人
min_pose_delta_deg = None    # 关闭
max_bg_hist        = None    # 关闭
```

理由（`verify.py:83-86`）：CREMA-D 是**正面、单一背景**的录播素材，
在这里要求"背景要不同、姿态差要大"会因为一个与表情毫无关系的理由拒掉所有 pair。
实测印证：177 对的 pose delta 中位数只有 **4.7°**，远低于 `reference` spec 的 8.0° 门槛。

**表情侧的门一个都没放松**：`d_bs / d_deform / d_region / d_gaze / d_au / energy /
rank-1 / ratio / n_pass_testable / pass_rate / n_testable` 全部沿用同一份
`out/calibration/liveportrait` 校准（`demo2_cremad.py:48`）。

**身份门先自检再使用**（`demo2_cremad.py:129-153`）：每个演员取 2 个片段、最多 48 个，
跑 `identity_separation`，如果 `diff_q95 > max_identity_cos` 就把门抬到
`max(diff_q95, eer_threshold)` 并打印，理由是"让门去拒同人 pair，而不是拒掉一切"。
本次实测 `diff_q95 = 0.115 < 0.25`，**门没被抬，仍是 0.25**。

### 6.5 为什么产量必然极低，以及为什么这条线只做评测

**2 / 177 = 1.1%**（`out/demo2_cremad/summary.md`）。原因是结构性的，不是调参问题：
"同一句台词 + 同一个情绪标签 + 同一个强度"是一个**粗粒度约束**
（`demo2_cremad.py:8-13`）。两个演员拿到这个指令，仍然会做出**肉眼可见不同**的脸：
时序不同（DTW 只对齐语音，不对齐表情峰值）、幅度不同、肌肉选择不同（一个皱眉一个咬唇）。
情绪标签的粒度比"逐帧表情相同"粗了好几个数量级。

拒绝理由的分布（`out/demo2_cremad/manifest.jsonl` 统计）证实了这一点：
90 对是 "0 accepted testable frames"、31 对是 "1 accepted testable frame"、
81 对是 "frame pass rate 0.00 < 0.0658"。也就是说**绝大多数 pair 一帧都对不上**，
不是"差一点点没过门"。全体 177 对的逐帧通过率中位数是 **0.0108**，`rank1_rate` 中位数 **0.173**。

**所以这条路线的定位是**（`demo2_cremad.py:8-13`）：
(a) 展示真实跨演员数据能给什么、不能给什么；
(b) 提供**人工标注、与所有被校准指标独立**的硬负样本（同情绪不同强度）；
(c) 高置信度评测集。**不做训练规模**——1.1% 的产量意味着要 1 万对就得筛约 90 万个候选，
而 CREMA-D 的规模按 `fetch_cremad.py:5-7` 的描述是 91 演员 × 12 句 × 6 情绪 × 4 强度
（并非全组合都存在），远远撑不起这个量级的候选池，更别提这些候选还得两两同组。

---

## 7. 结果与证据

### 7.1 absolute 2/6 vs relative 0/6 —— 为什么这是承重的证据

`out/demo1_liveportrait/summary.md` 的对照表：

| | absolute（表情相同） | relative（对照） |
| --- | ---: | ---: |
| 接受 | **2 / 6** | **0 / 6** |
| 中位 `d_deform` | 0.0201 | 0.0242 |
| 中位 rank-1 率 | 0.353 | 0.270 |
| 中位 `ratio` | 4.332 | 4.837 |
| 中位逐帧通过率 | 0.079 | 0.026 |

**为什么这个对照是承重的。** 两个 variant 的输入完全一样（同一个 driver、同一批裁剪好的 source、
同一套抽取参数、同一份校准、同一段验证代码），**唯一的差别是一个命令行 flag**，
而这个 flag 在上游源码里的效果是"表情张量里有没有 source 相关项"（5.3 节）。
如果验证器给两组差不多的分数，那它就是在跟着文件名/元数据走；
它给出 2/6 对 0/6、并且在四个独立指标上一致地偏向 absolute，
说明**它测的是构造本身**。

逐对明细（`out/demo1_liveportrait/manifest.jsonl`）：

| variant | ref | tgt | 类型 | n_frames | n_pass | pass_rate | n_pass_testable | n_testable | rank1 | 结果 / 拒绝理由 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| absolute | D | S_s18 | real-gen | 78 | 17 | 0.2179 | 13 | 30 | 0.577 | **接受** |
| absolute | S_s18 | S_s32 | gen-gen | 78 | 10 | 0.1282 | 8 | 32 | 0.346 | **接受** |
| absolute | S_s13 | S_s18 | gen-gen | 76 | 6 | 0.0789 | 6 | 70 | 0.382 | 拒：可测通过帧 6 < 8 |
| absolute | S_s13 | S_s32 | gen-gen | 76 | 5 | 0.0658 | 5 | 70 | 0.303 | 拒：5 < 8；通过率（见 7.5） |
| absolute | D | S_s13 | real-gen | 76 | 6 | 0.0789 | 4 | 29 | 0.303 | 拒：4 < 8；pose delta 4.4° < 8° |
| absolute | D | S_s32 | real-gen | 78 | 3 | 0.0385 | 2 | 30 | 0.359 | 拒：2 < 8；通过率 |
| relative | D | S_s18 | real-gen | 78 | 10 | 0.1282 | 6 | 30 | 0.397 | 拒：6 < 8 |
| relative | D | S_s13 | real-gen | 76 | 9 | 0.1184 | 8 | 29 | 0.263 | 拒：pose delta 4.6° < 8° |
| relative | S_s13 | S_s18 | gen-gen | 76 | 3 | 0.0395 | 3 | 67 | 0.276 | 拒：3 < 8；通过率 |
| relative | S_s13 | S_s32 | gen-gen | 76 | 1 | 0.0132 | 1 | 67 | 0.250 | 拒：1 < 8；通过率 |
| relative | S_s18 | S_s32 | gen-gen | 78 | 0 | 0.0000 | 0 | 27 | 0.218 | 拒：0 < 8；通过率 |
| relative | D | S_s32 | real-gen | 78 | 0 | 0.0000 | 0 | 30 | 0.295 | 拒：0 < 8；通过率 |

值得注意：涉及 s13 的 pair 里 `n_frames` 是 76（不是 78），因为 `verify_pair` 只统计
**双侧同时有效**的对齐帧（`verify.py:213-214`）。生成出来的 `S_s13--D_d0.mp4` 实测确实是 78 帧
（ffmpeg 计数），所以那 2 帧是在 s13 一侧没通过有效性检查——`ok = f.ok & good`
（`neutral.py:103`），即 landmark 缺失或 `canonicalize` 抛 `ValueError`（退化眼距/竖轴）。
另外 relative 最好的一对（D vs S_s18，`pass_rate` 0.1282）恰好等于 absolute 里被接受的
S_s18–S_s32；它被拒是因为 `n_pass_testable = 6 < 8`。所以两组的分离不是天壤之别，是稳定的偏移。

### 7.2 身份渗漏检查

absolute 模式已知的代价是 driver 脸型渗漏，而**这个失败对任何表情指标都不可见**
（`identity.py:6-9`），所以它有专门的双向检查（`demo1_liveportrait.py:379-389`）：

| 输出 | cos(输出, 自己的 source) | cos(输出, driver) |
| --- | ---: | ---: |
| absolute:S_s13 | 0.861 | −0.097 |
| absolute:S_s18 | 0.872 | 0.023 |
| absolute:S_s32 | 0.857 | 0.065 |
| relative:S_s13 | 0.903 | −0.105 |
| relative:S_s18 | 0.915 | −0.005 |
| relative:S_s32 | 0.895 | −0.016 |

门槛是 `≥ 0.45` 和 `≤ 0.30`。absolute 的三段输出都是 **0.857–0.872 对 −0.097…+0.065**：
**没有观测到渗漏**。relative 的身份保持略好（0.895–0.915），方向符合预期
（它的表情增量里本来就带着 source 自己的 `exp`），但差距只有约 0.04。

### 7.3 Demo 2 的身份门自检

`out/demo2_cremad/summary.md`：

| | 余弦 |
| --- | ---: |
| 同一演员，中位数 | **0.894** |
| 不同演员，中位数 | **0.021** |
| 不同演员，q95 | 0.115 |
| 等错误率阈值（EER threshold） | **0.160** |
| 实际使用的门 | **0.25** |

这两个分布几乎不重叠（0.894 vs 0.021），EER 阈值 0.160 落在中间，
而实际使用的 0.25 在 EER 之上、在同演员分布之下。
所以当一对 pair 因 "identity cosine" 被拒时，那个拒绝**是有意义的**。
本次运行里全体 177 对的 `identity_cos` 中位数 0.017、最大 0.191，
**没有任何一对因身份被拒**——这也是为什么这一项没出现在拒绝理由表里。

### 7.4 人脸像素表与 s13

`out/demo1_liveportrait/summary.md` 的表（数值 = 裁剪框边长 ≈ 人脸跨度 × 1.9，见 3.7 末的修正）：

| clip | face px（框边长） | 换算人脸跨度 ≈ |
| --- | ---: | ---: |
| D | 526 | ~277 |
| absolute:S_s18 | 307 | ~162 |
| absolute:S_s32 | 281 | ~148 |
| **absolute:S_s13** | **120** | **~63** |
| relative:S_s18 | 303 | ~159 |
| relative:S_s32 | 273 | ~144 |
| relative:S_s13 | 118 | ~62 |
| src:S_s13 | 169 | ~89 |
| src:S_s18 | 301 | ~158 |
| src:S_s32 | 272 | ~143 |

s13 在解码帧里人脸只有约 63 px，被 `INTER_CUBIC` 放到 512 也不会创造出不存在的信息。
后果在数据里看得很清楚：**6 对涉及 s13 的 pair（两个 variant 各 3 对）全部被拒**。
而且当 s13 站在**参考侧**时，`n_testable` 反常地高：
`absolute:S_s13` 对 s18/s32 是 70、70，`relative:S_s13` 对 s18/s32 是 67、67，
而其他 pair 只有 27–32。（`D vs *:S_s13` 的 29 是因为参考侧是 driver，`d_self` 由 D 决定。）
`n_testable` 高说明 `d_self`（参考侧自身 ±3 帧的形变差）普遍超过 0.006 —— 在一段本该有静止段的视频里，
这更像是**测量抖动被当成了"在动"**，而不是真的每一帧都在动。
低分辨率同时抬高了 `d_self`（虚假的可测帧）和 `d_deform`（真实的失配），
所以 s13 自己限制了自己的可达粒度。这就是 `demo1_liveportrait.py:337-339`
那句"一对里较小的那侧是瓶颈"的实证。

### 7.5 逐帧通过率约 13% 意味着什么、不意味着什么

先把三个不同的 13% 分清楚，因为它们来自不同的地方：

| 数字 | 出处 | 含义 |
| --- | ---: | --- |
| **0.1282** | `pairs/01_.../metrics.json` 的 `pass_rate`（`pairs/README.md` 写作 12.8%） | 交付的那一对生成 pair 的逐帧通过率 |
| **0.0789** | `liveportrait.json:226` `pair_level.aligned_median` | 6 个跨身份正样本 pair 的通过率**中位数** |
| **0.1347** | `augment.json:225` `pair_level.aligned_median` | 24 个**同一张脸**增广孪生 pair 的通过率中位数 |

**13% 不意味着"质量只有 13%"。** 它是七/八个门的合取在**帧**这个粒度上的通过率，
而两个最紧的门在设计上就会拒掉大量帧：

- `g_rank1 = 0.378`（`liveportrait.json:219`）：要求 A 的第 t 帧在 B 的 ±12 帧窗口里的最近邻
  就落在 t±1。落在 t±2 就算失败。
- `g_energy = 0.442`（`liveportrait.json:221`）：两侧都必须离自己的中性脸足够远，
  且这个门槛是从 CREMA-D 搬来的（4.5 节）。

`0.378 × 0.442 ≈ 0.167`，再乘上五个距离门各约 0.897–0.939 的通过率（它们并不完全独立），
落到 0.079–0.135 这个量级是**算得出来的**，不是数据坏。

**13% 意味着的是**："只有约 13% 的帧能被现有测量手段**严格证明**表情相同"。
剩下的帧分两类：`testable = False` 的静止帧（rank-1 检验在那里按构造无法成立）、
以及 rank-1 落到 ±2 以外的帧（可能真错，也可能是测量噪声——AUC 0.505–0.645 说明我们分不清）。

**它也不意味着"通过的帧一定对、被拒的帧一定错"。** 帧级判据的 AUC 只有 0.5–0.65。
唯一有统计效力的结论在 pair 级：对齐组 q05 = 0.0453 vs 平移组 q95 = 0.0262，两组不重叠，
AUC 0.993、precision 1.000。所以正确的读法是：
**"这一对整体上被证明是逐帧对齐的"，而不是"这 10 帧被证明相同、那 68 帧被证明不同"。**

如果只要求情绪类别一致，通过率会接近 100%，但那正是明确不要的"差不多"。

### 7.6 交付的三对

`pairs/` 下的成品（数据来自各自的 `metrics.json`）：

| 目录 | 路线 | 帧数 | pass_rate | n_pass_testable / n_testable | rank1_rate | identity_cos | pose delta | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `01_generated_S_s18_vs_S_s32` | 生成 | 78 | 0.1282 | 8 / 32 | 0.346 | **0.138** | 37.8° | `bg_hist` −0.003；`source_id` 0.857；`driver_id` 0.065 |
| `02_real_1004_vs_1014` | 真实 | 101 | 0.0792 | 8 / 69 | 0.554 | 0.108 | 10.7° | `dtw_cost` 0.2168，TAI/ANG/XX |
| `03_real_1002_vs_1005` | 真实 | 94 | 0.1170 | 11 / 62 | 0.277 | 0.035 | 11.2° | `dtw_cost` 0.3566，TIE/FEA/XX |

> `pairs/README.md` 第 17 行把 01 的"身份余弦"写成 0.081。`metrics.json` 里这一对的
> `identity_cos` 是 **0.1378**；0.0806 是 `summary.md` 里两个被接受 absolute pair 的
> **中位数**（0.138 与 0.023 的中位数）。结论不变（两个数都远低于 0.25 门槛），但标注对象错了。

### 7.7 一个刀锋级的浮点问题（不影响结论，但应当知道）

`min_pass_rate` 的拟合值是 `0.06578947603702545`（`liveportrait.json:10`）。
这个数正好是 **`float32(5/76)`**：`fit_threshold` 的候选网格来自 `np.unique` 拼接的 **float32** 数组
（`calibrate.py:310`、`calibrate_pair_level` 里 `rates()` 返回 `np.float32` 数组），
而 6 个正样本 pair 里有一个的通过率恰好是 5/76。

于是 `absolute:S_s13 vs absolute:S_s32` 这一对，它的真实通过率是
`5/76 = 0.06578947368421052`（float64），被拿去和 `0.06578947603702545` 比较：
**小了约 2.4 × 10⁻⁹，判为不合格。** 这也解释了 `pair_level.recall = 0.8333 = 5/6`——
被用来定阈值的那个正样本自己没过。

好消息是这一对同时还因 `n_pass_testable = 5 < 8` 被拒，**所以最终 2/6 的结论不变**。
但如果有一天 `min_pass_frames` 被放松，这个 float32 舍入就会成为一个真实的、
难以察觉的 off-by-epsilon 缺陷。

---

## 8. 已知局限与放大到真实数据集的路径

### 8.1 会坏的地方

1. **逐帧粒度没有达成。** 单指标在 3 帧尺度上 AUC 0.505–0.645（4.1、1.2 节）。
   而且五个距离阈值**全部是退化回退值**（正样本 q90，4.4 节），
   `max_ratio = 4.43` 让"赢过时序邻帧"这个原始严格条款失效（3.1 节末）。
   现在真正提供严格性的只有 `g_rank1` 和 pair 级聚合。
2. **`min_energy` 跨语料搬运**（4.5 节）：CREMA-D 的 35% 分位数被套到 LivePortrait 输出上，
   导致 `g_energy` 成为第二瓶颈门，其中一部分是阈值不匹配而非数据问题。
   正确做法是按目标语料自己的 energy 分布重算，代码支持（`--energy-percentile`），但本次没做。
3. **测量分辨率是硬天花板**（3.7、7.4 节）。s13 在约 63 px 上限制了自己；
   低分辨率还会**伪造可测帧**（`n_testable` 70 vs 30），污染 rank-1 的分母。
   建议加一条硬门：`face_px` 低于某个值直接不进 pair 池。目前代码只是把 `face_px` 记进 manifest。
4. **校准正样本只有 6 对。** `liveportrait.json` 的 `n_pos = 462` 帧、
   `pair_level.aligned_n = 6`。用 6 个样本拟合 pair 级阈值，置信区间极宽
   （7.7 节的刀锋现象就是小样本的直接后果）。
5. **单一生成器。** Track A 只有 LivePortrait。它的隐式关键点表示、
   `stitching` 模块、45/63 的表情覆盖范围，全部会以系统性偏置的形式印在数据里。
6. **每 clip 一个 neutral 的耦合**（3.2 节末）：Demo 1 里 neutral 来自被评估的那同一段视频，
   减法会连共享的表情基线一起减掉，压低 `energy`。

### 8.2 没有被证明的东西

- **"通过的帧确实表情相同"**没有独立验证。唯一的独立证据是 pair 级的对齐/平移分离
  和人眼可看的 `pairs/01_.../montage.png`。没有人工 FACS 标注做交叉验证。
- **rank-1 率**虽然被记录（`rank1_rate`），但**pair 级没有对它设门**（4.1 节）。
  `rank1_only_pair_level`（`liveportrait.json:201-210`）显示如果单用它就能做到 AUC 1.000
  （对齐中位数 0.491 vs 平移 0.065，阈值 0.329），但这个阈值**没有被接进 `DatasetSpec`**。
  这是一个明显可拿的强化，尚未实施。
- **增广层级的噪声地板对比**：`calibrate.py:175-190` 支持 `--all-tiers` 输出
  mild/medium/heavy 三档的地板，但**两份落盘 json 里都没有 `noise_floor_by_tier`**，
  说明这次没跑。"过度增广会淹没真实表情变化"这个论断只有代码注释
  （`augment.py:43-46`）支撑，没有数字。
- **M4 感知指标**（README 提到的 FaRL/SigLIP2 + FEC triplet 微调）完全未实现。

### 8.3 合成数据的偏置，以及三元组设计部分缓解了什么

偏置来源有三层：单一生成器（8.1.5）、LivePortrait 特定的表情参数化
（45/63 覆盖 + `stitching`）、以及"两侧都是生成视频"带来的分布同质化
（同一个解码器 G、同样的 warping 伪影、同样的贴回边界）。

**三元组 D / A′ / B′ 的设计**（`demo1_liveportrait.py:296-297`）针对第三层：
一个 driver D 加两路输出 A′、B′ 可以产出 3 种 pair —— `(D, A′)`、`(D, B′)`、`(A′, B′)`。
前两种的 ref 侧是**真人实拍**，一半的数据不带生成伪影，**免费**缓解同质化。
实测有效：被接受的两对里有一对正是 `(D, absolute:S_s18)`，而且它是全场最好的一对
（`pass_rate` 0.2179、`rank1_rate` 0.577）。
代价是 `(D, ·)` 这类 pair 更容易撞上 pose 门（D 与输出共享 source 的姿态？不——
实测 `D vs S_s13` 的 pose delta 只有 4.4°，因为 s13 本身几乎不动头，
所以它被 `min_pose_delta_deg = 8.0` 拒掉了）。

**没有被缓解的**：第一、二层偏置需要第二个生成器做交叉验证，本次没有。

### 8.4 搬到 Linux GPU 机器上要改什么

代码层面要改的很少，环境层面的坑几乎全是 macOS 特有的：

| 现在的约束 | 原因 | Linux GPU 上怎么处理 |
| --- | --- | --- |
| `mediapipe==0.10.21` | 1.0.x 在 macOS 上初始化 Metal helper 时段错误（`requirements.txt:19`） | 可以解钉，Metal 不参与 |
| `numpy<2` | LivePortrait 内置的 InsightFace 代码在 2.x 上 `TypeError`（`requirements.txt:4-5`） | 仍需保留，除非修补 vendored 代码 |
| `onnxruntime`（非 silicon） | `onnxruntime-silicon==1.16.3` 无现代 Python wheel 且已弃维（`requirements.txt:6-8`） | 换 `onnxruntime-gpu`；`identity.py:110` 的 `providers=["CPUExecutionProvider"]` 要改成 CUDA |
| `torch>=2.9` | 需要原生 MPS `grid_sampler_3d`（`requirements.txt:9-11`） | CUDA 一直有，版本可放宽 |
| 不能用 `--flag_do_torch_compile` | `torch.compile` 在 MPS 上崩（`requirements.txt:12`） | **应该打开**，这是主要加速点 |
| `au.py:57` 自动选 `mps` | — | 会自动落到 `cpu`；应加 `cuda` 分支 |
| ffmpeg 走 imageio-ffmpeg | 免 Homebrew（`requirements.txt:13-14`） | 可保留，`child_env()` 那个软链 hack 也可以留着 |

真正需要新写的是**并行化**：现在 `run_liveportrait` 是逐 (source, driver) 串行子进程
（`demo1_liveportrait.py:199-201`），约 0.9–1.1 s/帧。要规模化得：
按 driver 批量做一次驱动模板（LivePortrait 支持 `.pkl` 模板复用）、
多进程/多 GPU 分发 source、以及给特征抽取加缓存（`extract_cached` 已存在于
`landmarks.py:337-349`，但 Demo 1 没用它，Demo 2 用了）。

### 8.5 产量公式

Track A 的候选 pair 数是纯组合的。设 `D` 个 driver、每个 driver 配 `S` 个 source：

```
候选 pair 数 = D · [ C(S,2) + S ]
                     ↑gen-gen    ↑real-gen（ref 侧是实拍）
```

本次 `D = 1, S = 3` → `3 + 3 = 6`，与 manifest 一致。
接受数 = 候选数 × 接受率，本次接受率 `2/6 = 0.333`。

但这个 0.333 不能直接外推，因为两条拒绝理由在规模化时表现完全不同：

- **`face_px` 太小**（4 对里的主因）：可以**在配对前**过滤掉，不消耗生成算力。
  只用 `face_px ≥ 280` 的 source（本次是 s18、s32），`S_eff = 2`，接受率会显著上升
  （本次这个子集里 `S_s18–S_s32` 与 `D–S_s18` 都被接受，`D–S_s32` 因通过率被拒 → 2/3 = 0.667）。
- **pose delta 不够**（2 对）：也是 source 侧属性（头不动的 source），可以预筛。
- **`n_pass_testable < 8`**：这是真正的表情侧筛除，无法预筛。

所以更有用的形式是把可预筛的部分挪到前面：

```
产出 pair 数 ≈ D · [ C(S_ok, 2) + S_ok ] · p_expr
    S_ok  = 满足 face_px ≥ 阈值 且 头部有足够运动 的 source 数
    p_expr = 表情侧接受率（本次可观测的样本量下约 0.5–0.67，样本 n=3，置信区间极宽）
```

算力侧：每对 pair 需要 `S_ok` 次生成，每次 `n` 帧 × 约 1 s/帧（M3 Pro，GPU 上应显著更快）。
即"生成成本随 `S_ok` 线性增长，而 pair 数随 `S_ok²` 增长"——
**这是构造路线相对检索路线在规模上的根本优势**：
Demo 2 那条线要 90 万个候选才能凑 1 万对，而 Track A 只要 142 个 source（`C(142,2) ≈ 10^4`）
配一个 driver 就有 1 万个候选，并且每个候选的表情相等性是构造保证的。

### 8.6 规模化时该换的数据

README 第 226-230 行列的方向，逐条说明它解决 8.1/8.2 里的哪一条：

- **MEAD**（1080p、7 视角、60 演员）：直接解决 8.1.3（测量分辨率）和 8.1.4（样本量），
  同时 7 视角给了免费的姿态多样性。
- **第二个生成器**：解决 8.3 第一、二层偏置。
- **FEC strong-agreement triplet 微调 M4**：解决 8.1.1（帧级判别力）。这是唯一
  经人类判断验证过的路线。
- **DISFA**（27 人看同一段刺激视频、每帧由认证 FACS coder 标 12 个 AU 的 0–5 强度）：
  解决 8.2 第一条（"通过的帧确实相同"缺少独立验证）——它能提供人工标注的评测集。

---

## 9. 走过的弯路

这一节是为了让重建这套东西的人不必再踩一遍。每条都给出证据位置。

### 9.1 一开始按"测量并检索"来做

**表现**：抽特征 → 算相似度 → 取最近的 pair。
**为什么失败**：帧级指标在 3 帧尺度上 AUC 0.505–0.645（1.2 节），
分不清"表情相同"和"差 3 帧"，所以"最近"这个词在需要的粒度上没有意义。
**证据**：`expverify/__init__.py:1-8` 把结论写成了包的设计前提；
`out/calibration/liveportrait.json` 的 `auc_by_offset` 是数字证据。
**改法**：改成构造路线，验证器只负责拒绝。

### 9.2 阈值用"同一张脸的增广孪生"校准 → Demo 1 第一版全军覆没

**表现**：先用 `--positives augment` 校准，然后拿去判 Demo 1 的跨身份 pair，全部被拒。
**为什么失败**：增广孪生测的是**同一张脸的噪声地板**，它对"两张不同的脸做同一个表情
会差多少"一无所知。而后者必然更大，因为两张脸不会把同一个表情渲染成同一套几何
（`pipelines/calibrate.py:70-78`）。
**数字证据**（两份 json 直接对照）：`d_bs` 的噪声地板阈值是 **0.0092**（`augment.json:3`），
而跨身份正样本的中位数是 **0.0167**（`liveportrait.json:33`）——
**阈值卡在正样本分布的正中间偏下，任何跨身份 pair 都够不着。**
**改法**：`pipelines/calibrate.py:120-125` 加了 `--positives liveportrait`，
生产网关用跨身份正样本校准（两个 demo 的默认都指向 `out/calibration/liveportrait`）。
两份都留着：差值就是"跨越身份"的代价（1.2 节的比值表）。

### 9.3 校准 AUC 掉到 0.5 以下（噪声机制不匹配）

**表现**：一个明显有用的指标算出低于随机的 AUC。
**根因**：负样本取的是**单次抽取内**的 `d(t, t+k)`。FaceLandmarker 跑在 VIDEO 模式
（`landmarks.py:158`），跨帧携带跟踪状态，同一次抽取内的 landmark 被**时序平滑**、噪声相关；
而正样本比较两次**独立**抽取，携带完整独立噪声。负样本被系统性低估。
**数字证据**（3.6 节的表）：同段负样本(+3f)中位数 `d_deform` **0.0116**
＜ 正样本中位数 **0.0204**；四个指标全部如此。距离小的一侧是负样本 ⇒ AUC 必然 < 0.5。
**改法**：`collect_paired_negatives`（`calibrate.py:141-160`）——
负样本改成**同一对正样本平移 k 帧**，两侧处在同一个噪声机制里。
旧做法作为诊断保留在 `collect_temporal_negatives`（`calibrate.py:163-179`），
并且它的中位数被显式写进 `report.within_run_median`，所以这个坑在产物文件里永久可见。

### 9.4 增广过猛

**表现**：重档增广（噪点 σ=7、JPEG q=62、降采样 0.75、旋转 4°）让 landmark 检测退化，
噪声地板超过好几帧真实表情变化的量。
**改法**：`augment.py:47-82` 把增广分成 `mild / medium / heavy` 三档，
理由写在 `augment.py:43-46`："Heavy presets degrade the landmark detector enough to swamp
several frames' worth of real expression change, which makes them a stress test, not a calibration reference."
**实际出货的只有 `mild`**：纯色彩、无几何、无重采样、JPEG q=95（`augment.py:50-59`）。
> 这一条**只有代码注释支撑，没有数字**：`--all-tiers` 会输出 `noise_floor_by_tier`
> （`pipelines/calibrate.py:175-190`），但两份落盘 json 里都没有这个字段，说明本次没跑。标为未证实。

另外，风格化明确不用 img2img 扩散（`augment.py:11-16`）：没有表情保持约束、没有时序模型，
会逐帧改动嘴形和眼神，对一份以表情相等为唯一卖点的数据集是破坏性的。
色彩分级 / 噪点 / 几何抖动完全不触碰人脸几何，而且几何抖动会被
`descriptors.py` 的规范化坐标系**可证明地移除**（相似变换被 Umeyama 吸收）。

### 9.5 3 点人脸坐标系

**表现**：下巴处出现巨大的幻影位移。
**根因**：y 轴由额顶点 `p[10]` 决定，那是个**无纹理点**，它的定位误差**旋转整张脸**，
在离原点最远处放大（`descriptors.py:158-162`）。
**改法**：13 点 Umeyama + 广义 Procrustes 共享引用（3.3 节）。
老函数 `face_frame` 保留，但只作为拟合共享引用的 bootstrap，
`canonicalize` 里只在 `_REFERENCE is None` 时才回退到它（`descriptors.py:247-249`）。

### 9.6 `d_region` 与 `d_deform` 共线

**表现**：合取门里有两个门在说同一件事，等于同一个门按了两次。
**根因**：未归一化的"最差分区 RMS"被运动幅度最大的嘴唇主导，全局 RMS 也是。
**改法**：除以该分区自身的运动量（3.5 节的公式）。
**证据**：改造后 `d_deform ~ d_region` 相关系数 **+0.194**（`liveportrait.json:133`），
而且 `d_region` 的 AUC 高于 `d_deform`（+3f 0.574 vs 0.553；+10f 0.743 vs 0.666）。
> "改造前 0.987"（README:157）/"r = 0.99"（`verify.py:133`）只在文档和注释里，
> 两份落盘 json 都是改造后版本，无法复现。标为未证实。

### 9.7 MediaPipe 在 macOS 上段错误

**表现**：初始化时直接 segfault。
**改法**：钉 `mediapipe==0.10.21`，注释写明"1.0.x segfaults on macOS initialising its Metal helper"
（`requirements.txt:19`）。这是一个**版本钉死**而不是代码修复，Linux 上可以解钉。

### 9.8 `numpy<2`

**表现**：LivePortrait 里 vendored 的 InsightFace 代码抛 `TypeError`。
**改法**：`requirements.txt:4-5, 16` 钉 `numpy<2`，实装 1.26.4。
连带影响：整个仓库的所有代码都只能用 numpy 1.x API。

### 9.9 ArcFace 拿到了整帧、而 landmark 在 512×512 裁剪空间里

**表现**：身份门槛静默失效——不同人的余弦一起升高，几乎所有 pair 都因 "identity cosine" 被拒。
README 第 172 行记录当时"177 对里 176 对因 identity cosine 被拒"。
**两个独立的根因，都是"不报错只出垃圾"型**：

1. **坐标空间不匹配**：几何计算全在裁剪空间，而 ArcFace 对齐和背景比较消费原始帧。
   `landmarks.py:322-324`："feeding them crop-space coordinates silently produces garbage
   rather than an error."
   **改法**：`landmarks_in_original()`（`landmarks.py:319-334`）显式做逆映射，
   两个 demo 里都在调 ArcFace 之前调它（`demo1_liveportrait.py:270`、`demo2_cremad.py:124-125`）。
2. **对齐估计器选错**：`align112` 原本用 `cv2.estimateAffinePartial2D`，
   它的鲁棒估计在随机子集上拟合，而只有 5 个点时挑到退化子集的概率很高。
   轻微跑偏**不抛异常**，只是把所有 embedding 拉向平均脸。
   `identity.py:81-85` 写得很清楚："A slightly misaligned chip does not throw -- it quietly pulls
   every embedding toward a mean face, which inflates cosine similarity between different people
   and silently disables the identity gate."
   **改法**：`similarity_2d()`（`identity.py:78-96`）自己实现**精确的**最小二乘 2-D 相似变换（Umeyama）。

**修好之后的证据**：Demo 2 的身份自检 —— 同演员中位数 0.894、不同演员中位数 0.021、
q95 0.115、EER 0.160（7.3 节）；本次运行 177 对里**没有任何一对因身份被拒**。
另外为了防止这类失效再次静默发生，加了 `identity_separation()`（`identity.py:143-171`）
作为常驻自检，并且 `demo2_cremad.py:146-153` 会在门槛明显不适配当前分辨率时自动抬高门槛
并打印说明，而不是"拒掉一切"。

### 9.10 s13 人脸太小 → 整段无效

**表现**：一整段视频返回全无效帧，**没有报错**。
**根因**：MediaPipe 发布的是**短距 BlazeFace**，需要人脸占输入的相当比例；
全身镜头里头部只占画面高度约 7%，检测直接返回空（`landmarks.py:214-221`）。
**改法**：`locate_face()` 由粗到细分块搜索（3.7 节），三个尺度、50% 重叠、
按到 `(W/2, H/3)` 先验的距离排序，命中后把坐标映回原帧。
`run_face_crop` 在 probe 全灭时调它（`landmarks.py:262-268`）。
**代价**：救回来的脸仍然只有约 63 px，s13 的四对 pair 全部被拒（7.4 节）。
**这个坑救回了"能不能测"，救不回"测得准不准"。**

### 9.11 其他环境坑（`requirements.txt:1-14` 逐条记录）

- `onnxruntime-silicon==1.16.3`（LivePortrait 的 `requirements_macOS.txt` 钉的）
  没有现代 Python 的 wheel，且被它自己的维护者弃养 → 装官方 `onnxruntime`。
- `torch < 2.9` 没有原生 MPS `grid_sampler_3d`（pytorch/pytorch#160541），
  LivePortrait 的 warping module 会落到 CPU fallback → 钉 `torch>=2.9`，实装 2.13.0。
- `--flag_do_torch_compile` 在 MPS 上崩 → 不要加。
- ffmpeg 通过 imageio-ffmpeg 的自带二进制提供 → 不需要 Homebrew；
  LivePortrait 的前置检查靠 `child_env()` 造软链绕过（5.2 节）。

### 9.12 `max_ratio` 手选 0.8

**表现**：按文档意图设 `max_ratio = 0.80`（要求跨身份匹配赢过自身时序邻帧），静默拒掉一切。
**根因**：`calibrate.py:420-423`——即使是**构造保证相同**的正样本，
它们的 ratio 中位数也在 1.0 附近（`liveportrait.json:186` 记录为 1.4717）。
一个 0.8 的常数要求正样本做到连正样本自己都做不到的事。
**改法**：改成拟合。**代价**：拟合值 4.4321 让这个门几乎失效（3.1 节末的 ⚠️）。
这是一个**诚实但不令人满意**的结局：它没有被调参掩盖，但它也确实不再提供严格性。

### 9.13 pair 级 `min_pass_rate` 的 float32 刀锋

见 7.7 节。**新发现，此前未记录在任何文档里**：
拟合出的 `min_pass_rate` 是某个正样本自身通过率的 float32 表示，
比 float64 真值大约 2.4 × 10⁻⁹，于是那个正样本自己被拒。
本次不影响结论（该 pair 另有硬性失败），但它是一个潜伏的 off-by-epsilon。
修法很简单：在 `fit_threshold` 里把候选网格转成 float64，或在比较时留一个相对容差。

---

## 10. 本文中未能证实 / 与已有描述不一致的地方

汇总，便于核查。

### 10.1 无法在代码或产物中证实

| 说法 | 出处 | 状态 |
| --- | --- | --- |
| FEC 基准数字（AU 距离 40.7–47.1%、情绪 embedding 53.3%、人类 87.5%） | `expverify/__init__.py:4-7`、README:6-7 | 外部文献引用，仓库无复现代码与产物 |
| `d_region` 改造前与 `d_deform` 相关 **0.987**（README）/ **0.99**（`verify.py:133`） | 注释与 README | 两份落盘 json 均为改造后版本，无法复现 |
| `d_region` 改造前 AUC 0.556(+3f) / 0.664(+10f) | README:160 | 不在任何产物文件里。数值接近 `d_deform` 自身 AUC（0.553/0.666），与"改造前几乎共线"自洽，但只是推断 |
| 各增广层级的噪声地板（"过度增广"的量化证据） | `augment.py:43-46` | 两份 json 都没有 `noise_floor_by_tier`，本次未跑 `--all-tiers` |
| "177 对里 176 对因 identity cosine 被拒"（修 bug 前） | README:172 | 落盘的 manifest 是修好之后的运行，0 对因身份被拒；旧运行未保留 |
| Demo 1 速度 0.9–1.1 s/帧 | README:75 | `run_liveportrait` 会打印 s/frame（`demo1:144-145`），但日志未落盘 |
| M3 是 "EfficientNet-B0 + graph AU head" | `au.py:7-8` | 本仓库只调 `openface.multitask_model.MultitaskPredictor`，架构描述来自上游，未在此核实 |

### 10.2 与任务描述/现有文档不一致，需要修正的

1. **"表情张量被完整拷贝、无 source 相关项"过强。**
   absolute+exp 模式下 `delta_new` 起点是 `x_s_info['exp'].clone()`
   （`live_portrait_pipeline.py:319`），之后只覆盖 **45 / 63** 个分量，
   剩 18 个（关键点 0、7、10 全部，以及 3、4、5、8、9 的部分坐标）保留 source 自己的值。
   此外 `stitching(x_s, x_d_i_new)` 是吃 `x_s` 的学习模块。
   准确说法：**被驱动的那 45 个表情分量在所有 source 上逐元素相同。**
   （README:115 的"21 个关键点里有 8 个不被覆盖"是对的；`pairs/README.md:20-21`
   的"表情参数逐帧完全相同"需要加上"驱动子集"这个限定。）

2. **"~13% 逐帧通过率"要区分三个不同的数字**（7.5 节）：
   交付 pair 01 是 **0.1282**；跨身份正样本 pair 级中位数是 **0.0789**（`liveportrait.json`）；
   **0.1347** 是同一张脸增广孪生的中位数（`augment.json`）。任务描述里的"~13%"最接近前者。

3. **五个距离阈值全部是退化回退值**（4.4 节）。
   `separated: false` + `target_precision_unreachable: true` + `recall` 五个指标完全相同 = 0.8983，
   说明它们都等于正样本 q90，**不是**"能以 95% precision 区分"的阈值。
   任务描述把它们称为"calibrated threshold currently in force"是对的（它们确实在生效），
   但它们的判别含义需要这个限定。

4. **`max_ratio` 的语义被校准反转**（3.1 节末）。
   `verify.py` 文档描述的"必须赢过时序邻帧（ratio < 1）"与实际生效的 `max_ratio = 4.4321` 相矛盾。
   `g_ratio` 在正样本上通过率 0.939，接近空门。真正的严格性来自 `g_rank1` 和 pair 级聚合。

5. **`min_energy` 来自 CREMA-D，不是来自被评估的语料**（4.5 节）。
   两份 json 的 `min_energy` 逐位相同（0.023649911954998968），证明它与 `--positives` 的选择无关。

6. **`face px` 是裁剪框边长，不是人脸高度**（3.7 节末）。
   `summary.md` 的列名 "face height in decoded pixels" 高估了 1.9 倍
   （`margin = 1.9`）。s13 的真实人脸跨度约 63 px，不是 120 px。

7. **`pairs/README.md:17` 的"身份余弦 0.081"标错了对象**（7.6 节）。
   pair 01 的 `identity_cos` 是 0.1378；0.0806 是两个被接受 absolute pair 的中位数。

8. **`rank1_rate` 在 pair 级没有门**（4.1、8.2 节）。
   `liveportrait.json:201-210` 显示单用它可达 AUC 1.000、阈值 0.329，
   但这个值没有被接进 `DatasetSpec`，`verify.py` 里也没有对应的检查。

9. **落盘的 CREMA-D 子集与 `fetch_cremad.py` 默认参数不一致**（2.3 节）：
   实际是 6 个句子、强度 XX 371 + HI 60；脚本默认是 3 个句子、`--intensities HI`。

10. **Demo 1 用的是 `run_face_crop` 直调，不走 `extract_cached`**
    （`demo1_liveportrait.py:227-228` vs `landmarks.py:337-349`）。
    Demo 2 通过 `build_corpus` 用了缓存。这不是错误，但意味着重跑 Demo 1
    会重新抽取全部 10 段特征。

11. **新发现：`min_pass_rate` 的 float32 刀锋**（7.7、9.13 节）。
    `absolute:S_s13 vs absolute:S_s32` 的通过率 `5/76` 被一个等于 `float32(5/76)` 的阈值
    以 2.4 × 10⁻⁹ 的差距拒掉。本次不改变结论。
