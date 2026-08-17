# 表情可控 pair 数据集：构造方法与两个轻量 demo

目标是成对视频：两侧表情逐帧一致，其余（身份、姿态、背景、光照、运动）都不同。

核心结论先说清楚：**表情一致性必须在共享隐变量里构造出来，验证器只用来拒绝，不用来寻找。**
在 FEC 基准（人类三选二判断"哪两张表情最像"、明确要求忽略身份）上，AU 距离类指标只有
40.7–47.1%，情绪 embedding 类 53.3%，而人类中位数 87.5%。检索/匹配路线的天花板就在那里，
不是工程问题。本仓库因此分两层：

- **构造层**：一次真人表演 → 扇出到 N 个外观，**被驱动的那 45 个表情分量**逐帧逐元素相同（Demo 1，
  精确覆盖范围见下文"Demo 1"）。
- **验证层**（`expverify/`）：一套互相独立、合取判定的指标，只负责把"差不多"的数据拒掉。

> **已知的不一致 / 请注意**
> `METHOD.md` 第 10 节逐条列出了本仓库里**无法在代码或产物中证实**的说法，以及与既有文档不一致
> 需要修正的表述，核查以那一节为准。最需要先知道的一条：五个距离阈值
> （`d_bs`/`d_deform`/`d_gaze`/`d_region`/`d_au`）全部带 `target_precision_unreachable: true`，
> 是"放过 90% 已知真阳性"的**包络阈值**，不是判别性阈值（见下文"阈值校准"）。

## 结果速览（M3 Pro 实测）

| | Demo 1 absolute | Demo 1 relative（对照） | Demo 2 CREMA-D |
| --- | ---: | ---: | ---: |
| 候选 pair | 6 | 6 | 177 |
| 通过 | **2 (33%)** | **0 (0%)** | **2 (1.1%)** |
| 中位 d_deform | 0.0201 | 0.0242 | — |
| 中位 rank-1 率 | 0.353 | 0.270 | 0.416（通过者）|
| 中位逐帧通过率 | 0.079 | 0.026 | 0.098（通过者）|

relative 是**对照组**而不是竞争方案：它多了 source 相关项，表情本就不该相同，验证器
必须说不。两者若打平，说明判定跟着文件名走而不是跟着构造走。

三条最值得记住的量化结论：

1. **逐帧粒度上，现有指标分不清相邻 3 帧。** 以"构造上表情相同"为正样本、同一对 pair 平移
   3 帧为硬负样本，单指标 AUC 只有 0.505–0.645；把偏移拉到 10 帧，d_gaze 才升到 0.909。
2. **但 pair 级聚合能完全分开**：AUC 0.993，precision 1.000，对齐/平移两组不重叠。所以
   "严格逐帧一致"目前只能在**整段统计**意义上被证明，不能逐帧证明——这是指标的上限，
   诚实报出来比调参掩盖有用。
3. **真实跨演员数据只有约 1% 可用**。CREMA-D 同句同情绪同强度、音频 DTW 对齐之后，177 对
   里只有 2 对过关。它适合当高置信评测集，不适合做规模。

## 目录

```
expverify/            验证器（本仓库的主要产出）
  landmarks.py        MediaPipe 478 landmark / 52 blendshape / 头部姿态；人脸裁剪与小脸搜索
  descriptors.py      Umeyama 刚性对齐、身份归一化形变场、blendshape 通道计划
  neutral.py          per-person neutral 估计（两趟中位数）与描述子构建
  verify.py           rank-1 时序可辨识性 + 合取网关 + 数据契约（DatasetSpec）
  calibrate.py        阈值拟合、硬负样本、冗余性检查、pair 级 min_pass_rate
  identity.py         ArcFace 身份（只用于证明"人不同"）与身份可分性自检
  au.py               M3：OpenFace 3.0 AU 激活
  audio.py            音频 log-mel + DTW（Track B 对齐）
  augment.py          仅改外观的增广（用于测噪声地板）
  scene.py            姿态差、背景相似度
  report.py           manifest.jsonl、逐帧曲线图、markdown 汇总
pipelines/
  fetch_cremad.py     按单文件 LFS 拉 CREMA-D 子集（每条约 260 KB）
  fetch_liveportrait.py  只下 liveportrait/ 与 insightface/ 权重（约 660 MB，跳过 1.4 GB animals）
  calibrate.py        校准入口（两种正样本源）
  demo1_liveportrait.py  Track A：一次表演扇出到 N 个外观
  demo2_cremad.py     Track B：真实跨演员，零生成模型
```

## 安装

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

`requirements.txt` 里记了几个踩过的坑：`numpy` 必须 `<2`（LivePortrait 内置的 InsightFace
代码在 2.x 上崩）；`mediapipe` 钉 `0.10.21`（1.0.x 在 macOS 上初始化 Metal helper 时段错误）；
用官方 `onnxruntime` 而不是 LivePortrait 自带 requirements 里那个已经无人维护的
`onnxruntime-silicon==1.16.3`；torch 需 ≥ 2.9 才有原生 MPS `grid_sampler_3d`；
`--flag_do_torch_compile` 在 MPS 上崩，不要加。ffmpeg 走 `imageio-ffmpeg` 自带的二进制，
不需要 Homebrew。

## 跑法

```bash
# Demo 1：LivePortrait 生成（M3 Pro 上约 0.9–1.1 s/帧）
git clone https://github.com/KwaiVGI/LivePortrait third_party/LivePortrait
.venv/bin/python -m pipelines.fetch_liveportrait
.venv/bin/python -m pipelines.demo1_liveportrait --au

# 拉 CREMA-D 子集（每条约 260 KB，不需要 clone 7.5 GB）
.venv/bin/python -m pipelines.fetch_cremad --out data/cremad --actors 12

# 校准。两种正样本源都跑，差值就是"跨越身份"的代价；
# 生产网关用 liveportrait 那份（两个 demo 的默认值）
.venv/bin/python -m pipelines.calibrate --positives liveportrait --au \
    --out out/calibration/liveportrait
.venv/bin/python -m pipelines.calibrate --positives augment --tier mild --au \
    --out out/calibration/augment

# Demo 2：真实跨演员，零生成模型
.venv/bin/python -m pipelines.demo2_cremad --au
```

校准依赖 Demo 1 的输出当正样本，而 Demo 1 又会读校准结果，所以第一次跑 Demo 1 时它会
报一句 "no calibration; using spec defaults" 然后照常生成——生成本身不依赖阈值。
校准完再跑一次 `demo1_liveportrait --skip-generation` 即可用新阈值重判。

每个 demo 输出 `manifest.jsonl`（含 schema 说明 `manifest.schema.md`）、`summary.md`
与逐对对比图。被拒的 pair 一并写进 manifest：没有它们，通过率无法审计，而且它们本身
就是已校准的硬负样本。

## Demo 1：A 级构造，golden config 在源码层确认

`LivePortrait` video-to-video + `--animation-region exp` + `--no-flag-relative-motion`。
非相对模式下驱动序列 `x_d_exp_lst[i] = driving_template['motion'][i]['exp']`
（`third_party/LivePortrait/src/live_portrait_pipeline.py:228`）不含任何 source
相关项，所以所有身份拿到的是**同一条驱动表情序列**；但它被写进的那个张量并不是整体拷贝——
起点是 source 自己的 `exp`：

```319:319:third_party/LivePortrait/src/live_portrait_pipeline.py
            delta_new = x_s_info['exp'].clone()
```

absolute 分支只覆盖其中一部分（`flag_is_source_video` 为真时取平滑后的驱动序列）：

```367:373:third_party/LivePortrait/src/live_portrait_pipeline.py
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "exp":
                    for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                        delta_new[:, idx, :] = x_d_exp_lst_smooth[i][idx, :] if flag_is_source_video else ...
                    delta_new[:, 3:5, 1] = x_d_exp_lst_smooth[i][3:5, 1] if flag_is_source_video else ...
                    delta_new[:, 5, 2]   = x_d_exp_lst_smooth[i][5, 2]   if flag_is_source_video else ...
                    delta_new[:, 8, 2]   = x_d_exp_lst_smooth[i][8, 2]   if flag_is_source_video else ...
                    delta_new[:, 9, 1:]  = x_d_exp_lst_smooth[i][9, 1:]  if flag_is_source_video else ...
```

`exp` 的形状是 `(1, 21, 3)`，共 63 个数，上面五行只覆盖 **45 / 63**：13 个关键点
（1,2,6,11–20）的全部 3 个坐标 = 39，加上关键点 3、4 的 y、5 的 z、8 的 z、9 的 y/z，共 6 个。
剩下 **18 / 63 来自 source**：关键点 0、7、10 完整保留 source 自己的值（9 个），
关键点 3、4 的 x/z（4 个）、5 的 x/y（2 个）、8 的 x/y（2 个）、9 的 x（1 个）也保留。
所以"表情张量被整体拷贝、字节相同、无 source 相关项"是过强的说法，准确的说法是：
**被驱动的那 45 个分量在所有 source 上逐元素相同**。此外 `stitching(x_s, x_d_i_new)`
（`live_portrait_pipeline.py:411`）是一个吃 `x_s` 的学习模块，最终关键点在数值上仍依赖 source。

这不削弱构造路线的逻辑：表情相等仍然是生成过程的数学后果（同一条序列的同一份拷贝，
且这个子集就是 LivePortrait 定义的"表情区域"），也仍然解释 absolute / relative 的对照——
relative 分支第一项显式是 `source_template_dct['motion'][i]['exp']`
（`live_portrait_pipeline.py:214`），按构造就不该表情相同。只是这条保证的**范围**比原先写的窄：
它覆盖 45 个被驱动分量，不覆盖整个表情表示。

`scale_new / t_new / R_new` 则全部来自 source，各自保留自己的头部姿态、位移、背景与光照。
21 个关键点里有 8 个（0,3,4,5,7,8,9,10）不被完整覆盖，身份不被抹掉。

两个必须处理的实现细节：

1. `n_frames = min(len(source), len(driving))` 之后要进 Kalman 平滑，**所有 source 必须预先
   裁到与 driver 完全等长**，否则平滑窗口不同，"完全相同"退化成"几乎相同"。
2. 绝对模式的已知代价是 driver 人脸形状渗漏。这个失败对任何表情指标都不可见，所以单独查：
   实测 `cos(输出, 自己的 source)` 为 0.86–0.92，`cos(输出, driver)` 为 −0.10…+0.07，
   **没有观测到渗漏**（relative 模式身份保持略好，0.90 vs 0.86，方向符合预期）。

原子单元设计成三元组而非 pair：driver D + 两个 source 输出 A′、B′，可产出 `(D,A′)`、`(D,B′)`、
`(A′,B′)`。前两种的 ref 侧是真人实拍，免费缓解"两侧都是生成视频"的分布偏置。

## 验证层的设计

### rank-1 时序可辨识性

绝对阈值说不出"足够细"。阈值只能说"这两张脸相差小于 ε"，说不了"这个匹配比视频自身的
逐帧表情变化还细"——而后者才是真正的要求。所以对每个参考帧 t，在目标视频 `[t-W, t+W]`
内找最近帧，要求 argmin 落在 t±1（`g_rank1`），并把对齐距离与**同一段视频自己** t±k 帧的
距离作比（`ratio = d_deform / d_self`，门是 `g_ratio`）。

**但要说清 `g_ratio` 现在的实际状态。** "必须赢过时序邻帧"（即 `ratio < 1`）是设计意图；
手选 `max_ratio = 0.80` 会全拒（连构造保证相同的正样本 ratio 中位数也是 1.4717），所以改成拟合，
而**实际生效的值是 4.4321**，它在正样本上的通过率 **0.939**，已经接近空门，只拦严重错配。
细粒度严格性实际来自 `g_rank1`（正样本通过率 **0.378**）加 pair 级聚合，不是来自 ratio。

同步视频对最难的负样本本来就是时序相邻帧：身份、姿态、光照、背景全同，只差几十毫秒的
肌肉运动。要求跨身份匹配赢过它们，就是"差不多的我不要"的可执行定义。静止段无法支持这个
检验（保持不动的表情确实等于自己的邻帧），这些帧被标为 **untestable** 而不是悄悄放行。

实测：对齐组 exact 23.4% / ±1 47.5%，平移 −5 帧组的 argmin 平均偏移 +3.20、平移 +5 帧组
−4.27 —— 负控制跟着平移走，说明这个排序检验确实在测东西。

### 指标集与冗余性

全部先做 per-person neutral 减法（Baltrušaitis FG2015 两趟中位数法）。这一步不可省：
不做的话度量被脸型主导，而不是被肌肉活动主导。

| | 指标 | 来源 |
| --- | --- | --- |
| M1 | ARKit blendshape 子集（剔除已知失效通道，保留左右非对称特征）| MediaPipe，Apache-2.0 |
| M2 | 身份归一化关键点形变场 + 分区相对失配 | MediaPipe |
| M3 | AU 激活（8 通道）| OpenFace 3.0，研究许可 |

合取而非加权平均：加权平均会让一个自信的指标掩盖另一个的否决，这正是严格验证器要防的
失败模式。但合取只有在指标真的独立时才有意义，所以校准会算两两相关：

- 一开始 `d_region` 与 `d_deform` 相关系数 **0.987** ——"最差分区的 RMS"被嘴唇主导，
  和全局 RMS 说的是同一件事，等于同一个网关按了两次。
- 改成**分区失配除以该分区自身的运动量**（"有没有哪块脸在成比例地做错事"）之后，相关系数
  降到 0.194，而且 AUC 反而升了（+3 帧 0.556→0.574，+10 帧 0.664→0.743）。
- M3 与所有几何指标的相关系数 ≤ 0.25，确实是独立的一路——它读像素，M1/M2 读的是同一套
  landmark 回归，本来就可能一起错。

### 双侧过滤

表情相似度**高** AND 其余因素相似度**低**，是一份可检查的数据契约：ArcFace 身份余弦要低、
姿态差要大、背景要不同、表现力要超过语料分位数（近中性帧匹配是平凡的，指标虚高而信息量为零）。

身份门槛自带可分性自检，因为这里踩过一个安静的坑：`align112` 原本用
`cv2.estimateAffinePartial2D(..., LMEDS)`，只有 5 个点时鲁棒估计会挑到退化子集，
对齐轻微跑偏**不会报错**，只会把所有 embedding 拉向平均脸，于是不同人的余弦一起升高、
身份门槛静默失效（当时 177 对里 176 对因"identity cosine"被拒）。改成精确最小二乘
相似变换后：同演员中位数 0.894，不同演员中位数 0.021（q95 0.115），EER 阈值 0.160。

### 阈值校准：两种正样本源

- `--positives augment`：同一张脸的外观增广孪生体，量的是**噪声地板**。
- `--positives liveportrait`：Demo 1 的输出，不同的人、同一个表情张量，量的是**跨身份地板**。

两者之差就是"跨越身份"的代价，而这个代价**每个指标都不一样**（正样本距离中位数）：

| 指标 | 噪声地板（同一张脸）| 跨身份地板（表情相同的两个人）| 倍数 |
| --- | ---: | ---: | ---: |
| `d_bs` | 0.0044 | 0.0167 | 3.8× |
| `d_gaze` | 0.0552 | 0.1502 | 2.7× |
| `d_au` | 0.1029 | 0.1427 | 1.4× |
| `d_region` | 0.987 | 1.090 | 1.1× |
| `d_deform` | 0.0209 | 0.0204 | **1.0×** |

最后一行值得停一下：**M2 形变场在单张脸上的测量噪声，已经和两个不同的人做同一个表情时的
全部差异一样大**。也就是说 M2 没有剩余预算去分辨"表情不同"，它读到的几乎全是噪声——
这解释了为什么它在 3 帧尺度上的 AUC 只有 0.553。反过来，M1 blendshape 在固定脸上重复性
好得多（3.8×），所以它的跨身份差异是真信号。

这也定位了第一版 Demo 1 全军覆没的原因：用噪声地板拟合的 `d_bs` 阈值是 0.0092，
而跨身份正样本的中位数是 0.0167 —— 阈值卡在正样本分布的正中间，任何跨身份 pair 都够不着。
所以生产网关必须用跨身份正样本校准。

硬负样本取**同一对 pair 平移 k 帧**，保证正负样本处在同一个噪声机制里——否则 MediaPipe
VIDEO 模式的时序平滑会让同段内负样本显得异常接近，AUC 会掉到 0.5 以下。

**必须报的一件事：五个距离阈值全部是退化回退值，不是按 95% precision 拟合出来的工作点。**
`fit_threshold` 找不到同时满足 `precision ≥ 0.95` 与 `recall ≥ 0.5` 的阈值时，会退回
**正样本分布的 q90**，并在 report 里打上 `target_precision_unreachable: true`——让失败可见，
而不是静默清空数据集。`out/calibration/liveportrait.json` 里 `d_bs / d_deform / d_gaze /
d_region / d_au` **五个全部**带 `separated: false` + `target_precision_unreachable: true`，
且 `recall` 五个完全相同 = **0.8983（415/462）**，这就是它们都走了同一条回退分支的证据。

所以 `d_bs ≤ 0.047431`、`d_deform ≤ 0.034109`、`d_gaze ≤ 0.310878`、`d_region ≤ 1.352156`、
`d_au ≤ 0.219957` 的含义是**"正样本包络"阈值**（放过 90% 的已知真阳性），
**不是**"能以 95% precision 区分同表情与差 3 帧"的判别阈值。它们的作用是拦粗差。
`augment.json` 的情况完全一样（五个指标同为 `recall = 0.8996`、`target_precision_unreachable: true`）。
判别力实际来自 `g_rank1`（正样本通过率 0.378）、`g_energy`（0.442）和 pair 级聚合（AUC 0.993）。

**`min_energy` 是从 CREMA-D 搬过来的，不是按被评估的语料算的。** 它取语料 energy 的 35% 分位数，
而这里的"语料"永远是 CREMA-D，与 `--positives` 选什么无关：两份校准文件里 `min_energy`
**逐位相同**（`0.023649911954998968`）。这个阈值被套到 LivePortrait 输出上，而后者 energy 明显更低
（交付的 pair 01 `med_energy` 只有 **0.0088**），于是 `g_energy` 在正样本上通过率只有 **0.442**，
成为第二个瓶颈门——这一部分是跨语料阈值搬运的产物，不是数据质量问题。正确做法是按目标语料自己的
energy 分布重算（代码支持 `--energy-percentile`），本次没做。

## 明确不走的路

- **换脸（inswapper / SimSwap）**：模型已被 InsightFace 下架、仅限非商业；128×128 是唇部
  微动与虹膜细节的硬天花板；更关键的是表情保持只是重建损失的软副产品——眼睛只占人脸 5.6%
  面积却承载 40% 的人类注视，是被最严重欠优化的区域。而业界修补换脸表情崩坏的标准工具
  **就是 LivePortrait**，那不如直接用它当生成器。
- **img2img 扩散做风格化**：无表情保持约束、无时序模型，会逐帧改动嘴形眼神。对一份以表情
  相等为唯一卖点的数据集是破坏性的。风格/背景变化改用 ffmpeg 色彩分级 + LUT + 噪点 +
  分割换背景，零模型时间且完全不触碰人脸几何。
- **FLAME / pytorch3d**：Mac 上 pytorch3d 官方 CPU-only 且无 arm64 wheel；渲染非照片级；
  FLAME 需注册且非商业。最重的路径换最少的视觉收益。

## 已知限制

- **逐帧粒度未达成**：单指标在 3 帧尺度上 AUC ≈ 0.5–0.65。要把上限抬上去，唯一已知办法是
  M4——在 FaRL/SigLIP2 backbone 上用 FEC strong-agreement triplet 微调一个感知指标
  （这是唯一经人类判断验证过的一路，可望从 ~53% 抬到 ~80%+）。这一步需要 GPU，不在本次范围。
- **测量分辨率就是粒度上限**：脸在解码帧里占多少像素直接决定 landmark 精度。Demo 1 的
  summary 里列了每段的 **crop box 边长**（118–526 px），一对里较小的那侧是瓶颈。
  注意这一列不是人脸高度：`face_px = crop_box[2]`，而框边长 = 人脸跨度 × `margin (1.9)`，
  所以真实人脸跨度要除以 1.9——driver 约 **277 px**（不是 526），s13 约 **63 px**（不是 120）。
  s13 是全身镜头，人脸只占画面高度约 7%，MediaPipe 的短距 BlazeFace 整帧检测直接返回空——
  不是报错，是安静地给出全无效的一段。现在用由粗到细的分块搜索兜底。
- **单一生成器偏置**：Track A 目前只有 LivePortrait 一个生成器，需要引入第二个做交叉验证。
- **样本量小**：Demo 1 只有 3 个 source × 1 个 driver，校准正样本只有 6 对。

## 放大路径（GPU 侧）

MEAD（1080p、7 视角、60 演员）扩身份规模；FEC triplet 微调 M4 感知指标；第二个生成器摊薄
单一生成器偏置；DISFA（27 人看同一段刺激视频、每帧由认证 FACS coder 标了 12 个 AU 的 0–5
强度）作人工标注评测集。

## 许可证

若数据集要商用，干净栈只有：CREMA-D（**ODbL，可商用**）+ MediaPipe（Apache-2.0）+
LivePortrait 代码（MIT）。注意 LivePortrait 附带的 InsightFace 检测模型是非商业的，商用需
替换；OpenFace 3.0 与 ArcFace 权重均为研究许可，本仓库只在离线验证环节使用，
**不进交付物**。
