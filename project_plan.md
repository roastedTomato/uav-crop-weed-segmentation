# 项目规划 / Project Plan

**UAV Crop-Weed Segmentation for Precision Agriculture**

中文：本文件统一保存项目说明和待办框架。代码、运行结果及必要的技术解释写在 `project_todo.ipynb`，全部使用英文。以下待办用于规划，不记录历史完成情况。

English: This file contains the project description and task framework. Code, outputs, and necessary technical explanations belong in `project_todo.ipynb`, in English only. The checklist is a planning framework, not a record of previous work.

## 1. 确认数据和来源 / Define the Problem and Review the Data

### 1.1 问题定义 / Problem Definition

**应用场景与目标 / Application and Goal**

中文：利用无人机拍摄的高粱田 RGB 图像，识别作物、杂草及背景/土壤的位置。输出用于展示图像内的杂草分布，为精准农业中的杂草监测提供实验性支持。

English: Identify crop, weed, and background/soil regions in UAV RGB images of sorghum fields. The output visualizes weed distribution within an image and provides experimental support for weed monitoring in precision agriculture.

**研究问题 / Research Question**

中文：在按原始 UAV 图像隔离的数据划分下，RGB-only 分割模型能否有效区分作物与杂草，并优于简单基线？

English: With data splits separated by original UAV image, can an RGB-only segmentation model distinguish crops from weeds and outperform simple baselines?

**问题类型 / Problem Type**

中文：多类别语义分割，为每个有效像素分配一个类别。

English: Multiclass semantic segmentation, assigning one class to each valid pixel.

**输入 / Input**

中文：UAV RGB 图像或由原始图像裁剪的图块，形状为 H × W × 3。仅使用红、绿、蓝通道。训练时提供对齐的人工分割标签；预测时仅输入 RGB 图像。输入尺寸与预处理在步骤 3 确定。

English: UAV RGB images or patches cropped from original images, with shape H × W × 3. Only red, green, and blue channels are used. Training requires aligned human-annotated masks; inference requires only RGB images. Input size and preprocessing will be defined in Step 3.

**输出 / Output**

中文：H × W × 3 的类别概率图，沿类别维度取最大概率，得到 H × W 的整数分割图。

English: An H × W × 3 class probability map. Taking the highest-probability class at each pixel produces an H × W integer segmentation mask.

**类别 / Classes**

| ID | 中文 | English | 定义 / Definition | 标签 RGB / Mask RGB |
| --- | --- | --- | --- | --- |
| 0 | 背景/土壤 | Background / Soil | 标注中的土壤与背景 / Annotated soil and background | (199, 199, 199) |
| 1 | 作物/高粱 | Crop / Sorghum | 标注中的高粱作物 / Annotated sorghum plants | (31, 119, 180) |
| 2 | 杂草 | Weeds | 所有标注杂草统一为一类 / All annotated weeds treated as one class | (255, 127, 14) |

中文：类别与颜色依据为[数据集说明](https://www.kaggle.com/datasets/bouhadjer/crop-weed-segmentation-uav-rgb-indices)和原始 readme。类别 ID 是项目约定。边缘补齐区域应使用独立忽略标记，不属于上述三类。

English: Classes and colors follow the [dataset description](https://www.kaggle.com/datasets/bouhadjer/crop-weed-segmentation-uav-rgb-indices) and original readme. Class IDs are project conventions. Padded regions must use a separate ignore label and do not belong to these three classes.

**评价目标 / Evaluation Goal**

中文：以三个类别的 Mean IoU 为主要指标，单独关注 weed-class IoU，辅以 Dice 和各类别 IoU。比较多数类别基线、小型 encoder-decoder CNN 和 U-Net。指标实现与实验设置在后续步骤确定。

English: Use Mean IoU across the three classes as the primary metric, with particular attention to weed-class IoU. Report Dice and per-class IoU as supporting metrics. Compare a majority-class baseline, a small encoder-decoder CNN, and U-Net. Metric implementations and experimental settings will be defined in later steps.

**范围与约束 / Scope and Constraints**

中文：使用 Python 3.11+、TensorFlow/Keras 和 Jupyter notebook，遵循《Deep Learning with Python》第三版的项目流程。核心范围为 RGB-only 分割、模型对比与错误分析。不包含植被指数、杂草物种识别、实例计数、地理定位、自动喷药或无人机部署。学生必须理解并验证所有代码。

English: Use Python 3.11+, TensorFlow/Keras, and a Jupyter notebook, following the workflow in Deep Learning with Python, third edition. The core scope covers RGB-only segmentation, model comparison, and error analysis. Vegetation indices, weed species identification, instance counting, geolocation, automatic spraying, and UAV deployment are outside the core scope. The student must understand and verify all code.

### 1.2 数据获取 / Data Acquisition

**数据集与获取方式 / Dataset and Acquisition Method**

中文：数据集为 Crop-Weed Segmentation UAV RGB+Indices，[发布页面](https://www.kaggle.com/datasets/bouhadjer/crop-weed-segmentation-uav-rgb-indices)。通过 Kaggle 公开下载接口获取 ZIP；已有本地压缩包时复用，避免重复下载。仅提取原始 RGB 图像、对应标签和说明文件，暂不提取植被指数及预先增强的 patches。

English: The dataset is Crop-Weed Segmentation UAV RGB+Indices ([dataset page](https://www.kaggle.com/datasets/bouhadjer/crop-weed-segmentation-uav-rgb-indices)). Obtain the ZIP through Kaggle's public download endpoint, reusing an existing local archive when available. Extract only original RGB images, corresponding masks, and documentation, leaving vegetation indices and pre-augmented patches in the archive.

**存放路径 / Storage Paths**

| 内容 / Content | 项目内路径 / Project-relative Path |
| --- | --- |
| 原始下载 / Downloaded archive | `data/downloads/dataset.zip` |
| 原始 RGB 与标签 / Original RGB images and masks | `data/raw/` |
| 获取代码 / Acquisition code | `project_todo.ipynb`, section 1.2 |

**实际运行记录 / Execution Record — 2026-10-01**

中文：复用本地已下载 ZIP，并重新计算 SHA-256。文件大小为 2,590,700,651 bytes，校验值与固定快照一致。读取 42 个原始图像/标签文件及 3 个说明文件；已存在的提取文件与压缩包内容逐字节一致，因此无需重复写入。

English: Reused the downloaded local ZIP and recomputed its SHA-256. The archive is 2,590,700,651 bytes and matches the pinned snapshot. Read 42 original image/mask files and three documentation files. Existing extracted files matched the archive byte for byte, so no rewriting was necessary.

SHA-256: `1cd9eddab9c5e132f679f0b223591c2ef549282920b447a693525f973e4775e3`

中文：校验值用于识别本项目所使用的快照，并非作者提供的官方校验值。若未来下载版本不同，代码会停止，需查明版本变化后再更新快照。来源与许可、文件质量和数据泄漏问题分别在 1.3–1.5 处理。本次验证了本地复用与提取路径，未重新执行网络下载分支。

English: This checksum identifies the snapshot used in this project; it is not an official checksum supplied by the author. If a future download differs, the code stops until the version change is reviewed. Licensing, file quality, and leakage are addressed in Sections 1.3–1.5. This run verified local reuse and extraction; the network download branch was not rerun.

### 1.3 来源与许可 / Provenance and Licensing

**原始研究与数据 / Original Research and Data**

中文：原始研究为 Genze, N., Ajekwe, R., Güreli, Z., Haselbeck, F., Grieb, M., & Grimm, D. G. (2022), *Deep learning-based early weed segmentation using motion blurred UAV images of sorghum fields*, Computers and Electronics in Agriculture, 202, 107388，[论文 DOI](https://doi.org/10.1016/j.compag.2022.107388)。作者在 [Mendeley Data 第 4 版](https://data.mendeley.com/datasets/4hh45vkp38/4) 发布数据，并提供 [GitHub 代码](https://github.com/grimmlab/UAVWeedSegmentation)。数据发布说明指出图像来自德国南部的高粱试验田，使用 UAV 拍摄并进行人工标注。因此，它不是新西兰田地数据，不能据此声称已验证新西兰农业场景的性能。

English: The original study is Genze, N., Ajekwe, R., Güreli, Z., Haselbeck, F., Grieb, M., & Grimm, D. G. (2022), *Deep learning-based early weed segmentation using motion blurred UAV images of sorghum fields*, Computers and Electronics in Agriculture, 202, 107388 ([paper DOI](https://doi.org/10.1016/j.compag.2022.107388)). The authors released [Mendeley Data version 4](https://data.mendeley.com/datasets/4hh45vkp38/4) and [GitHub code](https://github.com/grimmlab/UAVWeedSegmentation). The data description identifies a sorghum experimental field in Southern Germany, with UAV imagery and manual annotations. This is not New Zealand field data, so performance in New Zealand agricultural settings remains untested.

**修改版来源 / Modified Dataset Provenance**

中文：本项目实际下载的是 [Bouhadjer 的 Kaggle 修改版](https://www.kaggle.com/datasets/bouhadjer/crop-weed-segmentation-uav-rgb-indices)。本地 `data/raw/License.txt` 与 `readme.txt` 说明其包含 RGB、标签、RGB 植被指数和切图/增强版本，并关联题为 *Dual-Branch Deep Learning with RGB-Based Vegetation Indices for Precise Weed Segmentation in UAV Crop Images* 的研究。本次未查到该题名可独立核实的正式出版记录或 DOI，因此只记录为发布者的来源说明，不将其作为已核实的正式论文引用。原始研究和本项目使用的数据修改版需要分别注明。

English: This project downloaded [Bouhadjer's modified Kaggle dataset](https://www.kaggle.com/datasets/bouhadjer/crop-weed-segmentation-uav-rgb-indices). The local `data/raw/License.txt` and `readme.txt` describe RGB images, masks, RGB vegetation indices, and patched/augmented versions, associated with a study titled *Dual-Branch Deep Learning with RGB-Based Vegetation Indices for Precise Weed Segmentation in UAV Crop Images*. No independently verifiable publication record or DOI for that exact title was located in this review. Treat it as the publisher's provenance statement rather than a verified published paper. Credit the original research and the modified dataset separately.

**许可证与署名 / Licensing and Attribution**

| 依据 / Evidence | 查验结论 / Finding |
| --- | --- |
| 原始 Mendeley Data 第 4 版 / Original Mendeley Data version 4 | 页面明确标注 MIT / Explicitly lists MIT |
| Kaggle 发布页面 / Kaggle data card | 前次查验记录为 CC BY 4.0；本次网页工具未返回正文 / Previously recorded as CC BY 4.0; the page body was unavailable through the web tool in this review |
| 本地 `data/raw/License.txt` / Local license file | 自定义授权文字允许研究、教育和商业使用，要求注明原始作者和修改者 / Custom permission text allows research, educational, and commercial use with attribution to original creators and the modifier |

中文：页面许可标记和包内授权文本不是同一份许可证全文。本项目保留包内许可证，报告与 README 同时引用原始论文、原始数据 DOI 和实际使用的 Kaggle 页面，并注明 RGB-only 选择及后续预处理修改。下载包和原始数据不随课程代码仓库重新发布，提供来源与获取方法。若将来重新分发数据，需进一步确认适用条款并保留所要求的版权、许可和署名文字。包内修改者姓名写作 “Mohaned El Amine BOUHADJER”，与此前页面记录的拼写不同；其 example.com 邮箱不作为真实联系地址，署名优先链接 Kaggle 发布者页面并说明包内写法。

English: The page's license label and the bundled permission text are not the same license document. Preserve the bundled license. Cite the original paper, original data DOI, and the Kaggle dataset actually used in the report and README, and describe RGB-only selection and subsequent preprocessing changes. Do not republish downloaded or original data in the course code repository; provide source links and acquisition instructions. Any future redistribution requires confirming the applicable terms and preserving required copyright, license, and attribution notices. The bundled modifier name is spelled “Mohaned El Amine BOUHADJER,” differing from the previously recorded page spelling. Do not use its example.com email as a real contact address; link to the Kaggle publisher and document the bundled spelling.

**伦理与适用限制 / Ethics and Intended Use**

中文：项目使用公开农业影像，不新增无人机采集或个人数据收集。展示前仍需检查图片是否包含可识别个人或敏感信息；当前未完成逐张隐私审查。输出仅用于课程实验和杂草分布可视化，不直接用于自动喷药。报告需披露 AI 辅助，并如实说明地域差异、标注局限和来源相关性。原始数据页面说明拍摄有约 10% 的重叠；具体跨集合空间重叠需在 1.5 单独查验，不能因文件名不同就认定完全独立。

English: The project uses public agricultural imagery without collecting new UAV or personal data. Check images for identifiable people or sensitive information before displaying them; a full image-by-image privacy review has not been completed. Outputs are for coursework and weed-distribution visualization, not direct automatic spraying. Disclose AI assistance and describe geographical differences, annotation limitations, and source correlations honestly. The original data page reports approximately 10% capture overlap. Cross-split spatial overlap requires separate review in Section 1.5; distinct filenames do not establish complete independence.

查验日期 / Review date: 2026-10-01. 本节为资料核验，不需要新增代码。 / This section is a documentary review and requires no additional code.

### 1.4 文件检查 / File Inspection

**检查方法 / Inspection Method**

中文：本节保留三项必要检查。先按文件名的来源编号建立 RGB 与标签的一对一对应关系（处理额外测试图的 `_img` / `_msk` 后缀），不靠排序配对；读取每对文件，确认尺寸一致；再检查标签每个像素是否属于三个规定 RGB 颜色。最后仅从 trainval 展示三组原图、标签和前景叠加图，目视检查是否存在明显错位。没有计算用于训练的测试集统计量，也没有进行模型评估或调整划分。

English: This section implements three essential checks. Match RGB images and masks by source identifiers, handling the `_img` / `_msk` suffixes in the extra test sets rather than relying on sorting. Load each pair and verify equal dimensions, then validate every mask pixel against the three expected RGB colors. Display three RGB/mask/foreground-overlay examples from trainval only and visually review alignment. No test-derived training statistics, model evaluation, or split adjustments are performed.

**自动检查结果 / Automatic Check Results — 2026-10-01**

| 原始集合 / Original Group | 配对数量 / Pairs | 尺寸（宽×高）/ Dimensions (W×H) | 结果 / Result |
| --- | ---: | --- | --- |
| trainval | 12 | 5472 × 3648 | 配对、尺寸、颜色通过 / Pairing, dimensions, colors passed |
| test | 7 | 5472 × 3648 | 配对、尺寸、颜色通过 / Pairing, dimensions, colors passed |
| extra_bbch15 | 1 | 2560 × 2816 | 配对、尺寸、颜色通过 / Pairing, dimensions, colors passed |
| extra_bbch19 | 1 | 2560 × 2816 | 配对、尺寸、颜色通过 / Pairing, dimensions, colors passed |

中文：全部 21 对原始图像与标签均可读取，没有缺少或多出的对应标签，尺寸全部一致；未知标签颜色像素为 0。三种规定颜色分别为土壤/背景 (199, 199, 199)、高粱 (31, 119, 180)、杂草 (255, 127, 14)。没有要求每张标签都包含全部三类。本节不检查已有增强 patches；后续切图流程需要单独验证其产出。

English: All 21 original image-mask pairs loaded successfully, with no unmatched images or masks and matching dimensions throughout. There were zero pixels with unexpected mask colors. Expected colors are soil/background (199, 199, 199), sorghum (31, 119, 180), and weeds (255, 127, 14). Individual masks are not required to contain all three classes. Existing augmented patches are outside this inspection; subsequent patch generation requires its own validation.

**叠加图抽查 / Visual Alignment Review**

中文：抽查 `trainval_01`、`trainval_03` 和 `trainval_08` 各一个 768 × 768 局部窗口。代码按窗口中的前景标注面积选择可见植物较多的区域，仅用于检查对齐，不用于筛选训练数据或改动划分。我于 2026-10-01 手动检查并确认这三组图像与标签对齐没有问题。蓝色作物与橙色杂草标注覆盖对应植物区域。部分人工标注边界较粗，图像有运动模糊。结论仅适用于这三个窗口，不代表所有原图或边缘区域的标注质量都已目视核验；也不证明类别判断完全正确或不存在数据泄漏。

English: Reviewed one 768 × 768 crop each from `trainval_01`, `trainval_03`, and `trainval_08`. The code selects plant-containing windows by annotated foreground area for alignment inspection only, without filtering training data or changing splits. On 2026-10-01, I manually reviewed these three examples and confirmed no image-mask alignment issues. Blue crop and orange weed annotations cover corresponding plant regions. Some manual boundaries are coarse and the imagery contains motion blur. This finding applies only to these three windows, not to all originals or edge regions, and does not establish perfect class labeling or absence of leakage.

![Alignment review: trainval originals only](artifacts/figures/file_inspection_overlays.png)

**代码与记录 / Code and Records**

中文：英文代码与运行输出位于 `project_todo.ipynb` 的 1.4；检查结果及抽查坐标位于 `artifacts/FILE_INSPECTION.json`；叠加图位于 `artifacts/figures/file_inspection_overlays.png`。目视确认状态已更新为 `visual_review: confirmed_by_student`，并记录日期与结论。这是我的手动确认，而非自动对齐检测。重新运行检查代码会重新生成待审查记录，新生成图片需要再次确认。

English: English code and execution outputs are in Section 1.4 of `project_todo.ipynb`. Inspection results and crop coordinates are saved in `artifacts/FILE_INSPECTION.json`; the figure is in `artifacts/figures/file_inspection_overlays.png`. The status is now `visual_review: confirmed_by_student`, with the review date and finding recorded. This is my manual confirmation, not an automatic alignment detector. Rerunning the inspection code regenerates a pending review record; newly generated figures require confirmation again.

### 1.5 来源与泄漏风险 / Source Mapping and Leakage Risks

**查验范围与方法 / Review Scope and Method — 2026-10-02**

中文：本节核验原图来源、已有 patches 的对应关系、发布者的划分流程，以及拍摄元数据和空间相关性。只进行来源核验，不训练模型、不读取模型测试表现，也不修改划分。代码位于英文 notebook 的 1.5。

English: This section reviews original-image provenance, existing patch mappings, the publisher's splitting workflow, capture metadata, and spatial correlations. It performs provenance checks only, without training models, inspecting model test performance, or modifying splits. The code is in Section 1.5 of the English notebook.

**1. Patch 与原图对应关系 / Patch-to-Original Mapping**

中文：实际文件名由原图名称加图块序号组成，例如 `trainval_01165.png` 对应原图 `trainval_01` 的第 165 号图块（序号从 0 开始）。代码解析来源编号，按每行 22 个、步长 256 的裁剪网格恢复坐标，逐像素比较原图标签裁剪与已发布 patch 标签的有效区域。12 张 trainval 原图各 330 个未增强图块，合计 3,960 个；7 张 test 原图各 330 个，合计 2,310 个。全部 6,270 个未增强 patch 标签均通过，序号连续，均有同名 RGB patch。

English: Actual filenames combine an original-image name with a zero-based patch index. For example, `trainval_01165.png` refers to patch 165 of `trainval_01`. The code parses source IDs, reconstructs coordinates on a 256-pixel-stride grid with 22 columns, and compares each published patch mask against the original mask crop over valid pixels. All 6,270 unaugmented masks passed: 3,960 from 12 trainval originals and 2,310 from seven test originals, with 330 patches per original. Indices are continuous and each mask has a corresponding RGB patch filename.

中文：这验证了未增强标签图块的来源与裁剪位置，不等于逐像素验证 JPEG RGB 图块或全部增强变换。边缘比较排除原图外补齐区域。额外 BBCH15/19 的图块对应关系未在本节核验。项目后续仍优先从原始 RGB 与标签重新切图，保存明确来源和坐标，仅在训练集在线增强。

English: This verifies provenance and coordinates of unaugmented mask patches, not pixel equality of JPEG RGB crops or all augmentation transformations. Comparisons exclude padding outside the original image. Extra BBCH15/19 patch mappings were not checked here. The project will still favor repatching original RGB images and masks with explicit source IDs and coordinates, with online augmentation restricted to training data.

**2. 原始集合与划分依据 / Original Groups and Split Evidence**

中文：数据集已经分好了 12 张 trainval 原图、7 张 test 原图和两张额外测试图。代码检查了全部 21 张 RGB 原图，没有完全相同的图片。作者的 [切图代码](https://github.com/grimmlab/UAVWeedSegmentation/blob/main/save_patches.py#L10) 也是先从各文件夹读取原图，再切成小图块。

English: The dataset already separates 12 trainval originals, seven test originals, and two extra test images. Code checks found no identical RGB images among all 21 originals. The author's [patching code](https://github.com/grimmlab/UAVWeedSegmentation/blob/main/save_patches.py#L10) also reads originals from their existing folders before cutting them into smaller patches.

中文：但 trainval 还没有分成我们要用的训练集和验证集。如果直接随机分配小图块，同一张原图的图块可能同时进入这两个集合。作者的 [训练代码](https://github.com/grimmlab/UAVWeedSegmentation/blob/main/train.py#L23) 直接对图块列表进行划分，没有明确要求同一原图的图块放在一起，因此我们需要自行保证这一点。

English: We still need to divide trainval into our training and validation sets. Randomly assigning individual patches could put patches from the same original image into both sets. The author's [training code](https://github.com/grimmlab/UAVWeedSegmentation/blob/main/train.py#L23) splits a list of patches without explicitly keeping patches from each original together, so we need to ensure this ourselves.

中文：第二步直接按原图分配训练集和验证集，让每张原图的所有小图块和增强版本跟随它进入同一集合。第一步已完成图像检查，第二步不再重复检查。

English: Step 2 will directly assign original images to training or validation, keeping all patches and augmented versions of each original in the same set. Image inspection was covered in Step 1 and will not be repeated in Step 2.

**3. 拍摄时间、GPS 与空间风险 / Capture Times, GPS, and Spatial Risks**

**检查原因 / Reason**

中文：不同原图也可能拍到同一块地面。GPS 检查发现 `trainval_11` 与 `test_05` 距离较近，因此需要查看图片是否重叠。

English: Different original images may show the same ground area. GPS screening found that `trainval_11` and `test_05` were close together, so their images needed review for overlap.

**人工检查 / Manual Review**

中文：我于 2026-10-02 手动检查了全部 21 张原始图像，未发现图像之间存在重叠。

English: On 2026-10-02, I manually inspected all 21 original images and found no overlap between them.

**结论 / Conclusion**

中文：全部原始图像通过目视重叠检查。该结论来自我的人工检查，不等于已验证不同航次完全独立。

English: All original images passed my visual overlap check. This finding comes from manual inspection and does not establish independence between flights.

**结论与第二步要求 / Conclusion and Requirements for Step 2**

中文：已确认主要未增强标签图块的原图来源，没有完全重复的大图；我手动检查全部原始图像后也未发现重叠。接下来直接把 trainval 按原图分为训练集和验证集。同次拍摄的图像可能有相似光照和地面环境，报告中需说明这一限制，可能影响模型泛化能力。

English: Main unaugmented mask patches have verified original-image sources, and no originals are exact duplicates. My manual review of all original images also found no overlap. Next, divide trainval into training and validation by original image. Images from the same capture session may share similar lighting and ground conditions; the report should acknowledge this limitation, which may affect the model generalization capability.

## 2. 数据划分 / Data Splitting

中文：使用固定随机种子 42，将 trainval 的 12 张原图分为 9 张训练图和 3 张验证图，并取得各自对应的标签路径。

English: Use random seed 42 to split the 12 trainval originals into nine training images and three validation images, with their corresponding mask paths. 
## 3. 数据探索与预处理 / Data Exploration and Preprocessing

- [ ] 展示 RGB、标签和叠加图，检查质量与类别比例。 / Display RGB images, masks, and overlays; inspect quality and class proportions.
- [ ] 检查缺失、错误颜色和尺寸不一致。 / Check missing files, invalid colors, and dimension mismatches.
- [ ] 确定输入尺寸、归一化和整数标签编码。 / Define input size, normalization, and integer label encoding.
- [ ] 标签缩放用最近邻，忽略补齐像素。 / Use nearest-neighbor mask resizing and exclude padded pixels.
- [ ] 建立 TensorFlow 加载、batch 和预取流程。 / Build TensorFlow loading, batching, and prefetching pipelines.
- [ ] 仅训练集增强，图像与标签几何变换同步。 / Augment training data only and synchronize image-mask geometric transforms.
- [ ] 仅训练集计算统计量与类别权重；检查 batch 形状和数值。 / Compute statistics and class weights from training data only; inspect batch shapes and values.

## 4. 基线模型 / Baseline Models

- [ ] 建立训练集多数类别预测基线。 / Build a majority-class baseline using the training set.
- [ ] 定义 IoU、Dice 和忽略像素处理。 / Define IoU, Dice, and ignored-pixel handling.
- [ ] 实现小型 Keras encoder-decoder CNN。 / Implement a small Keras encoder-decoder CNN.
- [ ] 设置损失、优化器、batch size、轮数与 early stopping。 / Configure loss, optimizer, batch size, epochs, and early stopping.
- [ ] 固定种子，记录环境、训练设置和曲线。 / Fix seeds and record the environment, training settings, and curves.
- [ ] 验证保存、加载与预测，准备 checkpoint。 / Verify saving, loading, and inference; prepare the checkpoint submission.

## 5. U-Net / U-Net Training

- [ ] 实现并解释 encoder、decoder 和 skip connections。 / Implement and explain the encoder, decoder, and skip connections.
- [ ] 使用相同划分和指标，M4 Pro 小规模试跑。 / Use the same splits and metrics; perform a small trial on the M4 Pro.
- [ ] 确认设备、内存与训练时长。 / Check device availability, memory requirements, and training time.
- [ ] 根据验证集选模型，保存权重、配置和曲线。 / Select models using validation data; save weights, configurations, and curves.
- [ ] 检查预测尺寸、类别和重新加载后的结果。 / Verify output dimensions, classes, and predictions after reloading.

## 6. 对比与错误分析 / Comparison and Error Analysis

- [ ] 比较多数类别基线、CNN 和 U-Net。 / Compare the majority-class baseline, CNN, and U-Net.
- [ ] 做有无数据增强的 ablation，保持其他设置一致。 / Run an augmentation ablation with other settings held constant.
- [ ] 模型选择结束后评估测试集，不用测试结果调参。 / Evaluate the test set after model selection; do not tune on test results.
- [ ] 报告 Mean IoU、各类别 IoU、weed-class IoU 和 Dice。 / Report Mean IoU, per-class IoU, weed-class IoU, and Dice.
- [ ] 整理对比表、曲线、指标图和预测图。 / Prepare comparison tables, curves, metric plots, and prediction examples.
- [ ] 分析小杂草漏检、类别混淆、阴影误检与边界误差。 / Analyze missed small weeds, class confusion, shadow-related false positives, and boundary errors.
- [ ] 讨论原图数量、来源相关性、不平衡与泛化限制。 / Discuss source-image count, correlations, imbalance, and generalization limitations.

## 7. 代码与报告 / Code and Report

- [ ] 从头运行 notebook，整理依赖、种子、划分与配置。 / Run the notebook from the beginning and document dependencies, seeds, splits, and configurations.
- [ ] 编写 README 的环境、数据获取和复现步骤。 / Document setup, data acquisition, and reproduction in the README.
- [ ] 撰写引言与相关工作、数据与预处理、模型与训练、结果、错误分析、AI 披露、结论与参考文献。 / Write introduction and related work, data and preprocessing, models and training, results, error analysis, AI disclosure, conclusion, and references.
- [ ] 使用真实结果，核实引用与许可，并理解所有提交代码。 / Use actual results, verify references and licensing, and understand all submitted code.
- [ ] 导出 fp_final_Lastname_Firstname.pdf，整理课程 GitHub 仓库。 / Export fp_final_Lastname_Firstname.pdf and organize the course GitHub repository.

## 8. 展示与提交 / Presentation and Submission

- [ ] 准备 8 分钟展示和 2 分钟 Q&A。 / Prepare an eight-minute presentation and two-minute Q&A.
- [ ] 展示问题、数据、模型、结果、局限与 AI 使用。 / Present the problem, data, models, results, limitations, and AI usage.
- [ ] 准备 demo 及离线结果备用，练习解释代码和关键概念。 / Prepare a demo and offline backup results; practice explaining code and key concepts.
- [ ] 核对评分细则、报告、仓库链接与复现说明，完成 Moodle 提交。 / Check the rubric, report, repository link, and reproduction instructions; submit through Moodle.
