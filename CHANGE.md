# CHANGE

Builder 每次交付更新本文件；回复末尾再贴同一段。Reviewer 对照这里是否诚实。

## 2026-09-19 — 旁路更好时改用 ToonOut∩MODNet

- 做了：第一步仍先跑 isnet（可换 seed 三次）。若 ToonOut/MODNet 彼此更一致、或 isnet 明显多包了一块（旁路几乎是 isnet 的子集且多出来 ≥10%），角色 mask 改用两个旁路的交集；交集太小则取结构更好的那个。原 isnet 落到 `01_character/isnet.png`。**没改 schema、没改 .venv**
- 怎么跑：`.\.venv\Scripts\python.exe -m pytest` ；GPU：`.\.venv\Scripts\python.exe -m layerforge run --input test_input/image_no_sag --out runs/flat --segment cascade --inpaint auto`
- 产物路径：`runs/flat/<uuid8>/steps/01_character/`。296a4352 / 9083053e `chosen_source=peer_and`，不再 FLAG（桌子/床从角色 mask 里拿掉了）。4034 / 458 / 640 / a163 仍用 isnet。908 `missing=[]`。296 仍 `missing=[eye_l,eye_r]`
- 未做 / 已知缺陷：旁路权重仍是 ToonOut + Xenova/modnet，不是官方 Anime-MODNet。640 头发 missing。前景遮挡没做。SAM3 无权重。ORT CUDA 仍缺 `cublasLt64_13.dll`

## 2026-09-19 — 第一步 anime-segmentation QA + 换 seed 重试

- 做了：isnet 出 mask 后做三项校验——连通域（最大块 <60% 或滤掉 <50px 后 N>15）、灰区 0.15–0.85 占比 >25% / 腐蚀 15px 内部高熵 >5%、旁路 ToonOut + MODNet Dice <0.85。不过就换 seed 再抠（ONNX 无随机，seed 只换阈值和输入噪声），最多 3 次；第三次仍失败写 `steps/01_character/FLAGGED.json`、橙色 overlay、`logs/character_qa.json`，`needs_click` 记 `character`。**没改 schema、没改 .venv、没用骨架裁 isnet**。权重：`scripts/download_models.py --only-character-qa`
- 怎么跑：`.\.venv\Scripts\python.exe scripts/download_models.py --only-character-qa` ；`.\.venv\Scripts\python.exe -m pytest` ；GPU：`.\.venv\Scripts\python.exe -m layerforge run --input test_input/image_no_sag --out runs/flat --segment cascade --inpaint auto`
- 产物路径：`runs/flat/<uuid8>/steps/01_character/qa.json`。9083053e 三次都失败并 FLAG（内部熵 + Dice 0.73/0.76，床被 toonout/modnet 抠掉而 isnet 连上）。458 / a163 seed0 过门 `missing=[]`。4034a197 `missing=[eye_r]`。64010c19 `missing=[hair_back,hair_front]`。296a4352 `missing=[eye_l,eye_r]`，Dice 刚过 0.85 **没 FLAG**（桌子连在身体上，是一块超大连通域）
- 未做 / 已知缺陷：贴着角色的同色家具（296 书桌）这三项抓不到。SkyTNT Anime-MODNet 没有公开 ONNX，旁路用同结构 Xenova/modnet。isnet 无真随机 seed。前景遮挡没做。SAM3 无权重。ORT CUDA 仍缺 `cublasLt64_13.dll`

## 2026-09-19 — 角色抠图只认 isnet，骨架不再覆盖

- 做了：`01_character` 改回 anime-segmentation 原 mask，DWPose 只写 `01_pose/` 并给 SAM 部件框/正点，不再用骨架裁角色，也不再用 pose person 扣 body 残差。296a4352 那种头发/袖子被骨架切掉的问题来自这步，不是 isnet。**没改 schema、没改 .venv**
- 怎么跑：`.\.venv\Scripts\python.exe -m pytest` ；GPU：`.\.venv\Scripts\python.exe -m layerforge run --input test_input/image_no_sag --out runs/flat --segment cascade --inpaint auto`
- 产物路径：`runs/flat/<uuid8>/steps/01_character/` 就是 isnet；骨架在 `steps/01_pose/overlay.png`。296a4352 头发/袖子/裙子回来了。4034a197 `missing=[eye_r]`。45825594 / 9083053e / a1639e70 `missing=[]`
- 未做 / 已知缺陷：9083053e 人躺在床上时 isnet 仍会连上家具，这次不靠骨架硬裁。前景遮挡没做。SAM3 无权重。ORT CUDA 仍缺 `cublasLt64_13.dll`

## 2026-09-19 — 形态学收边 + DWPose 提示身体

- 做了：mutex 后对非 overlay 做 `morph_open_px=2`，body 残差先把已切件膨胀 2px 再扣，去掉头发/衣服留在皮肤上的细边。加 `PoseEstimator` / DWPose（YOLOX+RTMPose ONNX）：骨架提示 face/body/arm 框和正点；二次元 YOLOX 经常检不到人时回退整图再估姿态。角色 mask 用骨架+躯干凸包+头部膨胀去裁 isnet（不把张开的四肢凸包填成床）。`body` 只表示姿态里的解剖身体/露肤，不是发缘或家具残差。`7d9f70f9` 已不测。本机重跑 `runs/flat`（新图 `296a4352` 替换了 7d9f）。**没改 schema、没改 .venv、没装 mmpose**
- 怎么跑：`.\.venv\Scripts\python.exe scripts/download_models.py --only-dwpose` ；`.\.venv\Scripts\python.exe -m pytest` ；GPU：`.\.venv\Scripts\python.exe -m layerforge run --input test_input/image_no_sag --out runs/flat --segment cascade --inpaint auto`
- 产物路径：`runs/flat/<uuid8>/`。先看 `steps/01_character/raw.png`（isnet）vs `rgba.png`（pose 裁过）和 `steps/01_pose/overlay.png`。4034a197：`04_segment/20_body` 基本是皮肤，细发缘没了，`missing=[eye_r]`，头发仍成层。9083053e：isnet 把床/窗帘连进角色（人躺在同色家具上），pose 裁掉大部分床和窗帘，头附近枕头/床头板还在；`missing=[clothes]`（睡裙常进 body）。45825594 `missing=[]`。a1639e70 `missing=[]`
- 未做 / 已知缺陷：前景遮挡（原 7d9f）没做。9083053e 头周围仍有枕头，睡裙没单独成衣服层。躺姿骨架膨胀盖不住裙摆时衣服会缺。`layers/20_body` 仍可能含 complete 层补绘。SAM3 无权重。ORT CUDA 仍缺 `cublasLt64_13.dll`

## 2026-09-19 — 修夜景外圈、切件顺序和过大框

- 做了：seam fill / residual 只在 anime-segmentation 角色 mask 里填缝，夜景不再贴到 body 外圈。切件改成衣服/脸 → 头发 → 眼睛，头发负点能打在衣服和脸上。SAM mask 裁回框（+12px）。过大的 DINO 框让位给更贴角色比例的框。hair_front 失败会进 missing。本机重跑 `runs/flat` 6 张。**没改 schema、没改 .venv、没装 SAM3**
- 怎么跑：`.\.venv\Scripts\python.exe -m pytest` ；GPU：`.\.venv\Scripts\python.exe -m layerforge run --input test_input/image_no_sag --out runs/flat --segment cascade --inpaint auto`
- 产物路径：`runs/flat/<uuid8>/`。45825594 / 9083053e / a1639e70 的 `missing=[]`；4034a197 补上 `hair_back` 只缺 `eye_r`；64010c19 切出了 `face`，外圈夜景不再贴上，头发仍在 body（`hair_front`/`hair_back` missing）。失败原因：`steps/04_segment/failures.json`
- 未做 / 已知缺陷：雪景图头发仍没单独成层（DINO 要么只框刘海要么框整个人）。`7d9f70f9` 眼睛仍缺。斗篷半透明里的雪是角色 mask 里的原图像素，不是 seam fill。SAM3 无权重。ORT CUDA 仍缺 `cublasLt64_13.dll`

## 2026-09-19 — 落盘每步预览，扁图跑到 runs/flat

- 做了：每步写到 `steps/`（01 角色抠图、02 tags、03 DINO 框、04 SAM 切件、05 refine、06 occlusion、07 inpaint 洞）。契约仍是 `layers/` `masks/` `preview/` `manifest.json`，没改 schema。`--input` 可以是扁图目录。job id 用 `grok-image-` 的 uuid 前 8 位。本机重跑 `test_input/image_no_sag` 全部 6 张单图。**没改切件算法、没改 .venv**
- 怎么跑：`.\.venv\Scripts\python.exe -m pytest` ；GPU：`.\.venv\Scripts\python.exe -m layerforge run --input test_input/image_no_sag --out runs/flat --segment cascade --inpaint auto`
- 产物路径：`runs/flat/<uuid8>/`（本机 6 张单图），先看 `steps/01_character/rgba.png` 和 `steps/04_segment/`
- 未做 / 已知缺陷：夜景 seam fill 外圈还在（这次只落盘，没修）。SAM3 无权重。ORT CUDA 仍缺 `cublasLt64_13.dll`

## 2026-09-19 — 角色切换成 anime-segmentation 官方预处理

- 做了：`cascade.character` 改为 `anime_segmentation`。预处理对齐 SkyTNT `inference.get_mask`（RGB/255、等比缩放、居中 pad 1024，再 crop 回原图）。权重仍用已有 `model/aniseg/isnetis.onnx`，不重下。registry 按 YAML 建 CharacterCut，`aniseg` 作别名。本机重跑雪景图。**没改 schema、没改 .venv、没装 PyTorch ckpt**
- 怎么跑：`.\.venv\Scripts\python.exe -m pytest` ；GPU：`.\.venv\Scripts\python.exe -m layerforge run --input test_input/image_no_sag/grok-image-64010c19-6759-4ab7-92fa-b0bede4bb0d7.jpg --out runs --segment cascade --inpaint auto --job-id sam_64010c19`
- 产物路径：`runs/sam_64010c19`（schema v2；layers: body/clothes/eye_l/eye_r；`missing: [face, hair_back]`）。角色剪影把树林/极光背景去掉了；衣服层是外套本身
- 未做 / 已知缺陷：没有换成官方 CLI 默认的 `isnet_is.ckpt`（仍是同一份 isnetis ONNX）。`hair_front` 进了库存但没成层，头发仍在 body。斗篷边缘还有雪景光晕。face 仍 missing。SAM3 无权重。ORT CUDA 仍缺 `cublasLt64_13.dll`

## 2026-09-19 — body 打洞不再留黑缝

- 做了：`_body_from_residual` 只按 hair/clothes/face 的真实 mask 扣，不再 `expand_px` 膨胀打洞。refine 后把距已有非 overlay 层 ≤ `seam_fill_px`（默认 24）的未认领前景缝归最近层。修 `sam_flat3` stack 发/颈/领口黑洞。**没改 schema、没改 .venv、没顺手修 eye_r / hair_back**
- 怎么跑：`.\.venv\Scripts\python.exe -m pytest && .\.venv\Scripts\python.exe -m layerforge run --input test_input/image_no_sag/grok-image-4034a197-286e-45f1-8049-60f91c89d8a8.jpg --out runs --segment cascade --inpaint auto --job-id sam_flat3`
- 产物路径：`runs/sam_flat3`（schema v2；前景 hole=0；preview/stack 发颈领口贴合原图）
- 未做 / 已知缺陷：`eye_r`、`hair_back` 仍 missing。不再膨胀打洞后，发缘和领口残片回到 `20_body`（宁可贴层上也不留洞）。SAM3 无权重。ORT CUDA 仍缺 `cublasLt64_13.dll`

## 2026-09-19 — 成对切眼 + body 打洞 + 保存 WDTagger

- 做了：眼睛按 pair 切（family query + 左右 query 分开、脸裁切再检、镜面补框，失败再回退 `_cut_one`）。`hair_back` 默认进库存（bald 除外）；body 残差打掉膨胀后的 hair/clothes/face，大块未认领前景不再折进 body。WDTagger 预处理改成 v3 白边正方形 + BICUBIC + BGR 0–255 NHWC，结果写入 `logs/tags.json`（不改 manifest schema）。本机重跑 `sam_flat3`。**没改 .venv、没升 schema、没实现 future/**
- 怎么跑：`.\.venv\Scripts\python.exe -m pytest && .\.venv\Scripts\python.exe -m layerforge run --input test_input/image_no_sag/grok-image-4034a197-286e-45f1-8049-60f91c89d8a8.jpg --out runs --segment cascade --inpaint auto --job-id sam_flat3`
- 产物路径：`runs/sam_flat3`（schema v2；`logs/tags.json`：1girl/white_hair/short_hair 等；layers: body/clothes/face/eye_l/mouth/hair_front/acc×2）
- 未做 / 已知缺陷：`eye_r` 仍 missing（头侧倾 + 右眼埋在头发里，镜像/脸裁切没出可用框）。头发整块进了 `hair_front`，`hair_back` missing。body 不再含整头白发和外套，但仍有颈侧发丝和胸前绑带。SAM3 仍无权重。ORT CUDA 仍缺 `cublasLt64_13.dll`

## 2026-09-19 — DINO API + manifest v2 + sam_flat2

- 做了：Grounding DINO 后处理按 transformers 签名传 `threshold`（旧版仍走 `box_threshold`）。`missing` / `needs_click` 升到 `layerforge.manifest.v2`，v1 读入会补默认值。本机跑通 `--segment cascade --inpaint auto --job-id sam_flat2`。**没改 .venv、没装 SAM3**
- 怎么跑：`.\.venv\Scripts\python.exe -m pytest && .\.venv\Scripts\python.exe -m layerforge run --input test_input/image_no_sag/grok-image-4034a197-286e-45f1-8049-60f91c89d8a8.jpg --out runs --segment cascade --inpaint auto --job-id sam_flat2`
- 产物路径：`runs/sam_flat2`（schema v2；layers: body/clothes/face/eye_l；`missing: [eye_r]`）
- 未做 / 已知缺陷：右眼未切出。SAM3 仍无权重，文本步跳过走 SAM2。ORT CUDA 仍缺 `cublasLt64_13.dll`，AniSeg/WD 回退 CPU EP（警告，不挡这次跑通）。头发未进库存（tagger 没给 bangs/long_hair 或未过阈值）

## 2026-09-18 — 下载 cascade 检测权重

- 做了：`scripts/download_models.py --only-detect` 增加 SAM3 尝试；本机已写入 AniSeg / WDTagger / Grounding DINO。**没改切件算法、没改 .venv**
- 怎么跑：`.\.venv\Scripts\python.exe scripts/download_models.py --only-detect`
- 产物路径：`model/aniseg/isnetis.onnx`、`model/wdtagger/model.onnx`、`model/grounding_dino/model.safetensors`（不进 git）
- 未做 / 已知缺陷：`facebook/sam3` 是门控仓库，当前 HF 账号 403，`model/sam3/` 仍空。去 https://huggingface.co/facebook/sam3 申请后重跑同一命令。没有 SAM3 时 cascade 会跳过文本步、走 SAM2

## 2026-09-18 — 检测级联切件（不假定姿态）

- 做了：扁图默认 `cascade`：AniSeg 抠角色 → WDTagger 对照 taxonomy 做库存 → 只对库存件 Grounding DINO 出框 → SAM3 文本或 SAM2 框+正负点出 mask；每件可用性打分，不合格换 query/后端，最多 4 次，失败写入 `manifest.missing` / `needs_click`。taxonomy 增加 queries / required / required_if_tags / skip_if_tags / cut_priority。mutex 按优先级（eye > face > hair_front > hair_back）。Imagine 跳过 1–4，只检查可用性并补切缺件。去掉姿态 `region_prompts`。**没改 .venv、没实现 future/ 视觉对齐**
- 怎么跑：`.\.venv\Scripts\python.exe -m pytest` ；GPU：`scripts/download_models.py` 后 `.\.venv\Scripts\python.exe -m layerforge run --input test_input/image_no_sag/grok-image-4034a197-286e-45f1-8049-60f91c89d8a8.jpg --out runs --segment cascade --inpaint auto --job-id sam_flat2`
- 产物路径：`runs/sam_flat2`（本机，需 AniSeg/WD/DINO 权重）；契约测试不加载这些模型
- 未做 / 已知缺陷：本机还没有 AniSeg/WDTagger/DINO/SAM3 权重时 cascade 会报错（SAM2 仍可用 `--segment sam2.hinted`）。SAM3 包未装则跳过文本步。`[gpu]` 声明了 onnxruntime-gpu，需要自己装。线稿对齐未做

## 2026-09-18 — seam 不得画出遮挡层

- 做了：`plan_occlusion` 的 seam 膨胀之后裁回 occluder 可见区（且不含本层 visible）。修复 `runs/sam_flat1` 里 `preview/diff.png` 发丝/脸/校徽描边：那是后层 occluded 盖住前层原图像素，不是 reproject 失效（各层 visible 与原图差为 0）。**没改 SAM、region_prompts、.venv**
- 怎么跑：`.\.venv\Scripts\python.exe -m pytest && .\.venv\Scripts\python.exe -m layerforge run --input test_input/image_no_sag/grok-image-4034a197-286e-45f1-8049-60f91c89d8a8.jpg --out runs --segment sam2.hinted --inpaint auto --job-id sam_flat1`
- 产物路径：`runs/sam_flat1`（重跑后 `preview/diff.png` 可见区应近黑）
- 未做 / 已知缺陷：`50_face` 仍是区域框切到的头发/耳环；`10_hair_back` 仍会在脸洞里补绘（body 吃了残差脸，taxonomy 认为头发在 body 下）。这是 SAM/切件问题，这轮不改

## 2026-09-18 — 项目内 .venv

- 做了：本机虚拟环境放到仓库 `.venv/`；gitignore 增加 `venv/` `env/`（`.venv/` 本来就忽略）。README 和 `.vscode/settings.json` 都指向 `.venv\\Scripts\\python.exe`。**没改 SAM / 形态学算法**
- 怎么跑：`.\.venv\Scripts\python.exe -m pip install -e ".[dev]"` 后用这个 python 跑 pytest / layerforge
- 产物路径：本机 `.venv/`（不进 git）
- 未做 / 已知缺陷：mask 质量问题仍在（区域框切件 + mutex，SAM 结果几乎不做开闭运算）。`.venv` 里的 GPU 包要按 README 再装一遍，不会从 Anaconda 自动拷过来

## 2026-09-18 — 扁图 role + 真补绘

- 做了：扁图 SAM 按 `segment_sam2.yaml` 的 `region_prompts` 切 `hair_back` / `face` / `clothes`（multimask 择一，不平均）；剩余 `unknown_*` 用几何+肤色启发式标 taxonomy role，标不出的归 `acc`。默认 `inpaint.name=auto`：洞面积 ≤ `small_hole_max_px` 走 LaMa（失败则 OpenCV Telea），更大走 SD1.5；只在 occluded 上补，可见区 reproject。SD 优先加载 `*inpainting*.ckpt`。补了 assign_roles / router / region box 测试。本机 `sam_flat1` 已跑通
- 怎么跑：`python -m pytest && python -m layerforge run --input test_input/image_no_sag/grok-image-4034a197-286e-45f1-8049-60f91c89d8a8.jpg --out runs --segment sam2.hinted --inpaint auto --job-id sam_flat1`
- 产物路径：`runs/sam_flat1`（`hair_back` `body` `clothes` `face` `acc_*`，无 `unknown_*`）；hair/clothes 走了 `sd15.anime`，face 小洞走了 Telea
- 未做 / 已知缺陷：`simple-lama-inpainting` 钉死 numpy<2，本机未装，小洞目前是 Telea 回退。区域框+启发式仍可能把碎块标成 acc。`[gpu]` extra 仍未声明 sam2；git 历史里仍有测试图 / skill

## 2026-09-18 — acc 去重 + 清空 logs

- 做了：overlay 同 role 且 mask IoU≥0.5，或小块 75% 落在已留大块里，只留一块（Imagine 不同标签贴到同一像素 / 嵌在大饰品里不再堆 `acc_*`）；左右鞋等空间分离的仍多分。每次 run 开始截断 `logs/run.jsonl`，不再追加旧次。补了 refine 与 log 回归
- 怎么跑：`python -m pytest && python -m layerforge run --input test_input/image_partial_sag/image0 --out runs --dry-run --job-id imagine_image0`
- 产物路径：`runs/imagine_image0`（本机）；dry-run 后 `acc` 从约 33 层降到 19 层（不同位置的鞋/饰品仍各一层）
- 未做 / 已知缺陷：扁图 `unknown_*` 未标 role；补绘仍是 identity；`[gpu]` extra 未声明 sam2；git 历史里仍有测试图 / skill。剩余 19 个 acc 是不同位置的件，不是同一贴点重复

## 2026-09-18 — 非产品目录不进 git

- 做了：`.cursor/` 与 `skills/` 加入 gitignore 并从跟踪移除（本机保留）。pytest（`tests/`）和 `scripts/download_models.py` 仍进仓库：前者是契约测试，后者是装权重，都是产品用法
- 怎么跑：`git ls-files` 不应再看到 `.cursor/` 或 `skills/`
- 产物路径：https://github.com/obrs09/layer-pipeline
- 未做 / 已知缺陷：git 历史里仍有这些文件和首个 commit 的测试图；彻底抹掉需要 rewrite + force push。acc 去重、扁图 unknown_*、真补绘、导出清空 logs 仍未做

## 2026-09-18 — 布局测试改回本机真实图

- 做了：`tests/test_imagine_layout.py` 再读本机 `test_input/`（Imagine `segments/` 与扁图目录）；图仍 gitignore，不进 GitHub。没有本机目录时 skip，避免别人 clone 后 pytest 红
- 怎么跑：`python -m pytest tests/test_imagine_layout.py`
- 产物路径：本机 `test_input/`（不进 git）
- 未做 / 已知缺陷：首个 commit 的 git 历史里仍有这些图；要从 GitHub 历史里彻底抹掉需要 rewrite + force push

## 2026-09-18 — 测试图不进 git

- 做了：`test_input/` 加入 gitignore 并从 git 跟踪中移除（本机文件保留）
- 怎么跑：`python -m pytest`
- 产物路径：https://github.com/obrs09/layer-pipeline
- 未做 / 已知缺陷：首个 commit 的 git 历史里仍有这些图；要从 GitHub 历史里彻底抹掉需要 rewrite + force push。布局测试当时误改成临时目录，下一节已改回真实图

## 2026-09-18 — git 远程 + CHANGE 落盘

- 做了：初始化 git，远程 `https://github.com/obrs09/layer-pipeline.git`；把 CHANGE 写成根目录 `CHANGE.md`；AGENTS / Builder 规则改为交付必须改这个文件
- 怎么跑：`git clone https://github.com/obrs09/layer-pipeline.git && python -m pip install -e ".[dev]" && python -m pytest`
- 产物路径：本仓库（不含 `model/` 权重与 `runs/`）
- 未做 / 已知缺陷：权重仍只在本机 `model/`；`runs/` 不进 git，Reviewer 仍要在本机看 preview；补绘仍是 identity

## 2026-09-18 — 可见区 reproject 与导出清理（Reviewer FAIL 的 BLOCKERS）

- 做了：Imagine/SAM 可见区强制 reproject 回原图；未覆盖前景并入 body；导出前清空 layers/masks/preview；补了可见区回归和脏目录测试
- 怎么跑：`python -m pytest && python -m layerforge run --input test_input/image_partial_sag/image0 --out runs --dry-run --job-id imagine_image0`
- 产物路径：`runs/imagine_image0`、`runs/sam_imagine0`、`runs/sam_flat0`（本机，不进 git）
- 未做 / 已知缺陷：未改 acc 去重、未把 unknown_* 标成真实 role、未改 pyproject `[gpu]` extra、补绘仍是 identity

## 2026-09-18 — SAM2 GPU 分割

- 做了：下载 SAM2/LaMa/SD1.5 到本机 `model/`；装 CUDA torch + sam2；扁图用前景框 box prompt 跑 `sam2.hinted`；Imagine 路径 SAM 只在原分块内收边
- 怎么跑：`python -m layerforge run --input test_input/image_no_sag/grok-image-4034a197-286e-45f1-8049-60f91c89d8a8.jpg --out runs --segment sam2.hinted --inpaint identity --job-id sam_flat0`
- 产物路径：`runs/sam_flat0`、`runs/sam_imagine0`；权重在 `model/sam2`、`model/lama`、`model/sd15`
- 未做 / 已知缺陷：权重不进 git；扁图子块仍是 unknown_*；Imagine 重复标签仍会堆 acc
