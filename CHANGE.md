# CHANGE

Builder 每次交付更新本文件；回复末尾再贴同一段。Reviewer 对照这里是否诚实。

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
