# CHANGE

Builder 每次交付更新本文件；回复末尾再贴同一段。Reviewer 对照这里是否诚实。

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
