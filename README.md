# LayerForge

单张二次元图或 Grok Imagine 分块图 → 遮挡补全后的分图层文件。

当前阶段 **v0**：不训练、不做视频、不绑骨。

- 策划书：`docs/PROJECT_PLAN.md`
- 双 Agent：`AGENTS.md`
- 每次交付：`CHANGE.md`
- GitHub：https://github.com/obrs09/layer-pipeline.git

Cursor 规则和 Skill 只留本机（`.cursor/`、`skills/`，不进 git）。清单见 `docs/SKILLS.md`。

## 跑

解释器固定为仓库里的 `.venv`（不进 git）。Windows：

```text
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m layerforge run --input <本机图片或 Imagine 目录> --out runs --dry-run
```

测试图放本机 `test_input/`（不进 git）。Imagine 分块目录或单张扁图都可以当 `--input`。

有 GPU 且权重已放到 `model/` 后：

```text
.\.venv\Scripts\python.exe scripts/download_models.py
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install sam2 hydra-core iopath diffusers transformers accelerate safetensors onnxruntime-gpu huggingface_hub
.\.venv\Scripts\python.exe -m layerforge run --input <flat.png> --out runs --segment cascade --inpaint auto
```

## 模型路径

把权重放进仓库根目录的 `model/`（不进 git）：

```text
model/aniseg/isnetis.onnx
model/wdtagger/model.onnx
model/wdtagger/selected_tags.csv
model/grounding_dino/     # Grounding DINO snapshot
model/sam2/*.pt
model/sam3/*.pt           # optional; cascade falls back to SAM2
model/lama/*.pt
model/sd15/               # Diffusers 目录或 .safetensors
```

扁图默认走 cascade（anime-segmentation → WDTagger 库存 → DINO 框 → SAM3/SAM2），不假定姿态。Imagine 已分块只做可用性检查，不合格才补切。本机需要 GPU 才能跑检测 / SAM / SD。检测不到 CUDA 会直接报错。小洞走 LaMa（失败则 OpenCV Telea），大洞走 SD1.5。`--dry-run` 仍是 identity。

## Imagine 分块（实际导出）

Grok Imagine 并不是策划书里的 `source.png + parts/part_00.png`。当前 `test_input` 是：

```text
<job>/
  grok-image-<uuid>.jpg
  segments/
    <label>.png
    <label>/<label>-N.png
```

分块是裁切后的 RGBA，ingest 会做模板匹配贴回原图画布。
