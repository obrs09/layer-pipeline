# LayerForge

单张二次元图或 Grok Imagine 分块图 → 遮挡补全后的分图层文件。

当前阶段 **v0**：不训练、不做视频、不绑骨。

- 策划书：`docs/PROJECT_PLAN.md`
- 双 Agent：`AGENTS.md`
- 每次交付：`CHANGE.md`
- 推荐 Skill：`docs/SKILLS.md`（仓库副本在 `skills/`，Cursor project skills 在 `.cursor/skills/`）
- GitHub：https://github.com/obrs09/layer-pipeline.git

## 跑

```text
python -m pip install -e ".[dev]"
python -m layerforge run --input <本机图片或 Imagine 目录> --out runs --dry-run
python -m pytest
```

测试图放本机 `test_input/`（不进 git）。Imagine 分块目录或单张扁图都可以当 `--input`。

有 GPU 且权重已放到 `model/` 后：

```text
python scripts/download_models.py
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install sam2 hydra-core iopath
python -m layerforge run --input <flat.png> --out runs --segment sam2.hinted --inpaint identity
```

## 模型路径

把权重放进仓库根目录的 `model/`（不进 git）：

```text
model/sam2/*.pt
model/lama/*.pt
model/sd15/          # Diffusers 目录或 .safetensors
```

本机需要 GPU 才能跑 SAM2 / SD1.5。检测不到 CUDA 会直接报错，不会偷偷用 CPU 跑 SD。

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
