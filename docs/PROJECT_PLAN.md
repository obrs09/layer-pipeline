# 二次元分层补绘流水线 — 项目策划书

**代号：** LayerForge  
**阶段：** v0（单图 / Imagine 分块图 → 补绘完成的分图层文件）  
**不做：** 视频、模型训练、自动绑骨出可播放 Live2D/Spine  
**协作：** 本机 Cursor + Grok 4.6；双 Agent（实现 / 验收）  
**日期：** 2026-09-18

---

## 1. 一句话目标

把一张二次元立绘（或 Grok Imagine 已经切好的分块图）变成一套 **命名稳定、遮挡补全、可再导入 PS / Cubism / 后续 Spine** 的图层包。切件和补绘的模型都可替换，视频和绑骨只留接口。

## 2. 成功标准（v0 必须同时满足）

1. 输入一张 PNG/WebP，或一个 Imagine「分块」目录，输出一个图层包。
2. 图层包至少包含：每层 RGBA PNG、`manifest.json`、可选 PSD。
3. 可见区域尽量使用原图像素（reproject），只对「被挡住应延长」的区域补绘。
4. 换分割后端或换 inpaint 后端不改 `manifest` 字段，只改 adapter。
5. 有一条 CLI：`python -m layerforge run --input ... --out ...`
6. 有一份验收清单，实现 Agent 提交前必须被验收 Agent 勾完。

非目标（明确写进范围外，防止膨胀）：

- 不训练 SAM / SD
- 不输出可运行的 `.moc3` / Spine 动画（v0 只预留 schema）
- 不在 Grok Bot 云电脑上跑 GPU 推理
- 不做 WebUI 完整产品（v0 可留 FastAPI 空路由）

## 3. 输入 / 输出

### 3.1 输入（v0）

| 类型 | 说明 | Ingest |
|---|---|---|
| A. 单张扁图 | 一张角色 PNG，透明或有背景 | `ingest.flat_image` |
| B. Imagine 分块 | 从 Imagine 下载的多块切图 + 原图 | `ingest.imagine_parts` |

Imagine 分块的约定（先写死，以后再适配真实导出结构）：

```
imagine_job/
  source.png          # 原图，必须有
  parts/
    part_00.png       # 单块，可带 alpha 或白底
    part_01.png
    ...
  optional_meta.json  # 若 Imagine 将来给 box/label 则读
```

没有 label 时：用文件名序号 + 视觉模型（可选，v0 可关）猜 `role`，猜不出则为 `unknown_N`，留给人工改 `manifest`。

### 3.2 输出（v0 合同）

```
out/<job_id>/
  source.png
  layers/
    00_hair_back.png
    10_body.png
    20_face.png
    30_eye_l.png
    ...
  masks/
    00_hair_back.png          # 可见区
    00_hair_back.occluded.png # 补绘区（若有）
  preview/
    stack.png                 # 全层合成对照原图
    diff.png                  # 合成 vs 原图
  manifest.json
  layers.psd                  # 可选
  logs/run.jsonl
```

`manifest.json` 是全项目的稳定契约，下游（Spine / WebUI / 视觉对齐）只读它。

```json
{
  "schema": "layerforge.manifest.v2",
  "job_id": "...",
  "source": "source.png",
  "canvas": {"w": 2048, "h": 2048},
  "backend": {
    "segment": "sam2.hinted",
    "inpaint": "sd15.anime",
    "compose": "reproject_v1"
  },
  "layers": [
    {
      "id": "30_eye_l",
      "role": "eye_l",
      "order": 30,
      "file": "layers/30_eye_l.png",
      "mask_visible": "masks/30_eye_l.png",
      "mask_occluded": "masks/30_eye_l.occluded.png",
      "bbox": [x, y, w, h],
      "source": "imagine_part|sam|manual",
      "complete": true,
      "notes": "",
      "needs_click": false
    }
  ],
  "missing": []
}
```

角色表（v0 最小集，可在 `taxonomy.yaml` 扩展，不要写死在代码里）：

`bg, hair_back, body, arm_l, arm_r, clothes, face, brow_l, brow_r, eye_l, eye_r, mouth, hair_front, acc`

## 4. 流水线（v0）

```
Ingest
  → Normalize（尺寸、色彩、透明底）
  → Segment（扁图用 SAM 类；Imagine 分块则跳过大部分切分）
  → Refine masks（互斥、去碎块、正负点规则）
  → Occlusion plan（按 role 膨胀/向关节延长，得到补绘 mask）
  → Inpaint（只在 occluded mask 上）
  → Reproject（可见区贴回原图像素）
  → Compose + QA preview
  → Export PNG pack / PSD / manifest
```

关键原则：

- **SAM 不准时不平均多次结果。** 用 multimask 择一、正负点、分层切、互斥规则。
- **挡住的区域不找 SAM。** 只做 occlusion plan + inpaint。
- **每个阶段都是函数 + 磁盘产物。** 中断可从该阶段重跑。

## 5. 仓库结构（可读 + 可扩展）

```
layerforge/
  README.md
  AGENTS.md                 # 双 Agent 协议
  pyproject.toml
  taxonomy.yaml             # 图层角色与绘制顺序
  configs/
    default.yaml
    backends/
      segment_sam2.yaml
      segment_sam3.yaml
      inpaint_sd15.yaml
      inpaint_lama.yaml
  src/layerforge/
    cli.py
    pipeline.py             # 只编排，不写算法
    contracts.py            # manifest / layer dataclass
    ingest/
      flat.py
      imagine_parts.py
    backends/
      segment/
        base.py             # Protocol
        sam_hinted.py
        noop_from_parts.py  # Imagine 已切好
      inpaint/
        base.py
        sd15.py
        lama.py
      export/
        png_pack.py
        psd.py
    ops/
      refine.py
      occlusion.py
      reproject.py
      qa.py
    future/                 # 空模块 + 接口注释，禁止在 v0 实现
      video.py
      align_vision.py
      spine_json.py
      webui.py
  tests/
    test_contracts.py
    test_pipeline_dryrun.py
    fixtures/               # 小图，无版权角色或自绘
  skills/                   # Cursor / Grok skill 草稿
  docs/
    PROJECT_PLAN.md         # 本文件
```

扩展规则：新能力 = 新 backend 或 `future/` 里填实现，**禁止**改已发布 schema 的字段含义。要加字段就出下一版并写迁移。当前契约是 `manifest.v2`（v1 读入时补 `missing` / `needs_click`）。

## 6. 后端替换方式

所有重模型只通过 Protocol：

```python
class SegmentBackend(Protocol):
    def segment(self, image, hints: SegmentHints) -> list[LayerMask]: ...

class InpaintBackend(Protocol):
    def inpaint(self, image, mask, prompt: str) -> Image: ...
```

`configs/default.yaml` 里写名字，CLI `--segment sam2.hinted --inpaint sd15.anime` 覆盖。

v0 推荐默认：

- 切件：SAM 2 + 人工/自动 hints（点、框、role）
- 已有 Imagine 分块：`noop_from_parts`
- 补绘：SD1.5 二次元 inpaint；小洞走 LaMa
- 设备：本机 CUDA；检测不到 GPU 时明确报错，不要偷偷 CPU 跑 SD

## 7. 分期

| 阶段 | 交付 | 不做 |
|---|---|---|
| **v0** 现在 | 扁图 + Imagine 分块 → 补全图层包 + manifest + preview | 视频、训练、Spine 动画、WebUI |
| **v0.1** | PSD 导出稳定、taxonomy 可配、干跑测试 | |
| **v1** | 视觉模型对齐（层名/左右眼）、简单 WebUI | |
| **v2** | Spine JSON 草稿（slots/attachments 骨架） | 完整绑骨权重 |
| **v3** | 视频：SAM2 跟踪 → 选关键帧再走 v0 | 全帧自动绑骨 |

## 8. Cursor + Grok 4.6 怎么做

### 8.1 本机约定

- 推理只在本机（或你已有的 GPU 机）跑，Cursor 只写代码、跑测试、看 preview。
- 模型权重不进 git，路径写在 `configs/local.yaml`（gitignore）。
- 大图和输出进 `runs/`（gitignore），fixtures 才进仓库。

### 8.2 双 Agent（代替 Grok Bot 云端协作）

见根目录 `AGENTS.md`。摘要：

- **实现 Agent（Builder）：** 只改 `src/`、测试、配置；每次 PR 级改动附 `CHANGE` 三段（改了什么 / 怎么跑 / 风险）。
- **验收 Agent（Reviewer）：** 不写功能代码。对照本策划书 + `AGENTS.md` 验收清单。不通过则只提问题列表，实现 Agent 再改。
- 你是唯一可以「合并」的人。两个 Agent 都不得跳过清单。

Cursor 用法建议：

1. 开两个 Composer：一个选 Builder 规则，一个选 Reviewer 规则（`.cursor/rules`）。
2. Builder 做完说「请 Reviewer 验收 `runs/<id>`」。
3. Reviewer 只读 diff + 输出目录 + 测试，给出 PASS/FAIL。

没有 Cursor 原生「永久双 bot」时，用规则文件模拟即可，比把 GPU 塞进 Grok Bot 云电脑现实。

## 9. 风险

| 风险 | 对策 |
|---|---|
| SAM 头发粘脸 | 分层切 + exclude 点 + 互斥；不平均 |
| 补绘改掉可见区 | reproject 强制贴回 |
| 模型太大装进仓库 | 权重外置 |
| 范围膨胀到绑骨 | `future/` 锁住，v0 禁止实现 |
| Imagine 导出格式变 | 只改 `ingest.imagine_parts` |

## 10. 立刻可做的第一周任务（给 Builder）

1. 初始化 `pyproject.toml` + 空 pipeline + manifest schema 测试
2. `ingest.flat` + `ingest.imagine_parts`（后者先支持「原图 + parts/*.png」）
3. `SegmentBackend` + `noop_from_parts`（先打通 Imagine 路径，不依赖 GPU）
4. `occlusion.py` 按 taxonomy 做膨胀 mask
5. `InpaintBackend` 先做 `identity`（不补，只拷贝）和 `lama` 可选
6. export PNG pack + preview stack/diff
7. 再接 `sam_hinted` 与 `sd15`

顺序刻意：先契约和 Imagine 已切路径，再接重模型。这样没 GPU 也能验收结构。
