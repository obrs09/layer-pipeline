# 推荐 Skill（给 Cursor / Grok 用）

原则：Skill 只写「模型本来就不会稳定遵守」的流程、文件约定、命令。不要把通用 Python 知识写进 Skill。

建议放在本机 `skills/` 或 Cursor project skills（`.cursor/skills/`），**不要进 git**。Grok 侧可用 skill-creator 同步一份。

## 现在就该有（v0）

### 1. `layerforge-pipeline`

触发：跑分层、切件、补绘、出包、看策划书阶段。

内容：v0 范围、manifest 契约、阶段顺序、禁止事项（不训练、不视频、不平均 SAM）。
相当于把 `PROJECT_PLAN.md` 压成 Agent 每次能加载的短规程。

### 2. `imagine-parts-ingest`

触发：Imagine 分块、Grok 切好的 parts、下载的分段图。

内容：目录约定（`source.png` + `parts/`）、无 label 时的命名、如何生成 hints、和扁图 ingest 的差异。
Imagine 导出一变只改这个 Skill + `ingest.imagine_parts`。

### 3. `sam-hint-protocol`

触发：SAM 漏件、头发没分开、两件粘连、正负点。

内容：禁止平均；multimask 择一；分层切；exclude 点；互斥优先级；碎块阈值。
这是操作规程，不是再讲一遍 SAM 论文。

### 4. `occlusion-inpaint`

触发：被挡住、补绘、袖内、发后、SD1.5、LaMa。

内容：可见 mask vs occluded mask；只在膨胀区 inpaint；reproject 贴回；小洞 LaMa、大洞 SD1.5；提示词模板（延续材质、对齐线稿、不要新图案）。

### 5. `layer-qa-review`

触发：验收、Reviewer、preview、diff、manifest。

内容：与 `AGENTS.md` 验收清单一致，外加「看 stack/diff 时怎么判断可见区被改」。

## 下一阶段再写（先别做实现）

### 6. `vision-align`（v1）

用视觉模型给 unnamed part 标 `role`、检查左右眼、层序是否反了。只读图 + 写 manifest，不改像素。

### 7. `spine-manifest-export`（v2）

`manifest.v1` → Spine 最小 JSON（skeleton/slots/attachments 文件名映射）。不写权重。

### 8. `video-keyframe-track`（v3）

SAM2 跟踪 → 选最展开的一帧 → 丢回 v0 流水线。视频不直接出绑定。

### 9. `layerforge-webui`（v1+）

FastAPI / Gradio 路由与 CLI 参数一一对应，禁止在 UI 里另写一套切件逻辑。

## 不要做成 Skill 的

- 普通 pytest / git 流程
- PyTorch 安装说明（放 README）
- 「如何训练 LoRA」（v0 范围外）
