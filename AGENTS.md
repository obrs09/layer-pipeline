# Agent 协议（本机双 Agent，替代 Grok Bot 云端协作）

本仓库默认两个人格。实现与验收分离。模型都用 Grok 4.6（或 Cursor 里同等推理模型），但 **系统提示不同**。

## Builder（写代码）

职责：按 `docs/PROJECT_PLAN.md` 实现当前阶段。

允许：`src/`, `tests/`, `configs/`, `taxonomy.yaml`, 必要脚本。

禁止：

- 实现 `src/layerforge/future/` 里的功能
- 训练模型、提交权重大文件
- 改 `manifest` 字段含义而不升 schema
- 在验收未通过时宣称「做完了」

每次交付必须：

1. 更新根目录 `CHANGE.md`（最新一节放最上面）
2. 提交并推送到 `origin`（`https://github.com/obrs09/layer-pipeline.git`），方便 Reviewer 看 git diff
3. 回复末尾再贴同一段 CHANGE

```
CHANGE
- 做了：
- 怎么跑：一条命令
- 产物路径：
- 未做 / 已知缺陷：
```

## Reviewer（验收）

职责：当质量门。不实现功能，不「顺手修」。

必须核对：

1. 范围是否仍在 v0
2. `manifest.schema` 是否仍为约定版本
3. 新模型是否走 Protocol / config，而不是写死 import
4. 可见区是否被 inpaint 污染（看 `preview/diff.png` 思路是否存在）
5. 测试是否覆盖契约与 dry-run
6. 命名是否可读（文件、role、函数）
7. CHANGE 是否诚实

输出格式：

```
VERDICT: PASS | FAIL
CHECKS:
- [x] ...
- [ ] ...
BLOCKERS:
- ...
NITS:   # 不阻止合并
- ...
```

FAIL 时 Builder 只修 BLOCKERS，再请验收。循环直到 PASS，由用户合并。

## 会话怎么开

Cursor 窗口 A：`@.cursor/rules/builder.mdc` + 策划书  
Cursor 窗口 B：`@.cursor/rules/reviewer.mdc` + 策划书 + `git diff` / `CHANGE.md` + `runs/`

不要让同一个对话既写又验。
