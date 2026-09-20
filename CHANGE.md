# CHANGE

Builder 每次交付更新本文件；回复末尾再贴同一段。Reviewer 对照这里是否诚实。

## 2026-09-19 — 脸选区膨胀 1px，并修拆脸逻辑

- 做了：`cascade.face_split.expand_px` 3→1。顺手修逻辑：（1）肤色种子排除已有头发层，并偏向 Lab a≥132 的暖色，避免刘海把皮肤中心拉白；（2）发色固定进 `hair_front`，不再因为 kept 里只有 `hair_back` 就把刘海塞进后发，切出后从 missing 拿掉；（3）refine 第二次拆脸若已有 neck 就不再按下巴重切；（4）脖子收窄相对整张脸最宽处，不相对嘴下局部峰值。**没改 schema、没改 .venv**
- 怎么跑：`.\.venv\Scripts\python.exe -m pytest`（126 passed）；GPU：`.\.venv\Scripts\python.exe -m layerforge run --input test_input/image_no_sag --out runs/flat --segment cascade --inpaint auto`
- 产物路径：`runs/flat/<uuid8>/steps/04_segment/01_face.png`、`face_split.json`。4034 剥发 18k→25k；a163 0.6k→9k。`missing`：6 张都是 `[]`（640 原先 `[hair_front]`，脸上剥下的刘海建成了 `hair_front`）
- 未做 / 已知缺陷：4034 / a163 脸缘仍有白发线稿。296 蓝发还沾一点。脖子仍是下巴下一小条。颜色对不上的像素仍可能被 fill 折进 body。SAM3 无权重。ORT CUDA 仍缺 `cublasLt64_13.dll`

## 2026-09-19 — 脸切完后按颜色拆脸 / 脖子 / 头发

- 做了：切完 face + eyes + mouth 后 `split_face_colors`：从脸上扣掉眼/嘴，Lab 就近分成肤色/发色/其他；肤色用宽度收窄找下巴，脖子写成 taxonomy `neck`（order 22，不进 DINO 库存）；发色并进 `hair_front`（没有则 `hair_back`）；选区膨胀 `expand_px=3`。refine 的 fill 不再把眼/嘴 overlay 洞填回脸，并在 fill 后再拆一次，免得 seam 把头发贴回去。配置在 `cascade.face_split`。**没改 schema、没改 .venv**
- 怎么跑：`.\.venv\Scripts\python.exe -m pytest`（124 passed）；GPU：`.\.venv\Scripts\python.exe -m layerforge run --input test_input/image_no_sag --out runs/flat --segment cascade --inpaint auto`
- 产物路径：`runs/flat/<uuid8>/layers/22_neck.png`、`steps/04_segment/face_split.json`。6 张都有 neck。4034 脸上扣了眼/嘴洞，剥走 18k 发色像素；296 剥走 10k。`missing`：640 `[hair_front]`，其余 `[]`
- 未做 / 已知缺陷：白发/浅发仍会留在脸上（4034 右侧刘海、a163 刘海、296 蓝发）。脖子只是下巴下一小条。颜色对不上的像素仍可能被 fill 折进 body。640 `hair_front` 仍缺。SAM3 无权重。ORT CUDA 仍缺 `cublasLt64_13.dll`

## 2026-09-19 — 2.4 按切开顺序落盘；taxonomy 含头发 query

- 做了：`04_segment` 文件名改成切开序号 `00_clothes` / `01_face` / …（不是 taxonomy 绘制 `order` 的 `10_hair_back`）。`cut_order.json` 记 planned vs 实际 kept。DINO 框按 `_trace_box` 顺序编号，`03_boxes/{index}_{role}.png` 单框 + `overlay.png` 全框。`layers/` `masks/` 仍用 `{order}_{role}`，**没改 schema、没改切件算法、没改 .venv**。头发在 taxonomy：`hair_back.queries=["hair","anime hair","long hair","back hair"]`，`hair_front.queries=["bangs","front hair",…]`，`_cut_one` 用这些英文 query 跑 Grounding DINO。
- 怎么跑：`.\.venv\Scripts\python.exe -m pytest`
- 产物路径：新 run 的 `runs/flat/<uuid8>/steps/03_boxes/` 与 `steps/04_segment/`。旧 run（如 296a4352）仍是绘制序文件名，需重跑才变成切开序。
- 未做 / 已知缺陷：没 GPU 重跑 6 张。残差头发 / body 在 SAM 循环之后，会出现在 cut_order 末尾。05_refine / 06_occlusion / 最终包仍按 taxonomy order。640 `hair_front` 仍缺。SAM3 无权重。ORT CUDA 仍缺 `cublasLt64_13.dll`

## 2026-09-19 — 姿态用 DWPose wholebody，不再收成 OpenPose-18

- 做了：`DwPoseEstimate` 不再把 RTMPose 133 点转成 OpenPose-18（丢掉手/脸、人造 `neck`）。overlay / `keypoints.json` / SAM 部件框走 COCO-17 + 脚/脸/手；手臂正点末位是手掌根（`lhand_00` / `rhand_00`），脸框用 68 个脸点。角色 mask 仍不被骨架裁。**没改 schema、没改 .venv**
- 怎么跑：`.\.venv\Scripts\python.exe -m pytest`（117 passed）；GPU：`.\.venv\Scripts\python.exe -m layerforge run --input test_input/image_no_sag --out runs/flat --segment cascade --inpaint auto`
- 产物路径：`runs/flat/<uuid8>/steps/01_pose/overlay.png`（身体 + 彩色手指 + 脸点）。`keypoints.json` 无 `neck`，有 `lhand_*` / `rhand_*` / `face_*`。`missing`：640 `[hair_front]`，其余 `[]`
- 未做 / 已知缺陷：二次元脸上 68 点会挤在五官附近。手杖仍在 body。640 `hair_front` 仍缺。908 `arm_l` 仍切不出。SAM3 无权重。ORT CUDA 仍缺 `cublasLt64_13.dll`

## 2026-09-19 — 修 BLOCKER：残差头发不再收手杖

- 做了：`_hair_from_residual` 不再把「质心落在 DINO 头发框里」当成头发。头发框只裁搜索区，连通块必须贴着脸（脸高×`hair_reach`）。手腕点从残差里挖掉（`cascade.hair_punch_pose_roles` / `hair_punch_pose_pad` 48px，YAML），把手杖从贴头的那坨拆开。不用整条手臂框——640 的 `arm_l` 框会把右侧长发一起切掉。测试覆盖「框内孤立手杖」和「经手腕走廊粘上的手杖」。**没改 schema、没改 .venv、只修 BLOCKER**
- 怎么跑：`.\.venv\Scripts\python.exe -m pytest`（116 passed）；GPU：`.\.venv\Scripts\python.exe -m layerforge run --input test_input/image_no_sag --out runs/flat --segment cascade --inpaint auto`
- 产物路径：`runs/flat/<uuid8>/`。640 `masks/10_hair_back.png` 无手杖（68408px，头发+耳罩+一点领口皮草）；手杖/手套在 `20_body`。`hair∩body=0`。458 hair_back 94644、clothes 204950（`hair∩clothes=0`）。`missing`：640 `[hair_front]`，其余 `[]`。6 张 diff 全黑、occluded 不压低层、hole ≤63（296=0，908=1，458=31，4034=61，a163=63）
- 未做 / 已知缺陷：640 `hair_front` 仍缺（刘海在 face）；手杖仍在 body（acc staff `too_large:0.877`）；hair_back 仍带耳罩和领口皮草，不是「只含头发」。458 残差头发仍带肩甲/打火机边。a163 `10_hair_back` 可见 139px。296 手机下半仍在 body。908 `arm_l` 仍切不出。SAM3 无权重。ORT CUDA 仍缺 `cublasLt64_13.dll`

## 2026-09-19 — 清已知缺陷：眼睛、640 头发、手机、发梢、NITs

- 做了：（1）眼睛 `min_box_cover` 0.5→0.4：DINO 眼框含眉/眼皮，低垂的眼 SAM mask 只盖 0.45–0.49，看过 mask 是完整的眼。（2）`hair_back.exclude_roles` 加 `clothes`：SAM 在衣服上有负点、overlap 校验生效，458 的头发不再把外套一起吐出来。（3）`_hair_from_residual`：tagger 说有头发但 SAM 切不出时，用角色残差里「贴着脸（脸高×`hair_reach` 0.6）或落在 DINO 头发框内」的块当 `hair_back`；候选先裁到该区域，且丢掉覆盖角色 >55% 的框，免得手杖/下摆沿着轮廓细缝连进来。（4）taxonomy 新增 `tag_queries`：acc 只在 WDTagger 看到 phone/staff 时才加 smartphone/staff query；acc 的通用 query 也只在饰品 tag 触发时跑（否则 296 肩膀被当 ribbon）。acc 的 `box_cover` 按 框∩角色 算（`box_cover_in_character`，只对 acc；对 face 开会把脖子吃进脸）。（5）peer_and 之后沿边回收 isnet 灰区（P≤0.85、12px 内、且和主体连通）的发梢；AND 结果先清 <200px 碎点。（6）两旁路彼此一致且 isnet 差得远（Dice 缺口 >0.05）时不再换 seed 重试，qa.json 记 `skipped_retries`。（7）peer_and 结构回退时保留 Dice 原因（`dice_reasons`）。（8）内部熵阈值 0.05→0.08：640 卡在 0.0499/0.0500 两次跑结果不同。（9）skill 文档改 v2（`.cursor/` 不进 git）；删了 `runs/flat/7d9f70f9`。途中抓到一个自己引入的 bug：回收发梢带来孤岛让 peer_and 结构校验失败、替换被放弃，908 的 body 吞了整张床——已修（只回收连通像素，失败退回未回收 mask）。**没改 schema、没改 .venv**
- 怎么跑：`.\.venv\Scripts\python.exe -m pytest`（114 passed）；GPU：`.\.venv\Scripts\python.exe -m layerforge run --input test_input/image_no_sag --out runs/flat --segment cascade --inpaint auto`
- 产物路径：`runs/flat/<uuid8>/`。`missing`：296 `[]`（原 `[eye_l, eye_r]`），4034 `[]`（原 `[eye_r]`），640 `[hair_front]`（原 `[hair_back, hair_front]`），其余 `[]`。296 `90_acc` = 手机（10294px，peer 只留了角色 mask 内的部分）。640 `10_hair_back` 85689px 只含头发。458 hair_back 201k→95k、clothes 76k→205k（外套回到衣服层）。908 一次 isnet 就走旁路，peer_and 回收 5846px 发梢，body 384k。6 张 diff 全黑、occluded 不压低层、hole ≤63px
- 未做 / 已知缺陷：640 `hair_front` 仍缺（刘海在 face 层里）；手杖仍在 body（DINO staff 框太大，acc 被 too_large 拒）。296 手机只有角色 mask 内的一半成 acc，body 下面仍有手机像素（v0 闭集没有道具角色）。908 `arm_l` 仍切不出（box_cover 0.26）。SAM3 无权重。ORT CUDA 仍缺 `cublasLt64_13.dll`

## 2026-09-19 — 修 Reviewer BLOCKER：occluded 不压低层可见区；角色 mask 不留洞

- 做了：（1）`taxonomy.yaml` 里 `clothes.occluded_by` 去掉 `arm_l/arm_r`（arm order 25/26 在 clothes 40 之下）；`load_taxonomy` 校验 occluder 的 order 必须高于被遮挡层，否则报错；`plan_occlusion` 只认更高 order 的 occluder，且 hole 永不落在任何更低 order 层的 visible 上，pipeline 对结果再断言一次（`occluded_over_lower_visible == 0`）。（2）refine 末尾加 `fill_unclaimed_domain`：角色 mask 内没被非 overlay 层认领的像素——薄缝（离已有层 ≤ `seam_fill_px`）归最近层，成块的进 body（没 body 层就新建）；`logs/run.jsonl` 多一条 `coverage`，填之前的洞落 `steps/05_refine/unclaimed.png`。新增 `tests/test_taxonomy.py`、occlusion/refine 契约测试，102 passed。**没改 schema、没改 .venv、只修 BLOCKERS**
- 怎么跑：`.\.venv\Scripts\python.exe -m pytest` ；GPU：`.\.venv\Scripts\python.exe -m layerforge run --input test_input/image_no_sag --out runs/flat --segment cascade --inpaint auto`
- 产物路径：`runs/flat/<uuid8>/`。6 张 `preview/diff.png` 全黑（nonzero=0），`mask_occluded ∩ 更低层 mask_visible` 全部为 0。角色 mask 未覆盖像素：296 / 640 / 908 = 0；4034 / 458 / a163 = 66 / 42 / 63 px（角色 mask 里亮度 ≥250 且连到边框的像素，被 luma 前景排除，边级别）。a1639e70 小腿+胸口回到 `20_body`（41009 → 101890 px）；64010c19 漏检的头发进了 body（仍 `missing=[hair_back, hair_front]`）；296a4352 手+手机进了 body。`missing` 不变：296 `[eye_l, eye_r]`，4034 `[eye_r]`，640 `[hair_back, hair_front]`，其余 `[]`
- 未做 / 已知缺陷：296 的手机现在算 body（没有检测器说它是 acc）。640 头发在 body 里不是 hair 层。NITs 未动：peer_and 吃掉半透明发梢、QA 失败仍跑三次 isnet、`structural_score` 覆盖 Dice 原因、`layer-qa-review/SKILL.md` 仍写 v1、`runs/flat/7d9f70f9` 旧产物未删。SAM3 无权重。ORT CUDA 仍缺 `cublasLt64_13.dll`

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
