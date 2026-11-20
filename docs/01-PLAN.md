# 01 · 选赛与四道闸

## 选赛（2026-11-21）

候选从 AME-1134 父目录下的 v2 清单（113 场）与 AME-1203 候选里逐条排。先按“能不能真干”过一遍硬条件：

| 条件 | 为什么必须 |
|---|---|
| 免费 | 用户的既有口径：不为比赛付费 |
| 个人可报、不需组队/企业 | 一个人推完 |
| 不依赖本机拿不到的数据或凭据 | Kaggle / Zindi 数据要登录，API 无凭据（本机实测 401）；Firebase / Gemini 域名本机不通 |
| 不需要指定 SDK 的账号 | Kiro / Forge / QNX / 华为云码道 都要求在对方平台上跑 |
| 截止日还有余量 | 一次会话内能收口 |

### 排除记录（本轮实际查过、否掉的）

| 候选 | 否掉的理由 |
|---|---|
| CartesianForms 重构建筑（M32） | 报名要付费（专业组 $40），且必须在 FORMAS.AI 平台上做工作板；任务书要登录 |
| Kiroween 2026（G16） | 必须用 Kiro（IDE/CLI），要 Kiro/AWS 账号与 credits |
| Codegeist 2026（G26） | Forge 应用必须装进 Atlassian Cloud，要开发者账号 |
| QNX Global Hackathon（G23） | 要 QNX 8.0 目标环境，本机无法验证运行 |
| GenAI for Good（G21） | 至少要 2 人非正式团队 |
| Name That LLM / PTCG / Envisaged / NFL（Kaggle） | 数据下载要 Kaggle 凭据，本机 API 401；且官方 slug 未锁定 |
| 美图秀秀 × 站酷（M30） | 指定必须用美图平台出图 |
| 数字武当杯（M31）、瓦猫（M23）、江苏足球（M33）、济宁（M36）、无锡（M37）等 | 已被本工作流的其它赛道做过（M/P/AG/AF/AD） |

### 选定

**Open Quantum Challenge（OQC）· Season 2 — QEC Decoder**，主办 FINAL-Bench（VIDRAFT 构建），
作品名 **《一票之差》**（判决靠投票，名次也只差一票），
承载在 Hugging Face Space `FINAL-Bench/open-quantum-challenge`。

选它的理由不是“题目好听”，而是**它是唯一一个判据完全客观、数据完全公开、本机能独自跑通的**：

- 公开数据 `FINAL-Bench/oqc-data`：`qec/detectors.npy`（10000×120）+ `qec/dem.txt`；
- 公开 harness `github.com/final-bench/quantum-bench-harness`，含冻结电路生成器与评分脚本；
- 评分键私有，但**口径可复现**：我自己采样的 MWPM 基线 0.0146～0.0158，与榜单锚点 0.0153 一致；
- 榜单可读：`/api/leaderboard?board=qec` 返回 4 条人类成绩（最好 0.0083）。

Season 1（Fermi-Hubbard 基态能量）顺手做完，但**不作为主攻**：榜单上 7 个人已经全部顶到
官方锚点 −36.4448（理论下界），没有获胜空间；它在本轮的作用是交叉验证与审计（见 04）。

## 四道闸

| 闸 | 问题 | 本轮的判据 | 结果 |
|---|---|---|---|
| A 资格 | 能不能报、要不要钱、要不要到现场 | 官方页与 harness 原文；个人可报、免费、纯线上、无需量子硬件 | 过 |
| B 实现 | 东西是不是真的对 | `verify.py` 34 条断言；MWPM 复现官方锚点；相关性/集成/混合三层单调改进；BP+OSD 修正量必然满足综合式 | 过 |
| C 交付 | 交的东西合不合格式 | `out/predictions_qec.json` 10000 个 `shot_*` 键、全 0/1、method ≤200 字符、摘要与侧文件一致 | 过 |
| D 复核 | 有没有自欺 | 留出种子 505 全程未参与调参；三族解码器在同一批镜头上做配对比较；审计公开冻结键的可复现性 | 条件性（数字在 04，见“未解决”） |

闸 D 的“条件性”在这里的具体含义：**我无法验证真实榜单分数**（键私有），只能给出留出种子上的
无偏估计与置信区间；任何“我们第几名”的说法都只是拿估计值去和别人的单次测量比。
