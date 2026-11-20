# 02 · 赛制（原文核实，2026-11-21）

来源：
- Space 卡片：`https://huggingface.co/spaces/FINAL-Bench/open-quantum-challenge`（README 由 `resolve/main/README.md` 取得）
- 挑战博文：`huggingface.co/blog/FINAL-Bench/the-open-quantum-challenge-quantum-simulation-and`
- 公开 harness：`github.com/final-bench/quantum-bench-harness`（README、`bench_qec.py`、`score_predictions.py`、`frozen/surface_d5.py`）
- 公开数据：`huggingface.co/datasets/FINAL-Bench/oqc-data`（`qec/task.json`、`qec/dem.txt`、`qec/detectors.npy`）
- 实时榜单：`https://final-bench-open-quantum-challenge.hf.space/api/leaderboard?board=qec`

## 两个赛季

| | Season 1 | Season 2 |
|---|---|---|
| 板 | QSIM | QEC |
| 题 | Fermi-Hubbard 基态能量（9 个系统） | 旋转表面码 d=5、p=0.005 的逻辑错误率 |
| 指标 | 总能量越低越好；低于精确基态视为非物理，拒收 | 逻辑观测量判错的比例，越低越好 |
| 开幕 / 截止 | 2026-10-11 / **2027-02-28** | 2026-10-11 / **2027-03-31** |
| 奖金 | $1,000（600/300/100） | $1,000（600/300/100） |

原文（README）：

> Season 1 — Quantum Simulation (Fermi-Hubbard ground-state energy). Opens 2026-10-11, deadline 2027-02-28.
> Season 2 — QEC Decoder (rotated surface code d=5, p=0.005, logical error rate). Opens 2026-10-11, deadline 2027-03-31.
> Prize: $1,000 per season (🥇600 / 🥈300 / 🥉100).

## 提交格式（`qec/task.json`）

```json
{
  "board": "QEC",
  "metric": "logical error rate = fraction of shots whose predicted observable bit is wrong",
  "submit_format": {"predictions": {"shot_<i:05d>": "0 or 1"}, "hf_user": "...", "method": "..."}
}
```

`gates.py` 只查形状：`predictions` 非空、≤60000 项、每项能被 `int()` 且 ∈{0,1}；键名以 `shot_`
开头的比例过半即可（错板守卫）。`app.py` 会把 `method` 截到 200 字符、`hf_user` 取 OAuth 名字。

## 评分链路（`app.py` 注释原文）

> Entries land in a private dataset ledger (FINAL-Bench/oqc-ledger); a worker on the GPU box
> polls that ledger, scores each entry against a frozen private answer key, and writes the
> rolled-up leaderboard back.

即：提交进私有账本 → GPU worker 用私有键评分 → 回写榜单。所以**提交后不会立刻知道分数**，
几分钟后看 `/api/leaderboard`。

## 配额

`app.py` 里有每日配额（`DAILY_CAP`，按 KST 日切，两季合计）。参赛者账号每天一次机会，
交之前必须自己把口径验到位——这正是本轮把 `verify.py` 写死的原因。

## 资格

博文原文：**"open to anyone with a GPU and an interest in quantum computing"**，不排除任何地区；
不要求量子硬件；不要求团队。Harness 与本机 CPU 即可完成（本作品全程未用 GPU）。

## 榜单快照（2026-11-21 06:33 UTC+8）

| 名次 | 参赛者 | LER | 方法（截断） |
|---|---|---|---|
| 锚点 | baseline-MWPM | 0.0153 | MWPM (pymatching) on the public DEM |
| 1 | Soonjae2 | 0.0083 | Coset-sum ML decoder … 1677 merged mechanisms |
| 2 | ntoledo145 | 0.0084 | Coset-sum ML … beam-truncated exact sweep |
| 3 | 1Dmeds | 0.0088 | Tesseract b10/o8 + hard-shot logical-coset summation |
| 4 | PhD-Shin | 0.0093 | Tesseract det_beam 65535 |

四个人的方法描述都指向同一件事：**在 1677 个"合并后"的机制上做逻辑陪集的求和**，
而不是在分解后的 1953 条边上做匹配。这解释了为什么纯匹配类解码器（包括相关匹配）
会卡在 0.011 附近（见 03）。
