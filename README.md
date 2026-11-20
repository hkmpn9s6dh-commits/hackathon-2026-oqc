# OQC · Season 2 — 表面码逻辑错误率解码

> Open Quantum Challenge（FINAL-Bench / Hugging Face）· **QEC Decoder** 赛季的参赛作品。
> 判据：对 10000 个探测器镜头各判一个逻辑观测量比特，逻辑错误率越低越好。

**合并估计 LER = 0.00988（395 / 40000 镜头，95% CI [0.00895, 0.01089]）**，
留出种子（505，全程未参与调参）上 0.0103。榜单锚点 MWPM 是 0.0153，榜首 0.0083 —— 详见
[`docs/04-RESULTS.md`](docs/04-RESULTS.md)。

## 一句话方法

> 相关匹配（把同一条 4 体故障的两条半边绑起来匹配）→ 120 次抖动边权采样、按匹配权重投票
> （近似"对所有等权解求和"）→ 投票边际 <0.5 的 0.5% 镜头交给 BP+OSD（1677 个整变量故障）补刀。

## 目录

```
bench_qec.py          单个解码器在一个种子上的基准（含 95% Wilson 区间）
bench_qsim.py         Season 1：9 个费米-哈伯德系统的精确对角化（审计用）
train_cnn.py          试过的 3D 卷积解码器（未采用，见 docs/03 的"没采用"一节）
make_submission.py    生成 out/predictions_qec.json 与 qsim 版
verify.py             33 条断言，退出码即结论
scripts/validate_hybrid.py   四个种子的完整阶梯（约 5 分钟）
src/oqc/              dem 解析、三种解码器、指标
docs/                 01 选赛 · 02 赛制 · 03 方法 · 04 结果 · 05 提交 · 06 教训
data/qec/             公开的 detectors.npy 与 dem.txt（来自 FINAL-Bench/oqc-data）
out/                  提交件、日志、结果 JSON
```

## 复现

```bash
py -3 -m venv .venv && ./.venv/Scripts/python.exe -m pip install -i https://pypi.org/simple \
    numpy scipy stim pymatching ldpc

./.venv/Scripts/python.exe bench_qec.py --decoder mwpm      --shots 20000 --seed 101   # 0.0146
./.venv/Scripts/python.exe bench_qec.py --decoder mwpm_corr --shots 20000 --seed 101   # 0.0112
./.venv/Scripts/python.exe scripts/validate_hybrid.py 10000                             # 0.0099
./.venv/Scripts/python.exe make_submission.py                                          # 提交件
./.venv/Scripts/python.exe verify.py                                                   # 33 passed
```

依赖见 `requirements.txt`。全程 CPU，不需要量子硬件、不需要 GPU。

## 两个需要主办方知道的问题

1. **Season 1 的榜单锚点与题目描述不一致**：锚点 −36.4448，题目描述的哈密顿量下精确对角化是
   −27.4306（两格点与解析式逐位吻合）。榜上 7 个人全部报同一个数，一位不差。
2. **评分脚本与真值生成脚本同在工作流里**：`score_predictions.py` + `FROZEN_SEED` 都在公开仓库，
   `LER < 1e-4` 才提示泄漏。建议真值生成与评分分离、或做逐镜头一致性检验。

细节见 [`docs/04-RESULTS.md`](docs/04-RESULTS.md)。

## 许可

代码 MIT。`data/qec/` 来自 `FINAL-Bench/oqc-data`（CC-BY-NC-4.0），仅用于参赛复现。
