# 05 · 交付与提交

## 这次要交的东西

一件：`out/predictions_qec.json`

```json
{
  "hf_user": "<你的 HF 用户名>",
  "method": "Ensemble correlated-MWPM: ... (≤200 字符)",
  "predictions": {"shot_00000": 0, "shot_00001": 1, ... "shot_09999": 0}
}
```

- 10000 个键，`shot_%05d`，值只能是 0 或 1；
- 文件 300 KB 左右；
- 比特数组摘要 `af32b951b89c02e3…`（`out/predictions_qec.sha256` 是完整值），重跑同一命令得到同一摘要。

## 提交步骤（需要你本人操作，约 3 分钟）

1. 打开 <https://huggingface.co/spaces/FINAL-Bench/open-quantum-challenge>，用 Hugging Face 账号登录
   （Space 开了 `hf_oauth`，提交接口要求登录）。
2. 切到 **Season 2 — QEC Decoder** 那一栏，把 `out/predictions_qec.json` 里的
   `predictions` 对象整个粘进去（或按页面提示上传）。
3. `hf_user` 填你的账号名，`method` 用文件里那一行（170 字符，没超 200 的上限）。
4. 点 Submit。成功会返回 `accepted: true` 和一个 `submission_id`。
5. **几分钟后**再看榜单：`https://final-bench-open-quantum-challenge.hf.space/api/leaderboard?board=qec`。
   worker 是异步评分的，不会立刻出分。
6. 把 `submission_id` 与榜单截图贴回 **AME-1532**，我把状态推到 `in_review`。

> 每日配额：一个账号一天一次（两季合计，按 KST 零点重置）。所以别在没把握的时候浪费机会——
> 本轮已经把能验的都验过了（33 条断言），可以直接交。

## Season 1 要不要交？

**建议不交**。三个理由：

1. 榜上 7 个人全部顶在锚点 −36.4448，没有名次可争；
2. 我的精确对角化给出 −27.4306，与锚点差 9.01，这说明锚点用的不是题目描述的那个模型
   （或锚点本身有问题）——交上去会被排在最后，没有信息量；
3. 交一个"和锚点不一致"的数，最大的价值是**逼主办方复核**。如果你愿意做这个实验，
   `out/predictions_qsim.json` 已经按格式生成好了，直接粘；但我不建议占用当天的配额。

## 复核

```bash
python verify.py          # 33 条断言，退出码即结论
python scripts/validate_hybrid.py 10000   # 四个种子的完整阶梯，约 5 分钟
```

`out/verify.log` 是最近一次运行的全文（33 passed, 0 failed）。
