# 公开输入（来自 FINAL-Bench/oqc-data，CC-BY-NC-4.0）

| 文件 | 来源 | 说明 |
|---|---|---|
| `qec/detectors.npy` | `huggingface.co/datasets/FINAL-Bench/oqc-data` | 10000 × 120 uint8，待解码的镜头 |
| `qec/dem.txt` | 同上 | stim DetectorErrorModel（分解版，1953 机制） |
| `qec/task.json` | 同上 | 提交格式与指标原文 |
| `qsim/systems.json` | 同上 | Season 1 的 9 个系统定义 |

`dem.txt` 与 `stim.Circuit.generated("surface_code:rotated_memory_z", d=5, rounds=5, p=0.005…)`
的 `detector_error_model(decompose_errors=True)` **逐字符相同**（`verify.py` 第 4 条断言）。
