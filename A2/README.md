# A2 高斯递归训练

先看 [现有结果简述](RESULTS.md)。本目录是整理后的高斯代码与结果汇总。

本目录包含单高斯 E0—E4、冻结对照、参数记忆（RB 与初始锚定）、必要的理论实现、核验与测试。`results_summary/` 中两张 CSV 是既有正式结果的汇总；原始数千条轨迹不纳入仓库。

## 安装与快速检查

进入本仓库的 `A2/` 目录（与 main.py 同级），使用 Python 3.11 或兼容环境：

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
python main.py study --smoke --workers 2
```

## 完整高斯主实验

```bash
python main.py study --all --workers 4
```

这会运行比例探索、长期替换、主验证和稳健性，并在 `results/` 保存轨迹、图表与报告。正式规模为 131 个配置、5260 条轨迹；计算量明显大于 smoke。每条轨迹按种子保存，同配置与代码哈希一致的完成结果可复用。

## 参数记忆

```bash
python main.py memory --seeds 30 --generations 1000 --workers 4
python main.py audit --output results/parameter_memory
```

默认在 `results/parameter_memory/` 保存新结果。快速工程检查可用：

```bash
python main.py memory --seeds 2 --generations 25 --workers 2 --output results/memory_smoke
```

## 分析与理论

完成相应主实验、获得原始轨迹后，可重新汇总或核验：

```bash
python main.py analyze --all
python main.py verify
python -m pip install -r theory/requirements.txt
python main.py theory
python main.py spectral
```

`results_summary/` 只有既有结果汇总，不能替代上述命令需要的原始轨迹。分析、核验和理论对照读取的是运行后新生成的 `results/`。

## 代码目录

- `main.py`：统一入口，`python main.py --help` 查看命令。
- `gaussian_core.py`：单高斯拟合、有限样本随机统计量、评价指标。
- `run_experiments.py` / `analyze_results.py`：E0—E4 实验和结果分析。
- `memory/`：参数积分、初始锚定、参数记忆试验和方向尾部审计。
- `theory/`：参数记忆依赖的矩理论，以及谱界实现与数值核验。
- `checks/`、`tests/`：保存轨迹核验和 18 项已有测试。

代码来自当前 A2，逐文件原样复制；校验信息在 `package_manifest.json`。文档中的两批正式结果来自既有主实验和参数记忆试验，本次发布只做工程验证，不重跑正式研究实验。
