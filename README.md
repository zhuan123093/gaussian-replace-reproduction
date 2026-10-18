# Gaussian recursive-training experiments

高维统计课程项目：合成数据递归训练中的模型坍塌。

- **[A1 复现 notebook](A1_cleaned.ipynb)**：多元高斯完全替换与原有三张结果图。
- **[A2 整理后的代码](A2/README.md)**：E0—E4、冻结对照、RB 参数积分与初始锚定。
- **[A2 结果简述](A2/RESULTS.md)**：主实验与参数记忆试验分别报告，附 CSV 汇总。

## 运行 A2

```bash
cd A2
python -m pip install -r requirements.txt
python -m pytest tests -q
python main.py study --smoke --workers 2
```

完整运行、参数记忆、结果分析与理论核验见 [A2 README](A2/README.md)。
已保存研究结果包括主实验 131 个配置、5260 条轨迹，以及参数记忆 600 条轨迹。
仓库提供代码和汇总，不提交大批原始轨迹。结论限定在已测试的高斯设定。
