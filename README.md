# DLM_Multitask_Reasoning

Code for controlled multi-question reasoning experiments in diffusion language models.

研究独立、共享中间量与上下游三类任务关系，实现模型适配、逐 token 解码、状态 patching、attention head bank 和离线审计。

**此公开仓库仅包含代码、配置、测试和运行说明。实验记录、成绩、报告、隐藏状态数组及原始结果包不公开。**

## 方法与范围

- iLLaDA / Dream 原生 tokenizer、chat template 与各自 logits 对齐。
- 算式任务：`C=x+y; U=u+r; V=v+s; A=C+U; B=2*C+V`。
- 单次或持续中间量残差、attention / MLP 输出及跨层 query-head 输出 patching。
- **Source** 提供状态，**Target** 接受状态；部分实现保留内部历史变量名。
- 记录并审计最终反事实、基线已有命中、非目标分支和自身替换对照。

中间量生成顺序由程序控制，接口为预留的数字槽；此设置本身不能证明自主规划、隐式复用或只计算一次。公开代码不附带实证结论。

## 安装与测试

Python 3.10+，建议虚拟环境；核心机械测试不需要模型权重。

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
```

部分测试需要可选 CPU PyTorch，未安装时跳过。GitHub Actions 安装 CPU PyTorch 运行完整测试。

## GPU 运行

[在 Colab 打开代码入口](https://colab.research.google.com/github/Twist-Shan/DLM_Multitask_Reasoning/blob/main/notebooks/Run_Mechanism_Headbank.ipynb)

需要自己的 GPU 与模型下载网络。运行前建议检查 notebook 中的 commit，并设置为你要复现的确切 SHA。运行环境、命令和输出路径见 [运行指南](docs/running.md)。

```text
src/dlm_coupling/  任务、解码、模型适配与 patching
scripts/          生成、运行、分析、审计和绘图工具
tests/            机械正确性测试
configs/          模型与实验配置
notebooks/        不含数据、结果或内嵌归档的 Colab 入口
docs/             运行说明
```

机制题由运行器生成，seeds 为 127/128/129；早期入口默认从 123 开始，均不按日期选取。旧单数字槽配置用于工程检查，不应直接当作多 token 机制实验配置。

## 来源

- [iLLaDA](https://huggingface.co/GSAI-ML/iLLaDA-8B-Instruct)
- [Dream](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B)

模型权重、第三方模型源码和上游任务仓库不随本仓库分发；运行时从指定来源加载。iGSM 导出工具需要另行提供上游生成器，参见其 `--help`。本项目原创代码暂未指定开放源代码许可证。
