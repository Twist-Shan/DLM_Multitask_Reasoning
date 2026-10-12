# 运行指南

## 组件与 head bank 实验

使用已配置 CUDA PyTorch 的 Linux / Colab 环境，建议 A100 40GB。模型使用 bfloat16、batch size 1，固定 iLLaDA revision `5769f049224b2cff61106ebce0429ede5deeb434`，启用官方 `trust_remote_code`。

```bash
python -m pip install -e .
python -m pip install 'transformers==4.46.2' 'tokenizers==0.20.3' 'huggingface-hub==0.36.0' 'accelerate>=0.34,<2' 'safetensors>=0.4' einops numpy matplotlib
python scripts/run_mechanism_pilot.py
python scripts/analyze_mechanism_pilot.py runs/mechanism-pilot --out runs/mechanism-pilot/analysis.json
python scripts/plot_mechanism_pilot.py runs/mechanism-pilot
python scripts/run_headbank_pilot.py
python scripts/analyze_headbank_pilot.py runs/headbank-pilot runs/mechanism-pilot --out runs/headbank-pilot/analysis.json
python scripts/plot_headbank_pilot.py runs/headbank-pilot
```

运行器自行生成发现/验证输入。组件发现 seed 127，组件验证 seed 128，head bank 独立验证 seed 129。每组包含 base、C/U/V 改动和同值不同操作数控制。C/U/V 由模型生成，然后程序允许 A/B 提交。每一步只提交一个最高置信度 token。

head bank 读取上一阶段 `runs/mechanism-pilot`。输出目录已存在时运行器拒绝覆盖；先保存已有结果，再选择新工作目录重新运行。脚本会保存模型环境、提交轨迹、Source / Target 对齐信息及离线审计所需字段。运行产生的 `runs/`、`data/`、`reports/` 均被 Git 忽略，不自动上传。

依赖安装不会替你选择合适的 CUDA PyTorch 构建；记录实际 Python、CUDA、PyTorch、模型 revision 及仓库 commit，才能区分软件环境差异。

## 其他入口

- `run_shared_causal.py`：共享中间量单次干预。
- `run_shared_followup.py`：持续干预与分支对照。
- `run_baseline_repair.py`：提示和生成接口校准。
- `generate_igsm_pairs.py` / `generate_igsm_explicit.py`：上游图导出与显式等式题目构造，先查看 `--help`。
- `run_igsm_explicit.py`：显式题目的单问/联合生成，需要自行生成数据路径。
- `run_answer_interp.py`：晚期答案 token 干预，需要先生成基础轨迹。

配套 `analyze_*` / `audit_*` / `plot_*` 读取你自己运行得到的文件。公开仓库不提供任何历史结果文件。部分绘图脚本按当前预设样本数标轴，修改样本量时应同步核对图注和分母。

## 解释边界

Source 提供 patching 状态，Target 接受该状态；Source 的部分 A/B token 在每步与 Target 对齐。持续交换成功仅回答所选干预是否足以改变输出，不直接证明组件必要性、最小性或自然计算复用。反事实指标必须扣除基线已有命中，并分别报告整体样本与基础正确子集。
