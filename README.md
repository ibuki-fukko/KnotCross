# 2D NURBS 曲线求交实验

本目录保存一套可重复运行的二维 NURBS 曲线求交实验代码、合成数据、结果和阶段报告。当前实现固定三次曲线、每条曲线 8 个控制点、开放节点向量和单位权重，学习预测 0 到 3 个普通孤立交点。该范围不包括相切、重合、尖点及任意阶数/控制点数，因此报告中不会把当前结果外推到这些情形。

## 实验结构

- `code/geometry/nurbs.py` 实现 NURBS 基函数、曲线求值、求导和数值参考求交。
- `code/data/generate_dataset.py` 生成有交点数量分层的训练、验证和测试数据；交点标签由折线段相交筛查后经数值细化获得。
- `code/models/mlp_numpy.py` 实现共享 MLP、全连接层反向传播和 Adam 更新，不依赖 PyTorch。
- `code/training/train_experiments.py` 训练共享完整模型、去几何损失、去残差场和非共享直接编码消融模型。
- `code/evaluation/evaluate_experiments.py` 在独立测试集上计算精确率、召回率、F1、交点数准确率、参数误差、几何残差和无交点误报率，并和数值方法及参数网格基线比较。
- `configs/experiment.json` 记录实验固定设置。
- `data/` 保存数据和标签来源元数据；`results/` 保存权重、训练日志和评测指标；`reports/` 保存阶段报告。

## 环境

实验代码只依赖 Python 3、NumPy 和标准库。推荐使用本机已安装的 Python 3.12 和 NumPy 2.x。无需联网下载模型，也无需 GPU。

## 完整重跑

在本目录打开 PowerShell 后运行以下命令。默认随机种子固定为 `20261025`，CPU 训练结果应一致；推理耗时会随电脑变化。

```powershell
python code/data/generate_dataset.py --output data/experiment_large.npz --train 1600 --validation 400 --test 600 --seed 20261025
python code/training/train_experiments.py --data data/experiment_large.npz --output results/large --epochs 20 --batch-size 32 --learning-rate 0.0005 --seed 20261025
python code/evaluation/evaluate_experiments.py --data data/experiment_large.npz --models results/large --output results/large/evaluation
python code/evaluation/plot_examples.py --data data/experiment_large.npz --models results/large --output results/large/examples
```

完整数据生成和训练在普通 CPU 上需要较长时间。如果运行时间有限，可先减少数据量或训练轮数；提交报告时应记录实际命令与轮数，不要把缩短训练的结果写成完整运行。

## 标签和指标说明

标签是参数对集合 `(u, v)`。数值标签生成器在采样折线上查找线段交叉，再对 `(u, v)` 做二维 Newton 细化；它适用于普通横截交点，不保证发现相切或极近的交点。为了避免训练/测试数据泄漏，三个数据集由互不相同的随机种子独立生成。

评测按参数对欧氏距离不超过 `0.035` 判定命中。这个阈值是归一化参数空间中的距离，不是平面几何距离。精确率、召回率和 F1 统计所有预测交点；几何残差统计匹配预测处 `||C1(u)-C2(v)||`；无交点误报率仅在真实无交点子集上统计。数值参考方法用在测试标签核验与传统基线比较，纯神经网络指标不会用数值算法补充漏检点。

## 结果边界

数据由程序随机生成，不是从论文图像数字化，也不是“覆盖所有可能曲线”。数值标签生成过程与传统求交基线共享部分几何求解思想，指标不能解释为对独立 CAD 内核的全面比较。当前实现把 NURBS 计算写成 NumPy 函数，训练损失中用中心差分近似曲线参数导数；这是一种可微几何残差训练的原型，不等同于复现 NURBS-Diff 论文的 PyTorch/CUDA 模块。正式论文前还需用独立高精度求交程序核验标签，并增加近距离交点、未见控制点数、不同权重/节点向量及多随机种子实验。

如果代码由 AI 辅助生成，汇报前请逐个运行脚本、理解输入输出与损失项，并按学校或导师的要求说明辅助方式；不要把未亲自检查的实现称为完全独立编写。
