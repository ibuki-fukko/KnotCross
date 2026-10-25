from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "large"
METRICS_PATH = RESULTS / "evaluation" / "metrics.json"
SUMMARY_PATH = RESULTS / "training_summary.json"
OUTPUT = ROOT / "reports" / "阶段性实验研究报告.docx"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else []
summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) if SUMMARY_PATH.exists() else []

def find(method):
    for item in metrics:
        if item.get("method") == method:
            return item
    return {}


def fmt(value):
    if value is None:
        return "未记录"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


doc = Document()
section = doc.sections[0]
section.top_margin = Cm(2.2)
section.bottom_margin = Cm(2.2)
section.left_margin = Cm(2.5)
section.right_margin = Cm(2.5)
styles = doc.styles
styles["Normal"].font.name = "宋体"
styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
styles["Normal"].font.size = Pt(10.5)
for name in ["Title", "Heading 1", "Heading 2"]:
    styles[name].font.name = "黑体"
    styles[name]._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")

title = doc.add_paragraph(style="Title")
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.add_run("二维 NURBS 曲线求交神经网络阶段性实验研究报告")
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.add_run("实验记录日期：2026年10月25日").italic = True

p = doc.add_paragraph()
p.add_run("报告目的。 ").bold = True
p.add_run("本报告记录当前研究方案从方法设计进入代码实现和实验验证后的阶段性结果。实验对象是两条二维三次 NURBS 曲线，网络尝试直接预测参数域中的全部普通孤立交点。报告把已经运行得到的结果与仍待验证的部分分开，便于下一次向指导老师汇报。")

doc.add_heading("一 实验问题与范围", level=1)
doc.add_paragraph("给定两条曲线 C1(u) 和 C2(v)，交点可以用参数对 (u, v) 表示，满足 C1(u)=C2(v)。本阶段将交点数量限制为 0、1、2、3 个，曲线固定为三次曲线，每条曲线包含 8 个控制点，使用开放节点向量 [0,0,0,0,0.2,0.4,0.6,0.8,1,1,1,1]，权重统一为 1。曲线在生成后做平移、旋转和尺度归一化。")
doc.add_paragraph("当前没有把相切、曲线重合、尖点、端点特殊交点、极近交点、不同次数、不同控制点数和不同权重作为正式测试结论。原因是这些情况需要独立的标签验证和单独的评价口径，不能与普通横截交点混在一起。")

doc.add_heading("二 数据和标签", level=1)
doc.add_paragraph("训练、验证和测试曲线由 code/data/generate_dataset.py 自动生成，并按照 0、1、2、3 个交点四类平均分配。测试使用与训练不同的随机种子。数据文件 data/experiment_large.npz 已生成，包含 1600 对训练曲线、400 对验证曲线和 600 对测试曲线。")
doc.add_paragraph("标签生成器先把每条 NURBS 曲线在参数区间内密集采样，把相邻采样点组成折线段，并检查两条折线段是否相交。得到初始参数对后，再用二维 Newton 细化，并按参数距离去重。这个过程适合普通横截交点，因此标签不是对所有退化几何情况的绝对真值。后续论文实验需要再用独立实现或高精度 CAD 内核抽样核验。")

doc.add_heading("三 网络和损失", level=1)
doc.add_paragraph("完整模型在 code/models/mlp_numpy.py 中实现。两条曲线先经过同一套共享 MLP 编码器，隐藏层为 128、128、64 个神经元。输入除了固定参数位置上的曲线采样点，还包括参数网格上的残差场。残差场每个网格位置保存 C1(ui)-C2(vj) 的横坐标差、纵坐标差、欧氏距离以及由距离得到的接近程度通道。两个曲线特征和残差场展平向量融合后，分别进入概率图头和交点槽位头。槽位头最多输出 3 个参数对，每个槽位同时给出参数 u、参数 v 和存在概率。")
doc.add_paragraph("训练损失包含四部分。第一部分监督参数域概率图；第二部分使用枚举匹配寻找预测槽位与真实交点的最低代价配对，从而不依赖交点顺序；第三部分把预测参数代入 NURBS 曲线计算，并加入 ||C1(u)-C2(v)|| 的几何残差；第四部分惩罚多个高置信度槽位集中在同一参数位置。训练时随机交换两条曲线，并同步交换标签参数 u、v。这里的 NURBS 计算和参数导数是 NumPy 原型，采用中心差分估计曲线参数导数，尚未复现 NURBS-Diff 的 PyTorch/CUDA 实现。")

doc.add_heading("四 对比实验", level=1)
doc.add_paragraph("当前已运行四个网络版本。shared_full 是完整模型；shared_no_geometry 去掉几何残差损失；shared_no_residual 把参数域残差场置零；direct 取消共享曲线编码器，直接把两条曲线采样和残差输入一个普通 MLP。另有 parameter_grid_only 基线，只在参数网格上找距离局部小值，不做连续细化；dense_sampling_newton_refinement 是用于标签核验和传统数值参考的密集采样加 Newton 细化方法。")

if metrics:
    doc.add_heading("五 已运行结果", level=1)
    doc.add_paragraph("以下网络指标来自 data/experiment_large.npz 的 600 对测试曲线，阈值只在 400 对验证曲线上选择。参数距离阈值为 0.035。数值参考方法在这批合成测试集上恢复了标签生成器的交点，因此不能据此声称它优于独立 CAD 内核；它只能作为当前标签过程的一致性检查。")
    table = doc.add_table(rows=1, cols=6)
    table.style = "Table Grid"
    headers = ["方法", "Precision", "Recall", "F1", "交点数准确率", "无交点误报率"]
    for cell, header in zip(table.rows[0].cells, headers):
        cell.text = header
    methods = ["dense_sampling_newton_refinement", "parameter_grid_only", "shared_full_slots", "shared_no_geometry_slots", "shared_no_residual_slots", "direct_slots"]
    for method in methods:
        item = find(method)
        row = table.add_row().cells
        values = [method, fmt(item.get("precision")), fmt(item.get("recall")), fmt(item.get("f1")), fmt(item.get("exact_count_accuracy")), fmt(item.get("no_intersection_false_positive_rate"))]
        for cell, value in zip(row, values):
            cell.text = value
    full = find("shared_full_slots")
    doc.add_paragraph(f"完整模型 shared_full_slots 的测试 F1 为 {fmt(full.get('f1'))}，召回率为 {fmt(full.get('recall'))}，匹配预测的平均参数误差为 {fmt(full.get('mean_parameter_error_matched'))}。在当前训练设置下，模型大多数预测没有落到真实交点附近，因此还不能作为可用求交器。")
    doc.add_paragraph("训练日志显示，shared_full 的训练损失从第 1 轮的 1.4463 降到第 20 轮的 0.7627，验证损失从第 1 轮的 1.3160 降到第 10 轮的 1.1113，第 20 轮为 1.1155。数据扩大后验证损失比小数据实验稳定，但最终交点指标仍然很低，说明损失下降没有转化为参数对定位能力。")
else:
    doc.add_heading("五 已运行结果", level=1)
    doc.add_paragraph("当前结果文件尚未生成，不能提前填写数值。")

doc.add_heading("六 当前结论", level=1)
doc.add_paragraph("当前阶段已经完成从数据生成、NURBS 求值、标签生成、共享 MLP、参数域残差场、无序槽位匹配、几何残差损失到独立测试指标的完整原型链条。实验结果没有支持神经网络已经超过传统数值方法，相反，当前网络的交点定位精度很低。这个结果说明方法框架可以继续研究，但网络结构和训练目标还没有达到论文实验要求。")
doc.add_paragraph("当前最需要处理的技术问题是网络对参数对的定位能力。可能原因包括合成交点在参数域中的分布不均、用展平残差场代替二维结构、固定槽位存在概率的监督不平衡、几何残差梯度尺度不稳定，以及标签生成器与训练目标之间存在误差。下一步应先建立一小批人工构造且交点参数已知的验证样本，再分开测试概率图、槽位回归和几何残差，避免一次同时改变多个部分。")

doc.add_heading("七 下一步实验", level=1)
doc.add_paragraph("下一步先把训练和验证集扩大到至少数千对，并使用多组随机种子重复训练；再把交点标签从固定填充的参数数组改为显式的集合标记，增加无交点类别的独立分类头。随后需要用二维残差场的局部区域提议交点候选，减少网络直接从展平向量回归全局参数的难度。完成这些修改后，再比较共享 MLP、加入轻量二维卷积、仅概率图和概率图加槽位回归四种结构。")
doc.add_paragraph("边界测试应单独生成近距离交点、端点交点、相切、不同权重和未见控制点数量的曲线，并报告查全率、查准率、F1、参数误差、平面几何残差、重复预测率和推理时间。只有在这些测试完成后，才能讨论模型对未见曲线的泛化能力。")

doc.add_heading("八 复现方法", level=1)
doc.add_paragraph("在项目根目录运行 README.md 中的三条命令即可重新生成数据、训练模型和评测。实际运行使用 Python 3.12、NumPy 2.3.5，代码不依赖 PyTorch、SciPy 或 GPU。随机种子、曲线形式、模型层数、损失权重、阈值选择方法和评价指标均记录在 configs/experiment.json 以及各结果目录中。不同电脑的运行时间会不同，但固定种子下的数值结果应保持接近。")

for paragraph in doc.paragraphs:
    paragraph.paragraph_format.first_line_indent = Cm(0.74)
    paragraph.paragraph_format.space_after = Pt(5)

# Reset title and headings indentation.
for paragraph in doc.paragraphs:
    if paragraph.style.name in {"Title", "Heading 1", "Heading 2"}:
        paragraph.paragraph_format.first_line_indent = Cm(0)

# Tables use a small font for the six-column summary.
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(8.5)

doc.save(OUTPUT)
print(OUTPUT)

