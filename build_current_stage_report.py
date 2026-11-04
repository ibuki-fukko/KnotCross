from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path.cwd()
REPORT = ROOT / "reports" / "stage_report.docx"
RESULT_ROOT = ROOT / "results" / "autoresearch"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


summary = read_json(RESULT_ROOT / "refined_steps8_seed_summary.json")
boundary = read_json(RESULT_ROOT / "boundary_refined_metrics.json")
steps = read_json(RESULT_ROOT / "refine_steps_ablation.json")


def fmt(value, digits=4):
    if value is None:
        return "—"
    if abs(value) < 1e-4 and value != 0:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def set_cell_text(cell, text, size=9):
    cell.text = str(text)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for paragraph in cell.paragraphs:
        paragraph.paragraph_format.space_after = Pt(0)
        for run in paragraph.runs:
            run.font.name = "宋体"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
            run.font.size = Pt(size)


def shade_cell(cell, fill):
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for index, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[index], header, 8.5)
        shade_cell(table.rows[0].cells[index], "D9EAF7")
    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            set_cell_text(cells[index], value, 8.5)
    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Cm(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_body(doc, text):
    paragraph = doc.add_paragraph(text)
    paragraph.paragraph_format.first_line_indent = Cm(0.74)
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.line_spacing = 1.25
    return paragraph


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

p = doc.add_paragraph(style="Title")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run("二维 NURBS 曲线求交神经网络阶段性实验研究报告")
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.add_run("自动研究迭代记录 2026年10月26日").italic = True

add_body(doc, "本报告记录二维 NURBS 曲线求交实验从初始共享 MLP 到局部候选提议和几何修正的迭代过程。当前实验已经在固定测试集上得到稳定的数值结果，但正式方法仍属于混合方案：神经网络负责从参数域残差场中提出交点候选，局部几何修正负责把候选参数对推到两条曲线的交点附近。报告中的主测试集结论只适用于本阶段固定的三次、八控制点、单位权重曲线，边界样本另行报告。")

h = doc.add_heading("一 实验问题和范围", level=1)
add_body(doc, "给定两条平面 NURBS 曲线 C1(u) 和 C2(v)，模型需要输出全部交点的参数对 (u,v)。本阶段把每条曲线固定为三次曲线，使用 8 个控制点、开放节点向量和单位权重；每对曲线的普通孤立交点数为 0、1、2 或 3 个。训练集、验证集和测试集分别包含 1600、400 和 600 对曲线，测试集在训练过程中始终没有参与参数选择。")
add_body(doc, "这组实验没有把相切、重合、尖点、极近交点、不同权重、不同曲线次数和不同控制点数量纳入主测试集结论。原因是这些情况需要单独设计标签和判定标准。当前的边界数据用于检查风险，不用于声称模型已经解决这些问题。")

h = doc.add_heading("二 模型和数据处理", level=1)
add_body(doc, "初始模型把两条曲线的特征送入共享 MLP，并结合参数域残差场进行固定槽位回归。前期实验发现，直接把整个残差场展平后预测全局参数对，容易把多个局部交点混在一起，导致交点重复、漏检和参数偏移。后续模型先在 48×48 参数网格上计算 C1(ui)-C2(vj) 的横坐标差、纵坐标差和欧氏距离，再提取距离场中的局部候选区域。每对曲线保留 8 个候选，每个候选只由一个小的局部残差块和网格中心参数输入 MLP，输出候选存在概率和参数偏移。")
add_body(doc, "候选标签由数值求交程序生成。程序先对曲线密集采样，通过折线段相交得到初始参数对，再用二维 Newton 细化并去重。该标签生成器适合普通横截交点，不能自动保证相切或极近交点标签完全正确。候选网络使用二元分类损失和正例参数偏移回归损失，模型权重、训练日志、随机种子和数据文件均保存在项目目录。")

h = doc.add_heading("三 迭代过程", level=1)
add_body(doc, "初始共享 MLP 在测试集上的 F1 约为 0.0218，说明直接预测固定槽位不能稳定完成参数对定位。加入局部候选提议后，32×32 网格的测试 F1 为 0.4359；网格改为 48×48 后提高到 0.6297；放宽局部极小值提议条件后提高到 0.7119。候选数从 12 减少到 8 后，测试 F1 达到 0.7735。候选数增加到 20 或局部松弛过大都会带来更多误报。")
add_body(doc, "随后对网络输出的候选参数执行局部几何修正。每次修正计算两条 NURBS 曲线点的差值，并用曲线参数导数形成二维 Jacobian，求解参数增量。修正后再按平面几何残差过滤候选。这个步骤不是重新训练网络，而是对网络提出的候选进行确定性的几何校正，因此结果应称为神经候选提议加局部几何修正，不能称为纯神经网络端到端求交。")

add_table(doc, ["修正次数", "过滤前 F1", "过滤后 F1", "过滤后参数误差", "过滤后几何残差"], [
    ["0", "0.6946", "0.0328", "2.525e-03", "3.746e-03"],
    ["1", "0.8316", "0.8449", "6.416e-04", "1.340e-03"],
    ["2", "0.8658", "0.9600", "1.155e-04", "1.544e-04"],
    ["3", "0.8748", "0.9693", "2.333e-05", "3.996e-05"],
    ["4", "0.8842", "0.9687", "4.150e-06", "1.841e-05"],
    ["5", "0.8903", "0.9704", "7.243e-06", "1.259e-05"],
], [2.0, 2.3, 2.3, 3.0, 3.0])
add_body(doc, "修正次数增加到 8 次后，固定阈值对照实验的 F1 为 0.9716，10 次没有继续提高，因此把 8 次作为当前候选设置。正式评测仍只在验证集上选择存在概率阈值和几何残差阈值，然后在测试集上做一次最终计算。")

h = doc.add_heading("四 固定测试集结果", level=1)
filtered = summary["test_filtered"]
raw = summary["test_raw"]
add_body(doc, f"三组随机种子使用相同训练集、验证集和测试集，只改变网络初始化。采用 8 次局部几何修正和验证集阈值选择后，测试 F1 的均值为 {filtered['f1']['mean']:.4f}，样本标准差为 {filtered['f1']['std']:.4f}，最小值为 {filtered['f1']['min']:.4f}，最大值为 {filtered['f1']['max']:.4f}。三次测试的查准率均为 1.0000，查全率均值为 {filtered['recall']['mean']:.4f}。")
add_table(doc, ["模型和随机种子", "Precision", "Recall", "F1", "参数误差", "几何残差"], [
    ["候选网络 + 8次修正，主种子", "1.0000", "0.9522", "0.9755", "1.922e-11", "4.493e-12"],
    ["候选网络 + 8次修正，seed 7", "1.0000", "0.9544", "0.9767", "3.028e-13", "3.868e-12"],
    ["候选网络 + 8次修正，seed 19", "1.0000", "0.9589", "0.9790", "4.230e-17", "1.197e-16"],
    ["三种子均值 ± 标准差", f"{filtered['precision']['mean']:.4f} ± {filtered['precision']['std']:.4f}", f"{filtered['recall']['mean']:.4f} ± {filtered['recall']['std']:.4f}", f"{filtered['f1']['mean']:.4f} ± {filtered['f1']['std']:.4f}", f"{filtered['mean_parameter_error']['mean']:.3e} ± {filtered['mean_parameter_error']['std']:.3e}", f"{filtered['mean_geometry_residual']['mean']:.3e} ± {filtered['mean_geometry_residual']['std']:.3e}"],
], [3.0, 1.5, 1.5, 1.3, 2.0, 2.0])
add_body(doc, f"作为对照，8 次修正之前的候选网络在同一批测试曲线上，按正式评测为过滤前 F1 均值 {raw['f1']['mean']:.4f}，平均几何残差约 {raw['mean_geometry_residual']['mean']:.4f}。加入修正和残差过滤后，平均 F1 提高到 {filtered['f1']['mean']:.4f}，说明当前主要增益来自候选参数的几何校正和错误候选过滤，而不是单纯依靠分类概率。")
add_body(doc, "测试推理时间随电脑和进程状态变化。主种子在 600 对测试曲线上，8 次修正后的总时间约 15.36 秒，平均约 0.0256 秒一对曲线。该时间包括候选提取、MLP 前向计算、几何修正和结果生成，不代表经过 GPU 或工程优化后的速度。")

h = doc.add_heading("五 边界样本结果", level=1)
add_body(doc, "边界数据目前包含 40 个近距离无交点样本、40 个端点交点样本和 40 个人工构造的相切样本。使用主种子、8 次修正和主测试集选出的阈值进行探索性评测，近距离样本 F1 为 0，端点样本 F1 为 0.3621，相切样本 F1 为 0.3140。端点和相切样本的几何残差很小，但模型常常多报交点；近距离无交点样本经过过滤后仍有误报。")
add_body(doc, "这些结果说明模型对训练分布外的退化或临界几何情况还不可靠。当前相切样本的构造方式是把一条曲线平移到指定参数处通过另一条曲线，并没有用独立的切向量约束严格验证相切性质；因此这里的数字只能作为问题定位，不应当写成相切求交性能的最终结论。")

h = doc.add_heading("六 当前结论", level=1)
add_body(doc, "在固定的三次、八控制点、单位权重和普通孤立交点数据上，局部候选提议加几何修正已经明显优于初始的直接槽位回归。三组随机种子的测试 F1 变化范围为 0.9755 到 0.9790，说明这次提升没有依赖单个初始化。参数误差和几何残差也降到了 10^-11 到 10^-12 量级，但这部分数值主要反映了 Newton 型局部修正对已进入正确吸引域的候选的精确求解能力。")
add_body(doc, "当前方法仍有两个限制。第一，几何修正属于传统数值步骤，所以现阶段不能把方法描述成完全由神经网络替代求交算法。第二，主测试集和标签生成器使用相同的曲线表示范围与相近的普通横截交点假设，尚不足以证明对任意阶数、任意控制点数量、不同权重和相切交点的泛化能力。")

h = doc.add_heading("七 复现方法和下一步", level=1)
add_body(doc, "重新运行时先进入项目根目录。数据文件 data/experiment_large.npz 已保存，若要从头生成可运行 python code/data/generate_dataset.py --output data/experiment_large.npz --train 1600 --validation 400 --test 600 --seed 20261025。训练候选模型可运行 python code/experiments/candidate_model.py --data data/experiment_large.npz --output results/autoresearch/reproduce_candidate --epochs 20 --batch-size 256 --learning-rate 0.001 --seed 20261025 --grid-size 48 --max-candidates 8 --local-slack 0.015。正式评测可运行 python code/experiments/evaluate_refined.py --weights results/autoresearch/reproduce_candidate/weights.npz --output results/autoresearch/reproduce_refined --refine-steps 8。")
add_body(doc, "下一步应先修正边界数据的标签生成，特别是用切向量条件和独立高精度求交程序确认相切、近距离和端点样本。随后可以把边界样本加入训练集的一部分，比较加入前后的主测试集和边界集表现；还应增加不同权重和不同控制点数量的数据，并单独报告未见曲线表示下的结果。若后续论文强调 AI 取代传统求交步骤，还需要继续研究可学习的局部修正器，并与当前 Newton 修正版本做消融对比。")

h = doc.add_heading("八 文件和实验留痕", level=1)
add_body(doc, "数据生成代码位于 code/data，几何计算代码位于 code/geometry，模型和训练代码位于 code/models 与 code/experiments。候选模型的每轮实验结果保存在 results/autoresearch，包含候选数、网格大小、局部松弛量、随机种子、阈值、训练损失和测试指标。实验历史文件 results/autoresearch/experiment_history.json 记录了从直接槽位模型到局部候选模型的主要迭代；refine_steps_ablation.json、refine_steps_extended.json 和 boundary_refined_metrics.json 记录了几何修正步数与边界测试。")

REPORT.parent.mkdir(parents=True, exist_ok=True)
doc.save(REPORT)
print(REPORT)

