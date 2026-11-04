"""Build the advisor-facing mechanism-exploration report and reproducibility guide."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from code.experiments.meeting_followup import geometric_residual, write_json
from code.geometry.nurbs import KNOTS

OUT = ROOT / "reports/meeting_followup_assets"
REPORT = ROOT / "reports/导师会议后机制探索实验报告.docx"
GUIDE = ROOT / "reports/导师会议后实验复现说明.md"
RESULTS = ROOT / "results/meeting_followup"

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 9})


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shade = OxmlElement("w:shd")
    shade.set(qn("w:fill"), fill)
    tc_pr.append(shade)


def set_run_font(run, name="宋体", size=10.5, bold=None, color=None):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)


def para(doc, text, style=None, after=5):
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.18
    run = p.add_run(text)
    set_run_font(run)
    return p


def heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(4)
    for run in p.runs:
        set_run_font(run, "黑体", 15 if level == 1 else 12, True, (38, 60, 68))
    return p


def formula(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    set_run_font(r, "Cambria Math", 11, False, (28, 52, 62))


def table(doc, headers, rows, widths=None, font_size=9):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"
    for j, text in enumerate(headers):
        cell = t.rows[0].cells[j]
        cell_shading(cell, "DCE9E6")
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        run = cell.paragraphs[0].add_run(str(text))
        set_run_font(run, "黑体", font_size, True)
    for row in rows:
        cells = t.add_row().cells
        for j, text in enumerate(row):
            cells[j].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            run = cells[j].paragraphs[0].add_run(str(text))
            set_run_font(run, size=font_size)
    if widths:
        for row in t.rows:
            for j, width in enumerate(widths):
                row.cells[j].width = Inches(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return t


def figure(doc, path, caption, width=6.25):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(7)
    run = cap.add_run(caption)
    set_run_font(run, "宋体", 9, False, (75, 85, 88))


def build_network_figure(path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    configs = [
        ("原始曲线输入", ["P1, P2\n32个控制点坐标", "q=(u0,v0)\n2维网格查询"], "34维"),
        ("局部网格输入", ["P1, P2\n32维", "q=(u0,v0)\n2维", "Rloc\n5x5x3=75维"], "109维"),
        ("全局网格输入", ["P1, P2\n32维", "q=(u0,v0)\n2维", "G12\n12x12x3=432维"], "466维"),
    ]
    for ax, (title, blocks, dim) in zip(axes, configs):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        ax.set_title(title, fontsize=12, color="#263c44", pad=8)
        y = 8.3
        for block in blocks:
            ax.text(1.5, y, block, ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.45", fc="#e9f0ee", ec="#7c9993", lw=1))
            y -= 2.0
        ax.annotate("", xy=(4.1, 5.0), xytext=(2.55, 6.4), arrowprops=dict(arrowstyle="->", color="#687a7d"))
        ax.text(5.9, 5.1, f"{dim}\n第一层全连接", ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", fc="#f4ead6", ec="#ac8b50", lw=1))
        ax.text(8.0, 5.1, "128 → 128 → 64\nReLU", ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", fc="#e8edf2", ec="#71869a", lw=1))
        ax.annotate("", xy=(5.0, 5.1), xytext=(4.0, 5.1), arrowprops=dict(arrowstyle="->", color="#687a7d"))
        ax.annotate("", xy=(7.0, 5.1), xytext=(6.8, 5.1), arrowprops=dict(arrowstyle="->", color="#687a7d"))
        ax.text(8.7, 2.1, "输出 (u_hat, v_hat, s)\n最终参数对 + 存在分数", ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", fc="#e7eee2", ec="#789269", lw=1))
        ax.annotate("", xy=(8.5, 2.9), xytext=(8.1, 4.2), arrowprops=dict(arrowstyle="->", color="#687a7d"))
    fig.suptitle("共享 MLP：每个网格查询独立预测最终参数对", fontsize=14, color="#263c44")
    fig.tight_layout(rect=[0, 0, 1, .92])
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_grid_figure(rows, path):
    names = ["均匀48", "端点聚集48", "节点区间对齐56"]
    keys = ["uniform", "cosine_endpoints", "knot_aligned"]
    radii = ["0.035", "0.02", "0.01", "0.005"]
    x = np.arange(len(radii)); width = .23
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    colors = ["#3b7a78", "#d38b3c", "#557a9f"]
    for j, (name, key, color) in enumerate(zip(names, keys, colors)):
        row = next(item for item in rows if item["grid"] == key)
        vals = [row["coverage_by_parameter_radius"][r] for r in radii]
        ax.bar(x+(j-1)*width, vals, width, label=name, color=color)
    ax.set_xticks(x, ["0.035", "0.02", "0.01", "0.005"])
    ax.set_ylim(0, 1.08); ax.set_ylabel("最近网格点距离不超过半径的根比例")
    ax.set_xlabel("参数平面距离半径")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=.2)
    ax.set_title("参数网格的几何覆盖（诊断统计）")
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def build_f1_figure(metrics, path):
    names = ["纯原始输入", "局部残差输入", "全局网格输入", "旧学习修正"]
    keys = ["direct_raw", "direct_raw_plus_grid_input", "direct_raw_plus_global_grid_input", "legacy_learned_correction"]
    categories = ["single", "multiple", "close_0p02", "close_0p005", "endpoint", "tangent", "near_disjoint"]
    labels = ["单交点", "三交点", "近根0.02", "近根0.005", "端点", "相切", "近距无交"]
    values = np.array([[metrics[k]["per_category"][c]["f1"] for c in categories] for k in keys])
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    im = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=max(.5, float(values.max())))
    ax.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(names)), names)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center",
                    color="white" if values[i,j] > .42 else "#263238", fontsize=8)
    fig.colorbar(im, ax=ax, label="F1（参数容差0.035）", shrink=.85)
    ax.set_title("未参与阈值选择的解析测试类别表现")
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def build_residual_figure(metrics, path):
    names = ["纯原始输入", "局部残差输入", "全局网格输入", "旧学习修正"]
    keys = ["direct_raw", "direct_raw_plus_grid_input", "direct_raw_plus_global_grid_input", "legacy_learned_correction"]
    tolerances = ["0.001", "0.0001", "1e-05", "1e-06"]
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    x = np.arange(len(tolerances)); width = .19
    floor = 1e-5
    colors = ["#3b7a78", "#d38b3c", "#557a9f", "#8d6b8a"]
    for j, (name, key, color) in enumerate(zip(names, keys, colors)):
        vals = [metrics[key]["prediction_geometry_residual"][t] for t in tolerances]
        vals = [0.0 if v is None else v for v in vals]
        visible_vals = [max(v, floor) for v in vals]
        positions = x+(j-1.5)*width
        bars = ax.bar(positions, [v-floor for v in visible_vals], width, bottom=floor, label=name, color=color)
        for bar, value, visible in zip(bars, vals, visible_vals):
            label = "0" if value == 0 else f"{100*value:.2f}%"
            ax.annotate(label, (bar.get_x()+bar.get_width()/2, visible),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(x, ["1e-3", "1e-4", "1e-5", "1e-6"])
    ax.set_yscale("log"); ax.set_ylim(floor, 1); ax.set_ylabel("全部返回预测中满足距离容差的比例（对数坐标）")
    ax.set_xlabel("几何残差阈值 εg")
    ax.legend(frameon=False, ncol=2); ax.grid(axis="y", alpha=.2)
    ax.set_title("同一探索测试集上的平面几何残差")
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def build_report():
    OUT.mkdir(parents=True, exist_ok=True)
    results = load_json(RESULTS/"experiment_results.json")
    holdout = load_json(RESULTS/"locked_holdout_results.json")
    grids = results["grid_audit"]
    quality = results["quality_audit"]
    test_metrics = {k:v["test"] for k,v in results["models"].items()}
    hold_metrics = holdout["models"]
    prior_cfg = load_json(RESULTS/"direct_raw_plus_grid_prior_corrected/config.json")
    prior_sel = load_json(RESULTS/"direct_raw_plus_grid_prior_corrected/validation_decode_selection.json")

    paths = {
        "network": OUT/"network_framework.png", "grid": OUT/"grid_coverage.png",
        "categories": OUT/"category_f1.png", "residual": OUT/"geometry_tolerance.png"}
    build_network_figure(paths["network"])
    build_grid_figure(grids, paths["grid"])
    build_f1_figure(hold_metrics, paths["categories"])
    build_residual_figure(test_metrics, paths["residual"])

    # New exact-root holdout label quality and near-disjoint distances.
    holdout_data = np.load(ROOT/"data/meeting_analytic_locked_holdout.npz")
    roots_residual, near_gaps = [], []
    for a, b, label, count, category in zip(holdout_data["curve_a"], holdout_data["curve_b"],
            holdout_data["labels"], holdout_data["counts"], load_json(ROOT/"data/meeting_analytic_locked_holdout.json")["metadata"]):
        if int(count):
            residual, _, _ = geometric_residual(np.broadcast_to(a, (int(count),8,2)),
                np.broadcast_to(b, (int(count),8,2)), label[:int(count)])
            roots_residual.extend(np.linalg.norm(residual, axis=1).tolist())
        if category["category"] == "near_disjoint":
            residual, _, _ = geometric_residual(a[None], b[None], np.array([[.5,.5]]))
            near_gaps.append(float(np.linalg.norm(residual)))
    tolerance_rows = [[f"{eps:.0e}", f"{np.mean(np.asarray(roots_residual)<=eps):.4f}",
                       str(int(np.sum(np.asarray(near_gaps)<=eps)))+" / 50"] for eps in (1e-3,1e-4,1e-5,1e-6)]

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(.65); sec.bottom_margin = Inches(.65)
    sec.left_margin = Inches(.72); sec.right_margin = Inches(.72)
    style = doc.styles["Normal"]
    style.font.name = "宋体"; style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体"); style.font.size = Pt(10.5)
    doc.core_properties.title = "导师会议后机制探索实验报告"
    doc.core_properties.subject = "二维NURBS曲线交点的直接MLP输出机制实验"
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(4)
    set_run_font(title.add_run("二维 NURBS 曲线求交机制探索实验报告"), "黑体", 20, True, (35, 57, 64))
    subtitle = doc.add_paragraph(); subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(subtitle.add_run("根据 2026 年 11 月 4 日导师讨论意见整理"), "宋体", 10, False, (83, 96, 99))

    heading(doc, "一、阶段结论", 1)
    para(doc, "本轮把网络输出改成最终参数对 (û, v̂)，并比较了三种输入：原始控制点与网格查询、再加局部残差块、再加低分辨率全局残差网格。还将原学习修正模型放到同一组解析难例上作对照。新直接输出模型在未参与调参的 350 对解析曲线上的 F1 为 0.0406、0.0909 和 0.1146；旧学习修正模型为 0.2619。新结构尚未学会稳定定位，不能写成已改进成功。")
    para(doc, "本轮的主要收获是把老师提出的结构问题变成了可测的对照，并找到了训练查询分布失配的证据。训练查询中正标签占 32.45%；在验证曲线的48×48网格上，距已知根不超过正例半径0.055的查询仅约1.22%，前者约为后者的26.6倍。未校正的局部输入模型在测试中平均几乎返回8个点，无交点误报率98.3%。随后只降低正例损失权重的验证对照没有提高F1，因此保留为失败实验，不拿它替代原配置。")
    para(doc, "目前只支持固定三次、8控制点、固定节点向量、单位权重的平面 B 样条特例；本阶段最多3个真实交点，推理最多返回8个预测。结果只说明该实验范围内的机制表现，不代表任意 NURBS 曲线的可靠求交。")

    heading(doc, "二、研究问题的符号表示", 1)
    para(doc, "两条曲线分别写作 C₁(u) 与 C₂(v)，参数 u,v∈[0,1]。对参数网格点 (uᵢ,vⱼ)，定义带方向的残差和欧氏距离：")
    formula(doc, "Fᵢⱼ = C₁(uᵢ) − C₂(vⱼ) = (Δxᵢⱼ, Δyᵢⱼ)，    dᵢⱼ = ‖Fᵢⱼ‖₂")
    para(doc, "交点要求两个坐标差都为零；计算机中的标签检查改为 ‖F‖₂≤εg。εg 分别取 10⁻³、10⁻⁴、10⁻⁵、10⁻⁶。这个阈值用于判断一个参考交点标签的几何残差是否足够小，也用于统计预测残差；推理程序不根据 εg 删除预测。端点是另一种标签：")
    formula(doc, "Iendpoint(u,v) = 1[ min(u, 1−u, v, 1−v) ≤ εp ]，本轮 εp=10⁻⁶")
    para(doc, "端点交点仍然是交点，端点标记与几何残差阈值不能混为一谈。近距离但没有真实根的曲线，也可能满足 ‖F‖₂≤εg，所以残差容差只能做数值质量分级，不能单独证明存在真实交点。")
    para(doc, "以网格锚点 (u₀,v₀) 为中心，局部残差块写成：")
    formula(doc, "Rᵢⱼ = [Δxᵢⱼ, Δyᵢⱼ, dᵢⱼ]，    Rloc(u₀,v₀) = {Rᵢ₊a,ⱼ₊b : a,b∈{−2,−1,0,1,2}}")
    para(doc, "因此局部块有 5×5×3=75 个数。边界外位置裁到 [0,1] 后取值；块展平后直接拼到MLP输入。全局对照把12×12个参数网格位置上的三个通道展平，得到432维。")

    heading(doc, "三、网络框架与训练", 1)
    figure(doc, paths["network"], "图1  三种输入形式进入同一共享 MLP；每个网格查询输出最终参数和存在分数。")
    table(doc, ["版本", "MLP 输入", "输入维数", "MLP 输出"], [
        ["原始曲线", "P₁(16坐标), P₂(16坐标), (u₀,v₀)", "34", "(û,v̂,s)"],
        ["局部残差", "原始曲线和查询 + Rloc", "109", "(û,v̂,s)"],
        ["全局网格", "原始曲线和查询 + 12×12残差图", "466", "(û,v̂,s)"]], font_size=8.5)
    para(doc, "每个网络有3层隐藏层，宽度为128、128、64，隐藏层用 ReLU，输出层两个参数经过 sigmoid 限制在 [0,1]，第三个输出经过 sigmoid 作为交点存在分数。同一个MLP在参数网格的每个查询点重复使用。原始控制点坐标未经额外标准化特征工程；不过曲线样本生成时已经做过统一尺度归一化。网格锚点仍作为查询输入，否则共享MLP不知道当前要检查参数域的哪个位置。")
    para(doc, "训练目标由存在分类、正候选参数误差和可微曲线残差组成：")
    formula(doc, "L = Lcls + 4 Lparam + Lgeo，    Lgeo = mean ‖C₁(û) − C₂(v̂)‖₂²")
    para(doc, "C₁ 与 C₂ 的基函数计算对输出参数可微，几何项的梯度经过 sigmoid 输出传回MLP参数；没有在推理阶段运行 Newton。每种结构用同一主种子 20261026、Adam 学习率 5×10⁻⁴、最多35轮，按验证损失保存检查点。候选存在阈值和去重半径只根据验证集校准；推理最多保留8个高分输出，并按参数平面半径去重。没有按几何残差阈值筛除。")

    heading(doc, "四、数据构造和质量核验", 1)
    para(doc, "训练/验证/探索测试分别由原合成曲线数据 1600/400/600 对与已知根解析曲线 840/210/210 对组成。新增解析数据每类均衡生成：单交点、三交点、参数间距0.02和0.005的近根、真实端点、二重根相切、近距离无交。解析根直接由多项式根给出，再转换为固定三次样条控制点；这部分是程序生成的合成数据，不是另一个AI生成模型的输出。曲线规格仍为三次、8控制点、节点向量 [0,0,0,0,0.2,0.4,0.6,0.8,1,1,1,1]、权重全1。")
    table(doc, ["几何容差 εg", "新解析根标签通过率", "近距离无交中距离≤εg"], tolerance_rows, font_size=9)
    para(doc, "新锁定解析集有350对、450个已知根。真根的几何残差接近浮点舍入误差，因而四个阈值都能通过；这不表示模型能达到这些求交精度。近无交样本的最小距离跨越约10⁻⁷到10⁻²，其中一部分在较宽εg下会被判成‘距离很近’，但它们的真值仍是无交。这正说明数据分类要同时保留解析根标签和最近距离，不能把距离阈值当作无误差真值。端点标记以参数距边界≤10⁻⁶独立生成。")
    para(doc, "原随机数据的参考标签由折线交叉搜索并离线细化，训练集标签有极少数残差落在10⁻⁵以上；因此数据质检保留四个阈值的通过率，而不是默认全部标签完美。新解析子集减少了对同一数值标注器的依赖。")

    heading(doc, "五、网格方案与覆盖", 1)
    para(doc, "比较了48点均匀划分、48点余弦端点聚集划分，以及沿5个非零节点区间每段细分11次的节点对齐划分（56个节点）。对已知根，测量最近网格查询与根在(u,v)参数平面的欧氏距离。")
    figure(doc, paths["grid"], "图2  网格覆盖统计；统计来自本轮探索数据，只作机制诊断。")
    table(doc, ["网格", "每轴点数", "每对查询数", "半径0.01覆盖", "半径0.005覆盖"], [
        ["均匀", 48, 2304, "69.1%", "23.1%"],
        ["余弦端点聚集", 48, 2304, "50.1%", "20.9%"],
        ["节点区间对齐", 56, 3136, "86.9%", "29.7%"]], font_size=8.5)
    para(doc, "三种网格在参数半径0.035下都覆盖了全部已知根；更严格半径下，节点对齐网格最好，但查询数比48×48增加约36%。余弦网格偏重端点，降低了对全参数域的覆盖。此处的覆盖只表示根附近有查询点，不代表MLP会把查询正确映射到根。近根间隔0.005仍远小于网格间距，单靠网格不能保证分开。")

    heading(doc, "六、测试结果", 1)
    para(doc, "下表的探索测试集由旧600对测试和新增解析测试合并，共810对。它在本轮被用于网格覆盖与失败分析，因此只作为探索结果；最终泛化结论仍需另生成从未查看的测试集。解析锁定测试集350对则在网络与阈值选定后生成，但随后已用于分析类别失配，同样不再视为最终未触碰测试。")
    rows = []
    labels = {"direct_raw":"原始曲线输入", "direct_raw_plus_grid_input":"局部残差输入",
              "direct_raw_plus_global_grid_input":"全局网格输入", "legacy_learned_correction":"旧学习修正"}
    for key, label in labels.items():
        m = test_metrics[key]
        rows.append([label, f"{m['f1']:.4f}", f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                     f"{m['f1_by_parameter_tolerance']['0.01']:.4f}", f"{m['prediction_geometry_residual_median']:.4f}"])
    table(doc, ["模型", "F1@0.035", "查准率", "查全率", "F1@0.01", "预测残差中位数"], rows, font_size=8.3)
    figure(doc, paths["categories"], "图3  新锁定解析测试集的分类别F1；参数匹配容差0.035。")
    figure(doc, paths["residual"], "图4  探索测试集全部预测满足不同平面距离阈值的比例。")
    para(doc, "新锁定解析集上，三个直接输出模型F1分别为0.0406、0.0909、0.1146；旧学习修正模型为0.2619。严格到参数容差0.01时，最高的新模型F1约0.0187。直接模型预测点的几何残差通过10⁻³的比例不超过约0.04%，通过10⁻⁴及更严阈值的比例为0；旧模型也只有3.1%预测通过10⁻³。参数匹配误差较小和几何两曲线点真正重合是不同指标，不能互相替代。")
    para(doc, "旧学习修正模型此前在原随机验证集记录的F1为0.8894；原验证集没有本轮新增的解析端点、相切和近距离无交平衡子集，因此该历史成绩不能与本报告的解析测试F1直接横向比较。这里的旧模型结果只表示它在新构造难例上的退化程度。")
    para(doc, "解析类别上，局部残差输入在端点类F1为0.1186、三交点类0.1055、近无交误报率100%；全局网格输入在单交点和相切上分别约0.144、0.170，但三交点召回不足1%，近无交误报率34%。旧模型单交点F1较高（0.947），但端点为0，相切约0.063。这显示不同结构的错误类型不同，整体F1不能说明已找到全部交点区域。")

    heading(doc, "七、单因素后续诊断", 1)
    prior_val_f1 = prior_sel["validation_full"]["f1"]
    para(doc, f"原查询生成方式使训练查询正例率为32.45%，存在明显的正例过采样。为了检查分数偏置，在局部残差结构和数据不变时，只把正例分类权重由5改为0.05。验证阈值范围随分数尺度扩到0.01–0.95后，最佳完整验证F1为{prior_val_f1:.4f}，低于原局部残差输入验证F1 0.0463；因此该权重改动不保留为推荐方案，也没有用已经查看过的测试集重新评估。")
    para(doc, "这次试验说明分类分数的校准确实受类别比例影响，但仅调权重没有解决参数定位。下一轮应先把训练查询采样改成与推理网格一致的分布，或者使用明确的先验校正/难负例策略，再分别观察存在分类和参数回归误差；固定种子即可，不需要多种子。")

    heading(doc, "八、结论和下一步", 1)
    para(doc, "本轮完成了老师提出的符号化残差表达、网络输入输出框图、网格方案比较、端点/相切/近根分类、几何容差1e-3至1e-6的标签质检，以及原始输入与局部/全局网格特征的直接输出MLP对照。MLP确实直接给出最终参数对，不依赖Newton修正；但参数网格查询、分数阈值和参数去重仍是模型外部流程，所以不是完全端到端的无规则算法。")
    para(doc, "现在最合理的研究结论是：‘直接把控制点输入MLP并输出最终交点参数’在当前数据和训练方式下没有达到可用求交效果；加入全局残差网格仅有有限改善。主要证据是输出过多、空集误报、严格参数容差F1低、几何残差大，以及相切/多交点类别不稳定。")
    para(doc, "下一步固定随机种子，先重做训练查询分布，让正负候选比例接近推理网格，同时把端点、近根、相切和近距离无交的生成比例保存在数据配置中；然后比较分类头校准与参数回归，记录候选区域召回率、每对输出数量、严格容差F1和几何残差。所有选型只看训练/验证集；新测试集应在模型锁定后另行生成，不再使用本轮已查看的解析测试数据。高精度阶段才另开MLP近似解加传统数值精化的独立路线，本轮没有实现或评估该组合。")

    doc.save(REPORT)
    guide = f'''# 导师会议后机制实验复现说明

本说明对应《导师会议后机制探索实验报告》。所有结果只用固定随机种子 `20261026`，没有多种子平均。训练只使用训练集，网格/阈值/去重方案按验证集选择；测试结果已用于本轮机制诊断，不能再称为最终未查看测试成绩。

## 环境

- Windows PowerShell
- Python：`E:\\study\\python\\python.exe`，Python 3.13.5
- NumPy 2.3.2、Matplotlib 3.10.6、python-docx 1.2.0
- CPU；固定 BLAS 线程数为1；不需要 PyTorch 或 GPU

进入项目目录 `E:\\study\\ThreeUp\\论文`，然后执行：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
E:\\study\\python\\python.exe -u code/experiments/meeting_followup.py
```

这一步生成三类解析已知根数据，训练三个直接输出模型，使用验证集72对固定分层子集校准分数阈值/参数去重半径/网格形式，再在完整验证集复核。然后运行锁定解析集评估：

```powershell
E:\\study\\python\\python.exe -u code/experiments/evaluate_meeting_holdout.py
```

它使用固定种子 `20261107`、每类50对、共350对的解析测试曲线。模型权重哈希、配置哈希、验证集选出的阈值和网格先写入 `results/meeting_followup/locked_holdout_lock.json`，随后才读取测试标签计算成绩。该测试后来被用于机制错误分析，故只作为本轮探索结果。

正例分类权重的单因素对照：

```powershell
E:\\study\\python\\python.exe -u code/experiments/run_prior_corrected_variant.py
```

该实验复用相同数据查询、网络结构、随机种子和局部残差输入，仅把正例分类权重从5改成0.05。它只用验证集重新校准，验证F1未改善；没有在已查看测试集上重选。

## 报告和关键文件

- 主实验代码：`code/experiments/meeting_followup.py`
- 锁定解析测试评估：`code/experiments/evaluate_meeting_holdout.py`
- 单因素分类权重实验：`code/experiments/run_prior_corrected_variant.py`
- 主数据：`data/experiment_large.npz`
- 解析训练/验证/探索测试：`data/meeting_analytic_train.npz`、`data/meeting_analytic_validation.npz`、`data/meeting_analytic_test.npz`
- 新锁定解析集：`data/meeting_analytic_locked_holdout.npz`
- 网络配置、权重、训练历史、验证选型和测试预测：`results/meeting_followup/`
- 运行日志：`logs/meeting_followup.log`、`logs/meeting_holdout.log`、`logs/meeting_prior_corrected.log`
- 图和报告素材：`reports/meeting_followup_assets/`

所有解析曲线用已知多项式根生成；原随机曲线参考标签由折线交叉搜索和离线细化生成。后者用于训练/验证的旧数据，并不等于解析证明。模型推理直接输出最终参数对，未调用Newton，也没有按几何残差阈值过滤；参数去重和每对最多8个输出仍是规则后处理。模型范围仍限三次、8控制点、固定开放节点、全1权重。

当前指标不理想，复现实验的目标是核对现阶段结果和失败原因，不应把这些数值写成已达到可靠求交。若重新开始新一轮研究，应重新生成未查看的测试集，并在分析前锁定配置和阈值。
'''
    GUIDE.write_text(guide, encoding="utf-8")
    print(REPORT)
    print(GUIDE)


if __name__ == "__main__":
    build_report()
