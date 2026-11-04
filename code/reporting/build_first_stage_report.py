"""Build the retrospective report from frozen results; never train or select a model.

Figures: E:/study/python/python.exe code/reporting/build_first_stage_report.py --figures
Word: bundled document runtime python .../build_first_stage_report.py --docx
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'reports/first_stage_assets'
OUT = ROOT / '正式提交/第一阶段研究报告.docx'
STUDY = ROOT / 'results/learned_study'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from code.geometry.vectorized_basis import basis_matrix
    from code.evaluation.set_metrics import match_metrics
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei'],
                         'axes.unicode_minus': False, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'baseline': '#90969b', 'selected': '#366d94'}
    final = read(STUDY / 'final_evaluation.json')
    fresh = final['fresh_holdout']['families']

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.55), layout='constrained')
    trials = [('原学习修正基线', 'depth_0p0'), ('中间步深度监督', 'depth_0p5'),
              ('正例权重改为5', 'positive_5p0'), ('刷新邻域 20轮', 'dynamic_True'),
              ('刷新邻域 80轮', 'refresh_duration_80'), ('最终步幅0.04', 'refresh_scale_0p04')]
    for ax, key, title, lim in [(axes[0], 'validation_metrics', '验证集 F1', (0.82, .91)),
                                (axes[1], 'validation_reference_final_loss', '统一口径验证损失', (.29, .39))]:
        vals = [read(STUDY / run / 'metrics.json') for _, run in trials]
        values = [v[key]['f1'] if isinstance(v[key], dict) else v[key] for v in vals]
        ax.barh(range(len(trials)), values, color=['#90969b'] * 5 + ['#366d94'])
        ax.set_yticks(range(len(trials)), [x[0] for x in trials] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_xlim(lim)
        ax.set_title(title, fontsize=11)
        for i, v in enumerate(values):
            ax.text(v + (lim[1] - lim[0]) * .015, i, f'{v:.4f}', va='center', fontsize=9)
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
    fig.savefig(ASSETS / 'validation_path.png', dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), layout='constrained')
    facts = {}
    for family, label in [('baseline', '基线'), ('selected', '改进')]:
        runs = fresh[family]['runs']
        values = np.array([[r['f1_by_parameter_tolerance'][str(t)] for t in [.005, .01, .02, .035]] for r in runs])
        axes[0].errorbar([.005, .01, .02, .035], values.mean(0), yerr=values.std(0),
                         label=label, marker='o', color=colors[family], capsize=3)
        residuals = [r['all_predictions_mean_geometry_residual'] for r in runs]
        axes[1].bar(label, np.mean(residuals), yerr=np.std(residuals), color=colors[family], capsize=5, width=.55)
        axes[1].text(label, np.mean(residuals) + .0035, f'{np.mean(residuals):.4f}', ha='center')
        facts[family] = {'f1_by_tolerance_mean': values.mean(0).tolist(),
                         'mean_geometry_residual': float(np.mean(residuals))}
    axes[0].set(xlabel='参数距离容差', ylabel='新留出集 F1', ylim=(0, 1))
    axes[0].set_xticks([.005, .01, .02, .035], ['0.005', '0.01', '0.02', '0.035'])
    axes[0].legend(frameon=False)
    axes[1].set(ylabel='全部返回预测的平均平面残差', ylim=(0, .05))
    for ax in axes:
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.savefig(ASSETS / 'tolerance_geometry.png', dpi=240)
    plt.close(fig)

    analytic = read(STUDY / 'analytic_evaluation.json')['families']
    categories = [('single', '单交点'), ('multiple', '三交点'), ('close_0p02', '近根 0.02'),
                  ('close_0p005', '近根 0.005'), ('endpoint', '端点'), ('tangent', '相切')]
    fig, ax = plt.subplots(figsize=(9.2, 3.5), layout='constrained')
    x = np.arange(len(categories))
    for shift, (family, label) in zip([-.18, .18], [('baseline', '基线'), ('selected', '改进')]):
        arr = np.array([[r['categories'][key]['f1'] for key, _ in categories] for r in analytic[family]])
        ax.bar(x + shift, arr.mean(0), width=.34, label=label, color=colors[family], yerr=arr.std(0), capsize=3)
        for ix, val in enumerate(arr.mean(0)):
            ax.text(ix + shift, val + .018, f'{val:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x, [name for _, name in categories])
    ax.set(ylabel='F1  参数容差 0.035', ylim=(0, 1.12))
    ax.legend(frameon=False, loc='upper right')
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    fig.savefig(ASSETS / 'analytic_boundary.png', dpi=240)
    plt.close(fig)

    archive = np.load(ROOT / 'data/learned_study_locked_test.npz')
    predictions = read(STUDY / 'refresh_scale_0p04/fresh_holdout_predictions.json')
    picks = read(STUDY / 'figures/example_selection.json')['indices']
    fig, axes = plt.subplots(3, 2, figsize=(9.2, 8.45), layout='constrained')
    for ax, index in zip(axes.ravel(), picks):
        a, b = archive['test_curve_a'][index], archive['test_curve_b'][index]
        t = np.linspace(0, 1, 280)
        ax.plot(*(basis_matrix(t) @ a).T, color='#366c96', label='曲线一', linewidth=1.4)
        ax.plot(*(basis_matrix(t) @ b).T, color='#bd7a33', label='曲线二', linewidth=1.4)
        count = int(archive['test_counts'][index])
        if count:
            ax.scatter(*(basis_matrix(archive['test_labels'][index, :count, 0]) @ a).T,
                       facecolors='none', edgecolors='#16874b', s=68, label='参考交点', zorder=5)
        if predictions[index]:
            uv = np.array(predictions[index])
            ax.scatter(*(basis_matrix(uv[:, 0]) @ a).T, marker='x', color='#c44242', s=38, label='预测点一', zorder=6)
            ax.scatter(*(basis_matrix(uv[:, 1]) @ b).T, marker='+', color='#854299', s=38, label='预测点二', zorder=6)
        met = match_metrics([predictions[index]], archive['test_labels'][index:index+1], archive['test_counts'][index:index+1])
        ax.set_title(f'样本 {index}   真交点 {count}   命中 {met["true_positive"]}   误报 {met["false_positive"]}   漏检 {met["false_negative"]}', fontsize=10)
        ax.set_aspect('equal', adjustable='datalim')
        ax.tick_params(labelsize=9)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=5, frameon=False, fontsize=9)
    fig.savefig(ASSETS / 'curve_examples.png', dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 2.5), layout='constrained')
    ax.set(xlim=(0, 10), ylim=(0, 3))
    ax.axis('off')
    boxes = [(0, 1.55, 2.0, '两条固定规格曲线\n48×48 残差场'),
             (2.65, 1.55, 1.85, '网格规则\n提出 8 个候选'),
             (5.15, 1.55, 2.05, '共享 MLP 修正\n4 步 每步刷新邻域'),
             (7.85, 1.55, 2.05, '网络分数接受\n参数去重后输出')]
    for x0, y0, width, text in boxes:
        ax.add_patch(FancyBboxPatch((x0+.05, y0), width-.1, 1.1, boxstyle='round,pad=.02',
                                   facecolor='#f1f3f4', edgecolor='#888888', linewidth=.9))
        ax.text(x0+width/2, y0+.55, text, ha='center', va='center', fontsize=10)
    for start, end in [(2.05, 2.6), (4.55, 5.1), (7.25, 7.8)]:
        ax.annotate('', xy=(end, 2.1), xytext=(start, 2.1), arrowprops={'arrowstyle': '->', 'color': '#555'})
    ax.annotate('仅训练时  参数标签与可微曲线几何损失反向更新网络', xy=(6.17, 1.5), xytext=(5, .35),
                ha='center', va='center', fontsize=11, arrowprops={'arrowstyle':'->', 'color':'#366d94'})
    fig.savefig(ASSETS / 'method_flow.png', dpi=240)
    plt.close(fig)
    source_paths = [ROOT/'正式提交/研究思路.docx', ROOT/'reports/阶段研究报告_学习修正_20261026.docx',
                    STUDY/'single_factor_results.csv', STUDY/'stages.json', STUDY/'final_evaluation.json',
                    STUDY/'analytic_evaluation.json', ROOT/'code/experiments/learned_correction.py',
                    ROOT/'code/experiments/candidate_model.py', ROOT/'results/large/evaluation/metrics.csv',
                    ROOT/'results/autoresearch/experiment_history.json', ROOT/'results/evaluation/metrics.csv',
                    ROOT/'results/full/evaluation/metrics.csv', ROOT/'code/data/generate_dataset.py',
                    ROOT/'code/data/locked_reference.py', ROOT/'code/data/analytic_cases.py',
                    ROOT/'code/geometry/nurbs.py']
    manifest = {'purpose': 'Retrospective report only; no training, no experimental file mutation',
                'results_cutoff': '2026-10-26', 'example_indices': picks, 'figure_data': facts,
                'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
    (ASSETS/'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    from first_stage_explained import explanatory_figure
    explanatory_figure(ASSETS/'residual_explained.png')
    print('Figures and source hashes saved:', ASSETS)


def build_docx():
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.top_margin, sec.bottom_margin = Inches(.65), Inches(.63)
    sec.left_margin, sec.right_margin = Inches(.76), Inches(.76)
    sec.header_distance, sec.footer_distance = Inches(.25), Inches(.27)
    for name in ['Normal', 'Title', 'Subtitle', 'Heading 1', 'Heading 2', 'Caption']:
        st = doc.styles[name]
        st.font.name = 'Times New Roman'
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.font.italic = False
        st.font.underline = False
        for border in st.element.xpath('./w:pPr/w:pBdr'):
            border.getparent().remove(border)
        st.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), '宋体' if name in ['Normal','Caption'] else '黑体')
    normal = doc.styles['Normal']
    normal.font.size = Pt(11)
    normal.paragraph_format.line_spacing = 1.23
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.widow_control = True
    for name, size in [('Title', 22), ('Subtitle', 12), ('Heading 1', 15), ('Heading 2', 12)]:
        doc.styles[name].font.size = Pt(size)
        doc.styles[name].paragraph_format.space_before = Pt(10 if name.startswith('Heading') else 0)
        doc.styles[name].paragraph_format.space_after = Pt(8)
    doc.styles['Caption'].font.size = Pt(9)
    doc.styles['Caption'].font.italic = False
    doc.styles['Caption'].paragraph_format.space_after = Pt(7)
    header = sec.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    rr = header.add_run('二维 NURBS 曲线求交研究  第一阶段')
    rr.font.size = Pt(8)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run('第 ')
    field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE'); footer._p.append(field)
    footer.add_run(' 页')
    for r in footer.runs: r.font.size = Pt(9)

    def p(text, bold=False):
        para = doc.add_paragraph()
        para.add_run(text).bold = bold
        return para

    def h(text): doc.add_paragraph(text, 'Heading 1')
    def sub(text): doc.add_paragraph(text, 'Heading 2')
    def page(text): doc.add_page_break(); h(text)
    def cap(text):
        para = doc.add_paragraph(text, 'Caption')
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    def picture(name, width=6.95):
        para = doc.add_paragraph()
        para.paragraph_format.space_after = Pt(2)
        para.add_run().add_picture(str(ASSETS/name), width=Inches(width))
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = True
    def table(headers, rows, widths, size=10):
        t = doc.add_table(rows=1, cols=len(headers))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for cell, w in zip(t.rows[0].cells, widths): cell.width = Inches(w)
        for cell, text in zip(t.rows[0].cells, headers): cell.text = text
        trpr = t.rows[0]._tr.get_or_add_trPr()
        repeat = OxmlElement('w:tblHeader'); trpr.append(repeat)
        for row in rows:
            cells = t.add_row().cells
            for cell, text, w in zip(cells, row, widths): cell.text = str(text); cell.width = Inches(w)
        for i, row in enumerate(t.rows):
            trpr = row._tr.get_or_add_trPr()
            cant = OxmlElement('w:cantSplit'); trpr.append(cant)
            for j, cell in enumerate(row.cells):
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tcpr = cell._tc.get_or_add_tcPr()
                borders = OxmlElement('w:tcBorders')
                for side in ['top','left','bottom','right']:
                    item = OxmlElement('w:'+side); item.set(qn('w:val'),'single'); item.set(qn('w:sz'),'4'); item.set(qn('w:color'),'D9D9D9'); borders.append(item)
                tcpr.append(borders)
                margins = OxmlElement('w:tcMar')
                for side in ['top','left','bottom','right']:
                    item = OxmlElement('w:'+side); item.set(qn('w:w'),'75' if side in ['top','bottom'] else '90'); item.set(qn('w:type'),'dxa'); margins.append(item)
                tcpr.append(margins)
                if i == 0:
                    shade = OxmlElement('w:shd'); shade.set(qn('w:fill'),'E8EDF1'); tcpr.append(shade)
                for para in cell.paragraphs:
                    para.paragraph_format.space_after = Pt(0)
                    para.paragraph_format.line_spacing = 1.13
                    para.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    for r in para.runs: r.font.size=Pt(size); r.bold=(i==0)
        doc.add_paragraph().paragraph_format.space_after = Pt(0)
        return t
    def equation(text):
        para = doc.add_paragraph()
        para.alignment=WD_ALIGN_PARAGRAPH.CENTER
        math = OxmlElement('m:oMath')
        run=OxmlElement('m:r'); tt=OxmlElement('m:t'); tt.text=text; run.append(tt); math.append(run); para._p.append(math)
        para.paragraph_format.space_after = Pt(7)

    # Page 1: purpose, headline results, evidence boundaries.
    doc.add_paragraph('第一阶段研究报告', 'Title')
    doc.add_paragraph('二维 NURBS 曲线求交的学习参数修正方法', 'Subtitle')
    p('整理日期  2026年10月29日    实验结果截止  2026年10月26日')
    from first_stage_explained import write_report
    write_report(p, h, sub, page, table, picture, cap)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.core_properties.title = '第一阶段研究报告'
    doc.core_properties.subject = '二维NURBS曲线求交学习参数修正 截至2026年10月26日'
    doc.core_properties.author = ''
    try:
        doc.save(OUT)
        print(OUT)
    except PermissionError:
        alternate = OUT.with_name(OUT.stem + '_修订版.docx')
        try:
            doc.save(alternate)
        except PermissionError:
            alternate = OUT.with_name(OUT.stem + '_讲解版.docx')
            doc.save(alternate)
        print('Original file is open; saved complete revision to:', alternate)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--figures', action='store_true')
    parser.add_argument('--docx', action='store_true')
    args=parser.parse_args()
    if args.figures: figures()
    if args.docx: build_docx()
