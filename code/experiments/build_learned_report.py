"""Create a measured research report and a reproduction guide."""

import json
import statistics
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "results/learned_study"
REPORTS = ROOT / "reports"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def average(runs, key):
    return statistics.mean(row[key] for row in runs)


def main():
    selected = Path(read(STUDY / "selection.json")["selected"])
    chosen = read(selected / "metrics.json")
    baseline = read(STUDY / "depth_0p0/metrics.json")
    config = read(selected / "requested_config.json")
    stages = read(STUDY / "stages.json")
    checks = read(STUDY / "checks.json")
    diagnosis = read(STUDY / "localization_diagnosis.json")
    final = read(STUDY / "final_evaluation.json")
    analytic = read(STUDY / "analytic_evaluation.json")["families"]
    seeds = read(STUDY / "seed_runs.json")
    repeated = read(STUDY / "reproducibility.json")
    assert read(STUDY / "single_factor_audit.json")["all_stages_one_factor"]
    REPORTS.mkdir(exist_ok=True)
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin = section.bottom_margin = Inches(0.75)
    section.left_margin = section.right_margin = Inches(0.8)
    normal = document.styles["Normal"]
    normal.font.name, normal.font.size = "SimSun", Pt(11)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.line_spacing = 1.25
    normal.paragraph_format.space_after = Pt(6)
    for name in ("Title", "Heading 1", "Heading 2"):
        style = document.styles[name]
        style.font.name = "SimHei"
        style.element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        style.font.color.rgb = RGBColor.from_string("203447")
    document.styles["Title"].font.size = Pt(21)
    document.styles["Heading 1"].font.size = Pt(14)
    footer = section.footer.paragraphs[0]
    footer.alignment = 2
    footer.add_run("学习参数修正实验 · 2026-10-26  |  ").font.size = Pt(9)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    markdown = []

    def paragraph(text):
        document.add_paragraph(text)
        markdown.append(text + "\n")

    def heading(text):
        document.add_heading(text, 1)
        markdown.append("## " + text + "\n")

    def table(headers, rows):
        target = document.add_table(rows=1, cols=len(headers))
        target.style = "Light Shading Accent 1"
        repeat = OxmlElement("w:tblHeader")
        target.rows[0]._tr.get_or_add_trPr().append(repeat)
        for cell, text in zip(target.rows[0].cells, headers):
            cell.text = str(text)
            for para in cell.paragraphs:
                para.paragraph_format.keep_with_next = True
        for values in rows:
            for cell, text in zip(target.add_row().cells, values):
                cell.text = str(text)
        for row in target.rows:
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.space_after = Pt(3)
                    para.paragraph_format.space_before = Pt(3)
                    for run in para.runs:
                        run.font.size = Pt(9.5)
        markdown.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"])
        markdown.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
        markdown.append("")

    def figure(name, caption):
        document.add_picture(str(STUDY / "figures" / name), width=Inches(6.7))
        document.add_paragraph(caption, "Caption")
        markdown.append(f"![{caption}](../results/learned_study/figures/{name})\n")

    document.add_heading("二维 NURBS 曲线求交\n学习参数修正阶段研究报告", 0)
    paragraph("实验日期：2026年10月26日。本文数字读取本轮保存的配置、权重和评测文件，不把历史评测修复计为模型进步。")
    heading("一、研究目标与本轮结论")
    paragraph(f"本轮以共享多层感知机（MLP）学习交点参数的修正量。每步由网络预测移动方向和距离，训练时通过曲线计算反向传播。推理不调用Newton，也不按最终几何残差阈值筛除输出。单种子验证F1从 {baseline['validation_metrics']['f1']:.4f} 提高到 {chosen['validation_metrics']['f1']:.4f}，增加 {(chosen['validation_metrics']['f1'] - baseline['validation_metrics']['f1']) * 100:.2f} 个百分点。两者均采用参数距离容差0.035和相同的一对一评测。")
    paragraph(f"统一最终状态验证损失从 {baseline['validation_reference_final_loss']:.6f} 变为 {chosen['validation_reference_final_loss']:.6f}。该损失固定使用分类正例权重7、参数权重4、几何权重1；各实验自身的损失权重不同，不能直接比较各自最低损失。选择规则首先看验证F1，相同才比较统一损失，并不要求每轮两个指标同时改善。")
    held_out = final["fresh_holdout"]["families"]
    paragraph(f"锁定后新生成的400对留出曲线上，三种子平均F1由 {held_out['baseline']['mean_f1']:.4f} 变为 {held_out['selected']['mean_f1']:.4f}，提高 {(held_out['selected']['mean_f1']-held_out['baseline']['mean_f1'])*100:.2f} 个百分点。但将参数容差收紧到0.005时，改进模型平均F1仅 {statistics.mean(run['f1_by_parameter_tolerance']['0.005'] for run in held_out['selected']['runs']):.4f}，还不能替代高精度数值求交。")
    paragraph("当前数据仅含三次、8控制点、固定开放节点和全1权重的平面曲线，属于NURBS的B样条特例。候选仍由参数网格提出，所以目前是学习型参数修正方法，尚不能称为完全端到端的全部交点预测，也没有验证任意次数、控制点数和权重。")
    heading("二、网络如何完成修正")
    paragraph("先在48×48参数网格计算两条曲线的横坐标差、纵坐标差和点间距离，按局部低谷规则取8个候选。每个输入包括当前两个参数和周围5×5邻域的三通道残差，共77维。共享MLP有128、128、64三个隐藏层，输出两个修正量和存在分数，约34,947个参数。修正量经双曲正切函数限幅，更新后把参数截到0至1。")
    paragraph(f"最终配置为 {config['steps']} 步修正，单步限幅 {config['step_scale']}，正例权重 {config['positive_weight']}，参数权重 {config['regression_weight']}，几何权重 {config['geometry_weight']}，深度监督 {config['deep_supervision']}，最多 {config['epochs']} 轮，学习率 {config['learning_rate']}，批量256。每次保存自身验证损失最低的一轮，主种子保存第 {chosen['best_epoch']} 轮。")
    paragraph("动态邻域在每次移动后重新计算局部残差块，使网络看到当前位置附近的几何，反向传播也包含该采样对参数的导数。曲线导数用于损失和特征的链式梯度，不用来构造Newton更新。几何残差让网络学到两条曲线上的预测点应当重合；参数标签约束它靠近指定根，避免仅凭两曲线靠近就算正确。")
    paragraph("深度监督指对中间状态也计算参数和几何损失。系数0只监督末步；0.5时各步权重从0.5线性增加到1，未归一化，因此会改变损失尺度。有限差分检查表明，中间损失的梯度已经接通，但实验没有显示0.5稳定提高F1，最终保留系数0。最终接受只依据验证集选择的网络分数阈值，再用0.025参数半径去重。加权分类分数尚未独立校准，不能直接当真实概率；去重可能合并极近真交点。")
    heading("三、数据和实现检查")
    paragraph("主数据为训练1600对、验证400对、历史测试600对，按0至3个交点均衡分层，生成种子20261025。标签由折线交叉搜索后离线数值细化获得；这一步含Newton，但只用于造标签，不参与网络推理，也不能保证发现相切或极近的所有根。候选对真值覆盖率为训练97.42%、验证97.83%，候选阶段本身可能漏点。")
    paragraph("历史600对测试在较早实验中被观察，虽然本轮没有用于调参，仍只作为描述性对照。锁定配置和分数阈值后，另生成新留出400对，种子98711351，使用121和241采样密度核对参考根并按固定规则拒收不一致样本。它与训练集无完全相同曲线对，但来自同类合成分布，不代表真实CAD或分布外泛化。")
    maximum = max(row["max_absolute_gradient_error"] for row in checks["gradient_checks"])
    paragraph(f"通过基函数、解析参数导数、端点单侧导数、前向缓存和候选向量化等价检查，以及8配置有限差分梯度检查，网络最大绝对梯度差 {maximum:.3g}。60组随机集合配对与穷举答案一致。关闭Newton、线性求解、矩阵求逆及训练几何损失函数后，静态与动态代表样本推理不变。这是已运行的隔离检查，不是任意路径的形式证明。")
    paragraph("旧配对评测可能少算容差内可匹配预测。修复后，同一历史基线权重、同一阈值的F1从0.82143变为0.86688，完全是评测修复。新结果统一采用maximum_cardinality_v2，先最大化容差内一对一命中数，再最小化平方参数误差。")
    heading("四、单因素实验")
    paragraph(f"已完成 {len(stages)} 个阶段，逐项审计确认每阶段相对父配置只改一个所标记参数，保留失败对照和权重。先比较深度监督0与0.5，再比较步数、步幅和正例权重；随后根据定位问题引入动态邻域，并在该分支延长训练。反复使用验证集可能过拟合验证集，因此最终另作多种子和新留出评估。")
    paths = [("无深度监督基线", STUDY / "depth_0p0"), ("深度监督0.5", STUDY / "depth_0p5"), ("正例权重5", STUDY / "positive_5p0"), ("动态邻域20轮", STUDY / "dynamic_True"), ("动态邻域80轮", STUDY / "refresh_duration_80"), ("动态80轮深度监督0.5", STUDY / "refresh_depth_0p5"), ("最终选定", selected)]
    rows = []
    for label, path in paths:
        item = read(path / "metrics.json")
        rows.append([label, f"{item['validation_metrics']['f1']:.4f}", f"{item['validation_reference_final_loss']:.4f}", item["best_epoch"]])
    table(["实验", "验证F1", "统一损失", "保存轮次"], rows)
    paragraph("完整结果在results/learned_study/single_factor_results.csv。上表展示路径中的代表结果；相隔多项修改的行不能当单因素因果对照。")
    figure("validation_experiments.png", "图1 代表配置的验证成绩和各自训练趋势。不同目标权重下的损失高度不能直接比较。")
    heading("五、学习修正的定位诊断")
    item = next(row for row in diagnosis if row["name"] == selected.name)
    paragraph(f"验证集正候选初始平均参数误差 {item['initial_positive_parameter_error']:.6f}，修正后 {item['corrected_positive_parameter_error']:.6f}。这里统计所有被赋正标签候选，不限于正确命中预测。保持网络分数不变、把坐标恢复初始候选时，F1为 {item['zero_update_f1_same_scores']:.4f}；学习修正后为 {chosen['validation_metrics']['f1']:.4f}；收紧到0.01容差时F1为 {item['strict_f1']:.4f}。零修正是事后干预诊断，分数仍依赖完整修正过程，不是独立训练的另一模型。")
    heading("六、多种子与锁定后的测试")
    rows = []
    for family, label in (("baseline", "基线"), ("selected", "改进")):
        for path in seeds[family]:
            item = read(Path(path) / "metrics.json")
            rows.append([label, item["seed"], f"{item['validation_metrics']['f1']:.4f}", f"{item['validation_reference_final_loss']:.4f}"])
    table(["方法", "种子", "验证F1", "统一损失"], rows)
    paragraph("固定训练种子20261026、7、19，各自只用验证集选择保存轮次和分数阈值。下表是相同测试数据上的三次训练均值与总体标准差，不衡量测试抽样不确定性，也不足以声称统计显著。锁定后没有按测试成绩再调参。")
    rows = []
    for values, label in ((final["families"], "历史600对"), (final["fresh_holdout"]["families"], "新留出400对")):
        for family, name in (("baseline", "基线"), ("selected", "改进")):
            item = values[family]
            rows.append([label, name, f"{item['mean_f1']:.4f} ± {item['std_f1_population']:.4f}", f"{average(item['runs'], 'precision'):.4f}", f"{average(item['runs'], 'recall'):.4f}"])
    table(["数据", "方法", "F1均值±标准差", "查准率", "查全率"], rows)
    fresh = final["fresh_holdout"]["families"]
    rows = []
    for family, label in (("baseline", "基线"), ("selected", "改进")):
        runs = fresh[family]["runs"]
        rows.append([label] + [f"{statistics.mean(row['f1_by_parameter_tolerance'][str(tolerance)] for row in runs):.4f}" for tolerance in (0.005, 0.01, 0.02, 0.035)])
    table(["新留出", "容差0.005", "容差0.01", "容差0.02", "容差0.035"], rows)
    paragraph("主容差0.035位于归一化参数空间，不能等同于高精度平面求交。为避免宽容差掩盖定位误差，同时报告严格容差和全部返回预测的平面残差，不只报告成功命中的误差。")
    rows = []
    for family, label in (("baseline", "基线"), ("selected", "改进")):
        runs = fresh[family]["runs"]
        rows.append([label, f"{average(runs, 'count_accuracy'):.4f}", f"{average(runs, 'empty_false_positive_rate'):.4f}", f"{average(runs, 'duplicate_prediction_rate'):.4f}", f"{1000 * average(runs, 'inference_seconds_per_pair'):.2f}"])
    table(["方法", "交点数准确率", "无交点误报率", "去重后重复率", "毫秒/对"], rows)
    paragraph("推理时间为CPU逐曲线对前向计时，包含候选构造、特征与网络更新、分数接受及去重，不包含离线标注和评测配对。只有单次顺序计时及三种子平均，未与传统算法在同等精度下比较，不能据此宣称更快。去重后的重复率定义为参数保留三位小数后的重复比例，不是原始槽位重复率。")
    for family, label in (("baseline", "基线"), ("selected", "改进")):
        runs = fresh[family]["runs"]
        paragraph(f"新留出集{label}的匹配参数平均误差 {average(runs, 'mean_parameter_error'):.6f}，全部预测平均平面残差 {average(runs, 'all_predictions_mean_geometry_residual'):.6f}，各次残差95%分位数均值 {average(runs, 'all_predictions_p95_geometry_residual'):.6f}。同分数零修正F1均值 {average(runs, 'zero_update_f1_same_threshold'):.4f}。")
    figure("fresh_holdout_summary.png", "图2 新留出集的容差敏感性和平面残差，误差线是三个种子的总体标准差。")
    heading("七、解析边界测试")
    paragraph("构造210对解析根已知的曲线，单交点、三交点、参数间隔0.02与0.005的近根、真正的端点、二重根相切、近距离不相交七类各30对。先规定可解析求根的三次多项式，再转换成同规格样条并共同旋转；控制点转换中的线性求解只用于离线造数据。真值独立于原数值标注器，且只在配置锁定后评估。")
    names = {"single": "单交点", "multiple": "三个交点", "close_0p02": "近根0.02", "close_0p005": "近根0.005", "endpoint": "端点", "tangent": "相切", "near_disjoint": "近距离不相交"}
    rows = []
    for category in sorted(analytic["selected"][0]["categories"]):
        key = "empty_false_positive_rate" if category == "near_disjoint" else "f1"
        values = [f"{statistics.mean(row['categories'][category][key] for row in analytic[family]):.4f}" for family in ("baseline", "selected")]
        rows.append([names.get(category, category), *values])
    table(["类别", "基线均值", "改进均值"], rows)
    paragraph("有交点类别填写F1；近距离不相交填写无交点曲线对误报率，越低越好。相切处梯度可能很弱，极近交点可能被0.025去重半径合并，主数据也没有针对这些情况训练，不能据普通测试声称已解决边界问题。旧边界脚本的部分名义相切和端点没有相应解析保证，本轮未沿用其标签。")
    selected_categories = {category: statistics.mean(run["categories"][category]["f1"] for run in analytic["selected"]) for category in ("endpoint", "tangent", "close_0p005", "multiple")}
    paragraph(f"本轮改进方法在端点类别F1为 {selected_categories['endpoint']:.4f}，相切为 {selected_categories['tangent']:.4f}，间隔0.005近根为 {selected_categories['close_0p005']:.4f}，解析三交点为 {selected_categories['multiple']:.4f}。这说明普通同分布留出成绩不能外推到这些曲线。当前只确认了测试现象；训练分布、候选覆盖、网络分数与去重各自造成多少失败，尚未分解验证。")
    heading("八、曲线实例与复现")
    figure("fresh_holdout_examples.png", "图3 每种交点数展示一个较好样本，再展示两个错误较多样本；固定选择索引已保存。")
    paragraph("绿圈是参考交点，红叉和紫色加号分别是预测参数在两曲线上的点。两个预测点不重合，说明仍有几何残差。图仅展示锁定模型，不参与选择。")
    paragraph(f"同机、同Python与NumPy、单BLAS线程、同随机种子独立重训后，权重数组最大绝对差 {repeated['weight_max_absolute_difference']:.3g}，验证参数输出最大差 {repeated['parameter_output_max_absolute_difference']:.3g}；验证F1原始 {repeated['f1_original']:.6f}、重复 {repeated['f1_repeat']:.6f}。依据解压数组而非压缩文件字节哈希核对。跨平台浮点和耗时可能不同。")
    paragraph("代码按geometry、data、models、experiments、evaluation分类。配置、权重、完整训练曲线、失败对照、验证输出、单因素审计、环境和源码快照均保留。操作见《学习修正实验_复现说明.md》，已有同名metrics结果会跳过，真正重训要用新输出目录。")
    heading("九、未完成内容与下一步")
    paragraph("当前为NumPy手写反向传播，借鉴NURBS-Diff把几何计算接入可微训练的思路，并非其PyTorch/CUDA模块复现。研究思路中的完整二维概率图、集合匹配训练和不确定度头未完成；评测一对一配对不等于训练实现了集合预测。非单位权重、变化节点、未见控制点数量及原文可靠算法的同精度对标也未完成，不能宣称优于原文。")
    paragraph("下一阶段需在新训练和验证数据中加入解析难例，另预留新的未查看测试集。先要求0.01与0.005严格容差下的学习修正优于同分数零修正诊断，再检查可学习候选与去重机制，不能用本轮边界测试选择新参数。旧高F1几何细化方案使用传统修正与残差筛选，保留为历史资料，不与本轮成绩混称。")
    paragraph("本轮使用项目自己的受控脚本。已安装codex-autoresearch技能缺少官方控制器，未运行其提交回滚流程，没有自动Git提交或伪造官方事件。代码与报告由AI辅助生成和检查；研究者应亲自复现、理解方法并按学校要求说明辅助方式，不能称为全部独立手工编写。")
    heading("参考文献")
    paragraph("Prasad A D, Balu A, Shah H, et al. NURBS-Diff: A Differentiable Programming Module for NURBS. Computer-Aided Design, 2022, 146:103199. DOI: 10.1016/j.cad.2022.103199。书目信息核对本地论文；原文不直接给出本实验的求交修正网络。")
    output = REPORTS / "阶段研究报告_学习修正_20261026.docx"
    document.save(output)
    output.with_suffix(".md").write_text("# 二维NURBS学习参数修正报告\n\n" + "\n".join(markdown), encoding="utf-8")
    flags = []
    for key, value in config.items():
        if key == "evaluate_test":
            continue
        if isinstance(value, bool):
            if value:
                flags.append("--" + key.replace("_", "-"))
        else:
            flags.extend(["--" + key.replace("_", "-"), str(value)])
    command = "& 'E:\\study\\python\\python.exe' -u code/experiments/learned_correction.py --output results/independent_selected_repeat " + " ".join("'" + item + "'" if " " in item else item for item in flags)
    guide = ["# 学习参数修正实验复现说明", "", "## 环境与检查", "", "在E:\\study\\ThreeUp\\论文打开PowerShell。实际训练解释器为E:\\study\\python\\python.exe，Python 3.13.5、NumPy 2.3.2，CPU计算，无需GPU或PyTorch。绘图另用matplotlib 3.10.6；报告用python-docx。保留现有数据，不重新造数据覆盖。", "", "```powershell", "$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'", "& 'E:\\study\\python\\python.exe' code/experiments/check_learned_correction.py", "```", "", "## 单模型真实重训", "", "下列命令从随机初始化训练，不读取旧权重，不评估测试；请使用新输出目录。", "", "```powershell", command, "```", "", f"期望主种子验证F1 {chosen['validation_metrics']['f1']:.10f}、统一验证损失 {chosen['validation_reference_final_loss']:.10f}、分数阈值{chosen['selected_probability_threshold']}。本轮已独立完整重训一次并逐数组核对，见results/learned_study/reproducibility.json。", "", "## 全部受控阶段", "", "```powershell", "& .\\code\\experiments\\reproduce_learned_study.ps1 -OutputDirectory 'results/learned_study_reproduction'", "```", "", "依次执行实现检查、单因素对照、三种子和独立重复。已有metrics.json的同名实验会跳过；想真重跑就换一个从未使用的输出目录。单模型最长80轮，全流程需要比单模型长得多。各组成步骤已在主研究执行；不要将这条说明命令误称为本轮又完整执行了一次新目录重跑。", "", "## 锁定后测试", "", "```powershell", "& .\\code\\experiments\\reproduce_learned_study.ps1 -OutputDirectory 'results/learned_study_reproduction' -FinalEvaluation", "```", "", "该开关先完成训练验证选型并锁定，才评估历史600对、新留出400对和解析210对。新留出数据已固定，请复用同一文件，不更换种子挑成绩。后续新研究需要另留未查看测试。", "", "## 文件与解释", "", "先读code/experiments/learned_correction.py理解前向、损失和反向传播，再读controlled_study.py理解父配置与选择规则。code/evaluation/set_metrics.py定义一对一匹配；code/data/analytic_cases.py定义解析真值。configs/learned_study/为最终配置，results/learned_study/single_factor_results.csv为全部对照，seed_runs.json指向六份种子实验，final_evaluation.json和analytic_evaluation.json是最终成绩。", "", "每候选77维，MLP预测参数修正和分数；动态邻域每步刷新，解析导数用于训练反向传播。推理没有Newton与最终残差筛选；离线数值标签中有Newton，要分开说明。0.035参数容差不代表高精度几何求交，单位权重样本不能代表任意NURBS。AI参与了代码和报告生成，汇报应区分亲自操作、理解的内容与辅助生成内容。"]
    (REPORTS / "学习修正实验_复现说明.md").write_text("\n".join(guide) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
