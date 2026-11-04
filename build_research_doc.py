from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

def font(run, name="宋体", size=11.5, bold=False):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold

def para(doc, text):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent=Inches(.34); p.paragraph_format.line_spacing_rule=WD_LINE_SPACING.ONE_POINT_FIVE; p.paragraph_format.space_after=Pt(7)
    font(p.add_run(text)); return p

def heading(doc, text):
    p=doc.add_paragraph(); p.paragraph_format.keep_with_next=True; p.paragraph_format.space_before=Pt(8); p.paragraph_format.space_after=Pt(7); p.paragraph_format.line_spacing=1.15
    font(p.add_run(text),"黑体",14,True); return p

d=Document(); s=d.sections[0]; s.page_width=Inches(8.5); s.page_height=Inches(11); s.top_margin=Inches(.8); s.bottom_margin=Inches(.75); s.left_margin=s.right_margin=Inches(.95)
n=d.styles["Normal"]; n.font.name="宋体"; n._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"),"宋体"); n.font.size=Pt(11.5)
t=d.add_paragraph(); t.alignment=WD_ALIGN_PARAGRAPH.CENTER; t.paragraph_format.space_before=Pt(10); t.paragraph_format.space_after=Pt(18); t.paragraph_format.keep_with_next=True; font(t.add_run("二维 NURBS 曲线交点智能求解研究思路"),"黑体",17,True)

para(d,"本研究拟借鉴 NURBS-Diff 的可微几何计算方法，探索由神经网络直接识别和定位两条平面 NURBS 曲线的有限个离散交点。网络在推理时直接输出交点参数和置信度，不把传统 Newton 迭代作为求交主体。考虑到网络可能漏掉交点或把近距离但不相交的曲线误报为相交，研究将从有明确范围的孤立交点问题开始，并用独立测试、困难样本和不确定性评估界定方法的适用范围。")
heading(d,"研究目标")
para(d,"研究目标是构建一套以机器学习模型直接求解二维 NURBS 曲线交点的方法。给定曲线 C₁(u) 和 C₂(v)，交点满足 C₁(u) = C₂(v)，因此可将求交写成参数域 (u，v) 上的二维非线性方程组求根问题。模型需要同时处理无交点、单交点和多个离散交点，并输出每个交点在两条曲线上的参数值，而不是只给出一个平面坐标或交点分类标签。")
para(d,"评价重点放在预测交点集合是否完整、位置是否准确，以及模型在训练时未见曲线上的表现。实验将特别考察多交点、近距离交点、相切附近和端点交点，并区分查全率、误报率、参数误差、几何残差和运行时间。首阶段不处理两条曲线存在连续重合段的情况，因为这种情况下公共点不是有限集合，和离散交点预测属于不同问题。")
para(d,"研究的理论依据来自三个方面。第一，NURBS 曲线在固定节点区间上是有理函数，曲线交点可表述为参数域中的方程组零点；第二，NURBS-Diff 说明几何计算可以通过可微模块进入神经网络训练；第三，已有神经网络多根求解和无序集合预测研究提供了输出多个解的建模参考。上述依据支持开展实验，但不保证网络必然找全根，也不构成对方法精度的预先结论。")
heading(d,"研究内容")
para(d,"首先实现二维 NURBS 曲线的可微评价，并以参数域残差场表示两条曲线间的关系。对采样参数 uᵢ、vⱼ，计算向量 F(uᵢ，vⱼ) = C₁(uᵢ) − C₂(vⱼ)，形成横坐标差、纵坐标差和欧氏距离等通道。相比只输入距离热图，带符号的两个坐标差能够保留方程两个分量的方向信息；交点对应两分量同时为零，因而网络可学习零点附近的结构，而不只是寻找“距离较近”的位置。")
para(d,"其次研究兼顾交点数量变化与交点顺序不固定的网络输出方式。网络以 MLP 或轻量卷积模块编码两条曲线及残差场，输出参数平面上的交点概率图，并通过有限数量的预测槽位回归交点参数、存在概率和不确定度。概率图帮助覆盖多个交点所在区域，预测槽位则给出连续参数值；训练时使用集合匹配，将预测结果与真实交点配对，不要求人为规定交点顺序。")
para(d,"为减少模型把两条输入曲线的顺序当作有意义特征，训练时随机交换曲线输入，并同步交换预测参数的标签。研究将比较共享 MLP、轻量卷积网络及其组合，优先采用在固定规模任务上能够稳定训练、参数量较小的结构，而不预设复杂网络一定更好。")
para(d,"训练数据将由程序生成的 NURBS 曲线对组成，覆盖无交点、单交点和多交点，并加入距离很近但实际不相交的困难负样本。训练标签由传统高精度程序生成，同时对一部分样本使用可控制地构造公共点的方法独立核验，降低单一教师程序漏解带来的标签偏差。训练、验证和测试按曲线实例、控制点配置和参数范围分组，避免同一曲线的轻微扰动同时进入训练集和测试集。")
para(d,"训练目标结合参数域交点区域分类、集合匹配后的参数回归、可微 NURBS 几何残差和重复预测惩罚。对网络预测的参数对 (û，v̂)，计算 C₁(û) 与 C₂(v̂) 的距离作为几何损失；对无交点曲线对，训练模型输出空交点集合。模型设定最大输出交点数，因此首阶段会明确限定交点数量上限，超过上限的样本用于后续扩展，不把模型容量造成的漏检误认为已解决全部交点问题。")
para(d,"研究还将通过消融实验分别去掉有符号残差通道、可微几何损失、参数域概率图或集合预测分支，观察各设计对查全率、近邻交点区分和未见曲线泛化的影响。传统细分、裁剪和 Newton 类求交方法作为对比基线，用来测量神经网络在精度、完整性和速度上的差异，但不作为所提网络的预测流程。")
heading(d,"研究方法")
para(d,"数学上令 F(u，v) = C₁(u) − C₂(v)，把网络的任务定义为预测参数域中 F 的全部孤立零点。第一阶段固定曲线次数、节点向量形式和控制点数量，使研究先聚焦于多根识别及定位。NURBS-Diff 的前向计算和梯度传播思路用于实现曲线评价和几何损失。由于原论文主要讨论几何点对控制点、权重及节点向量的导数，本研究还需单独实现或验证曲线点对参数 u、v 的导数，不能假设原模块无需修改就能直接用于求交训练。")
para(d,"数据生成时对曲线平移、旋转、尺度、权重和控制点分布作受控随机化，并通过曲线组合构造不同交点数量的样本。除用传统求交程序产生标签外，还应保留可构造已知交点的验证样本，并以另一种数值程序复核边界案例。划分数据时按完整曲线实例或参数配置分组；测试集加入训练阶段未出现的曲线组合、权重范围和交点间距，以免随机拆分相似样本造成过于乐观的泛化估计。")
para(d,"网络输入由曲线编码特征和参数域残差场组成，输出由交点概率图、有限个参数槽位、存在概率和不确定度构成。训练时先用网格标签学习交点区域，再用二分图匹配把预测槽位与真实交点配对，避免输出顺序影响损失。几何残差通过可微 NURBS 计算反向传播。推理阶段由网络直接给出交点，不用 Newton 迭代补点；残差与置信度用于评价、校准以及标记无法确定的样本。")
para(d,"模型评价通过预测交点与真实交点的一对一匹配进行，分别报告查准率、查全率、F1 值、无交点误报率、参数误差、最大几何残差和每对曲线的推理时间。结果还按交点数量、最小交点间距、曲线次数和权重范围分组，观察模型何时开始漏检。置信度评价采用覆盖率与误差关系，说明模型在拒绝低置信度样本后实际达到的准确程度；这一实验结果不应被表述为数学上的完备性保证。")
para(d,"研究按难度逐步推进。第一阶段实现固定规模曲线上的无交点、单交点和普通多交点预测；第二阶段测试近距离交点、未见曲线及消融结果；第三阶段再扩展到相切、尖点、端点和节点配置变化。若网络在某类情形持续漏检，应明确报告适用边界，不以传统程序事后补齐的交点冒充神经网络直接输出。")

f=s.footer.paragraphs[0]; f.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=f.add_run(); font(r,"宋体",9)
b=OxmlElement("w:fldChar"); b.set(qn("w:fldCharType"),"begin"); i=OxmlElement("w:instrText"); i.set(qn("xml:space"),"preserve"); i.text=" PAGE "; sep=OxmlElement("w:fldChar"); sep.set(qn("w:fldCharType"),"separate"); tx=OxmlElement("w:t"); tx.text="1"; e=OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"),"end"); r._r.extend([b,i,sep,tx,e])
d.core_properties.title="二维 NURBS 曲线交点智能求解研究思路"; d.core_properties.subject="研究目标、研究内容和研究方法"; d.core_properties.author=""; d.save("研究思路.docx")
print("Created 研究思路.docx")
