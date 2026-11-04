# 第一阶段研究报告生成记录

当前交付文件：`正式提交/第一阶段研究报告_修订版.docx`，13页、5幅图。实验结果截至2026年10月26日；本次只改写报告，没有新增训练或改动实验数据、模型和结果。

正文只保留几次影响研究方向的主要变化，不再逐轮列18个调参阶段。详细过程仍保存在原实验文件及旧版叙述模块中。对当前方法从数据到输出逐步解释，用同一组直线示意串联残差、参数移动与训练误差；示意图和数字明确标为说明用途，不冒充实测结果。

```powershell
& 'E:/study/python/python.exe' -X utf8 code/reporting/build_first_stage_report.py --figures
& 'C:/Users/hana/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -X utf8 code/reporting/build_first_stage_report.py --docx
```

文字与解释图位于`code/reporting/first_stage_explained.py`，排版与结果图由`build_first_stage_report.py`生成。原同名Word若在WPS或Word中打开，则另存修订版；修订版也被占用时，再保存为讲解版。

技能自带渲染器因缺少LibreOffice未能运行，已用Microsoft Word只读导出PDF，再将13页转成PNG逐页检查。检查文件在`tmp/first_stage_report/accessible`，不作为交付附件。`quality_check.json`记录交付文件哈希、图片数和源文件未变检查；修订前Word已备份到该临时目录。
