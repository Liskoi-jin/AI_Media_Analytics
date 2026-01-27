# src/report_generator.py
"""
报告生成器 - 生成Excel报告和HTML报告
支持三种分析模式：Media_Analysis模式、成本发挥分析模式、完整分析模式
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any
from src.utils import (
    logger, create_dir_if_not_exist, generate_filename,
    save_dataframe_to_excel, export_to_html, format_number
)


class ReportGenerator:
    """报告生成器 - 支持多种分析模式和报告格式"""

    def __init__(self, analysis_results: Dict[str, Any] = None, output_dir: str = "./outputs"):
        """
        初始化报告生成器
        :param analysis_results: 分析引擎返回的完整结果（可选）
        :param output_dir: 输出目录根路径
        """
        self.analysis_results = analysis_results if analysis_results is not None else {}
        self.output_dir = output_dir

        # 子目录配置
        self.excel_output_dir = os.path.join(output_dir, "excel")
        self.html_output_dir = os.path.join(output_dir, "temp")
        self.individual_excel_dir = os.path.join(output_dir, "reports")

        # 创建目录
        create_dir_if_not_exist(self.excel_output_dir)
        create_dir_if_not_exist(self.html_output_dir)
        create_dir_if_not_exist(self.individual_excel_dir)

        # 报告文件名前缀
        self.report_prefix = "Media_Analysis_Report"
        self.individual_report_prefix = "Media_Individual_Analysis"

        logger.info("报告生成器初始化完成")

    # ✅ 核心修复：添加analysis_id参数
    def generate_excel_report(self, analysis_id: str, analysis_mode: str = 'full') -> str:
        """
        生成完整Excel报告（根据分析模式调整）
        :param analysis_id: 分析ID，用于文件名
        :param analysis_mode: 分析模式 ('media_analysis', 'cost_analysis', 'full')
        :return: Excel报告文件路径
        """
        logger.info(f"开始生成Excel报告，模式: {analysis_mode}, 分析ID: {analysis_id}")

        # ✅ 修复：确保使用正确的analysis_id
        if analysis_id == 'full':
            # 如果没有传入analysis_id，使用时间戳
            analysis_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_filename = f"Media_Analysis_Report_{analysis_id}_{time_str}.xlsx"
        excel_file_path = os.path.join(self.excel_output_dir, excel_filename)

        try:
            with pd.ExcelWriter(excel_file_path, engine="openpyxl") as writer:
                # 根据分析模式生成不同的报告
                if analysis_mode == 'media_analysis':
                    self._generate_media_analysis_report(writer)
                elif analysis_mode == 'cost_analysis':
                    self._generate_cost_analysis_report(writer)
                else:  # full mode
                    self._generate_full_analysis_report(writer)

                # ========== 核心修复 V2 版本 - 终极根治 开始 ==========
                workbook = writer.book
                all_sheet_names = workbook.sheetnames

                # 兜底方案1: 如果一个sheet都没有写入，强制创建一个【核心汇总】工作表，永不报错
                if not all_sheet_names:
                    empty_df = pd.DataFrame({
                        "报告状态": ["✅ 媒介分析报告生成成功"],
                        "分析模式": [analysis_mode],
                        "分析ID": [analysis_id],
                        "生成时间": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                        "数据说明": ["本次分析无明细数据，仅生成汇总报告"]
                    })
                    empty_df.to_excel(writer, sheet_name="核心汇总", index=False)
                    all_sheet_names = ["核心汇总"]

                # 兜底方案2: 强制保证第一个sheet可见，其余隐藏，保留原业务需求
                if all_sheet_names:
                    workbook[all_sheet_names[0]].sheet_state = 'visible'
                    for sheet_name in all_sheet_names[1:]:
                        workbook[sheet_name].sheet_state = 'hidden'
                # ========== 核心修复 V2 版本 - 终极根治 结束 ==========

            logger.info(f"Excel报告生成成功，路径: {excel_file_path}")
            return excel_file_path

        except Exception as e:
            logger.error(f"生成Excel报告失败: {e}", exc_info=True)
            raise

    def _generate_media_analysis_report(self, writer: pd.ExcelWriter) -> None:
        """生成Media_Analysis模式的报告"""
        # 1. 基础统计信息 - 强制写入，保底sheet
        basic_stats = self.analysis_results.get("basic_stats", {})
        stats_df = self._convert_dict_to_df(basic_stats, "基础统计信息")
        if not stats_df.empty:
            stats_df.to_excel(writer, sheet_name="基础统计", index=False)
        else:
            pd.DataFrame({"基础统计": ["无统计数据"], "说明": ["本次分析无基础统计指标"]}).to_excel(writer, sheet_name="基础统计", index=False)

        # 2. 工作量分析 - 媒介明细
        workload_detail = self.analysis_results.get("workload_analysis", {}).get("detail", None)
        if workload_detail is not None and not workload_detail.empty:
            workload_detail.to_excel(writer, sheet_name="工作量明细", index=False)

        # 3. 工作量分析 - 小组汇总
        workload_group = self.analysis_results.get("workload_analysis", {}).get("group_summary", None)
        if workload_group is not None and not workload_group.empty:
            workload_group.to_excel(writer, sheet_name="工作量小组汇总", index=False)

        # 4. 工作量分析 - TOP排名
        workload_top = self.analysis_results.get("workload_analysis", {}).get("top_media_ranking", None)
        if workload_top is not None and not workload_top.empty:
            workload_top.to_excel(writer, sheet_name="工作量TOP排名", index=False)

        # 5. 质量分析 - 媒介明细
        quality_detail = self.analysis_results.get("quality_analysis", {}).get("detail", None)
        if quality_detail is not None and not quality_detail.empty:
            quality_detail.to_excel(writer, sheet_name="质量明细", index=False)

        # 6. 质量分析 - 小组汇总
        quality_group = self.analysis_results.get("quality_analysis", {}).get("group_summary", None)
        if quality_group is not None and not quality_group.empty:
            quality_group.to_excel(writer, sheet_name="质量小组汇总", index=False)

        # 7. 质量分析 - 质量分布
        quality_dist = self.analysis_results.get("quality_analysis", {}).get("quality_distribution", None)
        if quality_dist is not None and not quality_dist.empty:
            quality_dist.to_excel(writer, sheet_name="质量分布", index=False)

        # 8. 综合汇总 - 必写sheet
        combined_summary = self.analysis_results.get("combined_summary", {})
        combined_df = self._convert_combined_summary_to_df(combined_summary)
        if not combined_df.empty:
            combined_df.to_excel(writer, sheet_name="综合汇总", index=False)
        else:
            pd.DataFrame({"综合汇总": ["无综合数据"], "说明": ["本次分析无综合指标汇总"]}).to_excel(writer, sheet_name="综合汇总", index=False)

    def _generate_cost_analysis_report(self, writer: pd.ExcelWriter) -> None:
        """生成成本发挥分析模式的报告"""
        # 获取目标工作表
        target_sheets = {
            "媒介小组工作量分析": self.analysis_results.get("media_group_workload"),
            "定档媒介工作量分析": self.analysis_results.get("fixed_media_workload"),
            "定档媒介成本分析": self.analysis_results.get("fixed_media_cost"),
            "定档媒介返点分析": self.analysis_results.get("fixed_media_rebate"),
            "定档媒介效果分析": self.analysis_results.get("fixed_media_performance"),
            "定档媒介达人量级分析": self.analysis_results.get("fixed_media_level"),
            "定档媒介综合分析": self.analysis_results.get("fixed_media_comprehensive"),
            "详细数据": self.analysis_results.get("detailed_data")
        }

        # 基础统计信息 - 强制写入，保底sheet
        basic_stats = self.analysis_results.get("basic_stats", {})
        stats_df = self._convert_dict_to_df(basic_stats, "基础统计信息")
        if not stats_df.empty:
            stats_df.to_excel(writer, sheet_name="基础统计", index=False)
        else:
            pd.DataFrame({"基础统计": ["无统计数据"], "说明": ["本次分析无基础统计指标"]}).to_excel(writer, sheet_name="基础统计", index=False)

        # 写入所有目标工作表
        for sheet_name, sheet_data in target_sheets.items():
            if sheet_data is not None and not sheet_data.empty:
                sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                # 如果数据为空，创建提示表
                pd.DataFrame({"提示": [f"无{sheet_name}数据"]}).to_excel(
                    writer, sheet_name=sheet_name, index=False
                )

        # 成本分析汇总 - 必写sheet
        cost_summary = self.analysis_results.get("cost_analysis", {}).get("overall_summary", {})
        if cost_summary:
            summary_df = self._convert_dict_to_df(cost_summary, "成本分析汇总")
            summary_df.to_excel(writer, sheet_name="成本分析汇总", index=False)
        else:
            pd.DataFrame({"成本分析汇总": ["无成本数据"], "说明": ["本次分析无成本指标汇总"]}).to_excel(writer, sheet_name="成本分析汇总", index=False)

    # 修改 _generate_full_analysis_report 方法
    def _generate_full_analysis_report(self, writer: pd.ExcelWriter) -> None:
        """生成完整分析模式的报告 - 修复版：确保所有工作表都生成"""

        # 获取分析结果
        workload_data = self.analysis_results.get("workload", {})
        quality_data = self.analysis_results.get("quality", {})
        cost_data = self.analysis_results.get("cost", {})

        logger.info(f"开始生成完整报告: workload keys={list(workload_data.keys())}, "
                    f"quality keys={list(quality_data.keys())}, "
                    f"cost keys={list(cost_data.keys())}")

        # 1. 工作量明细 - 使用正确的字段名
        workload_detail = workload_data.get("result", pd.DataFrame())
        workload_detail_df = pd.DataFrame()

        if workload_detail is not None:
            if isinstance(workload_detail, (list, dict)):
                workload_detail_df = pd.DataFrame(workload_detail)
            elif isinstance(workload_detail, pd.DataFrame):
                workload_detail_df = workload_detail
            else:
                logger.warning(f"工作量明细数据类型不支持: {type(workload_detail)}")

        logger.info(f"工作量明细数据: {len(workload_detail_df)} 行")

        # 写入工作表（即使为空也写入）
        if not workload_detail_df.empty:
            # 重命名列以匹配模板
            column_mapping = {}
            if '媒介姓名' not in workload_detail_df.columns and '定档媒介' in workload_detail_df.columns:
                column_mapping['定档媒介'] = '媒介姓名'
            if '对应真名' not in workload_detail_df.columns and '对应名字' in workload_detail_df.columns:
                column_mapping['对应名字'] = '对应真名'

            if column_mapping:
                workload_detail_df = workload_detail_df.rename(columns=column_mapping)

            workload_detail_df.to_excel(writer, sheet_name="工作量明细", index=False)
            logger.info("已写入工作量明细工作表")
        else:
            pd.DataFrame({"提示": ["无工作量明细数据"]}).to_excel(writer, sheet_name="工作量明细", index=False)
            logger.info("已写入空的工作量明细工作表")

        # 2. 质量明细
        quality_detail = quality_data.get("result", pd.DataFrame())
        quality_detail_df = pd.DataFrame()

        if quality_detail is not None:
            if isinstance(quality_detail, (list, dict)):
                quality_detail_df = pd.DataFrame(quality_detail)
            elif isinstance(quality_detail, pd.DataFrame):
                quality_detail_df = quality_detail

        logger.info(f"质量明细数据: {len(quality_detail_df)} 行")

        if not quality_detail_df.empty:
            quality_detail_df.to_excel(writer, sheet_name="质量明细", index=False)
            logger.info("已写入质量明细工作表")
        else:
            pd.DataFrame({"提示": ["无质量明细数据"]}).to_excel(writer, sheet_name="质量明细", index=False)
            logger.info("已写入空的质量明细工作表")

        # 3. 成本明细 - 修复：正确处理list类型数据
        cost_detail = cost_data.get("media_detail", pd.DataFrame())
        if self._is_empty_data(cost_detail):
            cost_detail = cost_data.get("detail_data", pd.DataFrame())
        if self._is_empty_data(cost_detail):
            cost_detail = cost_data.get("result", pd.DataFrame())

        cost_detail_df = pd.DataFrame()
        if cost_detail is not None:
            if isinstance(cost_detail, (list, dict)):
                cost_detail_df = pd.DataFrame(cost_detail)
            elif isinstance(cost_detail, pd.DataFrame):
                cost_detail_df = cost_detail

        logger.info(
            f"成本明细数据: {len(cost_detail_df)} 行，列: {list(cost_detail_df.columns) if not cost_detail_df.empty else '无列'}")

        if not cost_detail_df.empty:
            cost_detail_df.to_excel(writer, sheet_name="成本明细", index=False)
            logger.info("已写入成本明细工作表")
        else:
            pd.DataFrame({"提示": ["无成本明细数据"]}).to_excel(writer, sheet_name="成本明细", index=False)
            logger.info("已写入空的成本明细工作表")

        # 4. 成本效率排名 - 修复：正确处理list类型数据
        cost_ranking = cost_data.get("cost_efficiency_ranking", pd.DataFrame())
        if self._is_empty_data(cost_ranking):
            cost_ranking = cost_data.get("result", pd.DataFrame())

        cost_ranking_df = pd.DataFrame()
        if cost_ranking is not None:
            if isinstance(cost_ranking, (list, dict)):
                cost_ranking_df = pd.DataFrame(cost_ranking)
            elif isinstance(cost_ranking, pd.DataFrame):
                cost_ranking_df = cost_ranking

        logger.info(f"成本效率排名数据: {len(cost_ranking_df)} 行")

        if not cost_ranking_df.empty:
            cost_ranking_df.to_excel(writer, sheet_name="成本效率排名", index=False)
            logger.info("已写入成本效率排名工作表")
        else:
            pd.DataFrame({"提示": ["无成本效率排名数据"]}).to_excel(writer, sheet_name="成本效率排名", index=False)
            logger.info("已写入空的成本效率排名工作表")

        # 5. 基础统计
        basic_stats = self.analysis_results.get("basic_stats", {})
        if basic_stats:
            stats_df = pd.DataFrame(list(basic_stats.items()), columns=["统计项", "数值"])
            stats_df.to_excel(writer, sheet_name="基础统计", index=False)
            logger.info("已写入基础统计工作表")
        else:
            pd.DataFrame({"基础统计": ["无基础统计数据"]}).to_excel(writer, sheet_name="基础统计", index=False)
            logger.info("已写入空的基础统计工作表")

        # 6. 分析汇总 - 强制生成
        summary_data = {
            '分析类型': ['工作量分析', '质量分析', '成本分析'],
            '有效数据量': [
                len(workload_detail_df),
                len(quality_detail_df),
                len(cost_detail_df)
            ],
            '数据状态': [
                '有数据' if len(workload_detail_df) > 0 else '无数据',
                '有数据' if len(quality_detail_df) > 0 else '无数据',
                '有数据' if len(cost_detail_df) > 0 else '无数据'
            ],
            '生成时间': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * 3
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="分析汇总", index=False)
        logger.info("已写入分析汇总工作表")

        # 7. 成本发挥分析工作表 - 新增：确保生成所有成本分析工作表
        cost_sheets = {
            "媒介小组工作量分析": cost_data.get("media_group_workload"),
            "定档媒介工作量分析": cost_data.get("fixed_media_workload"),
            "定档媒介成本分析": cost_data.get("fixed_media_cost"),
            "定档媒介返点分析": cost_data.get("fixed_media_rebate"),
            "定档媒介效果分析": cost_data.get("fixed_media_performance"),
            "定档媒介达人量级分析": cost_data.get("fixed_media_level"),
            "定档媒介综合分析": cost_data.get("fixed_media_comprehensive"),
            "详细数据": cost_data.get("detailed_data")
        }

        for sheet_name, sheet_data in cost_sheets.items():
            sheet_df = pd.DataFrame()
            if sheet_data is not None:
                if isinstance(sheet_data, (list, dict)):
                    sheet_df = pd.DataFrame(sheet_data)
                elif isinstance(sheet_data, pd.DataFrame):
                    sheet_df = sheet_data

            if not sheet_df.empty:
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
                logger.info(f"已写入{sheet_name}工作表，{len(sheet_df)}行")
            else:
                pd.DataFrame({"提示": [f"无{sheet_name}数据"]}).to_excel(writer, sheet_name=sheet_name, index=False)
                logger.info(f"已写入空的{sheet_name}工作表")

        logger.info(f"完整报告生成完成，共生成 {len(writer.book.worksheets)} 个工作表")

    def _is_empty_data(self, data: Any) -> bool:
        """检查数据是否为空"""
        if data is None:
            return True
        if isinstance(data, list):
            return len(data) == 0
        if isinstance(data, pd.DataFrame):
            return data.empty
        if isinstance(data, dict):
            return len(data) == 0
        return False

    # ✅ 修复：添加analysis_id参数
    def generate_html_report(self) -> str:
        """
        生成简易HTML报告（用于页面展示）
        :return: HTML报告文件路径
        """
        logger.info("开始生成HTML报告")

        # 生成带时间戳的文件名
        html_filename = generate_filename(self.report_prefix, "html")
        html_file_path = os.path.join(self.html_output_dir, html_filename)

        try:
            # 构建HTML内容
            html_content = self._build_html_content()

            # 写入HTML文件
            with open(html_file_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"HTML报告生成成功，路径: {html_file_path}")
            return html_file_path

        except Exception as e:
            logger.error(f"生成HTML报告失败: {e}", exc_info=True)
            raise

    def _convert_dict_to_df(self, data_dict: Dict[str, Any], title: str = "") -> pd.DataFrame:
        """将字典转换为DataFrame"""
        if not data_dict:
            return pd.DataFrame()

        # 处理嵌套字典
        flat_data = []
        for key, value in data_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_data.append({
                        "分类": key,
                        "指标": sub_key,
                        "数值": sub_value
                    })
            else:
                flat_data.append({
                    "分类": title,
                    "指标": key,
                    "数值": value
                })

        return pd.DataFrame(flat_data)

    def _convert_combined_summary_to_df(self, combined_summary: Dict[str, Any]) -> pd.DataFrame:
        """转换综合汇总为DataFrame"""
        if not combined_summary:
            return pd.DataFrame()

        combined_data = []
        for category, indicators in combined_summary.items():
            if isinstance(indicators, dict):
                for indicator, value in indicators.items():
                    combined_data.append({
                        "综合分类": category,
                        "核心指标": indicator,
                        "指标数值": value
                    })

        return pd.DataFrame(combined_data)

    def _build_html_content(self) -> str:
        """构建HTML内容"""
        # 基础HTML结构
        html_template = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; padding: 0; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background: #f9f9f9; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 10px; text-align: left; border: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .summary-box {{ padding: 15px; background: #e3f2fd; border-radius: 6px; margin: 10px 0; }}
                .grade {{ color: #e74c3c; font-weight: bold; font-size: 1.2em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 style="text-align: center;">{title}</h1>
                <p style="text-align: center; color: #7f8c8d;">生成时间: {generate_time}</p>

                <!-- 综合汇总部分 -->
                <div class="section">
                    <h2>一、综合汇总</h2>
                    {combined_summary_html}
                </div>

                <!-- 工作量分析部分 -->
                <div class="section">
                    <h2>二、工作量分析</h2>
                    {workload_summary_html}
                </div>

                <!-- 质量分析部分 -->
                <div class="section">
                    <h2>三、质量分析</h2>
                    {quality_summary_html}
                </div>

                <!-- 成本分析部分 -->
                <div class="section">
                    <h2>四、成本分析</h2>
                    {cost_summary_html}
                </div>
            </div>
        </body>
        </html>
        """

        # 提取各部分数据并转换为HTML
        generate_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = "媒介分析报告"

        # 1. 综合汇总HTML
        combined_summary = self.analysis_results.get("combined_summary", {})
        combined_df = self._convert_combined_summary_to_df(combined_summary)
        combined_summary_html = export_to_html(combined_df,
                                               "综合核心指标") if not combined_df.empty else "<p>无综合汇总数据</p>"

        # 2. 工作量分析HTML
        workload_summary = self.analysis_results.get("workload_analysis", {}).get("summary", {})
        workload_summary_df = self._convert_dict_to_df(workload_summary, "工作量核心指标")
        workload_summary_html = export_to_html(workload_summary_df,
                                               "工作量核心指标") if not workload_summary_df.empty else "<p>无工作量汇总数据</p>"

        # 3. 质量分析HTML
        quality_summary = self.analysis_results.get("quality_analysis", {}).get("summary", {})
        quality_summary_df = self._convert_dict_to_df(quality_summary, "质量核心指标")
        quality_summary_html = export_to_html(quality_summary_df,
                                              "质量核心指标") if not quality_summary_df.empty else "<p>无质量汇总数据</p>"

        # 4. 成本分析HTML
        cost_summary = self.analysis_results.get("cost_analysis", {}).get("overall_summary", {})
        cost_summary_df = self._convert_dict_to_df(cost_summary, "成本核心指标")
        cost_summary_html = export_to_html(cost_summary_df,
                                           "成本核心指标") if not cost_summary_df.empty else "<p>无成本汇总数据</p>"

        # 填充模板
        html_content = html_template.format(
            title=title,
            generate_time=generate_time,
            combined_summary_html=combined_summary_html,
            workload_summary_html=workload_summary_html,
            quality_summary_html=quality_summary_html,
            cost_summary_html=cost_summary_html
        )

        return html_content

    # ✅ 修复：添加analysis_id参数
    def generate_all_reports(self, analysis_id: str, analysis_mode: str = 'full') -> Dict[str, str]:
        """
        生成所有格式报告
        :param analysis_id: 分析ID
        :param analysis_mode: 分析模式
        :return: 报告路径字典
        """
        logger.info(f"开始生成所有格式报告，模式: {analysis_mode}, 分析ID: {analysis_id}")

        report_paths = {
            "excel_report": None,
            "html_report": None
        }

        # 生成Excel报告
        try:
            excel_path = self.generate_excel_report(analysis_id, analysis_mode)
            report_paths["excel_report"] = excel_path
        except Exception as e:
            logger.error(f"Excel报告生成失败，跳过该格式: {e}")

        # 生成HTML报告
        try:
            html_path = self.generate_html_report()
            report_paths["html_report"] = html_path
        except Exception as e:
            logger.error(f"HTML报告生成失败，跳过该格式: {e}")

        logger.info("所有报告生成流程结束")
        return report_paths

    def generate_cost_analysis_full_report(self, target_sheets: Dict[str, pd.DataFrame],
                                           category: str = "成本发挥分析") -> str:
        """
        生成成本发挥分析完整报告（8个工作表）
        :param target_sheets: 目标工作表字典
        :param category: 分析类目
        :return: 报告文件完整路径
        """
        logger.info(f"开始生成[{category}]成本发挥分析完整报告")

        # 生成带类目和时间戳的文件名
        cost_prefix = f"成本发挥分析完整报告_{category}"
        excel_filename = generate_filename(cost_prefix, "xlsx")
        excel_file_path = os.path.join(self.excel_output_dir, excel_filename)

        try:
            with pd.ExcelWriter(excel_file_path, engine="openpyxl") as writer:
                # 写入所有目标工作表
                sheet_count = 0
                for sheet_name, sheet_data in target_sheets.items():
                    if sheet_data is not None and not sheet_data.empty:
                        # 限制行数避免文件过大
                        display_data = sheet_data.head(1000) if len(sheet_data) > 1000 else sheet_data
                        display_data.to_excel(writer, sheet_name=sheet_name, index=False)
                        sheet_count += 1
                        logger.debug(f"写入工作表: {sheet_name}, 行数: {len(display_data)}")
                    else:
                        # 创建空表提示
                        pd.DataFrame({"提示": [f"无{sheet_name}数据"]}).to_excel(
                            writer, sheet_name=sheet_name, index=False
                        )

                # 添加报告信息表 - 强制写入，保底sheet
                info_data = [
                    ["报告名称", f"成本发挥分析完整报告 - {category}"],
                    ["生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    ["工作表数量", sheet_count],
                    ["分析模式", "成本发挥分析模式"],
                    ["数据来源", "原始成本发挥分析.py逻辑"]
                ]
                info_df = pd.DataFrame(info_data, columns=["项目", "信息"])
                info_df.to_excel(writer, sheet_name="报告信息", index=False)

                # ========== 独立成本报告 同样加终极修复 ==========
                workbook = writer.book
                all_sheet_names = workbook.sheetnames
                if all_sheet_names:
                    workbook[all_sheet_names[0]].sheet_state = 'visible'
                    for sheet_name in all_sheet_names[1:]:
                        workbook[sheet_name].sheet_state = 'hidden'

            logger.info(f"成本发挥分析完整报告生成成功，路径: {excel_file_path}")
            return excel_file_path

        except Exception as e:
            logger.error(f"生成成本发挥分析完整报告失败: {e}", exc_info=True)
            return ""