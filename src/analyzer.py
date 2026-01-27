# src/analyzer.py
"""
分析引擎主类 - 整合工作量、质量、成本分析，对齐Media_Analysis和成本发挥分析逻辑
支持三种分析模式：Media_Analysis模式、成本发挥分析模式、完整分析模式
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from src.workload_analyzer import WorkloadAnalyzer
from src.quality_analyzer import QualityAnalyzer
from src.cost_analyzer import CostAnalyzer
from src.utils import (
    logger, ID_TO_NAME_MAPPING, FLOWER_TO_NAME_MAPPING,
    NAME_TO_GROUP_MAPPING
)


class MediaAnalyzer:
    """媒介分析引擎 - 整合三大分析模块，支持不同分析模式"""

    def __init__(self, data_processor_result: Dict[str, Any], analysis_mode: str = 'full'):
        """
        初始化分析引擎
        :param data_processor_result: 数据处理器返回的结果字典
        :param analysis_mode: 分析模式 ('media_analysis', 'cost_analysis', 'full')
        """
        # 提取数据
        self.processed_df = data_processor_result.get('processed_data', pd.DataFrame())
        self.filtered_df = data_processor_result.get('filtered_data', pd.DataFrame())
        self.basic_stats = data_processor_result.get('stats', {})
        self.analysis_mode = analysis_mode

        # 核心映射表
        self.id_to_name_mapping = ID_TO_NAME_MAPPING
        self.flower_to_name_mapping = FLOWER_TO_NAME_MAPPING
        self.name_to_group_mapping = NAME_TO_GROUP_MAPPING

        # 分析结果存储
        self.analysis_results = {
            "analysis_mode": analysis_mode,
            "workload_analysis": None,  # 工作量分析结果
            "quality_analysis": None,  # 质量分析结果
            "cost_analysis": None,  # 成本分析结果
            "combined_summary": None,  # 综合汇总
            "basic_stats": self.basic_stats,  # 基础统计

            # Media_Analysis专用结果
            "media_workload_detail": None,  # 媒介工作量明细
            "media_quality_detail": None,  # 媒介质量明细

            # 成本发挥分析专用结果
            "media_group_workload": pd.DataFrame(),  # 修复：初始化空DF，避免None
            "fixed_media_workload": pd.DataFrame(),  # 修复：初始化空DF，避免None
            "fixed_media_cost": pd.DataFrame(),  # 修复：初始化空DF，避免None
            "fixed_media_rebate": pd.DataFrame(),  # 修复：初始化空DF，避免None
            "fixed_media_performance": pd.DataFrame(),  # 修复：初始化空DF，避免None
            "fixed_media_level": pd.DataFrame(),  # 修复：初始化空DF，避免None
            "fixed_media_comprehensive": pd.DataFrame(),  # 修复：初始化空DF，避免None
            "detailed_data": pd.DataFrame()  # 修复：初始化空DF，避免None
        }

        logger.info(f"媒介分析引擎初始化完成，模式: {analysis_mode}")

    # 在 analyzer.py 的 run_workload_analysis 方法中，确保正确传递映射表
    def run_workload_analysis(self, top_n: int = 10) -> Dict[str, Any]:
        """
        运行工作量分析（Media_Analysis逻辑）
        :param top_n: TOP媒介排名数量
        :return: 工作量分析结果
        """
        logger.info("启动工作量分析模块（Media_Analysis逻辑）")

        if self.processed_df.empty:
            logger.warning("无有效数据，工作量分析跳过")
            self.analysis_results["workload_analysis"] = {
                "summary": {"提示": "无有效数据进行工作量分析"},
                "detail": pd.DataFrame(),
                "group_summary": pd.DataFrame(),
                "top_media_ranking": pd.DataFrame()
            }
            return self.analysis_results["workload_analysis"]

        try:
            # 初始化工作量分析器（Media_Analysis逻辑）
            workload_analyzer = WorkloadAnalyzer(
                df=self.processed_df,
                known_id_name_mapping=self.id_to_name_mapping,
                config={"FLOWER_TO_NAME_MAPPING": self.flower_to_name_mapping}
            )

            # 执行分析
            workload_result = workload_analyzer.analyze(top_n=top_n)

            # 提取Media_Analysis专用结果
            self.analysis_results["media_workload_detail"] = workload_result.get("detail", pd.DataFrame())

            # 存储结果
            self.analysis_results["workload_analysis"] = workload_result
            logger.info("工作量分析模块执行完成")

            return workload_result

        except Exception as e:
            logger.error(f"工作量分析失败: {e}", exc_info=True)
            self.analysis_results["workload_analysis"] = {
                "summary": {"错误": f"工作量分析失败: {str(e)}"},
                "detail": pd.DataFrame(),
                "group_summary": pd.DataFrame(),
                "top_media_ranking": pd.DataFrame()
            }
            return self.analysis_results["workload_analysis"]

    def run_quality_analysis(self, use_original_state: bool = True) -> Dict[str, Any]:
        """
        运行质量分析（Media_Analysis逻辑）
        :param use_original_state: 是否使用原始状态计算过筛率
        :return: 质量分析结果
        """
        logger.info("启动质量分析模块（Media_Analysis逻辑）")

        if self.processed_df.empty:
            logger.warning("无有效数据，质量分析跳过")
            self.analysis_results["quality_analysis"] = {
                "summary": {"提示": "无有效数据进行质量分析"},
                "detail": pd.DataFrame(),
                "group_summary": pd.DataFrame(),
                "quality_distribution": pd.DataFrame()
            }
            return self.analysis_results["quality_analysis"]

        try:
            # 初始化质量分析器
            quality_analyzer = QualityAnalyzer(
                df=self.processed_df,
                known_id_name_mapping=self.id_to_name_mapping,
                config={"FLOWER_TO_NAME_MAPPING": self.flower_to_name_mapping}
            )

            # 执行分析
            quality_result = quality_analyzer.analyze(use_original_state=use_original_state)

            # 提取Media_Analysis专用结果
            self.analysis_results["media_quality_detail"] = quality_result.get("detail", pd.DataFrame())

            # 存储结果
            self.analysis_results["quality_analysis"] = quality_result
            logger.info("质量分析模块执行完成")

            return quality_result

        except Exception as e:
            logger.error(f"质量分析失败: {e}", exc_info=True)
            self.analysis_results["quality_analysis"] = {
                "summary": {"错误": f"质量分析失败: {str(e)}"},
                "detail": pd.DataFrame(),
                "group_summary": pd.DataFrame(),
                "quality_distribution": pd.DataFrame()
            }
            return self.analysis_results["quality_analysis"]

    def run_cost_analysis(self, top_n: int = 10) -> Dict[str, Any]:
        """
        运行成本分析（成本发挥分析逻辑）
        :param top_n: 成本效益TOP排名数量
        :return: 成本分析结果
        """
        logger.info("启动成本分析模块（成本发挥分析逻辑）")

        if self.processed_df.empty:
            logger.warning("无有效数据，成本分析跳过")
            self.analysis_results["cost_analysis"] = {
                "overall_summary": {"提示": "无有效数据进行成本分析"},
                "media_detail": pd.DataFrame(),
                "group_summary": pd.DataFrame(),
                "filtered_summary": pd.DataFrame(),
                "cost_efficiency_ranking": pd.DataFrame()
            }
            return self.analysis_results["cost_analysis"]

        try:
            # 初始化成本分析器
            cost_analyzer = CostAnalyzer(
                processed_df=self.processed_df,
                filtered_df=self.filtered_df
            )

            # 执行分析
            cost_result = cost_analyzer.analyze(top_n=top_n)

            # 提取成本发挥分析的所有目标工作表结果
            self._extract_cost_analysis_results(cost_result)

            # 存储结果
            self.analysis_results["cost_analysis"] = cost_result
            self.analysis_results["detailed_data"] = self.processed_df

            logger.info("成本分析模块执行完成（已提取所有目标工作表结果）")

            return cost_result

        except Exception as e:
            logger.error(f"成本分析失败: {e}", exc_info=True)
            self.analysis_results["cost_analysis"] = {
                "overall_summary": {"错误": f"成本分析失败: {str(e)}"},
                "media_detail": pd.DataFrame()
            }
            return self.analysis_results["cost_analysis"]

    # 找到 _extract_cost_analysis_results 方法，确保它获取无效数据详情：
    def _extract_cost_analysis_results(self, cost_result: Dict) -> None:
        """核心修复：从cost_analyzer返回的结果中提取工作表"""
        logger.info("提取成本发挥分析目标工作表结果")
        try:
            # 提取所有目标工作表
            self.analysis_results["media_group_workload"] = cost_result.get("media_group_workload", pd.DataFrame())
            self.analysis_results["fixed_media_workload"] = cost_result.get("fixed_media_workload", pd.DataFrame())
            self.analysis_results["fixed_media_cost"] = cost_result.get("fixed_media_cost", pd.DataFrame())
            self.analysis_results["fixed_media_rebate"] = cost_result.get("fixed_media_rebate", pd.DataFrame())
            self.analysis_results["fixed_media_performance"] = cost_result.get("fixed_media_performance",
                                                                               pd.DataFrame())
            self.analysis_results["fixed_media_level"] = cost_result.get("fixed_media_level", pd.DataFrame())
            self.analysis_results["fixed_media_comprehensive"] = cost_result.get("fixed_media_comprehensive",
                                                                                 pd.DataFrame())

            # ✅ 新增：提取无效数据详情
            self.analysis_results["invalid_data_detail"] = cost_result.get("invalid_data_detail", [])

            self.analysis_results["detailed_data"] = cost_result.get("detailed_data", pd.DataFrame())

            valid_sheet_count = len(
                [v for v in self.analysis_results.values() if isinstance(v, pd.DataFrame) and not v.empty])
            logger.info(
                f"成功提取 {valid_sheet_count} 个有效目标工作表，包含 {len(self.analysis_results['invalid_data_detail'])} 条无效数据详情")
        except Exception as e:
            logger.error(f"提取成本分析结果失败: {e}")

    def generate_combined_summary(self) -> Dict[str, Any]:
        """
        生成综合汇总信息（根据分析模式调整）
        :return: 综合汇总字典
        """
        logger.info("开始生成综合汇总信息")

        # 根据分析模式生成不同的汇总
        if self.analysis_mode == 'media_analysis':
            combined_summary = self._generate_media_analysis_summary()
        elif self.analysis_mode == 'cost_analysis':
            combined_summary = self._generate_cost_analysis_summary()
        else:  # full mode
            combined_summary = self._generate_full_summary()

        # 存储综合汇总
        self.analysis_results["combined_summary"] = combined_summary
        logger.info("综合汇总信息生成完成")

        return combined_summary

    def _generate_media_analysis_summary(self) -> Dict[str, Any]:
        """生成Media_Analysis模式的综合汇总"""
        combined_summary = {
            "分析模式": "Media_Analysis模式（工作量+工作质量）",
            "数据概览": self.basic_stats,
            "工作量核心指标": {},
            "质量核心指标": {},
            "综合评级": {}
        }

        # 提取工作量核心指标
        workload_summary = self.analysis_results.get("workload_analysis", {}).get("summary", {})
        if workload_summary and "提示" not in workload_summary and "错误" not in workload_summary:
            combined_summary["工作量核心指标"] = {
                "媒介总数": workload_summary.get("媒介总数", 0),
                "总处理量": workload_summary.get("总处理量", 0),
                "总定档量": workload_summary.get("总定档量", 0),
                "整体定档率": workload_summary.get("整体定档率", "0.00%"),
                "S级媒介数": workload_summary.get("S级媒介数", 0),
                "A级及以上媒介数": workload_summary.get("A级及以上媒介数", 0)
            }

        # 提取质量核心指标
        quality_summary = self.analysis_results.get("quality_analysis", {}).get("summary", {})
        if quality_summary and "提示" not in quality_summary and "错误" not in quality_summary:
            combined_summary["质量核心指标"] = {
                "媒介总数": quality_summary.get("媒介总数", 0),
                "总提报达人数": quality_summary.get("总提报达人数", 0),
                "总过筛人数": quality_summary.get("总过筛人数", 0),
                "整体过筛率": quality_summary.get("overall_screening_rate", "0.00%"),
                "优秀质量媒介数": quality_summary.get("优秀质量媒介数", 0),
                "优秀质量媒介占比": quality_summary.get("优秀质量媒介占比", "0.00%")
            }

        # 计算综合评级（Media_Analysis模式）
        workload_score = self._calculate_workload_score(workload_summary)
        quality_score = self._calculate_quality_score(quality_summary)
        total_score = round(workload_score + quality_score, 2)

        # 综合评级
        if total_score >= 90:
            combined_grade = "S级（卓越）"
        elif total_score >= 80:
            combined_grade = "A级（优秀）"
        elif total_score >= 70:
            combined_grade = "B级（良好）"
        elif total_score >= 60:
            combined_grade = "C级（合格）"
        elif total_score >= 50:
            combined_grade = "D级（较差）"
        else:
            combined_grade = "E级（很差）"

        combined_summary["综合评级"] = {
            "工作量得分": round(workload_score, 2),
            "质量得分": round(quality_score, 2),
            "综合得分": total_score,
            "综合评级": combined_grade
        }

        return combined_summary

    def _generate_cost_analysis_summary(self) -> Dict[str, Any]:
        """生成成本发挥分析模式的综合汇总"""
        combined_summary = {
            "分析模式": "成本发挥分析模式",
            "数据概览": self.basic_stats,
            "成本核心指标": {},
            "返点效益指标": {},
            "效果表现指标": {},
            "综合评级": {}
        }

        # 提取成本核心指标
        cost_summary = self.analysis_results.get("cost_analysis", {}).get("overall_summary", {})
        if cost_summary and "提示" not in cost_summary and "错误" not in cost_summary:
            combined_summary["成本核心指标"] = {
                "总媒介数": cost_summary.get("总媒介数", 0),
                "总达人数": cost_summary.get("总达人数", 0),
                "总成本": cost_summary.get("总成本", 0.0),
                "总返点金额": cost_summary.get("总返点金额", 0.0),
                "整体返点占报价比例": cost_summary.get("整体返点占报价比例(%)", "0.00%"),
                "整体成本占报价比例": cost_summary.get("整体成本占报价比例(%)", "0.00%")
            }

            combined_summary["返点效益指标"] = {
                "平均返点比例": cost_summary.get("整体返点占报价比例(%)", "0.00%"),
                "总返点金额": cost_summary.get("总返点金额", 0.0),
                "平均成本": cost_summary.get("平均成本", 0.0),
                "优秀评级媒介数": cost_summary.get("优秀评级媒介数", 0)
            }

        # 如果有效果分析数据
        fixed_performance = self.analysis_results.get("fixed_media_performance")
        if fixed_performance is not None and not fixed_performance.empty:
            avg_cpm = fixed_performance['平均CPM'].mean() if '平均CPM' in fixed_performance.columns else 0
            avg_cpe = fixed_performance['平均CPE'].mean() if '平均CPE' in fixed_performance.columns else 0
            avg_interaction = fixed_performance['平均互动量'].mean() if '平均互动量' in fixed_performance.columns else 0

            combined_summary["效果表现指标"] = {
                "平均CPM": f"{avg_cpm:.2f}",
                "平均CPE": f"{avg_cpe:.2f}",
                "平均互动量": f"{avg_interaction:.0f}",
                "优秀效果媒介数": len(fixed_performance[fixed_performance[
                                                            '效果评估'] == '效果卓越']) if '效果评估' in fixed_performance.columns else 0
            }

        # 计算成本效益评级
        cost_score = self._calculate_cost_score(cost_summary)

        # 综合评级（成本模式主要看成本效益）
        if cost_score >= 90:
            combined_grade = "S级（成本效益卓越）"
        elif cost_score >= 80:
            combined_grade = "A级（成本效益优秀）"
        elif cost_score >= 70:
            combined_grade = "B级（成本效益良好）"
        elif cost_score >= 60:
            combined_grade = "C级（成本效益一般）"
        elif cost_score >= 50:
            combined_grade = "D级（成本效益较差）"
        else:
            combined_grade = "E级（成本效益很差）"

        combined_summary["综合评级"] = {
            "成本效益得分": round(cost_score, 2),
            "综合评级": combined_grade
        }

        return combined_summary

    def _generate_full_summary(self) -> Dict[str, Any]:
        """生成完整模式的综合汇总"""
        combined_summary = {
            "分析模式": "完整分析模式（工作量+质量+成本）",
            "数据概览": self.basic_stats,
            "工作量核心指标": {},
            "质量核心指标": {},
            "成本核心指标": {},
            "综合评级": {}
        }

        # 提取工作量核心指标
        workload_summary = self.analysis_results.get("workload_analysis", {}).get("summary", {})
        if workload_summary and "提示" not in workload_summary and "错误" not in workload_summary:
            combined_summary["工作量核心指标"] = {
                "媒介总数": workload_summary.get("媒介总数", 0),
                "总处理量": workload_summary.get("总处理量", 0),
                "总定档量": workload_summary.get("总定档量", 0),
                "整体定档率": workload_summary.get("整体定档率", "0.00%"),
                "S级媒介数": workload_summary.get("S级媒介数", 0)
            }

        # 提取质量核心指标
        quality_summary = self.analysis_results.get("quality_analysis", {}).get("summary", {})
        if quality_summary and "提示" not in quality_summary and "错误" not in quality_summary:
            combined_summary["质量核心指标"] = {
                "媒介总数": quality_summary.get("媒介总数", 0),
                "整体过筛率": quality_summary.get("overall_screening_rate", "0.00%"),
                "优秀质量媒介数": quality_summary.get("优秀质量媒介数", 0)
            }

        # 提取成本核心指标
        cost_summary = self.analysis_results.get("cost_analysis", {}).get("overall_summary", {})
        if cost_summary and "提示" not in cost_summary and "错误" not in cost_summary:
            combined_summary["成本核心指标"] = {
                "总媒介数": cost_summary.get("总媒介数", 0),
                "总成本": cost_summary.get("总成本", 0.0),
                "整体返点占报价比例": cost_summary.get("整体返点占报价比例(%)", "0.00%"),
                "整体成本占报价比例": cost_summary.get("整体成本占报价比例(%)", "0.00%"),
                "优秀评级媒介数": cost_summary.get("优秀评级媒介数", 0)
            }

        # 计算综合得分
        workload_score = self._calculate_workload_score(workload_summary)
        quality_score = self._calculate_quality_score(quality_summary)
        cost_score = self._calculate_cost_score(cost_summary)

        # 权重分配：工作量30%，质量30%，成本40%
        total_score = round(workload_score * 0.3 + quality_score * 0.3 + cost_score * 0.4, 2)

        # 综合评级
        if total_score >= 90:
            combined_grade = "S级（卓越）"
        elif total_score >= 80:
            combined_grade = "A级（优秀）"
        elif total_score >= 70:
            combined_grade = "B级（良好）"
        elif total_score >= 60:
            combined_grade = "C级（合格）"
        elif total_score >= 50:
            combined_grade = "D级（较差）"
        else:
            combined_grade = "E级（很差）"

        combined_summary["综合评级"] = {
            "工作量得分": round(workload_score, 2),
            "质量得分": round(quality_score, 2),
            "成本得分": round(cost_score, 2),
            "综合得分": total_score,
            "综合评级": combined_grade
        }

        return combined_summary

    def _calculate_workload_score(self, workload_summary: Dict) -> float:
        """计算工作量得分"""
        if not workload_summary or "提示" in workload_summary or "错误" in workload_summary:
            return 0.0

        score = 0.0

        # 定档率得分（0-50分）
        rate_str = workload_summary.get("整体定档率", "0.00%").replace("%", "")
        try:
            rate = float(rate_str)
            score += min(rate * 0.5, 50)  # 每1%得0.5分，最多50分
        except:
            pass

        # S级媒介占比得分（0-30分）
        total_media = workload_summary.get("媒介总数", 0)
        s_level_count = workload_summary.get("S级媒介数", 0)
        if total_media > 0:
            s_ratio = s_level_count / total_media * 100
            score += min(s_ratio * 0.3, 30)  # 每1%得0.3分，最多30分

        # 处理量得分（0-20分）
        total_processing = workload_summary.get("总处理量", 0)
        if total_processing >= 100:
            score += 20
        elif total_processing >= 50:
            score += 15
        elif total_processing >= 20:
            score += 10
        elif total_processing >= 10:
            score += 5

        return min(score, 100)

    def _calculate_quality_score(self, quality_summary: Dict) -> float:
        """计算质量得分"""
        if not quality_summary or "提示" in quality_summary or "错误" in quality_summary:
            return 0.0

        score = 0.0

        # 过筛率得分（0-50分）
        rate_str = quality_summary.get("overall_screening_rate", "0.00%").replace("%", "")
        try:
            rate = float(rate_str)
            score += min(rate * 0.5, 50)  # 每1%得0.5分，最多50分
        except:
            pass

        # 优秀媒介占比得分（0-30分）
        total_media = quality_summary.get("媒介总数", 0)
        excellent_count = quality_summary.get("优秀质量媒介数", 0)
        if total_media > 0:
            excellent_ratio = excellent_count / total_media * 100
            score += min(excellent_ratio * 0.3, 30)  # 每1%得0.3分，最多30分

        # 提报量得分（0-20分）
        total_reporting = quality_summary.get("总提报达人数", 0)
        if total_reporting >= 100:
            score += 20
        elif total_reporting >= 50:
            score += 15
        elif total_reporting >= 20:
            score += 10
        elif total_reporting >= 10:
            score += 5

        return min(score, 100)

    def _calculate_cost_score(self, cost_summary: Dict) -> float:
        """计算成本得分"""
        if not cost_summary or "提示" in cost_summary or "错误" in cost_summary:
            return 0.0

        score = 0.0

        # 返点比例得分（0-40分）
        rebate_rate_str = cost_summary.get("整体返点占报价比例(%)", "0.00%").replace("%", "")
        try:
            rebate_rate = float(rebate_rate_str)
            score += min(rebate_rate * 2, 40)  # 每1%得2分，最多40分
        except:
            pass

        # 成本控制得分（0-30分）
        cost_rate_str = cost_summary.get("整体成本占报价比例(%)", "100.00%").replace("%", "")
        try:
            cost_rate = float(cost_rate_str)
            # 成本占比越低越好，100-cost_rate得到正向分数
            cost_control = max(0, 100 - cost_rate)
            score += min(cost_control * 0.3, 30)  # 每降低1%得0.3分，最多30分
        except:
            pass

        # 优秀媒介占比得分（0-30分）
        total_media = cost_summary.get("总媒介数", 0)
        excellent_count = cost_summary.get("优秀评级媒介数", 0)
        if total_media > 0:
            excellent_ratio = excellent_count / total_media * 100
            score += min(excellent_ratio * 0.3, 30)  # 每1%得0.3分，最多30分

        return min(score, 100)

    def run_full_analysis(self, workload_top_n: int = 10, cost_top_n: int = 10,
                          use_original_state: bool = True) -> Dict[str, Any]:
        """
        运行完整分析（根据分析模式调整）
        :param workload_top_n: 工作量TOP排名数量
        :param cost_top_n: 成本效益TOP排名数量
        :param use_original_state: 质量分析是否使用原始状态
        :return: 完整分析结果
        """
        logger.info(f"启动完整媒介分析流程，模式: {self.analysis_mode}")

        try:
            if self.analysis_mode == 'media_analysis':
                # Media_Analysis模式：只运行工作量+质量分析
                self.run_workload_analysis(top_n=workload_top_n)
                self.run_quality_analysis(use_original_state=use_original_state)

            elif self.analysis_mode == 'cost_analysis':
                # 成本发挥分析模式：只运行成本分析
                self.run_cost_analysis(top_n=cost_top_n)

            else:  # full mode
                # 完整模式：运行所有分析
                self.run_workload_analysis(top_n=workload_top_n)
                self.run_quality_analysis(use_original_state=use_original_state)
                self.run_cost_analysis(top_n=cost_top_n)

            # 新增：强制更新 detailed_data 为处理后的数据
            self.analysis_results["detailed_data"] = self.processed_df  # 关键修复：传递原始数据

            # 生成综合汇总
            self.generate_combined_summary()

            logger.info("完整媒介分析流程执行完成")
            return self.analysis_results

        except Exception as e:
            logger.error(f"完整分析执行失败: {e}", exc_info=True)
            raise

    def get_workload_detail(self) -> pd.DataFrame:
        """获取工作量明细数据"""
        return self.analysis_results.get("media_workload_detail", pd.DataFrame())

    def get_quality_detail(self) -> pd.DataFrame:
        """获取质量明细数据"""
        return self.analysis_results.get("media_quality_detail", pd.DataFrame())

    def get_cost_detail(self) -> pd.DataFrame:
        """获取成本明细数据"""
        return self.analysis_results.get("cost_analysis", {}).get("media_detail", pd.DataFrame())

    def get_target_sheets(self) -> Dict[str, pd.DataFrame]:
        """获取成本发挥分析的所有目标工作表"""
        return {
            "媒介小组工作量分析": self.analysis_results.get("media_group_workload", pd.DataFrame()),
            "定档媒介工作量分析": self.analysis_results.get("fixed_media_workload", pd.DataFrame()),
            "定档媒介成本分析": self.analysis_results.get("fixed_media_cost", pd.DataFrame()),
            "定档媒介返点分析": self.analysis_results.get("fixed_media_rebate", pd.DataFrame()),
            "定档媒介效果分析": self.analysis_results.get("fixed_media_performance", pd.DataFrame()),
            "定档媒介达人量级分析": self.analysis_results.get("fixed_media_level", pd.DataFrame()),
            "定档媒介综合分析": self.analysis_results.get("fixed_media_comprehensive", pd.DataFrame()),
            "详细数据": self.analysis_results.get("detailed_data", pd.DataFrame())
        }