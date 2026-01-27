# src/quality_analyzer.py
"""
工作质量分析器 - 完全对齐Media_Analysis.py的工作质量分析逻辑
输出表格字段：定档媒介ID,对应名字,所属小组,总提报达人数,过筛人数,过筛率(%),未过筛人数,未过筛率(%),质量评估,评估说明,主要状态分布
新增：按influencer_purpose分类分析（优质达人/高阅读达人）

【重要】所有工作质量数据始终按小组排序（数码→家居→快消→其他），用于工作质量分析页面
仪表盘TOP10由前端或调用方另行处理排序
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from src.utils import (
    logger, normalize_media_name, get_media_group,
    calculate_percentage, format_number, ID_TO_NAME_MAPPING, FLOWER_TO_NAME_MAPPING
)


class QualityAnalyzer:
    def __init__(self, df: pd.DataFrame, known_id_name_mapping: Dict = None, config: Dict = None):
        """
        初始化工作质量分析器（Media_Analysis工作质量分析逻辑）
        :param df: 处理后的DataFrame（必须包含'数据类型'标记）
        :param known_id_name_mapping: ID-真名映射表
        :param config: 配置字典
        """
        self.df = df.copy()
        self.known_id_name_mapping = known_id_name_mapping or {}
        self.config = config or {}

        # 存储分析结果
        self.result = {
            "summary": {},  # 汇总信息
            "detail": None,  # 详细数据（对应Media_Analysis输出）
            "group_summary": None,  # 小组汇总
            "quality_distribution": None,  # 质量分布
            "premium_detail": None,  # 优质达人质量明细
            "high_read_detail": None,  # 高阅读达人质量明细
        }

        logger.info("工作质量分析器初始化完成（Media_Analysis逻辑）")

    def analyze(self, use_original_state: bool = True) -> Dict[str, Any]:
        """
        执行完整工作质量分析（Media_Analysis逻辑）
        :param use_original_state: 是否使用原始状态计算过筛率
        :return: 分析结果
        """
        logger.info(f"开始执行Media_Analysis工作质量分析")
        try:
            # 1. 验证数据
            self._validate_data()

            # 2. 提取提报数据（工作质量分析只处理提报数据）
            reporting_df = self._extract_reporting_data()

            if reporting_df.empty:
                logger.warning("无提报数据可供工作质量分析")
                self.result["summary"] = {"提示": "无有效提报数据进行工作质量分析"}
                return self.result

            # 3. 处理媒介信息
            media_info_df = self._process_media_info(reporting_df)

            # 4. 计算媒介质量明细（核心逻辑）- 始终按小组排序
            media_quality_detail = self._calculate_media_quality(media_info_df, use_original_state)

            # 5. 计算汇总信息
            self.result["summary"] = self._calculate_quality_summary(media_quality_detail)

            # 6. 存储明细数据（格式化输出）
            self.result["detail"] = self._format_quality_detail(media_quality_detail)

            # 7. 计算小组汇总
            self.result["group_summary"] = self._calculate_group_summary(media_quality_detail)

            # 8. 计算质量分布
            self.result["quality_distribution"] = self._calculate_quality_distribution(media_quality_detail)

            # 9. 【新增】按influencer_purpose分类分析 - 也按小组排序
            self.result["premium_detail"] = self._calculate_purpose_specific_quality(
                media_info_df, use_original_state, purpose_filter='优质达人'
            )
            self.result["high_read_detail"] = self._calculate_purpose_specific_quality(
                media_info_df, use_original_state, purpose_filter='高阅读达人'
            )

            logger.info("工作质量分析执行完成")
            return self.result

        except Exception as e:
            logger.error(f"工作质量分析执行失败: {e}", exc_info=True)
            raise

    def _validate_data(self) -> None:
        """验证数据是否包含必要字段"""
        required_fields = ['定档媒介', '状态', '数据类型']
        missing_fields = [f for f in required_fields if f not in self.df.columns]

        if missing_fields:
            raise ValueError(f"数据缺少必要字段: {missing_fields}")

        # 检查是否有提报数据
        if '数据类型' in self.df.columns:
            reporting_count = (self.df['数据类型'] == '提报').sum()
            if reporting_count == 0:
                logger.warning("数据中无'提报'类型数据")

    def _extract_reporting_data(self) -> pd.DataFrame:
        """提取提报数据"""
        logger.info("提取提报数据")

        if '数据类型' in self.df.columns:
            reporting_df = self.df[self.df['数据类型'] == '提报'].copy()
        else:
            # 如果没有数据类型标记，根据状态判断
            def is_reporting_status(status):
                if pd.isna(status):
                    return False
                status_str = str(status).upper()
                # 非定档状态视为提报
                return 'CHAIN_RETURNED' not in status_str and 'SCHEDULED' not in status_str

            status_col = self.df['状态'] if '状态' in self.df.columns else self.df.get('原始状态', '')
            reporting_mask = status_col.apply(is_reporting_status)
            reporting_df = self.df[reporting_mask].copy()
            reporting_df['数据类型'] = '提报'

        logger.info(f"提取到 {len(reporting_df)} 条提报数据")
        return reporting_df

    def _process_media_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理媒介信息（工作质量分析专用）
        :return: 包含媒介信息的DataFrame
        """
        logger.info("处理媒介信息（工作质量分析）")
        df_processed = df.copy()

        # 1. 确定定档媒介ID（对应Media_Analysis的'定档媒介ID'字段）
        # 优先使用submit_media_user_id作为ID
        if 'submit_media_user_id' in df_processed.columns:
            df_processed['submit_media_user_id'] = df_processed['submit_media_user_id'].astype(str).str.replace('.0',
                                                                                                                '',
                                                                                                                regex=False)
            df_processed['定档媒介ID'] = df_processed['submit_media_user_id']
        else:
            df_processed['定档媒介ID'] = ''

        # 2. 确定对应名字（对应Media_Analysis的'对应名字'字段）
        # 逻辑：工作质量分析只看submit_media_user_id（提报人员ID）对应的真名

        df_processed['对应名字'] = '未知'

        # 检查是否存在必要的列
        has_submit_id = 'submit_media_user_id' in df_processed.columns

        # 方法：优先使用submit_media_user_id
        if has_submit_id:
            # 清理ID（去除.0后缀）
            df_processed['submit_media_user_id'] = df_processed['submit_media_user_id'].astype(str).str.replace('.0',
                                                                                                                '',
                                                                                                                regex=False)

            for idx, row in df_processed.iterrows():
                submit_id = str(row.get('submit_media_user_id', '')).strip()
                if submit_id and submit_id.lower() not in ['', 'nan', 'none', '未知']:
                    # 先尝试known_id_name_mapping
                    if submit_id in self.known_id_name_mapping:
                        real_name = self.known_id_name_mapping[submit_id]
                    # 再尝试全局ID_TO_NAME_MAPPING
                    elif submit_id in ID_TO_NAME_MAPPING:
                        real_name = ID_TO_NAME_MAPPING[submit_id]
                    else:
                        real_name = '未知'

                    if real_name != '未知':
                        df_processed.at[idx, '对应名字'] = real_name

        # 方法2：对于仍然未知的，尝试使用submit_media_user_name
        if 'submit_media_user_name' in df_processed.columns:
            for idx, row in df_processed.iterrows():
                if df_processed.at[idx, '对应名字'] == '未知':
                    submit_name = str(row.get('submit_media_user_name', '')).strip()
                    if submit_name and submit_name.lower() not in ['', 'nan', 'none', '未知']:
                        real_name = normalize_media_name(submit_name)
                        if real_name != '未知':
                            df_processed.at[idx, '对应名字'] = real_name

        # 方法3：对于仍然未知的，作为最后的手段使用schedule_user_name
        if 'schedule_user_name' in df_processed.columns:
            for idx, row in df_processed.iterrows():
                if df_processed.at[idx, '对应名字'] == '未知':
                    schedule_name = str(row.get('schedule_user_name', '')).strip()
                    if schedule_name and schedule_name.lower() not in ['', 'nan', 'none', '未知']:
                        real_name = normalize_media_name(schedule_name)
                        if real_name != '未知':
                            df_processed.at[idx, '对应名字'] = real_name

        # 最后清理
        df_processed['对应名字'] = df_processed['对应名字'].replace(['', 'nan', 'NaN', 'None'], '未知')

        # 3. 确定所属小组
        df_processed['所属小组'] = df_processed['对应名字'].apply(get_media_group)

        # 4. 确保定档媒介字段（兼容成本分析）
        if '定档媒介' not in df_processed.columns:
            df_processed['定档媒介'] = df_processed['对应名字']

        logger.info(f"媒介信息处理完成，唯一媒介数: {df_processed['对应名字'].nunique()}")
        return df_processed

    def _calculate_media_quality(self, df: pd.DataFrame, use_original_state: bool) -> pd.DataFrame:
        """
        计算媒介工作质量（核心逻辑）
        :param df: 输入数据
        :param use_original_state: 是否使用原始状态
        :return: 质量明细DataFrame（始终按小组排序）
        """
        logger.info("计算媒介工作质量明细（始终按小组排序）")

        # 选择使用的状态字段
        if use_original_state and '原始状态' in df.columns:
            status_col = '原始状态'
        else:
            status_col = '状态'

        # 标准化状态字段
        df['分析状态'] = df[status_col].fillna('UNKNOWN').astype(str).str.upper()

        # 定义过筛状态（Media_Analysis逻辑）
        def is_screening_passed(status):
            status_upper = str(status).upper()
            # Media_Analysis过筛逻辑：包含以下状态视为过筛
            passed_keywords = ['SCREENING_PASSED', 'CHAIN_RETURNED', 'SCHEDULED']
            return any(keyword in status_upper for keyword in passed_keywords)

        df['是否过筛'] = df['分析状态'].apply(is_screening_passed)

        # 统计主要状态分布
        def get_main_status_distribution(statuses):
            if len(statuses) == 0:
                return "无状态数据"

            # 统计状态频次
            status_counts = {}
            for status in statuses:
                status_str = str(status).upper()
                # 简化状态
                if 'SCREENING_PASSED' in status_str:
                    key = '过筛通过'
                elif 'CHAIN_RETURNED' in status_str:
                    key = '已发布'
                elif 'SCHEDULED' in status_str:
                    key = '已排期'
                elif 'SCREENING_FAILED' in status_str:
                    key = '过筛失败'
                elif 'REJECTED' in status_str:
                    key = '已拒绝'
                else:
                    key = '其他'

                status_counts[key] = status_counts.get(key, 0) + 1

            # 格式化输出
            distribution_items = []
            for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(statuses) * 100) if len(statuses) > 0 else 0
                distribution_items.append(f"{status}:{count}({percentage:.1f}%)")

            return "; ".join(distribution_items[:3])  # 只显示前3个主要状态

        # 按媒介分组统计
        media_stats = df.groupby(['定档媒介ID', '对应名字', '所属小组']).agg(
            总提报达人数=('是否过筛', 'count'),
            过筛人数=('是否过筛', 'sum'),
            状态列表=('分析状态', list)
        ).reset_index()

        # 计算未过筛人数
        media_stats['未过筛人数'] = media_stats['总提报达人数'] - media_stats['过筛人数']

        # 计算过筛率
        media_stats['过筛率'] = np.where(
            media_stats['总提报达人数'] > 0,
            (media_stats['过筛人数'] / media_stats['总提报达人数'] * 100).round(2),
            0.0
        )
        media_stats['过筛率(%)'] = media_stats['过筛率'].astype(str) + "%"

        # 计算未过筛率
        media_stats['未过筛率'] = np.where(
            media_stats['总提报达人数'] > 0,
            (media_stats['未过筛人数'] / media_stats['总提报达人数'] * 100).round(2),
            0.0
        )
        media_stats['未过筛率(%)'] = media_stats['未过筛率'].astype(str) + "%"

        # 质量评估（只根据过筛率评定）
        def evaluate_quality(screening_rate):
            if screening_rate >= 80:
                return "优秀"
            elif screening_rate >= 70:
                return "良好"
            elif screening_rate >= 60:
                return "一般"
            elif screening_rate >= 50:
                return "及格"
            else:
                return "较差"

        media_stats['质量评估'] = media_stats['过筛率'].apply(evaluate_quality)

        # 评估说明
        def generate_evaluation_explanation(quality_grade, screening_rate, total_count):
            base_explanations = {
                "优秀": "过筛率≥80%，工作质量优异",
                "良好": "过筛率≥70%，工作质量良好",
                "一般": "过筛率≥60%，工作质量一般",
                "及格": "过筛率≥50%，工作质量及格",
                "较差": "过筛率<50%，工作质量需要改进"
            }

            explanation = base_explanations.get(quality_grade, "未知评估")
            details = f"过筛率:{screening_rate:.1f}%，提报量:{total_count}人"

            # 针对性建议
            if quality_grade in ["及格", "较差"]:
                if screening_rate < 50:
                    details += "（建议：提高达人筛选标准）"
                elif total_count < 5:
                    details += "（建议：增加提报数量）"

            return f"{explanation}。{details}"

        media_stats['评估说明'] = media_stats.apply(
            lambda row: generate_evaluation_explanation(
                row['质量评估'], row['过筛率'], row['总提报达人数']
            ), axis=1
        )

        # 主要状态分布
        media_stats['主要状态分布'] = df.groupby(['定档媒介ID', '对应名字', '所属小组'])['分析状态'].apply(
            lambda x: get_main_status_distribution(x.tolist())
        ).reset_index(drop=True)

        # ========== 核心修改：始终按小组排序 ==========
        logger.info("工作质量数据始终按小组排序：数码媒介组→家居媒介组→快消媒介组→其他组")

        # 定义小组排序映射
        group_order_mapping = {
            '数码媒介组': 1,
            '家居媒介组': 2,
            '快消媒介组': 3
        }

        # 定义质量评估排序映射（用于小组内排序）
        eval_order = {'优秀': 1, '良好': 2, '一般': 3, '及格': 4, '较差': 5}

        # 添加排序字段
        media_stats['小组排序'] = media_stats['所属小组'].map(
            lambda x: group_order_mapping.get(x, 99)  # 其他组排在最后
        )
        media_stats['质量评估排序'] = media_stats['质量评估'].map(eval_order)

        # 排序：先小组，再质量评估，再过筛率，最后提报量
        media_stats = media_stats.sort_values(
            ['小组排序', '质量评估排序', '过筛率', '总提报达人数'],
            ascending=[True, True, False, False]  # 小组升序，质量评估升序，过筛率降序，提报量降序
        )

        # 删除临时排序列
        media_stats = media_stats.drop(['小组排序', '质量评估排序'], axis=1)

        # 记录排序结果（调试用）
        logger.info("小组排序结果：")
        for group in ['数码媒介组', '家居媒介组', '快消媒介组']:
            group_count = media_stats[media_stats['所属小组'] == group].shape[0]
            if group_count > 0:
                logger.info(f"  {group}: {group_count} 个媒介")

        # 其他组
        other_groups = media_stats[~media_stats['所属小组'].isin(['数码媒介组', '家居媒介组', '快消媒介组'])]
        if not other_groups.empty:
            logger.info(f"  其他组: {other_groups.shape[0]} 个媒介")

        # 重置索引
        media_stats = media_stats.reset_index(drop=True)

        logger.info(f"媒介工作质量计算完成，共 {len(media_stats)} 个媒介，按小组排序")
        return media_stats

    def _calculate_purpose_specific_quality(self, df: pd.DataFrame, use_original_state: bool,
                                            purpose_filter: str) -> pd.DataFrame:
        """
        【新增】按influencer_purpose分类计算工作质量
        :param df: 原始数据
        :param use_original_state: 是否使用原始状态
        :param purpose_filter: influencer_purpose过滤值 ('优质达人'或'高阅读达人')
        :return: 分类质量明细DataFrame（始终按小组排序）
        """
        logger.info(f"计算 {purpose_filter} 工作质量明细（按小组排序）")

        # 过滤数据
        if 'influencer_purpose' in df.columns:
            purpose_df = df[df['influencer_purpose'] == purpose_filter].copy()
        else:
            # 如果没有influencer_purpose列，返回空DataFrame
            logger.warning(f"数据中无'influencer_purpose'列，无法进行{purpose_filter}分析")
            return pd.DataFrame()

        if purpose_df.empty:
            logger.warning(f"无{purpose_filter}数据")
            return pd.DataFrame()

        # 选择使用的状态字段
        if use_original_state and '原始状态' in purpose_df.columns:
            status_col = '原始状态'
        else:
            status_col = '状态'

        # 标准化状态字段
        purpose_df['分析状态'] = purpose_df[status_col].fillna('UNKNOWN').astype(str).str.upper()

        # 定义过筛状态（Media_Analysis逻辑）
        def is_screening_passed(status):
            status_upper = str(status).upper()
            passed_keywords = ['SCREENING_PASSED', 'CHAIN_RETURNED', 'SCHEDULED']
            return any(keyword in status_upper for keyword in passed_keywords)

        purpose_df['是否过筛'] = purpose_df['分析状态'].apply(is_screening_passed)

        # 按媒介分组统计
        purpose_stats = purpose_df.groupby(['定档媒介ID', '对应名字', '所属小组']).agg(
            总提报达人数=('是否过筛', 'count'),
            过筛人数=('是否过筛', 'sum'),
            状态列表=('分析状态', list)
        ).reset_index()

        # 计算未过筛人数
        purpose_stats['未过筛人数'] = purpose_stats['总提报达人数'] - purpose_stats['过筛人数']

        # 计算过筛率
        purpose_stats['过筛率'] = np.where(
            purpose_stats['总提报达人数'] > 0,
            (purpose_stats['过筛人数'] / purpose_stats['总提报达人数'] * 100).round(2),
            0.0
        )
        purpose_stats['过筛率(%)'] = purpose_stats['过筛率'].astype(str) + "%"

        # 计算未过筛率
        purpose_stats['未过筛率'] = np.where(
            purpose_stats['总提报达人数'] > 0,
            (purpose_stats['未过筛人数'] / purpose_stats['总提报达人数'] * 100).round(2),
            0.0
        )
        purpose_stats['未过筛率(%)'] = purpose_stats['未过筛率'].astype(str) + "%"

        # 质量评估（只根据过筛率评定）
        def evaluate_quality(screening_rate):
            if screening_rate >= 80:
                return "优秀"
            elif screening_rate >= 70:
                return "良好"
            elif screening_rate >= 60:
                return "一般"
            elif screening_rate >= 50:
                return "及格"
            else:
                return "较差"

        purpose_stats['质量评估'] = purpose_stats['过筛率'].apply(evaluate_quality)

        # 评估说明
        def generate_evaluation_explanation(quality_grade, screening_rate, total_count, purpose):
            base_explanations = {
                "优秀": f"{purpose}过筛率≥80%，工作质量优异",
                "良好": f"{purpose}过筛率≥70%，工作质量良好",
                "一般": f"{purpose}过筛率≥60%，工作质量一般",
                "及格": f"{purpose}过筛率≥50%，工作质量及格",
                "较差": f"{purpose}过筛率<50%，工作质量需要改进"
            }

            explanation = base_explanations.get(quality_grade, "未知评估")
            details = f"过筛率:{screening_rate:.1f}%，提报量:{total_count}人"

            # 针对性建议
            if quality_grade in ["及格", "较差"]:
                if screening_rate < 50:
                    details += f"（建议：提高{purpose}筛选标准）"
                elif total_count < 3:
                    details += f"（建议：增加{purpose}提报数量）"

            return f"{explanation}。{details}"

        purpose_stats['评估说明'] = purpose_stats.apply(
            lambda row: generate_evaluation_explanation(
                row['质量评估'], row['过筛率'], row['总提报达人数'], purpose_filter
            ), axis=1
        )

        # 主要状态分布
        def get_main_status_distribution(statuses):
            if len(statuses) == 0:
                return "无状态数据"

            # 统计状态频次
            status_counts = {}
            for status in statuses:
                status_str = str(status).upper()
                if 'SCREENING_PASSED' in status_str:
                    key = '过筛通过'
                elif 'CHAIN_RETURNED' in status_str:
                    key = '已发布'
                elif 'SCHEDULED' in status_str:
                    key = '已排期'
                elif 'SCREENING_FAILED' in status_str:
                    key = '过筛失败'
                elif 'REJECTED' in status_str:
                    key = '已拒绝'
                else:
                    key = '其他'

                status_counts[key] = status_counts.get(key, 0) + 1

            # 格式化输出
            distribution_items = []
            for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(statuses) * 100) if len(statuses) > 0 else 0
                distribution_items.append(f"{status}:{count}({percentage:.1f}%)")

            return "; ".join(distribution_items[:3])

        purpose_stats['主要状态分布'] = purpose_df.groupby(['定档媒介ID', '对应名字', '所属小组'])['分析状态'].apply(
            lambda x: get_main_status_distribution(x.tolist())
        ).reset_index(drop=True)

        # ========== 核心修改：始终按小组排序 ==========
        logger.info(f"{purpose_filter} 数据始终按小组排序：数码媒介组→家居媒介组→快消媒介组→其他组")

        # 定义小组排序映射
        group_order_mapping = {
            '数码媒介组': 1,
            '家居媒介组': 2,
            '快消媒介组': 3
        }

        # 定义质量评估排序映射（用于小组内排序）
        eval_order = {'优秀': 1, '良好': 2, '一般': 3, '及格': 4, '较差': 5}

        # 添加排序字段
        purpose_stats['小组排序'] = purpose_stats['所属小组'].map(
            lambda x: group_order_mapping.get(x, 99)  # 其他组排在最后
        )
        purpose_stats['质量评估排序'] = purpose_stats['质量评估'].map(eval_order)

        # 排序：先小组，再质量评估，再过筛率，最后提报量
        purpose_stats = purpose_stats.sort_values(
            ['小组排序', '质量评估排序', '过筛率', '总提报达人数'],
            ascending=[True, True, False, False]  # 小组升序，质量评估升序，过筛率降序，提报量降序
        )

        # 删除临时排序列
        purpose_stats = purpose_stats.drop(['小组排序', '质量评估排序'], axis=1)

        # 重置索引
        purpose_stats = purpose_stats.reset_index(drop=True)

        # 添加分类标识列
        purpose_stats['达人类型'] = purpose_filter

        logger.info(f"{purpose_filter} 工作质量计算完成，共 {len(purpose_stats)} 个媒介，按小组排序")
        return purpose_stats

    def _format_quality_detail(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        格式化工作质量明细输出（确保字段顺序和格式）
        """
        required_columns = [
            '定档媒介ID', '对应名字', '所属小组', '总提报达人数',
            '过筛人数', '过筛率(%)', '未过筛人数', '未过筛率(%)',
            '质量评估', '评估说明', '主要状态分布'
        ]

        # 确保所有列都存在
        for col in required_columns:
            if col not in df.columns:
                if col in ['过筛率(%)', '未过筛率(%)']:
                    df[col] = '0.00%'
                elif col in ['质量评估', '评估说明']:
                    df[col] = '未知'
                elif col == '主要状态分布':
                    df[col] = '无数据'
                else:
                    df[col] = 0

        # 按指定顺序排列
        df = df[required_columns].copy()

        return df

    def _format_purpose_detail(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        格式化分类工作质量明细输出
        """
        if df.empty:
            return df

        required_columns = [
            '定档媒介ID', '对应名字', '所属小组', '达人类型', '总提报达人数',
            '过筛人数', '过筛率(%)', '未过筛人数', '未过筛率(%)',
            '质量评估', '评估说明', '主要状态分布'
        ]

        # 确保所有列都存在
        for col in required_columns:
            if col not in df.columns:
                if col in ['过筛率(%)', '未过筛率(%)']:
                    df[col] = '0.00%'
                elif col in ['质量评估', '评估说明', '达人类型']:
                    df[col] = '未知'
                elif col == '主要状态分布':
                    df[col] = '无数据'
                else:
                    df[col] = 0

        # 按指定顺序排列
        df = df[required_columns].copy()

        return df

    def _calculate_quality_summary(self, detail_df: pd.DataFrame) -> Dict[str, Any]:
        """计算工作质量分析汇总信息"""
        logger.info("计算工作质量分析汇总信息")

        if detail_df.empty:
            return {"提示": "无工作质量数据"}

        summary = {}

        # 基础统计
        summary['媒介总数'] = len(detail_df)
        summary['总提报达人数'] = int(detail_df['总提报达人数'].sum())
        summary['总过筛人数'] = int(detail_df['过筛人数'].sum())
        summary['总未过筛人数'] = int(detail_df['未过筛人数'].sum())

        # 整体过筛率
        if summary['总提报达人数'] > 0:
            overall_rate = (summary['总过筛人数'] / summary['总提报达人数'] * 100)
            summary['overall_screening_rate'] = f"{overall_rate:.2f}%"
            summary['总体过筛率(%)'] = round(overall_rate, 2)  # 兼容字段
        else:
            summary['overall_screening_rate'] = "0.00%"
            summary['总体过筛率(%)'] = 0.0

        # 质量评估统计
        if '质量评估' in detail_df.columns:
            quality_counts = detail_df['质量评估'].value_counts().to_dict()
            summary['质量评估分布'] = quality_counts

            summary['优秀质量媒介数'] = int(quality_counts.get('优秀', 0))
            summary['良好质量媒介数'] = int(quality_counts.get('良好', 0))
            summary['一般质量媒介数'] = int(quality_counts.get('一般', 0))
            summary['及格质量媒介数'] = int(quality_counts.get('及格', 0))
            summary['较差质量媒介数'] = int(quality_counts.get('较差', 0))

            summary['优秀质量媒介占比'] = f"{(summary['优秀质量媒介数'] / summary['媒介总数'] * 100):.2f}%"
            summary[
                '良好及以上质量媒介占比'] = f"{((summary['优秀质量媒介数'] + summary['良好质量媒介数']) / summary['媒介总数'] * 100):.2f}%"

        # 平均指标
        if summary['媒介总数'] > 0:
            summary['平均提报达人数'] = round(summary['总提报达人数'] / summary['媒介总数'], 1)
            summary['平均过筛人数'] = round(summary['总过筛人数'] / summary['媒介总数'], 1)

        # 小组分布（前5个）
        if '所属小组' in detail_df.columns:
            group_dist = detail_df.groupby('所属小组')['总提报达人数'].sum().sort_values(ascending=False).head(
                5).to_dict()
            summary['主要小组提报分布'] = group_dist

        logger.info(f"工作质量汇总计算完成，总媒介数: {summary['媒介总数']}")
        return summary

    def _calculate_group_summary(self, detail_df: pd.DataFrame) -> pd.DataFrame:
        """计算小组工作质量汇总"""
        logger.info("计算小组工作质量汇总")

        if detail_df.empty or '所属小组' not in detail_df.columns:
            return pd.DataFrame()

        group_summary = detail_df.groupby('所属小组').agg(
            媒介数量=('对应名字', 'nunique'),
            总提报达人数=('总提报达人数', 'sum'),
            总过筛人数=('过筛人数', 'sum'),
            总未过筛人数=('未过筛人数', 'sum'),
            优秀媒介数=('质量评估', lambda x: (x == '优秀').sum()),
            良好媒介数=('质量评估', lambda x: (x == '良好').sum())
        ).reset_index()

        # 计算小组指标
        group_summary['小组过筛率'] = np.where(
            group_summary['总提报达人数'] > 0,
            (group_summary['总过筛人数'] / group_summary['总提报达人数'] * 100).round(2),
            0.0
        )
        group_summary['小组过筛率(%)'] = group_summary['小组过筛率'].astype(str) + "%"

        group_summary['小组未过筛率'] = np.where(
            group_summary['总提报达人数'] > 0,
            (group_summary['总未过筛人数'] / group_summary['总提报达人数'] * 100).round(2),
            0.0
        )
        group_summary['小组未过筛率(%)'] = group_summary['小组未过筛率'].astype(str) + "%"

        # 计算优秀良好占比
        group_summary['优秀良好媒介数'] = group_summary['优秀媒介数'] + group_summary['良好媒介数']
        group_summary['优秀良好占比(%)'] = np.where(
            group_summary['媒介数量'] > 0,
            (group_summary['优秀良好媒介数'] / group_summary['媒介数量'] * 100).round(2),
            0.0
        )
        group_summary['优秀良好占比(%)'] = group_summary['优秀良好占比(%)'].astype(str) + "%"

        # 计算占比
        total_reporting = group_summary['总提报达人数'].sum()
        group_summary['提报量占比(%)'] = np.where(
            total_reporting > 0,
            (group_summary['总提报达人数'] / total_reporting * 100).round(2),
            0.0
        )
        group_summary['提报量占比(%)'] = group_summary['提报量占比(%)'].astype(str) + "%"

        # 按照指定小组顺序排序（数码媒介组→家居媒介组→快消媒介组→其他组）
        group_order_mapping = {
            '数码媒介组': 1,
            '家居媒介组': 2,
            '快消媒介组': 3
        }
        group_summary['小组排序'] = group_summary['所属小组'].map(
            lambda x: group_order_mapping.get(x, 99)
        )
        group_summary = group_summary.sort_values('小组排序', ascending=True)
        group_summary = group_summary.drop('小组排序', axis=1)

        # 重新排列列顺序
        column_order = [
            '所属小组', '媒介数量', '总提报达人数', '提报量占比(%)',
            '总过筛人数', '总未过筛人数', '小组过筛率(%)', '小组未过筛率(%)',
            '优秀媒介数', '良好媒介数', '优秀良好占比(%)'
        ]

        existing_columns = [col for col in column_order if col in group_summary.columns]
        group_summary = group_summary[existing_columns]

        logger.info(f"小组汇总计算完成，共 {len(group_summary)} 个小组")
        logger.info(f"小组汇总排序：{group_summary['所属小组'].tolist()}")
        return group_summary

    def _calculate_quality_distribution(self, detail_df: pd.DataFrame) -> pd.DataFrame:
        """计算质量评估分布"""
        logger.info("计算质量评估分布")

        if detail_df.empty or '质量评估' not in detail_df.columns:
            return pd.DataFrame()

        # 统计质量评估分布
        quality_dist = detail_df['质量评估'].value_counts().reset_index()
        quality_dist.columns = ['质量等级', '媒介数量']

        # 计算占比
        total_media = quality_dist['媒介数量'].sum()
        quality_dist['占比'] = np.where(
            total_media > 0,
            (quality_dist['媒介数量'] / total_media * 100).round(2),
            0.0
        )
        quality_dist['占比(%)'] = quality_dist['占比'].astype(str) + "%"

        # 按等级排序
        order = {'优秀': 1, '良好': 2, '一般': 3, '及格': 4, '较差': 5}
        quality_dist['排序'] = quality_dist['质量等级'].map(order)
        quality_dist = quality_dist.sort_values('排序')
        quality_dist = quality_dist.drop('排序', axis=1)

        # 重新排列列
        quality_dist = quality_dist[['质量等级', '媒介数量', '占比(%)']]

        logger.info(f"质量分布计算完成，共 {len(quality_dist)} 个等级")
        return quality_dist