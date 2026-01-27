# src/workload_analyzer.py - 修正版
"""
工作量分析器 - 完全对齐Media_Analysis.py的工作量分析逻辑
输出表格字段：媒介姓名,对应真名,所属小组,总处理量,定档量,已发布数,未发布数,其他状态数,定档率(%),定档率评估,产量评估,综合评估
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from src.utils import (
    logger, normalize_media_name, get_media_group,
    calculate_percentage, format_number, ID_TO_NAME_MAPPING
)


class WorkloadAnalyzer:
    def __init__(self, df: pd.DataFrame, known_id_name_mapping: Dict = None, config: Dict = None):
        """
        初始化工作量分析器（Media_Analysis工作量分析逻辑）
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
            "top_media_ranking": None,  # TOP排名
        }

        logger.info("工作量分析器初始化完成（Media_Analysis逻辑）")

    def analyze(self, top_n: int = 10) -> Dict[str, Any]:
        """
        执行完整工作量分析（Media_Analysis逻辑）
        :param top_n: TOP媒介排名数量
        :return: 分析结果
        """
        logger.info("开始执行Media_Analysis工作量分析")
        try:
            # 1. 验证数据
            self._validate_data()

            # 2. 提取定档数据（工作量分析只处理定档数据）
            scheduling_df = self._extract_scheduling_data()

            if scheduling_df.empty:
                logger.warning("无定档数据可供工作量分析")
                self.result["summary"] = {"提示": "无有效定档数据进行工作量分析"}
                return self.result

            # 3. 处理媒介信息（Media_Analysis逻辑）
            media_info_df = self._process_media_info(scheduling_df)

            # 4. 计算媒介工作量明细（核心逻辑）
            media_workload_detail = self._calculate_media_workload(media_info_df)

            # 5. 计算汇总信息
            self.result["summary"] = self._calculate_workload_summary(media_workload_detail)

            # 6. 存储明细数据（格式化输出）
            self.result["detail"] = self._format_workload_detail(media_workload_detail)

            # 7. 计算小组汇总
            self.result["group_summary"] = self._calculate_group_summary(media_workload_detail)

            # 8. 生成TOP排名
            self.result["top_media_ranking"] = self._generate_top_ranking(media_workload_detail, top_n)

            logger.info("工作量分析执行完成")
            return self.result

        except Exception as e:
            logger.error(f"工作量分析执行失败: {e}", exc_info=True)
            raise

    def _validate_data(self) -> None:
        """验证数据是否包含必要字段"""
        # 兼容字段名映射
        self._standardize_column_names()

        required_fields = ['定档媒介', '状态', '数据类型']
        missing_fields = [f for f in required_fields if f not in self.df.columns]

        if missing_fields:
            # 尝试使用替代字段
            field_alternatives = {
                '定档媒介': ['schedule_user_name', 'submit_media_user_name'],
                '状态': ['state', 'system_status', 'status'],
                '数据类型': []  # 如果没有数据类型字段，后面会根据状态判断
            }

            for field in required_fields:
                if field not in self.df.columns:
                    for alt in field_alternatives.get(field, []):
                        if alt in self.df.columns:
                            self.df[field] = self.df[alt]
                            break

            # 重新检查缺失字段
            missing_fields = [f for f in required_fields if f not in self.df.columns]
            if missing_fields:
                raise ValueError(f"数据缺少必要字段: {missing_fields}")

        # 检查是否有定档数据
        if '数据类型' in self.df.columns:
            scheduling_count = (self.df['数据类型'] == '定档').sum()
            if scheduling_count == 0:
                logger.warning("数据中无'定档'类型数据")

    def _standardize_column_names(self):
        """标准化列名，确保后续逻辑能够正常运行"""
        # 状态字段映射
        if 'state' in self.df.columns and '状态' not in self.df.columns:
            self.df['状态'] = self.df['state']
            logger.info("已将'state'字段映射为'状态'")

        # 系统状态字段映射
        if 'system_status' in self.df.columns and '状态' not in self.df.columns:
            self.df['状态'] = self.df['system_status']
            logger.info("已将'system_status'字段映射为'状态'")

        # 定档媒介字段映射
        if 'schedule_user_name' in self.df.columns and '定档媒介' not in self.df.columns:
            self.df['定档媒介'] = self.df['schedule_user_name']
            logger.info("已将'schedule_user_name'字段映射为'定档媒介'")

        # 确保数据类型字段存在
        if '数据类型' not in self.df.columns:
            # 根据状态判断数据类型
            def determine_data_type(status):
                if pd.isna(status):
                    return '其他'
                status_str = str(status).upper()
                if 'CHAIN_RETURNED' in status_str or 'SCHEDULED' in status_str or '已发布' in status_str or '未发布' in status_str:
                    return '定档'
                else:
                    return '其他'

            status_col = self.df['状态'] if '状态' in self.df.columns else self.df.get('state', '')
            self.df['数据类型'] = status_col.apply(determine_data_type)
            logger.info(f"已根据状态字段创建'数据类型'字段，定档数据: {(self.df['数据类型'] == '定档').sum()}条")

    def _extract_scheduling_data(self) -> pd.DataFrame:
        """提取定档数据"""
        logger.info("提取定档数据")
        logger.info(f"原始数据行数: {len(self.df)}")
        logger.info(f"原始数据列名: {list(self.df.columns)}")

        # 检查数据类型字段
        if '数据类型' in self.df.columns:
            logger.info(f"'数据类型'字段分布: {self.df['数据类型'].value_counts().to_dict()}")
            scheduling_df = self.df[self.df['数据类型'] == '定档'].copy()
            logger.info(f"通过'数据类型'提取到 {len(scheduling_df)} 条定档数据")
        else:
            logger.info("没有'数据类型'字段，尝试根据状态判断")

            # 如果没有数据类型标记，根据状态判断
            def is_scheduling_status(status):
                if pd.isna(status):
                    return False
                status_str = str(status).upper()
                result = ('CHAIN_RETURNED' in status_str or 'SCHEDULED' in status_str
                          or '已发布' in status_str or '未发布' in status_str)
                return result

            # 尝试多个可能的状态字段
            status_col_names = ['状态', 'state', 'system_status', 'status']
            status_col = None

            for col in status_col_names:
                if col in self.df.columns:
                    status_col = self.df[col]
                    logger.info(f"使用'{col}'字段作为状态字段")
                    logger.info(f"状态字段分布: {status_col.value_counts().to_dict()}")
                    break

            if status_col is None:
                logger.error("未找到任何状态字段")
                # 如果都没有，尝试寻找包含'status'的列
                for col in self.df.columns:
                    if 'status' in col.lower():
                        status_col = self.df[col]
                        logger.info(f"使用'{col}'字段作为状态字段")
                        logger.info(f"状态字段分布: {status_col.value_counts().to_dict()}")
                        break

            if status_col is None:
                logger.error("未找到状态字段，无法提取定档数据")
                return pd.DataFrame()

            # 测试状态判断
            logger.info("测试状态判断:")
            for val in status_col.unique()[:5]:
                logger.info(f"  状态值 '{val}' -> 是否定档: {is_scheduling_status(val)}")

            scheduling_mask = status_col.apply(is_scheduling_status)
            logger.info(f"定档数据筛选结果: {scheduling_mask.sum()} 条为True")

            scheduling_df = self.df[scheduling_mask].copy()
            scheduling_df['数据类型'] = '定档'

            logger.info(f"提取到 {len(scheduling_df)} 条定档数据")

            if '状态' not in scheduling_df.columns and status_col is not None:
                scheduling_df['状态'] = status_col[scheduling_mask]
                logger.info("已添加'状态'字段到提取的数据")

        logger.info(f"最终提取到的定档数据行数: {len(scheduling_df)}")
        if len(scheduling_df) > 0:
            logger.info(
                f"提取数据中的状态分布: {scheduling_df['状态'].value_counts().to_dict() if '状态' in scheduling_df.columns else '无状态字段'}")

        return scheduling_df

    def _process_media_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理媒介信息（Media_Analysis逻辑）
        :return: 包含媒介信息的DataFrame
        """
        logger.info("处理媒介信息")
        df_processed = df.copy()

        # 1. 确定媒介姓名（对应Media_Analysis的'媒介姓名'字段）
        # 逻辑：先看schedule_user_name（定档人员）对应的真名
        #       如果schedule_user_name是空的，再看submit_media_user_id（提报人员ID）对应的真名

        # 初始化媒介姓名列
        df_processed['媒介姓名'] = '未知'
        df_processed['对应真名'] = '未知'

        # 首先检查是否包含必要的列
        has_schedule_name = 'schedule_user_name' in df_processed.columns
        has_submit_id = 'submit_media_user_id' in df_processed.columns

        if not has_schedule_name and not has_submit_id:
            logger.warning("数据中既无schedule_user_name也无submit_media_user_id列")
            # 尝试使用其他字段
            if 'submit_media_user_name' in df_processed.columns:
                df_processed['媒介姓名'] = df_processed['submit_media_user_name'].apply(normalize_media_name)
            else:
                df_processed['媒介姓名'] = '未知'
        else:
            # 逐行处理媒介名称
            for idx, row in df_processed.iterrows():
                real_name = '未知'

                # 方法1：优先使用schedule_user_name
                if has_schedule_name:
                    schedule_name = str(row.get('schedule_user_name', '')).strip()
                    if schedule_name and schedule_name.lower() not in ['', 'nan', 'none', '未知']:
                        # 使用normalize_media_name函数获取真名
                        real_name = normalize_media_name(schedule_name)
                        if real_name != '未知':
                            df_processed.at[idx, '媒介姓名'] = real_name
                            df_processed.at[idx, '对应真名'] = real_name

                # 方法2：对于仍然未知的，使用submit_media_user_id
                if real_name == '未知' and has_submit_id:
                    submit_id = str(row.get('submit_media_user_id', '')).strip()
                    if submit_id and submit_id.lower() not in ['', 'nan', 'none', '未知']:
                        # 清理ID（去除.0后缀）
                        submit_id = submit_id.replace('.0', '')

                        # 先尝试known_id_name_mapping
                        if submit_id in self.known_id_name_mapping:
                            real_name = self.known_id_name_mapping[submit_id]
                        # 再尝试全局ID_TO_NAME_MAPPING
                        elif submit_id in ID_TO_NAME_MAPPING:
                            real_name = ID_TO_NAME_MAPPING[submit_id]

                        if real_name != '未知':
                            df_processed.at[idx, '媒介姓名'] = real_name
                            df_processed.at[idx, '对应真名'] = real_name

                # 方法3：对于仍然未知的，尝试使用submit_media_user_name
                if real_name == '未知' and 'submit_media_user_name' in df_processed.columns:
                    submit_name = str(row.get('submit_media_user_name', '')).strip()
                    if submit_name and submit_name.lower() not in ['', 'nan', 'none', '未知']:
                        real_name = normalize_media_name(submit_name)
                        if real_name != '未知':
                            df_processed.at[idx, '媒介姓名'] = real_name
                            df_processed.at[idx, '对应真名'] = real_name

        # 最后清理
        df_processed['媒介姓名'] = df_processed['媒介姓名'].replace(['', 'nan', 'NaN', 'None', 'null'], '未知')
        df_processed['对应真名'] = df_processed['对应真名'].replace(['', 'nan', 'NaN', 'None', 'null'], '未知')

        # 4. 确定所属小组
        df_processed['所属小组'] = df_processed['媒介姓名'].apply(get_media_group)

        # 5. 确保定档媒介字段（兼容成本分析）
        if '定档媒介' not in df_processed.columns:
            df_processed['定档媒介'] = df_processed['媒介姓名']

        # 记录一些调试信息
        unique_names = df_processed['媒介姓名'].unique()
        logger.info(f"媒介信息处理完成，唯一媒介数: {df_processed['媒介姓名'].nunique()}")
        logger.info(f"前10个媒介名称: {unique_names[:10] if len(unique_names) > 10 else unique_names}")
        logger.info(f"未知媒介数量: {(df_processed['媒介姓名'] == '未知').sum()}")

        return df_processed

    def _calculate_media_workload(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算媒介工作量（核心逻辑）
        输出字段：媒介姓名,对应真名,所属小组,总处理量,定档量,已发布数,未发布数,其他状态数,定档率(%),定档率评估,产量评估,综合评估
        """
        logger.info("计算媒介工作量明细")

        # 打印详细的调试信息
        logger.info(f"=== 开始计算媒介工作量 ===")
        logger.info(f"输入数据行数: {len(df)}")

        if '状态' in df.columns:
            logger.info(f"状态字段值分布: {df['状态'].value_counts().to_dict()}")
            logger.info(f"状态字段示例值: {df['状态'].head().tolist()}")

        # 确保状态字段存在
        if '状态' not in df.columns:
            logger.error("无状态字段可用，无法计算工作量")
            # 创建空结果
            return pd.DataFrame(columns=['媒介姓名', '对应真名', '所属小组', '总处理量', '定档量',
                                         '已发布数', '未发布数', '其他状态数',
                                         '定档率(%)', '定档率评估', '产量评估', '综合评估'])

        # 标准化状态字段 - 兼容中文和英文状态
        df['状态'] = df['状态'].fillna('UNKNOWN').astype(str)
        logger.info(f"标准化后状态分布: {df['状态'].value_counts().to_dict()}")

        # 定义状态分类 - 兼容中文和英文，并映射为中文状态
        def classify_status(status):
            status_str = str(status).upper()

            # 中文状态识别
            if '已发布' in status_str or 'CHAIN_RETURNED' in status_str:
                return '已发布'
            elif '未发布' in status_str or 'SCHEDULED' in status_str:
                return '未发布'
            else:
                return '其他'

        df['状态分类'] = df['状态'].apply(classify_status)

        # 打印状态分类结果
        logger.info(f"状态分类分布: {df['状态分类'].value_counts().to_dict()}")
        logger.info(f"状态分类示例 (前5行):")
        for idx, row in df.head().iterrows():
            logger.info(f"  原始状态: {row['状态']}, 分类: {row['状态分类']}")

        # 按媒介分组统计
        logger.info("开始按媒介分组统计...")

        # 确保所有分组列都存在
        required_group_cols = ['媒介姓名', '对应真名', '所属小组']
        for col in required_group_cols:
            if col not in df.columns:
                logger.error(f"缺少必需的分组列: {col}")
                df[col] = '未知'

        # 分组统计
        try:
            media_stats = df.groupby(required_group_cols).agg(
                总处理量=('状态分类', 'count'),
                已发布数=('状态分类', lambda x: (x == '已发布').sum()),
                未发布数=('状态分类', lambda x: (x == '未发布').sum()),
                其他状态数=('状态分类', lambda x: (x == '其他').sum())
            ).reset_index()

            logger.info(f"分组统计成功，共 {len(media_stats)} 个媒介")

        except Exception as e:
            logger.error(f"分组统计失败: {e}")
            # 创建空结果
            return pd.DataFrame(columns=['媒介姓名', '对应真名', '所属小组', '总处理量', '定档量',
                                         '已发布数', '未发布数', '其他状态数',
                                         '定档率(%)', '定档率评估', '产量评估', '综合评估'])

        if len(media_stats) > 0:
            logger.info(f"媒介统计结果 (前10个):")
            for idx, row in media_stats.head(10).iterrows():
                logger.info(f"  媒介: {row['媒介姓名']}, 总处理量: {row['总处理量']}, "
                            f"已发布: {row['已发布数']}, "
                            f"未发布: {row['未发布数']}, "
                            f"其他: {row['其他状态数']}")

            # 打印统计汇总
            logger.info(f"统计汇总:")
            logger.info(f"  总媒介数: {len(media_stats)}")
            logger.info(f"  总处理量: {media_stats['总处理量'].sum()}")
            logger.info(f"  总已发布数: {media_stats['已发布数'].sum()}")
            logger.info(f"  总未发布数: {media_stats['未发布数'].sum()}")
            logger.info(f"  总其他状态数: {media_stats['其他状态数'].sum()}")

        # 计算定档量（已发布 + 未发布）
        media_stats['定档量'] = media_stats['已发布数'] + media_stats['未发布数']

        # 计算定档率
        media_stats['定档率'] = np.where(
            media_stats['总处理量'] > 0,
            (media_stats['定档量'] / media_stats['总处理量'] * 100).round(2),
            0.0
        )
        media_stats['定档率(%)'] = media_stats['定档率'].astype(str) + "%"

        # 定档率评估（Media_Analysis逻辑）
        def evaluate_scheduling_rate(rate):
            if rate >= 80:
                return "优秀"
            elif rate >= 60:
                return "良好"
            elif rate >= 40:
                return "一般"
            elif rate >= 20:
                return "待改进"
            else:
                return "较差"

        media_stats['定档率评估'] = media_stats['定档率'].apply(evaluate_scheduling_rate)

        # 产量评估（Media_Analysis逻辑）
        def evaluate_output_volume(total_volume, scheduling_volume):
            if total_volume >= 50 and scheduling_volume >= 40:
                return "高产"
            elif total_volume >= 30 and scheduling_volume >= 20:
                return "中产"
            elif total_volume >= 10 and scheduling_volume >= 5:
                return "低产"
            else:
                return "微量"

        media_stats['产量评估'] = media_stats.apply(
            lambda row: evaluate_output_volume(row['总处理量'], row['定档量']), axis=1
        )

        # 综合评估（结合定档率和产量）
        def evaluate_comprehensive(rate_eval, output_eval):
            if rate_eval == "优秀" and output_eval in ["高产", "中产"]:
                return "S级"
            elif rate_eval in ["优秀", "良好"] and output_eval in ["高产", "中产", "低产"]:
                return "A级"
            elif rate_eval == "一般" or output_eval == "微量":
                return "B级"
            else:
                return "C级"

        media_stats['综合评估'] = media_stats.apply(
            lambda row: evaluate_comprehensive(row['定档率评估'], row['产量评估']), axis=1
        )

        # 按小组顺序排序（数码媒介组 -> 家居媒介组 -> 快消媒介组 -> 其他）
        # 注意：这个排序只用于明细展示，便于按小组查看数据，不影响TOP排名
        group_order = {'数码媒介组': 1, '家居媒介组': 2, '快消媒介组': 3}

        # 为每个小组分配排序值
        media_stats['小组排序'] = media_stats['所属小组'].apply(
            lambda x: group_order.get(x, 999)  # 其他小组排在最后
        )

        # 按综合评估排序
        eval_order = {'S级': 1, 'A级': 2, 'B级': 3, 'C级': 4}
        media_stats['评估排序'] = media_stats['综合评估'].map(eval_order)

        # 排序：先按小组排序，再按综合评估，再按定档率，最后按定档量
        # 备注：这个排序用于工作量明细页面，保持按小组分组查看的便利性
        media_stats = media_stats.sort_values(
            ['小组排序', '评估排序', '定档率', '定档量'],
            ascending=[True, True, False, False]
        )

        # 移除临时排序列
        media_stats = media_stats.drop(['小组排序', '评估排序'], axis=1)

        # 重置索引
        media_stats = media_stats.reset_index(drop=True)

        logger.info(f"媒介工作量计算完成，共 {len(media_stats)} 个媒介")
        logger.info(f"最终结果前5行:")
        for idx, row in media_stats.head().iterrows():
            logger.info(f"  {row['媒介姓名']}: 总处理量={row['总处理量']}, 定档量={row['定档量']}, "
                        f"已发布={row['已发布数']}, 未发布={row['未发布数']}, "
                        f"小组={row['所属小组']}")

        return media_stats

    def _format_workload_detail(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        格式化工作量明细输出（确保字段顺序和格式）
        """
        required_columns = [
            '媒介姓名', '对应真名', '所属小组', '总处理量', '定档量',
            '已发布数', '未发布数', '其他状态数',
            '定档率(%)', '定档率评估', '产量评估', '综合评估'
        ]

        # 确保所有列都存在
        for col in required_columns:
            if col not in df.columns:
                if col in ['定档率(%)']:
                    df[col] = '0.00%'
                elif col in ['定档率评估', '产量评估', '综合评估']:
                    df[col] = '未知'
                else:
                    df[col] = 0

        # 按指定顺序排列
        df = df[required_columns].copy()

        return df

    def _calculate_workload_summary(self, detail_df: pd.DataFrame) -> Dict[str, Any]:
        """计算工作量分析汇总信息"""
        logger.info("计算工作量分析汇总信息")

        if detail_df.empty:
            return {"提示": "无工作量数据"}

        summary = {}

        # 基础统计
        summary['媒介总数'] = len(detail_df)
        summary['总处理量'] = int(detail_df['总处理量'].sum())
        summary['总定档量'] = int(detail_df['定档量'].sum())
        summary['总已发布数'] = int(detail_df['已发布数'].sum())
        summary['总未发布数'] = int(detail_df['未发布数'].sum())

        # 整体定档率
        if summary['总处理量'] > 0:
            overall_rate = (summary['总定档量'] / summary['总处理量'] * 100)
            summary['整体定档率'] = f"{overall_rate:.2f}%"
            summary['总体过筛率(%)'] = round(overall_rate, 2)  # 兼容字段
        else:
            summary['整体定档率'] = "0.00%"
            summary['总体过筛率(%)'] = 0.0

        # 发布率（已发布占定档比）
        if summary['总定档量'] > 0:
            release_rate = (summary['总已发布数'] / summary['总定档量'] * 100)
            summary['已发布占定档比'] = f"{release_rate:.2f}%"
        else:
            summary['已发布占定档比'] = "0.00%"

        # 评级统计
        if '综合评估' in detail_df.columns:
            summary['S级媒介数'] = int((detail_df['综合评估'] == 'S级').sum())
            summary['A级媒介数'] = int((detail_df['综合评估'] == 'A级').sum())
            summary['B级媒介数'] = int((detail_df['综合评估'] == 'B级').sum())
            summary['C级媒介数'] = int((detail_df['综合评估'] == 'C级').sum())
            summary['A级及以上媒介数'] = summary['S级媒介数'] + summary['A级媒介数']

        # 小组分布（前5个）
        if '所属小组' in detail_df.columns:
            group_dist = detail_df['所属小组'].value_counts().head(5).to_dict()
            summary['主要小组分布'] = group_dist

        # 平均指标
        if summary['媒介总数'] > 0:
            summary['平均处理量'] = round(summary['总处理量'] / summary['媒介总数'], 1)
            summary['平均定档量'] = round(summary['总定档量'] / summary['媒介总数'], 1)

        logger.info(f"工作量汇总计算完成，总媒介数: {summary['媒介总数']}")
        return summary

    def _calculate_group_summary(self, detail_df: pd.DataFrame) -> pd.DataFrame:
        """计算小组工作量汇总"""
        logger.info("计算小组工作量汇总")

        if detail_df.empty or '所属小组' not in detail_df.columns:
            return pd.DataFrame()

        group_summary = detail_df.groupby('所属小组').agg(
            媒介数量=('媒介姓名', 'nunique'),
            总处理量=('总处理量', 'sum'),
            总定档量=('定档量', 'sum'),
            总已发布数=('已发布数', 'sum'),
            总未发布数=('未发布数', 'sum')
        ).reset_index()

        # 计算小组指标
        group_summary['小组定档率'] = np.where(
            group_summary['总处理量'] > 0,
            (group_summary['总定档量'] / group_summary['总处理量'] * 100).round(2),
            0.0
        )
        group_summary['小组定档率(%)'] = group_summary['小组定档率'].astype(str) + "%"

        group_summary['小组发布率'] = np.where(
            group_summary['总定档量'] > 0,
            (group_summary['总已发布数'] / group_summary['总定档量'] * 100).round(2),
            0.0
        )
        group_summary['小组发布率(%)'] = group_summary['小组发布率'].astype(str) + "%"

        # 计算占比
        total_processing = group_summary['总处理量'].sum()
        group_summary['处理量占比(%)'] = np.where(
            total_processing > 0,
            (group_summary['总处理量'] / total_processing * 100).round(2),
            0.0
        )
        group_summary['处理量占比(%)'] = group_summary['处理量占比(%)'].astype(str) + "%"

        # 排序 - 按指定的小组顺序排序
        group_order = {'数码媒介组': 1, '家居媒介组': 2, '快消媒介组': 3}
        group_summary['小组排序'] = group_summary['所属小组'].apply(
            lambda x: group_order.get(x, 999)
        )
        group_summary = group_summary.sort_values('小组排序', ascending=True)
        group_summary = group_summary.drop('小组排序', axis=1)

        # 重新排列列顺序
        column_order = [
            '所属小组', '媒介数量', '总处理量', '处理量占比(%)',
            '总定档量', '总已发布数', '总未发布数',
            '小组定档率(%)', '小组发布率(%)'
        ]

        existing_columns = [col for col in column_order if col in group_summary.columns]
        group_summary = group_summary[existing_columns]

        logger.info(f"小组汇总计算完成，共 {len(group_summary)} 个小组")
        return group_summary

    def _generate_top_ranking(self, detail_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """生成TOP媒介排名 - 按工作表现排序，不按小组"""
        logger.info(f"生成TOP{top_n}媒介排名")

        if detail_df.empty:
            return pd.DataFrame()

        # 创建一个副本来避免修改原始数据
        ranking_df = detail_df.copy()

        # ✅ 核心修复：只按工作表现排序，不按小组
        # 1. 按定档量降序（主要指标）
        ranking_df['定档量_排序'] = ranking_df['定档量'].rank(method='dense', ascending=False)

        # 2. 按定档率降序（次要指标）
        # 先提取定档率数值（移除%符号）
        ranking_df['定档率数值'] = ranking_df['定档率(%)'].str.replace('%', '').astype(float)
        ranking_df['定档率_排序'] = ranking_df['定档率数值'].rank(method='dense', ascending=False)

        # 3. 计算综合得分（定档量权重70%，定档率权重30%）
        ranking_df['综合得分'] = (
                ranking_df['定档量_排序'] * 0.7 +
                ranking_df['定档率_排序'] * 0.3
        )

        # 按综合得分升序排序（得分越低排名越高）
        top_media = ranking_df.sort_values('综合得分', ascending=True).head(top_n).copy()

        # 添加排名列
        top_media['排名'] = range(1, len(top_media) + 1)

        # 重新排列列顺序
        column_order = [
            '排名', '媒介姓名', '对应真名', '所属小组', '总处理量', '定档量',
            '定档率(%)', '综合评估'
        ]

        existing_columns = [col for col in column_order if col in top_media.columns]
        top_media = top_media[existing_columns]

        # 移除临时列
        temp_cols = ['定档量_排序', '定档率_排序', '综合得分', '定档率数值']
        for col in temp_cols:
            if col in top_media.columns:
                top_media = top_media.drop(col, axis=1)

        logger.info(f"TOP{top_n}排名生成完成")
        logger.info(f"小组分布: {top_media['所属小组'].value_counts().to_dict()}")
        return top_media.reset_index(drop=True)

    def get_workload_detail(self) -> pd.DataFrame:
        """获取工作量明细数据"""
        return self.result.get("detail", pd.DataFrame())

    def get_workload_summary(self) -> Dict[str, Any]:
        """获取工作量汇总信息"""
        return self.result.get("summary", {})

    def get_group_summary(self) -> pd.DataFrame:
        """获取小组汇总数据"""
        return self.result.get("group_summary", pd.DataFrame())

    def get_top_ranking(self) -> pd.DataFrame:
        """获取TOP媒介排名"""
        return self.result.get("top_media_ranking", pd.DataFrame())