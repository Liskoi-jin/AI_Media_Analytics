# src/data_processor.py
"""
数据处理器 - 整合数据清洗逻辑，支持两种清洗模式：
1. Media_Analysis模式：基础清洗用于工作量/质量分析
2. 成本发挥分析模式：完整清洗用于成本分析
"""
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from src.utils import (
    logger, safe_read_csv, safe_read_excel, read_data_file,
    clean_column_names, normalize_media_name, get_media_group,
    format_number, validate_dataframe, deduplicate_dataframe,
    ID_TO_NAME_MAPPING, FLOWER_TO_NAME_MAPPING, NAME_TO_GROUP_MAPPING
)


class DataProcessor:
    def __init__(self):
        """初始化数据处理器"""
        self.processed_data = None
        self.filtered_data = None
        self.stats = {}
        logger.info("数据处理器初始化完成")

    def process_for_media_analysis(self, file_paths: List[str], category: str = None) -> Dict[str, Any]:
        """
        为Media_Analysis（工作量/质量分析）处理数据
        :param file_paths: 文件路径列表
        :param category: 数据类别
        :return: 处理结果字典
        """
        logger.info(f"开始Media_Analysis模式数据处理，文件数: {len(file_paths)}")
        try:
            # 读取所有文件
            dataframes = []
            for file_path in file_paths:
                df = read_data_file(file_path)
                if df is not None and not df.empty:
                    dataframes.append(df)
                    logger.info(f"读取文件: {os.path.basename(file_path)}, 行数: {len(df)}")

            if not dataframes:
                raise ValueError("未读取到有效数据")

            # 合并数据
            merged_df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"合并后数据总行数: {len(merged_df)}")

            # Media_Analysis专用清洗（简化的基础清洗）
            cleaned_df = self._clean_for_media_analysis(merged_df)

            # 基础统计
            self.stats = self._calculate_basic_stats(cleaned_df, category)

            self.processed_data = cleaned_df
            logger.info(f"Media_Analysis模式数据处理完成，有效数据: {len(cleaned_df)} 行")

            return {
                'processed_data': cleaned_df,
                'filtered_data': pd.DataFrame() if self.filtered_data is None else self.filtered_data,
                'stats': self.stats
            }

        except Exception as e:
            logger.error(f"Media_Analysis数据处理失败: {e}", exc_info=True)
            raise

    def process_for_cost_analysis(self, file_paths: List[str], category: str = None) -> Dict[str, Any]:
        """
        为成本发挥分析处理数据（使用完整清洗逻辑）
        :param file_paths: 文件路径列表
        :param category: 数据类别
        :return: 处理结果字典
        """
        logger.info(f"开始成本发挥分析模式数据处理，文件数: {len(file_paths)}")
        try:
            # 执行完整数据清洗
            cleaned_df, filtered_df = self._clean_for_cost_analysis(file_paths)

            if cleaned_df is None or len(cleaned_df) == 0:
                raise ValueError("数据清洗后无有效数据")

            # 添加：确保所有成本发挥分析必需字段都存在
            cleaned_df = self._ensure_cost_analysis_fields(cleaned_df)

            # 计算基础统计
            self.stats = self._calculate_basic_stats(cleaned_df, category)

            # 添加成本分析专用字段
            cleaned_df = self._add_cost_analysis_fields(cleaned_df)

            self.processed_data = cleaned_df
            self.filtered_data = filtered_df

            logger.info(f"成本发挥分析模式数据处理完成")
            logger.info(
                f"保留数据: {len(cleaned_df)} 行，筛除数据: {len(filtered_df) if filtered_df is not None else 0} 行")

            return {
                'processed_data': cleaned_df,
                'filtered_data': filtered_df,
                'stats': self.stats
            }

        except Exception as e:
            logger.error(f"成本发挥分析数据处理失败: {e}", exc_info=True)
            raise

    def _ensure_cost_analysis_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保成本发挥分析所有必需字段都存在"""
        logger.info("确保成本发挥分析必需字段完整")

        df = df.copy()

        # 原始成本发挥分析.py 中的所有必需字段
        required_fields = [
            '成本无效', '手续费情况', '不含手续费的下单价', '返点金额', '返点比例',
            'cpm', 'cpe', 'cpv', '被筛除标志', '筛除原因',
            '数据异常', '数据异常原因'  # 新增字段
        ]

        for field in required_fields:
            if field not in df.columns:
                logger.warning(f"字段缺失: {field}，正在补充")
                if field in ['成本无效', '被筛除标志', '数据异常']:
                    df[field] = False
                elif field in ['手续费情况', '筛除原因', '数据异常原因']:
                    df[field] = ''
                elif field in ['不含手续费的下单价', '返点金额', '返点比例', 'cpm', 'cpe', 'cpv']:
                    df[field] = 0.0

        return df

    def _clean_for_media_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Media_Analysis专用清洗（简化版）
        :param df: 原始数据
        :return: 清洗后数据
        """
        logger.info("开始Media_Analysis专用数据清洗")
        df_clean = df.copy()

        # 1. 清理列名
        df_clean = clean_column_names(df_clean)

        # 2. 标准化关键字段名
        column_mapping = {
            'influencer_nickname': '达人昵称',
            'project_name': '项目名称',
            'schedule_user_name': 'schedule_user_name',
            'submit_media_user_id': 'submit_media_user_id',
            'submit_media_user_name': 'submit_media_user_name',
            'follower_count': '粉丝数',
            'state': '状态',
            'kol_koc_type': '达人量级',
            'note_type': '笔记类型(图文/视频)',
            'cooperation_quote': '报价',
            'order_amount': '下单价',
            'rebate_amount': '返点',
            'cost_amount': '成本',
            'influencer_source': '达人来源(媒介 BD/机构)',
            'influencer_purpose': '达人用途',
            'note_like_count': '笔记点赞数',
            'note_favorite_count': '笔记收藏数',
            'note_comment_count': '笔记评论数',
            'interaction_count': '互动量',
            'read_count': '阅读量',
            'exposure_count': '曝光量',
            'read_uv_count': '阅读uv数',
            'pgy_url': '蒲公英链接pgy_url',
            'note_url': '笔记链接note_url'
        }

        # 应用列名映射
        for old_col, new_col in column_mapping.items():
            if old_col in df_clean.columns and new_col not in df_clean.columns:
                df_clean[new_col] = df_clean[old_col]
                logger.debug(f"映射字段: {old_col} -> {new_col}")

        # 3. 处理定档媒介字段
        df_clean = self._process_media_field_simple(df_clean)

        # 4. 处理状态字段
        df_clean = self._process_status_field(df_clean)

        # 5. 清洗数值字段
        numeric_fields = ['粉丝数', '报价', '下单价', '返点', '成本', '笔记点赞数', '笔记收藏数',
                          '笔记评论数', '互动量', '阅读量', '曝光量', '阅读uv数']
        for field in numeric_fields:
            if field in df_clean.columns:
                df_clean[field] = pd.to_numeric(df_clean[field], errors='coerce')
                df_clean[field] = df_clean[field].where(df_clean[field] >= 0, np.nan)

        # 6. 处理数据类型标记（定档/提报）
        if '数据类型' not in df_clean.columns:
            # 根据状态判断数据类型
            def detect_data_type(status):
                if pd.isna(status):
                    return '未知'
                status_str = str(status).upper()
                if 'CHAIN_RETURNED' in status_str or 'SCHEDULED' in status_str:
                    return '定档'
                else:
                    return '提报'

            status_col = df_clean['状态'] if '状态' in df_clean.columns else df_clean.get('原始状态', '')
            df_clean['数据类型'] = status_col.apply(detect_data_type)

        logger.info(f"Media_Analysis清洗完成，数据形状: {df_clean.shape}")
        return df_clean

    def _process_media_field_simple(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        简化版的定档媒介字段处理（Media_Analysis专用）
        """
        logger.info("处理Media_Analysis定档媒介字段")
        df_processed = df.copy()

        # 创建定档媒介和小组字段
        df_processed['定档媒介'] = '未知'
        df_processed['定档媒介小组'] = '未知'

        # 优先使用schedule_user_name（花名）
        if 'schedule_user_name' in df_processed.columns:
            df_processed['schedule_user_name'] = df_processed['schedule_user_name'].fillna('').astype(str).str.strip()

            # 标准化花名
            df_processed['schedule_user_name'] = df_processed['schedule_user_name'].apply(
                lambda x: normalize_media_name(x) if x != '' else ''
            )

            # 花名不为空时，直接使用标准化后的花名作为定档媒介
            mask = df_processed['schedule_user_name'] != ''
            df_processed.loc[mask, '定档媒介'] = df_processed.loc[mask, 'schedule_user_name']

        # 其次使用submit_media_user_id映射
        if 'submit_media_user_id' in df_processed.columns:
            # 处理需要映射的记录
            need_mapping = df_processed['定档媒介'] == '未知'

            # ID映射逻辑
            def map_id_to_name(media_id):
                if pd.isna(media_id):
                    return None
                try:
                    # 清理ID（去除.0后缀）
                    clean_id = str(media_id).replace('.0', '').strip()
                    # 在映射表中查找
                    if clean_id in ID_TO_NAME_MAPPING:
                        return ID_TO_NAME_MAPPING[clean_id]
                    elif clean_id in ID_TO_NAME_MAPPING.values():
                        return clean_id
                    else:
                        return None
                except:
                    return None

            df_processed.loc[need_mapping, '定档媒介'] = df_processed.loc[need_mapping, 'submit_media_user_id'].apply(
                map_id_to_name
            )

        # 最后使用submit_media_user_name
        if 'submit_media_user_name' in df_processed.columns:
            # 处理仍然未知的记录
            still_unknown = df_processed['定档媒介'].isin(['未知', None, ''])
            df_processed.loc[still_unknown, '定档媒介'] = df_processed.loc[
                still_unknown, 'submit_media_user_name'].fillna('未知')

        # 标准化定档媒介名称
        df_processed['定档媒介'] = df_processed['定档媒介'].fillna('未知').astype(str).str.strip()
        df_processed['定档媒介'] = df_processed['定档媒介'].replace(['', 'nan', 'NaN', 'None', 'none'], '未知')

        # 确定小组
        df_processed['定档媒介小组'] = df_processed['定档媒介'].apply(get_media_group)

        logger.info(f"定档媒介处理完成，唯一媒介数: {df_processed['定档媒介'].nunique()}")
        return df_processed

    def _process_status_field(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理状态字段
        """
        logger.info("处理状态字段")
        df_processed = df.copy()

        # 保存原始状态
        if '状态' in df_processed.columns:
            df_processed['原始状态'] = df_processed['状态'].copy()

        # 标准化状态
        if '状态' in df_processed.columns:
            df_processed['状态'] = df_processed['状态'].fillna('UNKNOWN').astype(str).str.upper()

            # 统一常见状态
            status_mapping = {
                'CHAIN_RETURNED': '已发布',
                'SCHEDULED': '已排期/未发布',
                'SCREENING_PASSED': '筛选通过',
                'SCREENING_FAILED': '筛选失败',
                'REJECTED': '已拒绝',
                'UNKNOWN': '未知'
            }

            def standardize_status(status):
                status_upper = str(status).upper()
                for key, value in status_mapping.items():
                    if key in status_upper:
                        return value
                return status

            df_processed['状态'] = df_processed['状态'].apply(standardize_status)

        return df_processed

    def _clean_for_cost_analysis(self, file_paths: List[str]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        成本发挥分析专用清洗（完整逻辑）
        """
        logger.info("开始成本发挥分析完整数据清洗")

        all_dataframes = []
        all_filtered_dataframes = []

        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"处理文件 {i}/{len(file_paths)}: {os.path.basename(file_path)}")

            # 读取文件
            df = read_data_file(file_path)
            if df is None:
                logger.warning(f"无法读取文件: {file_path}")
                continue

            # 清洗单个文件
            cleaned_df, filtered_df = self._clean_single_file_for_cost(df)

            if cleaned_df is not None and len(cleaned_df) > 0:
                all_dataframes.append(cleaned_df)

            if filtered_df is not None and len(filtered_df) > 0:
                all_filtered_dataframes.append(filtered_df)

        # 合并保留数据
        if all_dataframes:
            if len(all_dataframes) > 1:
                final_df = pd.concat(all_dataframes, ignore_index=True)
            else:
                final_df = all_dataframes[0]
        else:
            final_df = pd.DataFrame()

        # 合并被筛除数据
        if all_filtered_dataframes:
            if len(all_filtered_dataframes) > 1:
                filtered_final_df = pd.concat(all_filtered_dataframes, ignore_index=True)
            else:
                filtered_final_df = all_filtered_dataframes[0]
        else:
            filtered_final_df = pd.DataFrame()

        logger.info(f"完整清洗完成，保留数据: {len(final_df)} 行，筛除数据: {len(filtered_final_df)} 行")
        return final_df, filtered_final_df

    def _clean_single_file_for_cost(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        成本分析专用单文件清洗
        """
        cleaned_df = df.copy()

        # 初始化字段（与原始代码一致）
        cleaned_df['不含手续费的下单价'] = np.nan
        cleaned_df['被筛除标志'] = False
        cleaned_df['筛除原因'] = ''
        cleaned_df['手续费情况'] = '未知'
        cleaned_df['成本无效'] = False  # 新增：成本无效标记

        # 清理列名
        cleaned_df.columns = cleaned_df.columns.str.strip()

        # 定义所需字段（按成本发挥分析要求）
        required_fields = [
            '达人昵称', '项目名称', 'submit_media_user_id', 'submit_media_user_name', 'schedule_user_name',
            '定档媒介', '定档媒介小组', '粉丝数', '点赞收藏量', '日常阅读中位数', '日常互动中位数',
            '合作阅读中位数', '合作互动中位数', '状态', '达人量级', '笔记类型(图文/视频)',
            '报价', '下单价', '返点', '成本', '手续费', '达人来源(媒介 BD/机构)', '达人用途',
            '笔记点赞数', '笔记收藏数', '笔记评论数', '互动量', '阅读量', '曝光量', '阅读uv数',
            '蒲公英链接pgy_url', '笔记链接note_url', '不含手续费的下单价', '被筛除标志', '筛除原因', '手续费情况',
            '成本无效'
        ]

        # 修复的列名映射
        column_mapping = {
            'influencer_nickname': '达人昵称',
            'project_name': '项目名称',
            'follower_count': '粉丝数',
            'like_and_favorite_count': '点赞收藏量',
            'read_median': '日常阅读中位数',
            'interaction_median': '日常互动中位数',
            'hz_read_median': '合作阅读中位数',
            'hz_interaction_median': '合作互动中位数',
            'state': '状态',
            'kol_koc_type': '达人量级',
            'note_type': '笔记类型(图文/视频)',
            'cooperation_quote': '报价',
            'order_amount': '下单价',
            'rebate_amount': '返点',
            'cost_amount': '成本',
            'influencer_source': '达人来源(媒介 BD/机构)',
            'influencer_purpose': '达人用途',
            'note_like_count': '笔记点赞数',
            'note_favorite_count': '笔记收藏数',
            'note_comment_count': '笔记评论数',
            'interaction_count': '互动量',
            'read_count': '阅读量',
            'exposure_count': '曝光量',
            'read_uv_count': '阅读uv数',
            'pgy_url': '蒲公英链接pgy_url',
            'note_url': '笔记链接note_url'
        }

        # 应用列名映射
        rename_dict = {}
        for old_col in cleaned_df.columns:
            if old_col in column_mapping:
                rename_dict[old_col] = column_mapping[old_col]

        if rename_dict:
            cleaned_df = cleaned_df.rename(columns=rename_dict)

        # 确保所有所需字段都存在
        for field in required_fields:
            if field not in cleaned_df.columns:
                if field in ['手续费', '不含手续费的下单价']:
                    cleaned_df[field] = 0
                elif field in ['被筛除标志', '成本无效']:
                    cleaned_df[field] = False
                elif field in ['筛除原因', '手续费情况']:
                    cleaned_df[field] = ''
                elif field == '定档媒介小组':
                    cleaned_df[field] = ''
                else:
                    cleaned_df[field] = np.nan

        # 1. 处理定档媒介字段（使用完整逻辑）
        cleaned_df = self._process_media_field_cost(cleaned_df)

        # 2. 检查下单价是否含手续费并进行数据筛除
        cleaned_df = self._check_and_filter_price_data(cleaned_df)

        # 3. 计算手续费
        cleaned_df = self._calculate_actual_service_charge(cleaned_df)

        # 4. 确保不含手续费的下单价字段完整
        cleaned_df = self._ensure_no_fee_price_complete(cleaned_df)

        # 5. 数据清洗处理
        cleaned_df = self._clean_numeric_data_cost(cleaned_df)
        cleaned_df = self._clean_text_data_cost(cleaned_df)
        cleaned_df = self._standardize_categorical_data_cost(cleaned_df)

        # 6. 标记成本为0的数据（不删除，与原始代码一致）
        cleaned_df = self._mark_zero_cost_data(cleaned_df)

        # 7. 计算关键指标
        cleaned_df = self._calculate_metrics_cost(cleaned_df)

        # 8. 清理URL字段
        cleaned_df = self._clean_url_fields(cleaned_df)

        # 所有数据都保留在同一个DataFrame中
        retained_df = cleaned_df.copy()  # 所有数据都保留
        filtered_df = pd.DataFrame()  # 返回空的被筛除数据框

        logger.info(f"单文件清洗完成: 保留全部 {len(retained_df)} 条数据")
        return retained_df, filtered_df

    def _mark_zero_cost_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标记成本为0或空的数据（与原始代码一致）
        """
        if '成本' not in df.columns:
            return df

        try:
            # 确保成本是数值型
            df['成本'] = pd.to_numeric(df['成本'], errors='coerce')

            # 统计不同类型的数据
            total_count = len(df)
            zero_cost_count = (df['成本'] == 0).sum()
            nan_cost_count = df['成本'].isna().sum()
            positive_cost_count = (df['成本'] > 0).sum()

            # 标记成本为0或NaN的数据
            df['成本无效'] = (df['成本'] == 0) | (df['成本'].isna())
            df['成本无效'] = df['成本无效'].fillna(False)

            logger.info(
                f"成本分布: 成本>0: {positive_cost_count} 条, 成本=0: {zero_cost_count} 条, 成本缺失: {nan_cost_count} 条")
            logger.info(f"标记为成本无效的数据: {df['成本无效'].sum()} 条")

        except Exception as e:
            logger.error(f"标记成本数据时出错: {e}")

        return df

    def _process_media_field_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        成本分析专用的定档媒介字段处理（完整逻辑）
        """
        logger.info("处理成本分析定档媒介字段")

        # 创建新的定档媒介和定档媒介小组列
        df['定档媒介'] = np.nan
        df['定档媒介小组'] = ''

        # 步骤1：先处理schedule_user_name（花名映射到真实名字）
        if 'schedule_user_name' in df.columns:
            df['schedule_user_name'] = df['schedule_user_name'].fillna('')
            df['schedule_user_name'] = df['schedule_user_name'].astype(str).str.strip()

            has_schedule_name = df['schedule_user_name'] != ''

            # 花名映射到真实名字
            def map_flower_to_name(flower_name):
                if pd.isna(flower_name) or flower_name == '':
                    return None
                flower_name = str(flower_name).strip()
                return FLOWER_TO_NAME_MAPPING.get(flower_name, None)

            df.loc[has_schedule_name, '定档媒介'] = df.loc[has_schedule_name, 'schedule_user_name'].apply(
                map_flower_to_name
            )

        # 步骤2：对于没有通过schedule_user_name映射到真实名字的记录，使用submit_media_user_id映射
        need_id_mapping = df['定档媒介'].isna()

        if need_id_mapping.sum() > 0 and 'submit_media_user_id' in df.columns:
            # 确保submit_media_user_id是数值型
            df['submit_media_user_id'] = pd.to_numeric(df['submit_media_user_id'], errors='coerce')

            # ID映射到真实名字
            def map_id_to_name(media_id):
                if pd.isna(media_id):
                    return None
                if isinstance(media_id, (int, np.integer)):
                    return ID_TO_NAME_MAPPING.get(int(media_id), None)
                try:
                    int_id = int(float(media_id))
                    return ID_TO_NAME_MAPPING.get(int_id, None)
                except:
                    return None

            df.loc[need_id_mapping, '定档媒介'] = df.loc[need_id_mapping, 'submit_media_user_id'].apply(
                map_id_to_name
            )

        # 步骤3：如果还有未映射的记录，设为"未知"
        still_missing = df['定档媒介'].isna()
        df.loc[still_missing, '定档媒介'] = "未知"

        # 步骤4：为每个定档媒介设置小组
        def map_name_to_group(media_name):
            if pd.isna(media_name) or media_name == '未知':
                return '未知'
            return NAME_TO_GROUP_MAPPING.get(str(media_name).strip(), 'other组')

        df['定档媒介小组'] = df['定档媒介'].apply(map_name_to_group)

        # 清理定档媒介字段
        df['定档媒介'] = df['定档媒介'].fillna('未知')
        df['定档媒介'] = df['定档媒介'].astype(str).str.strip()
        df['定档媒介'] = df['定档媒介'].replace(['', 'nan', 'NaN', 'None', 'none', 'null', 'NULL'], '未知')

        # 统计
        media_counts = df['定档媒介'].value_counts()
        group_counts = df['定档媒介小组'].value_counts()
        logger.info(f"定档媒介处理完成: {len(media_counts)} 个媒介, {len(group_counts)} 个小组")
        logger.info(f"小组分布: {dict(group_counts.head())}")

        return df

    def _check_and_filter_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        检查下单价是否含手续费，按照正确的逻辑处理
        规则：
        1. cost_amount = order_amount - rebate_amount 时：下单价含手续费
        2. cost_amount = order_amount*1.1 - rebate_amount 时：下单价不含手续费
        3. 其他情况：无法判断
        """
        logger.info("检查下单价是否含手续费（按照正确逻辑）")

        # 确保相关字段为数值型
        numeric_fields = ['下单价', '成本', '返点', '报价']
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')

        # 初始化字段（确保存在）
        if '被筛除标志' not in df.columns:
            df['被筛除标志'] = False
        if '筛除原因' not in df.columns:
            df['筛除原因'] = ''
        if '手续费情况' not in df.columns:
            df['手续费情况'] = '未知'
        if '成本无效' not in df.columns:
            df['成本无效'] = False
        if '不含手续费的下单价' not in df.columns:
            df['不含手续费的下单价'] = np.nan

        # 遍历每一行数据
        for idx in df.index:
            order_amount = df.at[idx, '下单价']
            cost_amount = df.at[idx, '成本']
            rebate_amount = df.at[idx, '返点']
            quote_amount = df.at[idx, '报价']

            # 检查是否有缺失值
            if (pd.isna(order_amount) or pd.isna(cost_amount) or
                    pd.isna(rebate_amount) or pd.isna(quote_amount)):
                df.at[idx, '手续费情况'] = '数据不全'
                df.at[idx, '不含手续费的下单价'] = '无法判断'
                df.at[idx, '成本无效'] = True
                df.at[idx, '筛除原因'] = '数据不全'
                continue

            tolerance = 0.01  # 容忍度，避免浮点数精度问题

            # 情况1：下单价 = 成本 + 返点（含手续费）
            # cost_amount = order_amount - rebate_amount
            if abs(cost_amount - (order_amount - rebate_amount)) <= tolerance:
                df.at[idx, '手续费情况'] = '含手续费'
                df.at[idx, '不含手续费的下单价'] = order_amount / 1.1

                # ✅ 修改：统一异常原因为"报价或下单价异常"
                if quote_amount < (order_amount / 1.1):
                    df.at[idx, '筛除原因'] = '报价或下单价异常'
                    df.at[idx, '成本无效'] = True
                else:
                    df.at[idx, '筛除原因'] = '正常'

            # 情况2：成本 = 下单价×1.1 - 返点（不含手续费）
            # cost_amount = order_amount*1.1 - rebate_amount
            elif abs(cost_amount - (order_amount * 1.1 - rebate_amount)) <= tolerance:
                df.at[idx, '手续费情况'] = '不含手续费'
                df.at[idx, '不含手续费的下单价'] = order_amount

                # ✅ 修改：统一异常原因为"报价或下单价异常"
                if quote_amount < order_amount:
                    df.at[idx, '筛除原因'] = '报价或下单价异常'
                    df.at[idx, '成本无效'] = True
                else:
                    df.at[idx, '筛除原因'] = '正常'

            # 其他情况：无法判断
            else:
                df.at[idx, '手续费情况'] = '无法判断'
                df.at[idx, '不含手续费的下单价'] = '无法判断'
                df.at[idx, '筛除原因'] = '数据异常'
                df.at[idx, '成本无效'] = True

            # 检查成本是否为0或缺失
            if pd.isna(cost_amount) or cost_amount == 0:
                df.at[idx, '成本无效'] = True
                if pd.isna(cost_amount):
                    df.at[idx, '筛除原因'] = '成本缺失'
                elif cost_amount == 0:
                    df.at[idx, '筛除原因'] = '成本为0'

            # 确保不被筛除
            df.at[idx, '被筛除标志'] = False

        # 统计
        total_records = len(df)
        case1_count = (df['手续费情况'] == '含手续费').sum()
        case2_count = (df['手续费情况'] == '不含手续费').sum()
        unknown_case = (df['手续费情况'] == '无法判断').sum() + (df['手续费情况'] == '数据不全').sum()
        cost_invalid_count = df['成本无效'].sum()
        filtered_count = df['被筛除标志'].sum()

        # 统计无效原因
        invalid_reasons = {}
        for reason in df['筛除原因'].unique():
            if reason and reason != '正常':
                count = (df['筛除原因'] == reason).sum()
                invalid_reasons[reason] = count

        logger.info(f"手续费情况统计:")
        logger.info(f"  总计记录数: {total_records}")
        logger.info(f"  含手续费: {case1_count} 条 ({case1_count / total_records * 100:.1f}%)")
        logger.info(f"  不含手续费: {case2_count} 条 ({case2_count / total_records * 100:.1f}%)")
        logger.info(f"  无法判断/数据不全: {unknown_case} 条 ({unknown_case / total_records * 100:.1f}%)")
        logger.info(f"  成本无效数据: {cost_invalid_count} 条 ({cost_invalid_count / total_records * 100:.1f}%)")
        logger.info(f"  被筛除数据: {filtered_count} 条 (应该为0)")
        logger.info(f"  无效数据原因分布: {invalid_reasons}")

        return df



    def _calculate_actual_service_charge(self, df: pd.DataFrame) -> pd.DataFrame:
        """根据手续费情况计算实际手续费"""

        def calculate_charge(row):
            if pd.isna(row['下单价']):
                return 0

            if row['手续费情况'] == '含手续费':
                return row['下单价'] * 0.1 / 1.1
            elif row['手续费情况'] == '不含手续费':
                return row['下单价'] * 0.1
            else:
                return row['下单价'] * 0.1

        df['手续费'] = df.apply(calculate_charge, axis=1)

        # 统计
        total_charge = df['手续费'].sum()
        avg_charge = df['手续费'].mean()
        logger.info(f"手续费统计: 总额 {total_charge:.2f} 元, 平均 {avg_charge:.2f} 元")

        return df

    def _ensure_no_fee_price_complete(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保不含手续费的下单价字段在所有记录中都有值"""
        if '不含手续费的下单价' not in df.columns or '下单价' not in df.columns:
            return df

        # 处理 '无法判断' 的情况
        mask = (df['不含手续费的下单价'] == '无法判断') | df['不含手续费的下单价'].isna()
        if mask.sum() > 0:
            # 对于无法判断的情况，保持 '无法判断' 字符串
            df.loc[mask & (df['不含手续费的下单价'] != '无法判断'), '不含手续费的下单价'] = df.loc[
                mask & (df['不含手续费的下单价'] != '无法判断'), '下单价']

            if '手续费情况' in df.columns:
                df.loc[mask & (df['手续费情况'] == ''), '手续费情况'] = '默认使用下单价'

        logger.info(f"不含手续费下单价处理完成")
        return df

    def _clean_numeric_data_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗数值型数据"""
        numeric_fields = [
            '粉丝数', '点赞收藏量', '日常阅读中位数', '日常互动中位数',
            '合作阅读中位数', '合作互动中位数', '报价', '下单价', '返点', '成本', '手续费', '不含手续费的下单价',
            '笔记点赞数', '笔记收藏数', '笔记评论数', '互动量', '阅读量', '曝光量', '阅读uv数'
        ]

        existing_numeric_fields = [col for col in numeric_fields if col in df.columns]

        for col in existing_numeric_fields:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if col not in ['报价', '下单价', '返点', '成本', '手续费', '不含手续费的下单价']:
                    df[col] = df[col].where(df[col] >= 0, np.nan)
            except:
                pass

        return df

    def _clean_text_data_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗文本数据"""
        text_fields = [
            '达人昵称', '项目名称', 'schedule_user_name', 'submit_media_user_name', '定档媒介', '定档媒介小组',
            '状态', '达人量级', '笔记类型(图文/视频)', '达人来源(媒介 BD/机构)', '达人用途',
            '筛除原因', '手续费情况'
        ]

        existing_text_fields = [col for col in text_fields if col in df.columns]

        for col in existing_text_fields:
            try:
                if col not in ['schedule_user_name', 'submit_media_user_name', '筛除原因', '手续费情况']:
                    df[col] = df[col].fillna('未知' if col != '定档媒介小组' else 'other组')
                elif col in ['筛除原因', '手续费情况']:
                    df[col] = df[col].fillna('')

                df[col] = df[col].astype(str).str.strip()

                if col not in ['schedule_user_name', 'submit_media_user_name', '筛除原因', '手续费情况']:
                    df[col] = df[col].replace(['', 'nan', 'NaN', 'None', 'none', 'null', 'NULL'],
                                              '未知' if col != '定档媒介小组' else 'other组')
                elif col in ['筛除原因', '手续费情况']:
                    df[col] = df[col].replace(['nan', 'NaN', 'None', 'none', 'null', 'NULL'], '')
            except:
                pass

        return df

    def _standardize_categorical_data_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化分类数据"""
        # 标准化达人量级
        if '达人量级' in df.columns:
            kol_mapping = {
                'KOL': 'KOL', 'KOC': 'KOC', '十万KOL': '十万KOL',
                '优质KOL': '优质KOL', '高阅读达人': '高阅读达人',
                'kol': 'KOL', 'koc': 'KOC',
                '头部达人': 'KOL', '腰部达人': 'KOL', '尾部达人': 'KOC'
            }

            def standardize_kol(x):
                return kol_mapping.get(str(x).strip(), str(x).strip())

            df['达人量级'] = df['达人量级'].apply(standardize_kol)

        # 标准化笔记类型
        if '笔记类型(图文/视频)' in df.columns:
            df['笔记类型(图文/视频)'] = df['笔记类型(图文/视频)'].astype(str)

            def standardize_note_type(x):
                if '图文' in x:
                    return '图文'
                elif '视频' in x:
                    return '视频'
                else:
                    return x

            df['笔记类型(图文/视频)'] = df['笔记类型(图文/视频)'].apply(standardize_note_type)

        # 标准化达人来源
        if '达人来源(媒介 BD/机构)' in df.columns:
            df['达人来源(媒介 BD/机构)'] = df['达人来源(媒介 BD/机构)'].astype(str)

            def standardize_source(x):
                if '媒介' in x or 'BD' in x:
                    return '媒介 BD'
                elif '机构' in x:
                    return '机构'
                else:
                    return x

            df['达人来源(媒介 BD/机构)'] = df['达人来源(媒介 BD/机构)'].apply(standardize_source)

        # 标准化达人用途
        if '达人用途' in df.columns:
            df['达人用途'] = df['达人用途'].astype(str).str.strip().fillna('未知')
            df['达人用途'] = df['达人用途'].replace(['', 'nan', 'NaN', 'None', 'none', 'null', 'NULL'], '未知')
            # 将"未知"重命名为"优质达人"
            df['达人用途'] = df['达人用途'].replace('未知', '优质达人')

        return df

    def _calculate_metrics_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算关键指标：cpm, cpe, cpv, 返点比例"""
        # 计算cpm(千次曝光成本=成本/曝光量*1000)
        if '成本' in df.columns and '曝光量' in df.columns:
            mask = df['成本'].notna() & df['曝光量'].notna() & (df['曝光量'] > 0)
            df['cpm'] = np.nan
            df.loc[mask, 'cpm'] = df.loc[mask, '成本'] / df.loc[mask, '曝光量'] * 1000

        # 计算cpe(互动成本=成本/互动量)
        if '成本' in df.columns and '互动量' in df.columns:
            mask = df['成本'].notna() & df['互动量'].notna() & (df['互动量'] > 0)
            df['cpe'] = np.nan
            df.loc[mask, 'cpe'] = df.loc[mask, '成本'] / df.loc[mask, '互动量']

        # 计算cpv(阅读成本=成本/阅读量)
        if '成本' in df.columns and '阅读量' in df.columns:
            mask = df['成本'].notna() & df['阅读量'].notna() & (df['阅读量'] > 0)
            df['cpv'] = np.nan
            df.loc[mask, 'cpv'] = df.loc[mask, '成本'] / df.loc[mask, '阅读量']

        # 计算返点金额和返点比例（原始公式）
        # 先确保有返点金额字段
        if '返点金额' not in df.columns:
            if '报价' in df.columns and '不含手续费的下单价' in df.columns and '返点' in df.columns:
                df['返点金额'] = df['报价'] - df['不含手续费的下单价'] + df['返点']
            else:
                df['返点金额'] = 0.0

        # 计算返点比例: 返点比例 = (报价 - 不含手续费的下单价 + 返点) / 报价
        if '不含手续费的下单价' in df.columns and '报价' in df.columns and '返点' in df.columns:
            mask = df['不含手续费的下单价'].notna() & df['报价'].notna() & (df['报价'] > 0) & df['返点'].notna()
            df['返点比例'] = np.nan

            # 使用原始公式：返点比例 = (报价 - 不含手续费的下单价 + 返点) / 报价
            numerator = df.loc[mask, '报价'] - df.loc[mask, '不含手续费的下单价'] + df.loc[mask, '返点']
            tolerance = 1e-10
            numerator_rounded = numerator.where(np.abs(numerator) >= tolerance, 0)
            df.loc[mask, '返点比例'] = numerator_rounded / df.loc[mask, '报价']
            df['返点比例'] = df['返点比例'].round(12)

            # 返点比例小于0的直接设为0，大于1的设为1
            negative_mask = df['返点比例'].notna() & (df['返点比例'] < 0)
            df.loc[negative_mask, '返点比例'] = 0

            high_mask = df['返点比例'].notna() & (df['返点比例'] > 1)
            df.loc[high_mask, '返点比例'] = 1

            # 统计
            valid_rebate_ratio = df['返点比例'].notna().sum()
            if valid_rebate_ratio > 0:
                avg_ratio = df['返点比例'].mean() * 100
                logger.info(f"返点比例统计: 有效数据 {valid_rebate_ratio} 条, 平均返点比例 {avg_ratio:.2f}%")

        return df

    def _clean_url_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗URL字段"""
        url_fields = ['蒲公英链接pgy_url', '笔记链接note_url']

        for col in url_fields:
            if col in df.columns:
                df[col] = df[col].fillna('')
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'NaN', 'None', 'none', 'null', 'NULL', '未知'], '')

        return df

    def _add_cost_analysis_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为成本分析添加专用字段
        """
        logger.info("添加成本分析专用字段")

        # 确保必要字段存在
        required_fields = [
            '达人昵称', '项目名称', '定档媒介', '定档媒介小组', '粉丝数',
            '状态', '达人量级', '笔记类型(图文/视频)', '报价', '下单价',
            '返点', '成本', '手续费', '不含手续费的下单价', '达人来源(媒介 BD/机构)',
            '达人用途', '笔记点赞数', '笔记收藏数', '笔记评论数', '互动量',
            '阅读量', '曝光量', '阅读uv数', '蒲公英链接pgy_url', '笔记链接note_url',
            '成本无效', '手续费情况', '返点金额', '返点比例', 'cpm', 'cpe',
            '数据异常', '数据异常原因'  # 新增字段
        ]

        for field in required_fields:
            if field not in df.columns:
                if field in ['手续费', '不含手续费的下单价', '返点比例', '返点金额', 'cpm', 'cpe']:
                    df[field] = 0.0
                elif field in ['定档媒介小组']:
                    df[field] = ''
                elif field in ['成本无效', '数据异常']:
                    df[field] = False
                elif field in ['手续费情况', '数据异常原因']:
                    df[field] = '未知'
                else:
                    df[field] = np.nan

        return df

    def _calculate_basic_stats(self, df: pd.DataFrame, category: str = None) -> Dict[str, Any]:
        """
        计算基础统计数据
        """
        logger.info("计算基础统计数据")
        stats = {}

        if df.empty:
            stats['数据状态'] = '无有效数据'
            return stats

        # 基础统计
        stats['总记录数'] = len(df)
        stats['唯一媒介数'] = df['定档媒介'].nunique() if '定档媒介' in df.columns else 0
        stats['唯一项目数'] = df['项目名称'].nunique() if '项目名称' in df.columns else 0
        stats['唯一达人数'] = df['达人昵称'].nunique() if '达人昵称' in df.columns else 0

        # 成本相关统计
        if '成本' in df.columns:
            stats['总成本'] = round(df['成本'].sum(), 2)
            stats['平均成本'] = round(df['成本'].mean(), 2) if len(df) > 0 else 0

        if '返点金额' in df.columns:
            stats['总返点金额'] = round(df['返点金额'].sum(), 2)
            stats['平均返点金额'] = round(df['返点金额'].mean(), 2) if len(df) > 0 else 0

        if '返点比例' in df.columns:
            valid_rebate = df['返点比例'].dropna()
            if len(valid_rebate) > 0:
                stats['平均返点比例'] = f"{valid_rebate.mean() * 100:.2f}%"

        # 成本无效统计
        if '成本无效' in df.columns:
            stats['成本无效数据'] = int(df['成本无效'].sum())
            stats['成本无效比例'] = f"{df['成本无效'].sum() / len(df) * 100:.1f}%"

        # 手续费情况统计
        if '手续费情况' in df.columns:
            fee_stats = df['手续费情况'].value_counts().to_dict()
            stats['手续费情况分布'] = fee_stats

        # 小组分布
        if '定档媒介小组' in df.columns:
            group_counts = df['定档媒介小组'].value_counts().to_dict()
            stats['小组分布'] = group_counts

        # 状态分布
        if '状态' in df.columns:
            status_counts = df['状态'].value_counts().to_dict()
            stats['状态分布'] = status_counts

        # 类别信息
        if category:
            stats['数据类别'] = category

        stats['处理时间'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        logger.info(f"基础统计计算完成，总记录数: {stats['总记录数']}")
        return stats

    def save_processed_data(self, output_dir: str, filename_prefix: str = "processed_data") -> str:
        """
        保存处理后的数据
        """
        if self.processed_data is None or self.processed_data.empty:
            logger.warning("无处理数据可保存")
            return ""

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.xlsx"
        filepath = os.path.join(output_dir, filename)

        try:
            # 保存数据
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                self.processed_data.to_excel(writer, sheet_name='处理数据', index=False)

                # 如果有筛除数据，也保存
                if self.filtered_data is not None and not self.filtered_data.empty:
                    self.filtered_data.to_excel(writer, sheet_name='筛除数据', index=False)

                # 保存统计信息
                stats_df = pd.DataFrame(list(self.stats.items()), columns=['统计项', '数值'])
                stats_df.to_excel(writer, sheet_name='数据统计', index=False)

            logger.info(f"数据已保存: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            return ""