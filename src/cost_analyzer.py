# src/cost_analyzer.py
"""
成本发挥分析器 - 完全对齐成本发挥分析.py的逻辑
输出所有目标工作表：
1. 媒介小组工作量分析
2. 定档媒介工作量分析
3. 定档媒介成本分析
4. 定档媒介返点分析
5. 定档媒介效果分析
6. 定档媒介达人量级分析
7. 定档媒介综合分析
8. 详细数据
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from src.utils import (
    logger, normalize_media_name, get_media_group,
    calculate_percentage, format_number, ID_TO_NAME_MAPPING, FLOWER_TO_NAME_MAPPING
)


class CostAnalyzer:
    def __init__(self, processed_df: pd.DataFrame, filtered_df: pd.DataFrame = None):
        """
        初始化成本发挥分析器
        :param processed_df: 清洗后的数据（来自data_processor.py）
        :param filtered_df: 被筛除的数据（现在为空DataFrame）
        """
        logger.info("成本发挥分析器初始化 - 使用所有数据进行分析")

        # 1. 验证数据并确保字段完整
        self.all_df = self._validate_and_fill_fields(processed_df)

        # 2. 标记无效数据（但不删除）
        self.all_df = self._mark_invalid_data(self.all_df)

        # 3. 不分离有效和无效数据，全部保留在all_df中
        # 仍保留这两个属性用于统计，但不用于主要分析
        self.valid_df = self.all_df[~self.all_df['成本无效']].copy()
        self.invalid_df = self.all_df[self.all_df['成本无效']].copy()

        # 4. 处理媒介信息（对所有数据）
        self.all_df = self._process_media_info_cost(self.all_df)

        # 5. 设置filtered_df属性（即使为空也要设置）
        if filtered_df is not None and not filtered_df.empty:
            self.filtered_df = filtered_df
        else:
            # 创建一个空的DataFrame，确保属性存在
            self.filtered_df = pd.DataFrame()

        logger.info(
            f"数据准备完成: 总数据 {len(self.all_df)} 条, 有效数据 {len(self.valid_df)} 条, 无效数据 {len(self.invalid_df)} 条")
        # 用于存储媒介ID到真名的映射
        self.media_mapping = {}

        # 存储所有分析结果
        self.result = {
            "overall_summary": {},  # 整体汇总
            "media_detail": None,  # 媒介明细
            "group_summary": None,  # 小组汇总
            "filtered_summary": None,  # 筛除数据汇总
            "cost_efficiency_ranking": None,  # 成本效益排名

            # 成本发挥分析专用结果字段
            "media_group_workload": None,  # 媒介小组工作量分析
            "fixed_media_workload": None,  # 定档媒介工作量分析
            "fixed_media_cost": None,  # 定档媒介成本分析
            "fixed_media_rebate": None,  # 定档媒介返点分析
            "fixed_media_performance": None,  # 定档媒介效果分析
            "fixed_media_level": None,  # 定档媒介达人量级分析
            "fixed_media_comprehensive": None,  # 定档媒介综合分析
            "detailed_data": None,  # 详细数据
            "invalid_data_detail": None  # 无效数据详情
        }

        logger.info("成本发挥分析器初始化完成")

    def _validate_and_fill_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证并填充成本分析所需字段"""
        logger.info("验证和填充成本分析字段")

        df = df.copy()

        # 原始成本发挥分析.py 中的必需字段
        required_fields = [
            '达人昵称', '项目名称', '定档媒介', '定档媒介小组',
            '粉丝数', '点赞收藏量', '日常阅读中位数', '日常互动中位数',
            '合作阅读中位数', '合作互动中位数', '状态', '达人量级',
            '笔记类型(图文/视频)', '报价', '下单价', '返点', '成本',
            '手续费', '不含手续费的下单价', '达人来源(媒介 BD/机构)',
            '达人用途', '笔记点赞数', '笔记收藏数', '笔记评论数',
            '互动量', '阅读量', '曝光量', '阅读uv数',
            '蒲公英链接pgy_url', '笔记链接note_url',
            '成本无效', '筛除原因', '手续费情况',  # 原始代码新增字段
            'schedule_user_name', 'submit_media_user_id', 'submit_media_user_name',  # 媒介映射字段
            '返点金额', '返点比例', 'cpm', 'cpe', 'cpv'  # 计算字段
        ]

        # 填充缺失字段
        for field in required_fields:
            if field not in df.columns:
                logger.warning(f"字段缺失: {field}，正在填充默认值")

                if field in ['成本', '不含手续费的下单价', '报价', '返点', '手续费', '互动量', '阅读量', '曝光量',
                             '返点金额', '返点比例', 'cpm', 'cpe', 'cpv']:
                    df[field] = 0.0
                elif field in ['成本无效']:
                    df[field] = False
                elif field in ['筛除原因', '手续费情况', '定档媒介', '定档媒介小组']:
                    df[field] = '' if field != '定档媒介小组' else 'other组'
                elif field in ['状态', '达人量级', '笔记类型(图文/视频)']:
                    df[field] = '未知'
                else:
                    df[field] = ''

        # 确保数值字段类型正确
        numeric_fields = ['成本', '不含手续费的下单价', '报价', '返点', '手续费',
                          '互动量', '阅读量', '曝光量', '阅读uv数',
                          '返点金额', '返点比例', 'cpm', 'cpe', 'cpv']
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0.0)

        return df

    # 修改这个方法为兼容模式：
    def _filter_invalid_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        兼容方法：如果需要过滤数据，就过滤掉无效的
        """
        logger.warning("注意：正在过滤数据，建议使用标记方式替代过滤")
        if '成本无效' in df.columns:
            return df[~df['成本无效']].copy()
        return df

    # 在 _filter_invalid_data 方法后添加这个方法：
    def _mark_invalid_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标记无效数据和异常数据（按照新逻辑）
        无效数据（不参与分析）：成本为0或缺失、数据不全
        异常数据（参与分析）：报价异常、数据异常、返点异常
        """
        logger.info("标记无效数据和异常数据（按照新逻辑）")

        df = df.copy()

        # 步骤1: 确保必要字段存在
        required_fields = ['成本', '报价', '不含手续费的下单价', '筛除原因', '返点比例']
        for field in required_fields:
            if field not in df.columns:
                if field in ['成本', '报价', '返点比例']:
                    df[field] = 0.0
                elif field == '不含手续费的下单价':
                    df[field] = 0.0
                elif field == '筛除原因':
                    df[field] = ''

        # 步骤2: 初始化标记字段
        df['成本无效'] = False  # 不参与分析的数据
        df['数据异常'] = False  # 参与分析但异常的数据
        df['成本无效原因'] = '有效数据'
        df['数据异常原因'] = '正常数据'

        # 步骤3: 检查各种情况

        # 情况1: 成本为0或缺失 → 无效数据（不参与分析）
        mask_zero_cost = (df['成本'] == 0) | (df['成本'].isna())
        df.loc[mask_zero_cost, '成本无效'] = True
        df.loc[mask_zero_cost, '成本无效原因'] = df.loc[mask_zero_cost, '成本无效原因'].apply(
            lambda x: '成本为0或缺失' if x == '有效数据' else f"{x}；成本为0或缺失"
        )

        # 情况2: 数据不全 → 无效数据（不参与分析）
        mask_data_incomplete = (
                (df['不含手续费的下单价'] == '无法判断') |
                (df['不含手续费的下单价'].isna()) |
                (df['报价'].isna()) |
                (df['下单价'].isna()) |
                (df['返点'].isna())
        )
        df.loc[mask_data_incomplete, '成本无效'] = True
        df.loc[mask_data_incomplete, '成本无效原因'] = df.loc[mask_data_incomplete, '成本无效原因'].apply(
            lambda x: '数据不全' if x == '有效数据' else f"{x}；数据不全"
        )

        # 情况3: 报价异常（报价 < 不含手续费的下单价） → 异常数据（参与分析）
        # 先确保数据类型正确
        try:
            df['报价'] = pd.to_numeric(df['报价'], errors='coerce')

            # 处理不含手续费的下单价
            def parse_no_fee_price(val):
                if isinstance(val, str) and val == '无法判断':
                    return np.nan
                try:
                    return float(val)
                except:
                    return np.nan

            df['不含手续费的下单价_数值'] = df['不含手续费的下单价'].apply(parse_no_fee_price)

            # 比较报价和不含手续费的下单价（只处理数值有效的数据）
            mask_valid_data = df['不含手续费的下单价_数值'].notna() & df['报价'].notna()
            mask_price_abnormal = mask_valid_data & (df['报价'] < df['不含手续费的下单价_数值'])

            # 排除已经标记为无效的数据
            mask_price_abnormal = mask_price_abnormal & ~df['成本无效']

            # ✅ 修改：统一标记为"报价或下单价异常"
            df.loc[mask_price_abnormal, '数据异常'] = True
            df.loc[mask_price_abnormal, '数据异常原因'] = '报价或下单价异常'

            # ✅ 修改：同时更新筛除原因字段
            df.loc[mask_price_abnormal, '筛除原因'] = '报价或下单价异常'

            # 删除临时列
            df = df.drop('不含手续费的下单价_数值', axis=1)
        except Exception as e:
            logger.error(f"检查报价异常时出错: {e}")

        # 情况4: 数据异常（"不含手续费的下单价"是"无法判断"但其他数据完整）→ 异常数据（参与分析）
        mask_data_abnormal = (df['不含手续费的下单价'] == '无法判断') & ~mask_data_incomplete
        df.loc[mask_data_abnormal, '数据异常'] = True
        df.loc[mask_data_abnormal, '数据异常原因'] = '数据异常（无法判断手续费情况）'

        # 情况5: 返点异常（返点比例异常）→ 异常数据（参与分析）
        if '返点比例' in df.columns:
            # 返点比例超过100%或小于-50%（极端异常）
            mask_rebate_abnormal = (
                    (df['返点比例'] > 1.0) |  # 超过100%
                    (df['返点比例'] < -0.5)  # 小于-50%（极端异常）
            )
            # 排除已经标记为无效的数据
            mask_rebate_abnormal = mask_rebate_abnormal & ~df['成本无效']

            df.loc[mask_rebate_abnormal, '数据异常'] = True

            def format_rebate_reason(ratio):
                return f"返点比例异常({ratio * 100:.1f}%)"

            df.loc[mask_rebate_abnormal, '数据异常原因'] = df.loc[mask_rebate_abnormal, '返点比例'].apply(
                lambda x: format_rebate_reason(x) if pd.notna(x) else '返点比例异常'
            )

        # 情况6: 筛除原因异常 → 异常数据（参与分析）
        if '筛除原因' in df.columns:
            # 获取筛除原因不为空且不是'正常'的数据
            abnormal_mask = df['筛除原因'].notna() & (df['筛除原因'] != '') & (df['筛除原因'] != '正常')
            # 排除已经标记为无效的数据
            abnormal_mask = abnormal_mask & ~df['成本无效']

            # ✅ 修改：对于报价相关的异常原因，统一为"报价或下单价异常"
            for idx in df[abnormal_mask].index:
                reason = str(df.at[idx, '筛除原因']).strip()
                # 检查是否为报价相关的异常
                if any(keyword in reason for keyword in ['报价', '下单价', '不含手续费']):
                    df.at[idx, '数据异常原因'] = '报价或下单价异常'
                    df.at[idx, '筛除原因'] = '报价或下单价异常'  # 同时更新筛除原因
                else:
                    df.at[idx, '数据异常原因'] = reason

            df.loc[abnormal_mask, '数据异常'] = True

        # 统计
        total_invalid = df['成本无效'].sum()
        total_abnormal = df['数据异常'].sum()
        total_valid = len(df) - total_invalid - total_abnormal

        # 统计无效原因
        invalid_reasons = {}
        for reason in df[df['成本无效']]['成本无效原因'].unique():
            if reason and reason != '有效数据':
                count = (df['成本无效原因'] == reason).sum()
                invalid_reasons[reason] = count

        # 统计异常原因
        abnormal_reasons = {}
        for reason in df[df['数据异常']]['数据异常原因'].unique():
            if reason and reason != '正常数据':
                count = (df['数据异常原因'] == reason).sum()
                abnormal_reasons[reason] = count

        logger.info(f"数据分类标记完成:")
        logger.info(f"  总数据: {len(df)} 条")
        logger.info(f"  有效数据: {total_valid} 条（参与分析）")
        logger.info(f"  无效数据: {total_invalid} 条（不参与分析）")
        logger.info(f"  异常数据: {total_abnormal} 条（参与分析但标记异常）")
        logger.info(f"  无效原因分布: {invalid_reasons}")
        logger.info(f"  异常原因分布: {abnormal_reasons}")

        return df

    def _process_media_info_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理成本分析中的媒介信息（简化版）"""
        logger.info("处理成本分析媒介信息")

        # 如果定档媒介字段已存在且不为空，直接使用
        if '定档媒介' in df.columns and not df['定档媒介'].isna().all():
            logger.info("使用已有的定档媒介字段")
            return df

        # 创建新的定档媒介和定档媒介小组列
        df['定档媒介_新'] = '未知'
        df['定档媒介小组_新'] = 'other组'

        # 检查是否存在必要的列
        has_schedule_name = 'schedule_user_name' in df.columns
        has_submit_id = 'submit_media_user_id' in df.columns

        # 处理每一行数据
        for idx, row in df.iterrows():
            real_name = '未知'

            # 方法1：优先使用schedule_user_name
            if has_schedule_name:
                schedule_name = str(row.get('schedule_user_name', '')).strip()
                if schedule_name and schedule_name.lower() not in ['', 'nan', 'none', '未知']:
                    # 使用normalize_media_name函数获取真名
                    real_name = normalize_media_name(schedule_name)

            # 方法2：对于仍然未知的，使用submit_media_user_id
            if real_name == '未知' and has_submit_id:
                submit_id = str(row.get('submit_media_user_id', '')).strip()
                if submit_id and submit_id.lower() not in ['', 'nan', 'none', '未知']:
                    # 清理ID（去除.0后缀）
                    submit_id = submit_id.replace('.0', '')

                    # 先尝试全局ID_TO_NAME_MAPPING
                    if submit_id in ID_TO_NAME_MAPPING:
                        real_name = ID_TO_NAME_MAPPING[submit_id]

            # 方法3：对于仍然未知的，尝试使用submit_media_user_name
            if real_name == '未知' and 'submit_media_user_name' in df.columns:
                submit_name = str(row.get('submit_media_user_name', '')).strip()
                if submit_name and submit_name.lower() not in ['', 'nan', 'none', '未知']:
                    real_name = normalize_media_name(submit_name)

            # 设置媒介名称
            df.at[idx, '定档媒介_新'] = real_name

            # 确定所属小组
            if real_name != '未知':
                group = get_media_group(real_name)
                df.at[idx, '定档媒介小组_新'] = group

        # 如果原有字段不存在，使用新的字段
        if '定档媒介' not in df.columns:
            df['定档媒介'] = df['定档媒介_新']
            df['定档媒介小组'] = df['定档媒介小组_新']
            df = df.drop(['定档媒介_新', '定档媒介小组_新'], axis=1)

        logger.info(f"成本分析媒介信息处理完成，唯一媒介数: {df['定档媒介'].nunique()}")
        return df

    # 找到 analyze 方法，修改第一部分的验证逻辑：
    def analyze(self, top_n: int = 10) -> Dict[str, Any]:
        """
        执行完整成本发挥分析
        使用有效数据 + 异常数据进行分析（排除无效数据）
        """
        logger.info("开始执行成本发挥分析（使用有效数据+异常数据）")

        try:
            # 1. 创建用于分析的DataFrame：有效数据 + 异常数据
            if self.all_df.empty:
                logger.warning("无数据进行分析")
                self.result["overall_summary"] = {"提示": "无数据进行分析"}
                return self.result

            # 排除无效数据（成本为0或缺失、数据不全）
            analysis_df = self.all_df[~self.all_df['成本无效']].copy()

            logger.info(f"分析数据准备: 总数据 {len(self.all_df)} 条, 排除无效数据后 {len(analysis_df)} 条")

            # 正确统计异常数据
            if '数据异常' in analysis_df.columns:
                abnormal_count = analysis_df['数据异常'].sum()
            else:
                abnormal_count = 0

            logger.info(f"  其中异常数据: {abnormal_count} 条")

            # 2. 计算所有目标工作表 - 使用排除无效数据后的数据
            self._calculate_all_target_sheets_with_df(analysis_df)

            # 3. 计算整体汇总（包含异常数据统计）
            self.result["overall_summary"] = self._calculate_overall_summary()

            # 4. 新增：计算无效数据详情
            self.result["invalid_data_detail"] = self._calculate_invalid_data_detail()

            # 5. 新增：计算异常数据详情
            abnormal_detail = self._calculate_abnormal_data_detail()
            self.result["abnormal_data_detail"] = abnormal_detail

            # ✅ 关键修复：确保异常数据统计也被存储
            if abnormal_detail:
                # 计算异常原因分布
                reason_dist = {}
                total_cost = 0
                for detail in abnormal_detail:
                    reason = detail.get('数据异常原因', '未知原因')
                    reason_dist[reason] = reason_dist.get(reason, 0) + 1
                    total_cost += detail.get('成本', 0)

                # 更新 overall_summary 中的异常数据统计
                if '异常数据条数' not in self.result["overall_summary"]:
                    self.result["overall_summary"]['异常数据条数'] = len(abnormal_detail)
                if '异常数据原因分布' not in self.result["overall_summary"]:
                    self.result["overall_summary"]['异常数据原因分布'] = reason_dist
                if '异常数据总成本(元)' not in self.result["overall_summary"]:
                    self.result["overall_summary"]['异常数据总成本(元)'] = total_cost

                # 创建独立的 abnormal_data_stats
                self.result["abnormal_data_stats"] = {
                    '异常数据条数': len(abnormal_detail),
                    '异常数据比例(%)': f"{(len(abnormal_detail) / len(self.all_df) * 100):.2f}%" if len(
                        self.all_df) > 0 else '0%',
                    '异常数据原因分布': reason_dist,
                    '异常数据总成本(元)': total_cost,
                    '参与分析数据条数': len(analysis_df),
                    '参与分析数据比例(%)': f"{(len(analysis_df) / len(self.all_df) * 100):.2f}%" if len(
                        self.all_df) > 0 else '0%'
                }

            # 6. 计算媒介明细
            self.result["media_detail"] = self._calculate_media_detail()

            # 7. 计算小组汇总
            self.result["group_summary"] = self._calculate_group_summary()

            # 8. 计算筛除数据汇总
            self.result["filtered_summary"] = self._calculate_filtered_summary()

            # 9. 生成成本效益排名
            self.result["cost_efficiency_ranking"] = self._generate_cost_efficiency_ranking(top_n)

            # 10. 设置详细数据 - 使用分析数据（有效+异常）
            self.result["detailed_data"] = analysis_df

            logger.info("成本发挥分析执行完成")
            return self.result

        except Exception as e:
            logger.error(f"成本发挥分析执行失败: {e}", exc_info=True)
            raise

    # 找到这个方法，修改为使用 all_df：
    def _calculate_all_target_sheets(self) -> None:
        """计算所有目标工作表（兼容旧版本）"""
        logger.info("开始计算所有目标工作表（使用所有数据）")

        # 排除无效数据（成本为0或缺失、数据不全）
        analysis_df = self.all_df[~self.all_df['成本无效']].copy()

        # 使用新的方法
        self._calculate_all_target_sheets_with_df(analysis_df)

    def _calculate_all_target_sheets_with_df(self, df: pd.DataFrame) -> None:
        """计算所有目标工作表（使用指定的DataFrame）"""
        logger.info("开始计算所有目标工作表（使用指定数据）")

        # 1. 媒介小组工作量分析
        self.result["media_group_workload"] = self._calculate_media_group_workload(df)

        # 2. 定档媒介工作量分析
        self.result["fixed_media_workload"] = self._calculate_fixed_media_workload(df)

        # 3. 定档媒介成本分析
        self.result["fixed_media_cost"] = self._calculate_fixed_media_cost(df)

        # 4. 定档媒介返点分析
        self.result["fixed_media_rebate"] = self._calculate_fixed_media_rebate(df)

        # 5. 定档媒介效果分析
        self.result["fixed_media_performance"] = self._calculate_fixed_media_performance(df)

        # 6. 定档媒介达人量级分析
        self.result["fixed_media_level"] = self._calculate_fixed_media_level(df)

        # 7. 定档媒介综合分析
        self.result["fixed_media_comprehensive"] = self._calculate_fixed_media_comprehensive(df)

        logger.info("所有目标工作表计算完成")

    def _calculate_media_group_workload(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算媒介小组工作量分析
        字段：媒介小组,总达人数,总项目数,媒介人数,已发布数,未发布数,发布率(%),总成本(元),平均成本(元),总返点金额(元),平均返点金额(元),平均返点比例(%),返点表现
        """
        logger.info("计算媒介小组工作量分析")

        if df.empty or '定档媒介小组' not in df.columns:
            return pd.DataFrame({'提示': ['没有足够的数据进行媒介小组分析']})

        analysis_results = []

        # 按定档媒介小组分组
        if '定档媒介小组' in df.columns:
            group_groups = df.groupby('定档媒介小组')

            for group_name, group_data in group_groups:
                group_analysis = {
                    '媒介小组': group_name,
                    '总达人数': len(group_data),
                    '总项目数': group_data['项目名称'].nunique() if '项目名称' in group_data.columns else 0
                }

                # 按定档媒介统计
                if '定档媒介' in group_data.columns:
                    media_count = group_data['定档媒介'].nunique()
                    group_analysis['媒介人数'] = media_count

                    # 状态分布
                    if '状态' in group_data.columns:
                        chained = len(group_data[group_data['状态'] == 'CHAIN_RETURNED'])
                        scheduled = len(group_data[group_data['状态'] == 'SCHEDULED'])
                        group_analysis['已发布数'] = chained
                        group_analysis['未发布数'] = scheduled
                        if (chained + scheduled) > 0:
                            group_analysis['发布率(%)'] = f"{(chained / (chained + scheduled) * 100):.1f}%"
                        else:
                            group_analysis['发布率(%)'] = "0%"

                # 成本统计
                if '成本' in df.columns:
                    group_analysis['总成本(元)'] = round(group_data['成本'].sum(), 2)
                    if len(group_data) > 0:
                        group_analysis['平均成本(元)'] = round(group_data['成本'].mean(), 2)
                    else:
                        group_analysis['平均成本(元)'] = 0

                # 返点统计
                if '返点金额' in df.columns:
                    group_analysis['总返点金额(元)'] = round(group_data['返点金额'].sum(), 2)
                    if len(group_data) > 0:
                        group_analysis['平均返点金额(元)'] = round(group_data['返点金额'].mean(), 2)
                    else:
                        group_analysis['平均返点金额(元)'] = 0

                # 返点比例统计
                if '返点比例' in df.columns:
                    valid_rebate = group_data[group_data['返点比例'].notna()]['返点比例']
                    if len(valid_rebate) > 0:
                        avg_rebate = valid_rebate.mean() * 100
                        group_analysis['平均返点比例(%)'] = f"{avg_rebate:.1f}%"

                        # 返点表现评估（与原始代码一致）
                        if avg_rebate >= 35:
                            group_analysis['返点表现'] = '优秀'
                        elif avg_rebate >= 25:
                            group_analysis['返点表现'] = '良好'
                        elif avg_rebate >= 20:
                            group_analysis['返点表现'] = '一般'
                        elif avg_rebate >= 10:
                            group_analysis['返点表现'] = '较差'
                        else:
                            group_analysis['返点表现'] = '很差'
                    else:
                        group_analysis['平均返点比例(%)'] = 'N/A'
                        group_analysis['返点表现'] = 'N/A'

                analysis_results.append(group_analysis)

        # 转换为DataFrame并筛选字段
        if analysis_results:
            result_df = pd.DataFrame(analysis_results)

            # 只保留指定字段
            required_columns = [
                '媒介小组', '总达人数', '总项目数', '媒介人数', '已发布数', '未发布数',
                '发布率(%)', '总成本(元)', '平均成本(元)', '总返点金额(元)',
                '平均返点金额(元)', '平均返点比例(%)', '返点表现'
            ]

            # 只保留存在的字段
            existing_columns = [col for col in required_columns if col in result_df.columns]
            result_df = result_df[existing_columns]

            # 确保字段顺序正确
            final_columns = []
            for col in required_columns:
                if col in result_df.columns:
                    final_columns.append(col)

            result_df = result_df[final_columns]

            # 按小组顺序排序：数码媒介组 → 家居媒介组 → 快消媒介组 → other组
            group_order = {'数码媒介组': 0, '家居媒介组': 1, '快消媒介组': 2}

            if '媒介小组' in result_df.columns:
                # 添加排序键
                result_df['_排序键'] = result_df['媒介小组'].apply(
                    lambda x: group_order.get(x, 99)  # 其他组排在最后
                )
                # 按排序键排序
                result_df = result_df.sort_values('_排序键')
                # 删除临时排序键
                result_df = result_df.drop('_排序键', axis=1)

            return result_df
        else:
            return pd.DataFrame({'提示': ['没有足够的数据进行媒介小组分析']})

    def _calculate_fixed_media_workload(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算定档媒介工作量分析
        字段：定档媒介,定档达人数,合作项目数,所属小组,总互动量,平均互动量,总阅读量,平均阅读量,总返点金额(元),平均返点金额(元),平均返点比例(%),返点表现评估,返点分布,总手续费(元),平均手续费(元),图文笔记数,视频笔记数,图文视频比,已发布数,未发布数,其他状态数,发布率(%),互动量最大值,互动量最小值
        """
        logger.info("计算定档媒介工作量分析")

        if df.empty or '定档媒介' not in df.columns:
            return pd.DataFrame({'错误': ['缺少定档媒介字段']})

        analysis_results = []

        # 按定档媒介分组
        media_groups = df.groupby('定档媒介')

        for media, group in media_groups:
            media_data = {
                '定档媒介': media,
                '定档达人数': len(group),
                '合作项目数': group['项目名称'].nunique() if '项目名称' in group.columns else 0
            }

            # 如果有媒介小组信息，也显示
            if '定档媒介小组' in group.columns and group['定档媒介小组'].iloc[0] != '未知':
                media_data['所属小组'] = group['定档媒介小组'].iloc[0]
            else:
                media_data['所属小组'] = '未知'

            # 互动量统计
            if '互动量' in df.columns:
                valid_interaction = group[group['互动量'].notna()]
                if len(valid_interaction) > 0:
                    media_data['总互动量'] = round(valid_interaction['互动量'].sum(), 0)
                    media_data['平均互动量'] = round(valid_interaction['互动量'].mean(), 0)
                    media_data['互动量最大值'] = round(valid_interaction['互动量'].max(), 0)
                    media_data['互动量最小值'] = round(valid_interaction['互动量'].min(), 0)
                else:
                    media_data['总互动量'] = 0
                    media_data['平均互动量'] = 0
                    media_data['互动量最大值'] = 0
                    media_data['互动量最小值'] = 0

            # 阅读量统计
            if '阅读量' in df.columns:
                valid_read = group[group['阅读量'].notna()]
                if len(valid_read) > 0:
                    media_data['总阅读量'] = round(valid_read['阅读量'].sum(), 0)
                    media_data['平均阅读量'] = round(valid_read['阅读量'].mean(), 0)
                else:
                    media_data['总阅读量'] = 0
                    media_data['平均阅读量'] = 0

            # 返点金额统计
            if '返点金额' in df.columns:
                valid_rebate_amount = group[group['返点金额'].notna()]
                if len(valid_rebate_amount) > 0:
                    media_data['总返点金额(元)'] = round(valid_rebate_amount['返点金额'].sum(), 2)
                    media_data['平均返点金额(元)'] = round(valid_rebate_amount['返点金额'].mean(), 2)
                else:
                    media_data['总返点金额(元)'] = 0
                    media_data['平均返点金额(元)'] = 0

            # 返点比例统计和评估
            if '返点比例' in df.columns:
                valid_rebate_ratio = group[group['返点比例'].notna()]['返点比例']
                if len(valid_rebate_ratio) > 0:
                    avg_rebate_ratio = valid_rebate_ratio.mean() * 100
                    media_data['平均返点比例(%)'] = f"{avg_rebate_ratio:.2f}%"

                    # 返点表现评估（与原始代码一致）
                    if avg_rebate_ratio < 10:
                        media_data['返点表现评估'] = '很差'
                    elif avg_rebate_ratio < 20:
                        media_data['返点表现评估'] = '较差'
                    elif avg_rebate_ratio < 25:
                        media_data['返点表现评估'] = '一般'
                    elif avg_rebate_ratio < 35:
                        media_data['返点表现评估'] = '良好'
                    else:
                        media_data['返点表现评估'] = '优秀'

                    # 返点比例分布统计
                    very_poor_count = len(valid_rebate_ratio[valid_rebate_ratio < 0.1])
                    poor_count = len(valid_rebate_ratio[(valid_rebate_ratio >= 0.1) & (valid_rebate_ratio < 0.2)])
                    normal_count = len(valid_rebate_ratio[(valid_rebate_ratio >= 0.2) & (valid_rebate_ratio < 0.25)])
                    good_count = len(valid_rebate_ratio[(valid_rebate_ratio >= 0.25) & (valid_rebate_ratio < 0.35)])
                    excellent_count = len(valid_rebate_ratio[valid_rebate_ratio >= 0.35])

                    media_data[
                        '返点分布'] = f"很差(<10%):{very_poor_count} 较差(10-20%):{poor_count} 一般(20-25%):{normal_count} 良好(25-35%):{good_count} 优秀(>35%):{excellent_count}"
                else:
                    media_data['平均返点比例(%)'] = 'N/A'
                    media_data['返点表现评估'] = 'N/A'
                    media_data['返点分布'] = 'N/A'

            # 手续费统计
            if '手续费' in df.columns:
                valid_service_charge = group[group['手续费'].notna()]
                if len(valid_service_charge) > 0:
                    media_data['总手续费(元)'] = round(valid_service_charge['手续费'].sum(), 2)
                    media_data['平均手续费(元)'] = round(valid_service_charge['手续费'].mean(), 2)
                else:
                    media_data['总手续费(元)'] = 0
                    media_data['平均手续费(元)'] = 0

            # 笔记类型统计
            if '笔记类型(图文/视频)' in df.columns:
                article_count = len(group[group['笔记类型(图文/视频)'] == '图文'])
                video_count = len(group[group['笔记类型(图文/视频)'] == '视频'])
                media_data['图文笔记数'] = article_count
                media_data['视频笔记数'] = video_count
                if video_count > 0:
                    media_data['图文视频比'] = f"{article_count}:{video_count}"
                else:
                    media_data['图文视频比'] = f"{article_count}:0"

            # 状态统计
            if '状态' in df.columns:
                chained_count = len(group[group['状态'] == 'CHAIN_RETURNED'])
                scheduled_count = len(group[group['状态'] == 'SCHEDULED'])
                other_count = len(group) - chained_count - scheduled_count
                media_data['已发布数'] = chained_count
                media_data['未发布数'] = scheduled_count
                media_data['其他状态数'] = other_count

                # 发布率
                total_published = chained_count + scheduled_count
                if total_published > 0:
                    media_data['发布率(%)'] = f"{(chained_count / total_published * 100):.1f}%"
                else:
                    media_data['发布率(%)'] = '0%'

            analysis_results.append(media_data)

        # 转换为DataFrame并筛选字段
        if analysis_results:
            result_df = pd.DataFrame(analysis_results)

            # 只保留指定字段
            required_columns = [
                '定档媒介', '定档达人数', '合作项目数', '所属小组', '总互动量', '平均互动量',
                '总阅读量', '平均阅读量', '总返点金额(元)', '平均返点金额(元)', '平均返点比例(%)',
                '返点表现评估', '返点分布', '总手续费(元)', '平均手续费(元)', '图文笔记数',
                '视频笔记数', '图文视频比', '已发布数', '未发布数', '其他状态数', '发布率(%)',
                '互动量最大值', '互动量最小值'
            ]

            # 只保留存在的字段
            existing_columns = [col for col in required_columns if col in result_df.columns]
            result_df = result_df[existing_columns]

            # 确保字段顺序正确
            final_columns = []
            for col in required_columns:
                if col in result_df.columns:
                    final_columns.append(col)

            result_df = result_df[final_columns]

            # 按所属小组顺序排序：数码媒介组 → 家居媒介组 → 快消媒介组 → 其他
            group_order = {'数码媒介组': 0, '家居媒介组': 1, '快消媒介组': 2}

            if '所属小组' in result_df.columns:
                # 添加排序键
                result_df['_排序键'] = result_df['所属小组'].apply(
                    lambda x: group_order.get(x, 99)  # 其他小组排在最后
                )
                # 按排序键和定档达人数排序
                result_df = result_df.sort_values(['_排序键', '定档达人数'], ascending=[True, False])
                # 删除临时排序键
                result_df = result_df.drop('_排序键', axis=1)

            return result_df
        else:
            return pd.DataFrame({'提示': ['没有足够的数据进行定档媒介工作量分析']})

    def _calculate_fixed_media_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算定档媒介成本分析
        字段：定档媒介,定档达人数,所属小组,总成本(元),平均成本(元),成本最大值(元),成本最小值(元),成本中位数(元),总报价(元),平均报价(元),报价最大值(元),报价最小值(元),总下单价(元),平均下单价(元),总节约金额(元),平均节约金额(元),成本占比(%),总返点金额(元),平均返点金额(元)
        """
        logger.info("计算定档媒介成本分析")

        if df.empty or '定档媒介' not in df.columns:
            return pd.DataFrame({'错误': ['缺少定档媒介字段']})

        analysis_results = []

        # 按定档媒介分组
        media_groups = df.groupby('定档媒介')

        for media, group in media_groups:
            media_data = {
                '定档媒介': media,
                '定档达人数': len(group),
                '有效数据条数': len(group[~group['成本无效']]),
                '无效数据条数': len(group[group['成本无效']])
            }

            # 如果有媒介小组信息，也显示
            if '定档媒介小组' in group.columns and group['定档媒介小组'].iloc[0] != '未知':
                media_data['所属小组'] = group['定档媒介小组'].iloc[0]
            else:
                media_data['所属小组'] = '未知'

            # 成本相关指标
            if '成本' in df.columns:
                media_data['总成本(元)'] = round(group['成本'].sum(), 2)
                if len(group) > 0:
                    media_data['平均成本(元)'] = round(group['成本'].mean(), 2)
                    media_data['成本最大值(元)'] = round(group['成本'].max(), 2)
                    media_data['成本最小值(元)'] = round(group['成本'].min(), 2)
                    media_data['成本中位数(元)'] = round(group['成本'].median(), 2)
                else:
                    media_data['平均成本(元)'] = 0
                    media_data['成本最大值(元)'] = 0
                    media_data['成本最小值(元)'] = 0
                    media_data['成本中位数(元)'] = 0

            # 报价相关指标
            if '报价' in df.columns:
                media_data['总报价(元)'] = round(group['报价'].sum(), 2)
                if len(group) > 0:
                    media_data['平均报价(元)'] = round(group['报价'].mean(), 2)
                    media_data['报价最大值(元)'] = round(group['报价'].max(), 2)
                    media_data['报价最小值(元)'] = round(group['报价'].min(), 2)
                else:
                    media_data['平均报价(元)'] = 0
                    media_data['报价最大值(元)'] = 0
                    media_data['报价最小值(元)'] = 0

            # 下单价相关指标
            if '下单价' in df.columns:
                media_data['总下单价(元)'] = round(group['下单价'].sum(), 2)
                if len(group) > 0:
                    media_data['平均下单价(元)'] = round(group['下单价'].mean(), 2)
                else:
                    media_data['平均下单价(元)'] = 0

            # 计算节约金额
            if '报价' in df.columns and '成本' in df.columns:
                media_data['总节约金额(元)'] = round((group['报价'].sum() - group['成本'].sum()), 2)
                if len(group) > 0:
                    media_data['平均节约金额(元)'] = round((group['报价'].mean() - group['成本'].mean()), 2)
                else:
                    media_data['平均节约金额(元)'] = 0

            # 计算成本占比
            if '成本' in df.columns:
                total_cost_all = df['成本'].sum()
                if total_cost_all > 0:
                    media_data['成本占比(%)'] = f"{media_data['总成本(元)'] / total_cost_all * 100:.2f}%"
                else:
                    media_data['成本占比(%)'] = '0%'

            # 返点金额统计
            if '返点金额' in df.columns:
                media_data['总返点金额(元)'] = round(group['返点金额'].sum(), 2)
                if len(group) > 0:
                    media_data['平均返点金额(元)'] = round(group['返点金额'].mean(), 2)
                else:
                    media_data['平均返点金额(元)'] = 0

            analysis_results.append(media_data)

        # 转换为DataFrame并筛选字段
        if analysis_results:
            result_df = pd.DataFrame(analysis_results)

            # 只保留指定字段
            required_columns = [
                '定档媒介', '定档达人数', '所属小组', '总成本(元)', '平均成本(元)',
                '成本最大值(元)', '成本最小值(元)', '成本中位数(元)', '总报价(元)',
                '平均报价(元)', '报价最大值(元)', '报价最小值(元)', '总下单价(元)',
                '平均下单价(元)', '总节约金额(元)', '平均节约金额(元)', '成本占比(%)',
                '总返点金额(元)', '平均返点金额(元)'
            ]

            # 只保留存在的字段
            existing_columns = [col for col in required_columns if col in result_df.columns]
            result_df = result_df[existing_columns]

            # 确保字段顺序正确
            final_columns = []
            for col in required_columns:
                if col in result_df.columns:
                    final_columns.append(col)

            result_df = result_df[final_columns]

            # 按所属小组顺序排序：数码媒介组 → 家居媒介组 → 快消媒介组 → 其他
            group_order = {'数码媒介组': 0, '家居媒介组': 1, '快消媒介组': 2}

            if '所属小组' in result_df.columns:
                # 添加排序键
                result_df['_排序键'] = result_df['所属小组'].apply(
                    lambda x: group_order.get(x, 99)  # 其他小组排在最后
                )
                # 按排序键和总成本排序
                result_df = result_df.sort_values(['_排序键', '总成本(元)'], ascending=[True, False])
                # 删除临时排序键
                result_df = result_df.drop('_排序键', axis=1)

            return result_df
        else:
            return pd.DataFrame({'提示': ['没有足够的数据进行定档媒介成本分析']})

    def _calculate_fixed_media_rebate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算定档媒介返点分析
        字段：定档媒介,定档达人数,所属小组,平均返点比例(%),返点比例最大值(%),返点比例最小值(%),返点比例中位数(%),返点表现评估,返点优化建议,总返点金额(元),平均返点金额(元),返点金额最大值(元),返点金额最小值(元),返点金额中位数(元)
        """
        logger.info("计算定档媒介返点分析")

        if df.empty or '定档媒介' not in df.columns:
            return pd.DataFrame({'错误': ['缺少定档媒介字段']})

        analysis_results = []

        # 按定档媒介分组
        media_groups = df.groupby('定档媒介')

        for media, group in media_groups:
            media_data = {
                '定档媒介': media,
                '定档达人数': len(group)
            }

            # 如果有媒介小组信息，也显示
            if '定档媒介小组' in group.columns and group['定档媒介小组'].iloc[0] != '未知':
                media_data['所属小组'] = group['定档媒介小组'].iloc[0]
            else:
                media_data['所属小组'] = '未知'

            # 返点金额分析
            if '返点金额' in df.columns:
                media_data['总返点金额(元)'] = round(group['返点金额'].sum(), 2)
                if len(group) > 0:
                    media_data['平均返点金额(元)'] = round(group['返点金额'].mean(), 2)
                    media_data['返点金额最大值(元)'] = round(group['返点金额'].max(), 2)
                    media_data['返点金额最小值(元)'] = round(group['返点金额'].min(), 2)
                    media_data['返点金额中位数(元)'] = round(group['返点金额'].median(), 2)
                else:
                    media_data['平均返点金额(元)'] = 0
                    media_data['返点金额最大值(元)'] = 0
                    media_data['返点金额最小值(元)'] = 0
                    media_data['返点金额中位数(元)'] = 0

            # 返点比例分析
            if '返点比例' in df.columns:
                # 计算有效返点比例数据
                valid_rebate_ratio = group[group['返点比例'].notna()]['返点比例']
                if len(valid_rebate_ratio) > 0:
                    media_data['平均返点比例(%)'] = f"{valid_rebate_ratio.mean() * 100:.2f}%"
                    media_data['返点比例最大值(%)'] = f"{valid_rebate_ratio.max() * 100:.2f}%"
                    media_data['返点比例最小值(%)'] = f"{valid_rebate_ratio.min() * 100:.2f}%"
                    media_data['返点比例中位数(%)'] = f"{valid_rebate_ratio.median() * 100:.2f}%"

                    # 返点比例分布
                    high_rebate = len(valid_rebate_ratio[valid_rebate_ratio >= 0.3])  # 30%以上
                    medium_rebate = len(
                        valid_rebate_ratio[(valid_rebate_ratio >= 0.2) & (valid_rebate_ratio < 0.3)])  # 20-30%
                    low_rebate = len(valid_rebate_ratio[valid_rebate_ratio < 0.2])  # 20%以下
                    media_data[
                        '返点比例分布'] = f"高(≥30%):{high_rebate}/中(20-30%):{medium_rebate}/低(<20%):{low_rebate}"
                else:
                    media_data['平均返点比例(%)'] = 'N/A'
                    media_data['返点比例最大值(%)'] = 'N/A'
                    media_data['返点比例最小值(%)'] = 'N/A'
                    media_data['返点比例中位数(%)'] = 'N/A'
                    media_data['返点比例分布'] = 'N/A'

            # 评估返点表现
            if '平均返点比例(%)' in media_data and media_data['平均返点比例(%)'] != 'N/A':
                avg_rebate = float(media_data['平均返点比例(%)'].rstrip('%'))
                if avg_rebate >= 35:
                    media_data['返点表现评估'] = '优秀'
                    media_data['返点优化建议'] = '保持高水平，可作为标杆'
                elif avg_rebate >= 25:
                    media_data['返点表现评估'] = '良好'
                    media_data['返点优化建议'] = '表现良好，仍有提升空间'
                elif avg_rebate >= 20:
                    media_data['返点表现评估'] = '一般'
                    media_data['返点优化建议'] = '需加强谈判，争取更高返点'
                elif avg_rebate >= 10:
                    media_data['返点表现评估'] = '较差'
                    media_data['返点优化建议'] = '返点偏低，需重新评估合作策略'
                else:
                    media_data['返点表现评估'] = '很差'
                    media_data['返点优化建议'] = '返点严重偏低，建议重新谈判或更换媒介'
            else:
                media_data['返点表现评估'] = 'N/A'
                media_data['返点优化建议'] = 'N/A'

            analysis_results.append(media_data)

        # 转换为DataFrame并筛选字段
        if analysis_results:
            result_df = pd.DataFrame(analysis_results)

            # 只保留指定字段
            required_columns = [
                '定档媒介', '定档达人数', '所属小组', '平均返点比例(%)', '返点比例最大值(%)',
                '返点比例最小值(%)', '返点比例中位数(%)', '返点比例分布', '总返点金额(元)',
                '平均返点金额(元)', '返点金额最大值(元)', '返点金额最小值(元)', '返点金额中位数(元)',
                '返点表现评估', '返点优化建议'
            ]

            # 只保留存在的字段
            existing_columns = [col for col in required_columns if col in result_df.columns]
            result_df = result_df[existing_columns]

            # 确保字段顺序正确
            final_columns = []
            for col in required_columns:
                if col in result_df.columns:
                    final_columns.append(col)

            result_df = result_df[final_columns]

            # 按所属小组顺序排序：数码媒介组 → 家居媒介组 → 快消媒介组 → 其他
            group_order = {'数码媒介组': 0, '家居媒介组': 1, '快消媒介组': 2}

            if '所属小组' in result_df.columns:
                # 添加排序键
                result_df['_排序键'] = result_df['所属小组'].apply(
                    lambda x: group_order.get(x, 99)  # 其他小组排在最后
                )
                # 提取平均返点比例数值用于排序
                result_df['平均返点比例_数值'] = result_df['平均返点比例(%)'].apply(
                    lambda x: float(x.rstrip('%')) if x != 'N/A' else 0
                )
                # 按排序键和平均返点比例排序
                result_df = result_df.sort_values(['_排序键', '平均返点比例_数值'], ascending=[True, False])
                # 删除临时排序键
                result_df = result_df.drop(['_排序键', '平均返点比例_数值'], axis=1)

            return result_df
        else:
            return pd.DataFrame({'提示': ['没有足够的数据进行定档媒介返点分析']})

    def _calculate_fixed_media_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算定档媒介效果分析
        字段：定档媒介,定档达人数,所属小组,总互动量,平均互动量,互动量最大值,互动量最小值,互动量标准差,总阅读量,平均阅读量,阅读量最大值,阅读量最小值,总曝光量,平均曝光量,效果评估,效果建议,平均CPE,CPE最小值,CPE最大值,CPE中位数,CPE评估,每元互动量,成本效益评估,平均CPM,CPM最小值,CPM最大值,CPM中位数,CPM标准差,CPM评估,CPM/CPE比率,性价比评估,性价比分析,平均互动率(%),互动率评估,每元阅读量
        """
        logger.info("计算定档媒介效果分析")

        if df.empty or '定档媒介' not in df.columns:
            return pd.DataFrame({'错误': ['缺少定档媒介字段']})

        # 检查是否已有CPM/CPE字段，如果没有则尝试计算
        if 'cpm' not in df.columns and '成本' in df.columns and '曝光量' in df.columns:
            mask = df['成本'].notna() & df['曝光量'].notna() & (df['曝光量'] > 0)
            df['cpm'] = np.nan
            df.loc[mask, 'cpm'] = df.loc[mask, '成本'] / df.loc[mask, '曝光量'] * 1000

        if 'cpe' not in df.columns and '成本' in df.columns and '互动量' in df.columns:
            mask = df['成本'].notna() & df['互动量'].notna() & (df['互动量'] > 0)
            df['cpe'] = np.nan
            df.loc[mask, 'cpe'] = df.loc[mask, '成本'] / df.loc[mask, '互动量']

        analysis_results = []

        # 按定档媒介分组
        media_groups = df.groupby('定档媒介')

        for media, group in media_groups:
            media_data = {
                '定档媒介': media,
                '定档达人数': len(group)
            }

            # 如果有媒介小组信息，也显示
            if '定档媒介小组' in group.columns and group['定档媒介小组'].iloc[0] != '未知':
                media_data['所属小组'] = group['定档媒介小组'].iloc[0]
            else:
                media_data['所属小组'] = '未知'

            # 互动效果
            if '互动量' in df.columns:
                valid_interaction = group[group['互动量'].notna()]
                if len(valid_interaction) > 0:
                    media_data['总互动量'] = round(valid_interaction['互动量'].sum(), 0)
                    media_data['平均互动量'] = round(valid_interaction['互动量'].mean(), 0)
                    media_data['互动量最大值'] = round(valid_interaction['互动量'].max(), 0)
                    media_data['互动量最小值'] = round(valid_interaction['互动量'].min(), 0)
                    media_data['互动量标准差'] = round(valid_interaction['互动量'].std(), 0)
                else:
                    media_data['总互动量'] = 0
                    media_data['平均互动量'] = 0
                    media_data['互动量最大值'] = 0
                    media_data['互动量最小值'] = 0
                    media_data['互动量标准差'] = 0

            # 阅读效果
            if '阅读量' in df.columns:
                valid_read = group[group['阅读量'].notna()]
                if len(valid_read) > 0:
                    media_data['总阅读量'] = round(valid_read['阅读量'].sum(), 0)
                    media_data['平均阅读量'] = round(valid_read['阅读量'].mean(), 0)
                    media_data['阅读量最大值'] = round(valid_read['阅读量'].max(), 0)
                    media_data['阅读量最小值'] = round(valid_read['阅读量'].min(), 0)
                else:
                    media_data['总阅读量'] = 0
                    media_data['平均阅读量'] = 0
                    media_data['阅读量最大值'] = 0
                    media_data['阅读量最小值'] = 0

            # 曝光效果
            if '曝光量' in df.columns:
                valid_exposure = group[group['曝光量'].notna()]
                if len(valid_exposure) > 0:
                    media_data['总曝光量'] = round(valid_exposure['曝光量'].sum(), 0)
                    media_data['平均曝光量'] = round(valid_exposure['曝光量'].mean(), 0)
                else:
                    media_data['总曝光量'] = 0
                    media_data['平均曝光量'] = 0

            # CPE分析
            if 'cpe' in df.columns:
                valid_cpe = group[group['cpe'].notna()]['cpe']
                if len(valid_cpe) > 0:
                    media_data['平均CPE'] = round(valid_cpe.mean(), 2)
                    media_data['CPE最小值'] = round(valid_cpe.min(), 2)
                    media_data['CPE最大值'] = round(valid_cpe.max(), 2)
                    media_data['CPE中位数'] = round(valid_cpe.median(), 2)

                    # CPE评估（与原始代码一致）
                    avg_cpe = media_data['平均CPE']
                    if avg_cpe <= 2:
                        media_data['CPE评估'] = '优秀'
                    elif avg_cpe <= 5:
                        media_data['CPE评估'] = '良好'
                    elif avg_cpe <= 10:
                        media_data['CPE评估'] = '一般'
                    elif avg_cpe <= 20:
                        media_data['CPE评估'] = '较差'
                    else:
                        media_data['CPE评估'] = '很差'
                else:
                    media_data['平均CPE'] = 0
                    media_data['CPE最小值'] = 0
                    media_data['CPE最大值'] = 0
                    media_data['CPE中位数'] = 0
                    media_data['CPE评估'] = 'N/A'

            # CPM分析
            if 'cpm' in df.columns:
                valid_cpm = group[group['cpm'].notna()]['cpm']
                if len(valid_cpm) > 0:
                    media_data['平均CPM'] = round(valid_cpm.mean(), 2)
                    media_data['CPM最小值'] = round(valid_cpm.min(), 2)
                    media_data['CPM最大值'] = round(valid_cpm.max(), 2)
                    media_data['CPM中位数'] = round(valid_cpm.median(), 2)
                    media_data['CPM标准差'] = round(valid_cpm.std(), 2)

                    # CPM评估（与原始代码一致）
                    avg_cpm = media_data['平均CPM']
                    if avg_cpm <= 50:
                        media_data['CPM评估'] = '优秀'
                    elif avg_cpm <= 100:
                        media_data['CPM评估'] = '良好'
                    elif avg_cpm <= 200:
                        media_data['CPM评估'] = '一般'
                    elif avg_cpm <= 300:
                        media_data['CPM评估'] = '较差'
                    else:
                        media_data['CPM评估'] = '很差'
                else:
                    media_data['平均CPM'] = 0
                    media_data['CPM最小值'] = 0
                    media_data['CPM最大值'] = 0
                    media_data['CPM中位数'] = 0
                    media_data['CPM标准差'] = 0
                    media_data['CPM评估'] = 'N/A'

            # CPM/CPE对比分析
            if '平均CPM' in media_data and '平均CPE' in media_data:
                valid_both = group[group['cpm'].notna() & group['cpe'].notna()]
                if len(valid_both) > 0:
                    media_data['CPM/CPE比率'] = round(
                        valid_both['cpm'].mean() / valid_both['cpe'].mean() if valid_both['cpe'].mean() > 0 else 0, 2)

                    # 性价比评估
                    cpm_score = 1 if media_data.get('平均CPM', 999) <= 100 else 0
                    cpe_score = 1 if media_data.get('平均CPE', 999) <= 5 else 0

                    if cpm_score == 1 and cpe_score == 1:
                        media_data['性价比评估'] = '高性价比'
                        media_data['性价比分析'] = 'CPM和CPE均表现优秀'
                    elif cpm_score == 1:
                        media_data['性价比评估'] = '曝光成本优势'
                        media_data['性价比分析'] = 'CPM优秀但CPE有待提升'
                    elif cpe_score == 1:
                        media_data['性价比评估'] = '互动成本优势'
                        media_data['性价比分析'] = 'CPE优秀但CPM有待提升'
                    else:
                        media_data['性价比评估'] = '需全面优化'
                        media_data['性价比分析'] = 'CPM和CPE均需优化'
                else:
                    media_data['CPM/CPE比率'] = 0
                    media_data['性价比评估'] = 'N/A'
                    media_data['性价比分析'] = 'N/A'

            # 计算每元互动量
            if '互动量' in df.columns and '成本' in df.columns:
                valid_combo = group[group['互动量'].notna() & group['成本'].notna() & (group['成本'] > 0)]
                if len(valid_combo) > 0:
                    interaction_per_yuan = valid_combo['互动量'].sum() / valid_combo['成本'].sum()
                    media_data['每元互动量'] = round(interaction_per_yuan, 4)

                    # 成本效益评估
                    if interaction_per_yuan >= 20:
                        media_data['成本效益评估'] = '优秀'
                    elif interaction_per_yuan >= 10:
                        media_data['成本效益评估'] = '良好'
                    elif interaction_per_yuan >= 5:
                        media_data['成本效益评估'] = '一般'
                    else:
                        media_data['成本效益评估'] = '较差'
                else:
                    media_data['每元互动量'] = 0
                    media_data['成本效益评估'] = 'N/A'

            # 计算互动率
            if '互动量' in df.columns and '阅读量' in df.columns:
                valid_interaction = group[group['互动量'].notna() & group['阅读量'].notna() & (group['阅读量'] > 0)]
                if len(valid_interaction) > 0:
                    interaction_rate = (valid_interaction['互动量'].sum() / valid_interaction['阅读量'].sum()) * 100
                    media_data['平均互动率(%)'] = f"{interaction_rate:.2f}%"

                    # 互动率评估
                    if interaction_rate >= 5:
                        media_data['互动率评估'] = '优秀'
                    elif interaction_rate >= 3:
                        media_data['互动率评估'] = '良好'
                    elif interaction_rate >= 1:
                        media_data['互动率评估'] = '一般'
                    else:
                        media_data['互动率评估'] = '较差'
                else:
                    media_data['平均互动率(%)'] = '0%'
                    media_data['互动率评估'] = 'N/A'

            # 计算每元阅读量
            if '阅读量' in df.columns and '成本' in df.columns:
                valid_combo = group[group['阅读量'].notna() & group['成本'].notna() & (group['成本'] > 0)]
                if len(valid_combo) > 0:
                    read_per_yuan = valid_combo['阅读量'].sum() / valid_combo['成本'].sum()
                    media_data['每元阅读量'] = round(read_per_yuan, 4)
                else:
                    media_data['每元阅读量'] = 0

            # 综合效果评估
            if '平均CPM' in media_data and '平均CPE' in media_data:
                # 简单评分逻辑（与原始代码一致）
                cpm_score = 30 if media_data.get('CPM评估') == '优秀' else (
                    25 if media_data.get('CPM评估') == '良好' else (
                        20 if media_data.get('CPM评估') == '一般' else (
                            15 if media_data.get('CPM评估') == '较差' else 5
                        )))

                cpe_score = 30 if media_data.get('CPE评估') == '优秀' else (
                    25 if media_data.get('CPE评估') == '良好' else (
                        20 if media_data.get('CPE评估') == '一般' else (
                            15 if media_data.get('CPE评估') == '较差' else 5
                        )))

                cost_score = 20 if media_data.get('成本效益评估') == '优秀' else (
                    15 if media_data.get('成本效益评估') == '良好' else (
                        10 if media_data.get('成本效益评估') == '一般' else 5
                    ))

                interaction_score = 20 if media_data.get('互动率评估') == '优秀' else (
                    15 if media_data.get('互动率评估') == '良好' else (
                        10 if media_data.get('互动率评估') == '一般' else 5
                    ))

                total_score = cpm_score + cpe_score + cost_score + interaction_score
                score_percentage = (total_score / 100) * 100

                if score_percentage >= 85:
                    media_data['效果评估'] = '优秀'
                    media_data['效果建议'] = '表现优异，可加大投放'
                elif score_percentage >= 75:
                    media_data['效果评估'] = '良好'
                    media_data['效果建议'] = '表现良好，保持稳定合作'
                elif score_percentage >= 65:
                    media_data['效果评估'] = '一般'
                    media_data['效果建议'] = '有提升空间，建议优化'
                elif score_percentage >= 50:
                    media_data['效果评估'] = '较差'
                    media_data['效果建议'] = '需重点优化，谨慎合作'
                else:
                    media_data['效果评估'] = '很差'
                    media_data['效果建议'] = '效果不理想，建议重新评估'
            else:
                media_data['效果评估'] = 'N/A'
                media_data['效果建议'] = 'N/A'

            analysis_results.append(media_data)

        # 转换为DataFrame并筛选字段
        if analysis_results:
            result_df = pd.DataFrame(analysis_results)

            # 只保留指定字段
            required_columns = [
                '定档媒介', '定档达人数', '所属小组', '总互动量', '平均互动量',
                '互动量最大值', '互动量最小值', '互动量标准差', '总阅读量',
                '平均阅读量', '阅读量最大值', '阅读量最小值', '总曝光量',
                '平均曝光量', '效果评估', '效果建议', '平均CPE', 'CPE最小值',
                'CPE最大值', 'CPE中位数', 'CPE评估', '每元互动量', '成本效益评估',
                '平均CPM', 'CPM最小值', 'CPM最大值', 'CPM中位数', 'CPM标准差',
                'CPM评估', 'CPM/CPE比率', '性价比评估', '性价比分析',
                '平均互动率(%)', '互动率评估', '每元阅读量'
            ]

            # 只保留存在的字段
            existing_columns = [col for col in required_columns if col in result_df.columns]
            result_df = result_df[existing_columns]

            # 确保字段顺序正确
            final_columns = []
            for col in required_columns:
                if col in result_df.columns:
                    final_columns.append(col)

            result_df = result_df[final_columns]

            # 按所属小组顺序排序：数码媒介组 → 家居媒介组 → 快消媒介组 → 其他
            group_order = {'数码媒介组': 0, '家居媒介组': 1, '快消媒介组': 2}

            if '所属小组' in result_df.columns:
                # 添加排序键
                result_df['_排序键'] = result_df['所属小组'].apply(
                    lambda x: group_order.get(x, 99)  # 其他小组排在最后
                )
                # 按排序键和总互动量排序
                result_df = result_df.sort_values(['_排序键', '总互动量'], ascending=[True, False])
                # 删除临时排序键
                result_df = result_df.drop('_排序键', axis=1)

            return result_df
        else:
            return pd.DataFrame({'提示': ['没有足够的数据进行定档媒介效果分析']})

    def _calculate_fixed_media_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算定档媒介达人量级分析
        字段：定档媒介,达人量级,达人数,所属小组,总成本(元),平均成本(元),总返点金额(元),平均返点金额(元),平均返点比例(%),总互动量,平均互动量,总阅读量,平均阅读量,平均CPE,平均CPM
        """
        logger.info("计算定档媒介达人量级分析")

        if df.empty or '定档媒介' not in df.columns or '达人量级' not in df.columns:
            return pd.DataFrame({'错误': ['缺少定档媒介或达人量级字段']})

        analysis_results = []

        # 按定档媒介和达人量级分组
        media_level_groups = df.groupby(['定档媒介', '达人量级'])

        for (media, level), group in media_level_groups:
            media_data = {
                '定档媒介': media,
                '达人量级': level,
                '达人数': len(group)
            }

            # 如果有媒介小组信息，也显示
            if '定档媒介小组' in group.columns and group['定档媒介小组'].iloc[0] != '未知':
                media_data['所属小组'] = group['定档媒介小组'].iloc[0]
            else:
                media_data['所属小组'] = '未知'

            # 成本相关
            if '成本' in df.columns:
                media_data['总成本(元)'] = round(group['成本'].sum(), 2)
                if len(group) > 0:
                    media_data['平均成本(元)'] = round(group['成本'].mean(), 2)
                else:
                    media_data['平均成本(元)'] = 0

            # 返点相关
            if '返点金额' in df.columns:
                media_data['总返点金额(元)'] = round(group['返点金额'].sum(), 2)
                if len(group) > 0:
                    media_data['平均返点金额(元)'] = round(group['返点金额'].mean(), 2)
                else:
                    media_data['平均返点金额(元)'] = 0

            # 返点比例
            if '返点比例' in df.columns:
                valid_rebate_ratio = group[group['返点比例'].notna()]['返点比例']
                if len(valid_rebate_ratio) > 0:
                    media_data['平均返点比例(%)'] = f"{valid_rebate_ratio.mean() * 100:.2f}%"
                else:
                    media_data['平均返点比例(%)'] = 'N/A'

            # 互动相关
            if '互动量' in df.columns:
                media_data['总互动量'] = round(group['互动量'].sum(), 0)
                if len(group) > 0:
                    media_data['平均互动量'] = round(group['互动量'].mean(), 0)
                else:
                    media_data['平均互动量'] = 0

            # 阅读相关
            if '阅读量' in df.columns:
                media_data['总阅读量'] = round(group['阅读量'].sum(), 0)
                if len(group) > 0:
                    media_data['平均阅读量'] = round(group['阅读量'].mean(), 0)
                else:
                    media_data['平均阅读量'] = 0

            # CPE
            if 'cpe' in df.columns:
                valid_cpe = group[group['cpe'].notna()]['cpe']
                if len(valid_cpe) > 0:
                    media_data['平均CPE'] = round(valid_cpe.mean(), 2)
                else:
                    media_data['平均CPE'] = 0

            # CPM
            if 'cpm' in df.columns:
                valid_cpm = group[group['cpm'].notna()]['cpm']
                if len(valid_cpm) > 0:
                    media_data['平均CPM'] = round(valid_cpm.mean(), 2)
                else:
                    media_data['平均CPM'] = 0

            analysis_results.append(media_data)

        # 转换为DataFrame并筛选字段
        if analysis_results:
            result_df = pd.DataFrame(analysis_results)

            # 只保留指定字段
            required_columns = [
                '定档媒介', '达人量级', '达人数', '所属小组', '总成本(元)',
                '平均成本(元)', '总返点金额(元)', '平均返点金额(元)', '平均返点比例(%)',
                '总互动量', '平均互动量', '总阅读量', '平均阅读量', '平均CPE',
                '平均CPM'
            ]

            # 只保留存在的字段
            existing_columns = [col for col in required_columns if col in result_df.columns]
            result_df = result_df[existing_columns]

            # 确保字段顺序正确
            final_columns = []
            for col in required_columns:
                if col in result_df.columns:
                    final_columns.append(col)

            result_df = result_df[final_columns]

            # 按所属小组顺序和达人量级排序
            group_order = {'数码媒介组': 0, '家居媒介组': 1, '快消媒介组': 2}

            if '所属小组' in result_df.columns:
                # 添加排序键
                result_df['_排序键'] = result_df['所属小组'].apply(
                    lambda x: group_order.get(x, 99)  # 其他小组排在最后
                )
                # 达人量级顺序（假设的顺序，可以根据实际情况调整）
                level_order = {'头部达人': 0, '腰部达人': 1, '初级达人': 2, '尾部达人': 3, '未知': 4}
                result_df['_量级键'] = result_df['达人量级'].apply(
                    lambda x: level_order.get(x, 99)
                )
                # 按排序键、量级键和达人数排序
                result_df = result_df.sort_values(['_排序键', '_量级键', '达人数'], ascending=[True, True, False])
                # 删除临时排序键
                result_df = result_df.drop(['_排序键', '_量级键'], axis=1)

            return result_df
        else:
            return pd.DataFrame({'提示': ['没有足够的数据进行定档媒介达人量级分析']})

    def _calculate_fixed_media_comprehensive(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算定档媒介综合分析
        字段：定档媒介,定档达人数,合作项目数,笔记类型分布,所属小组,总成本(元),平均成本(元),总返点金额(元),平均返点金额(元),平均返点比例(%),总互动量,平均互动量,总阅读量,综合得分,综合评价,合作建议,平均CPE,每元互动量,平均CPM
        """
        logger.info("计算定档媒介综合分析")

        if df.empty or '定档媒介' not in df.columns:
            return pd.DataFrame({'错误': ['缺少定档媒介字段']})

        analysis_results = []

        # 按定档媒介分组
        media_groups = df.groupby('定档媒介')

        for media, group in media_groups:
            media_data = {
                '定档媒介': media,
                '定档达人数': len(group),
                '合作项目数': group['项目名称'].nunique() if '项目名称' in group.columns else 0,
            }

            # 笔记类型分布
            if '笔记类型(图文/视频)' in group.columns:
                article_count = len(group[group['笔记类型(图文/视频)'] == '图文'])
                video_count = len(group[group['笔记类型(图文/视频)'] == '视频'])
                media_data['笔记类型分布'] = f"图文:{article_count}/视频:{video_count}"

            # 如果有媒介小组信息，也显示
            if '定档媒介小组' in group.columns and group['定档媒介小组'].iloc[0] != '未知':
                media_data['所属小组'] = group['定档媒介小组'].iloc[0]
            else:
                media_data['所属小组'] = '未知'

            # 成本维度
            if '成本' in df.columns:
                media_data['总成本(元)'] = round(group['成本'].sum(), 2)
                if len(group) > 0:
                    media_data['平均成本(元)'] = round(group['成本'].mean(), 2)
                else:
                    media_data['平均成本(元)'] = 0

            # 返点维度
            if '返点金额' in df.columns:
                media_data['总返点金额(元)'] = round(group['返点金额'].sum(), 2)
                if len(group) > 0:
                    media_data['平均返点金额(元)'] = round(group['返点金额'].mean(), 2)
                else:
                    media_data['平均返点金额(元)'] = 0

            if '返点比例' in df.columns:
                valid_rebate_ratio = group[group['返点比例'].notna()]['返点比例']
                if len(valid_rebate_ratio) > 0:
                    media_data['平均返点比例(%)'] = f"{valid_rebate_ratio.mean() * 100:.2f}%"
                else:
                    media_data['平均返点比例(%)'] = 'N/A'

            # 效果维度
            if '互动量' in df.columns:
                media_data['总互动量'] = round(group['互动量'].sum(), 0)
                if len(group) > 0:
                    media_data['平均互动量'] = round(group['互动量'].mean(), 0)
                else:
                    media_data['平均互动量'] = 0

            if '阅读量' in df.columns:
                media_data['总阅读量'] = round(group['阅读量'].sum(), 0)

            # CPE
            if 'cpe' in df.columns:
                valid_cpe = group[group['cpe'].notna()]['cpe']
                if len(valid_cpe) > 0:
                    media_data['平均CPE'] = round(valid_cpe.mean(), 2)
                else:
                    media_data['平均CPE'] = 0

            # CPM
            if 'cpm' in df.columns:
                valid_cpm = group[group['cpm'].notna()]['cpm']
                if len(valid_cpm) > 0:
                    media_data['平均CPM'] = round(valid_cpm.mean(), 2)
                else:
                    media_data['平均CPM'] = 0

            # 计算每元互动量
            if '互动量' in df.columns and '成本' in df.columns:
                valid_combo = group[group['互动量'].notna() & group['成本'].notna() & (group['成本'] > 0)]
                if len(valid_combo) > 0:
                    interaction_per_yuan = valid_combo['互动量'].sum() / valid_combo['成本'].sum()
                    media_data['每元互动量'] = round(interaction_per_yuan, 4)
                else:
                    media_data['每元互动量'] = 0

            # 综合评价（与原始代码一致）
            score = 0
            max_score = 0

            # 返点比例评分（30%权重）
            if '平均返点比例(%)' in media_data and media_data['平均返点比例(%)'] != 'N/A':
                rebate_pct = float(media_data['平均返点比例(%)'].rstrip('%'))
                if rebate_pct >= 35:
                    score += 30
                elif rebate_pct >= 25:
                    score += 25
                elif rebate_pct >= 20:
                    score += 20
                elif rebate_pct >= 10:
                    score += 15
                else:
                    score += 5
                max_score += 30

            # 每元互动量评分（30%权重）
            if '每元互动量' in media_data:
                interaction_per_yuan = media_data['每元互动量']
                if interaction_per_yuan >= 20:
                    score += 30
                elif interaction_per_yuan >= 15:
                    score += 25
                elif interaction_per_yuan >= 10:
                    score += 20
                elif interaction_per_yuan >= 5:
                    score += 15
                elif interaction_per_yuan >= 2:
                    score += 10
                else:
                    score += 5
                max_score += 30

            # CPM评分（20%权重）
            if '平均CPM' in media_data:
                avg_cpm = media_data['平均CPM']
                if avg_cpm <= 50:
                    score += 20
                elif avg_cpm <= 100:
                    score += 18
                elif avg_cpm <= 150:
                    score += 15
                elif avg_cpm <= 200:
                    score += 12
                elif avg_cpm <= 300:
                    score += 8
                else:
                    score += 5
                max_score += 20

            # 达人数规模评分（20%权重）
            if media_data['定档达人数'] >= 50:
                score += 20
            elif media_data['定档达人数'] >= 30:
                score += 18
            elif media_data['定档达人数'] >= 20:
                score += 15
            elif media_data['定档达人数'] >= 10:
                score += 12
            elif media_data['定档达人数'] >= 5:
                score += 8
            else:
                score += 5
            max_score += 20

            # 计算综合得分
            if max_score > 0:
                final_score = (score / max_score) * 100
                media_data['综合得分'] = round(final_score, 1)

                # 评级（与原始代码一致）
                if final_score >= 90:
                    media_data['综合评价'] = 'S级（卓越）'
                    media_data['合作建议'] = '重点合作，加大投放'
                elif final_score >= 80:
                    media_data['综合评价'] = 'A级（优秀）'
                    media_data['合作建议'] = '优先合作，保持稳定'
                elif final_score >= 70:
                    media_data['综合评价'] = 'B级（良好）'
                    media_data['合作建议'] = '正常合作，适度优化'
                elif final_score >= 60:
                    media_data['综合评价'] = 'C级（合格）'
                    media_data['合作建议'] = '谨慎合作，需要改进'
                elif final_score >= 50:
                    media_data['综合评价'] = 'D级（较差）'
                    media_data['合作建议'] = '限制合作，严格审查'
                else:
                    media_data['综合评价'] = 'E级（很差）'
                    media_data['合作建议'] = '停止合作，重新评估'
            else:
                media_data['综合得分'] = 'N/A'
                media_data['综合评价'] = 'N/A'
                media_data['合作建议'] = 'N/A'

            analysis_results.append(media_data)

        # 转换为DataFrame并筛选字段
        if analysis_results:
            result_df = pd.DataFrame(analysis_results)

            # 只保留指定字段
            required_columns = [
                '定档媒介', '定档达人数', '合作项目数', '笔记类型分布', '所属小组',
                '总成本(元)', '平均成本(元)', '总返点金额(元)', '平均返点金额(元)',
                '平均返点比例(%)', '总互动量', '平均互动量', '总阅读量',
                '综合得分', '综合评价', '合作建议', '平均CPE', '每元互动量', '平均CPM'
            ]

            # 只保留存在的字段
            existing_columns = [col for col in required_columns if col in result_df.columns]
            result_df = result_df[existing_columns]

            # 确保字段顺序正确
            final_columns = []
            for col in required_columns:
                if col in result_df.columns:
                    final_columns.append(col)

            result_df = result_df[final_columns]

            # 按所属小组顺序排序：数码媒介组 → 家居媒介组 → 快消媒介组 → 其他
            group_order = {'数码媒介组': 0, '家居媒介组': 1, '快消媒介组': 2}

            if '所属小组' in result_df.columns:
                # 添加排序键
                result_df['_排序键'] = result_df['所属小组'].apply(
                    lambda x: group_order.get(x, 99)  # 其他小组排在最后
                )
                # 按排序键和综合得分排序
                if '综合得分' in result_df.columns and result_df['综合得分'].dtype != object:
                    result_df = result_df.sort_values(['_排序键', '综合得分'], ascending=[True, False])
                else:
                    result_df = result_df.sort_values('_排序键')
                # 删除临时排序键
                result_df = result_df.drop('_排序键', axis=1)

            return result_df
        else:
            return pd.DataFrame({'提示': ['没有足够的数据进行定档媒介综合分析']})

    # 在 _calculate_overall_summary 方法开头添加无效数据统计：
    def _calculate_overall_summary(self) -> Dict[str, Any]:
        """计算整体成本汇总信息（包含详细的无效数据和异常数据统计）"""
        logger.info("计算整体成本汇总信息（包含详细的无效数据和异常数据分类统计）")

        summary = {}

        # ========== 数据质量统计 ==========
        summary['总数据条数'] = int(len(self.all_df))

        # 正确计算有效数据（不是成本无效的数据）
        if '成本无效' in self.all_df.columns:
            summary['无效数据条数'] = int(self.all_df['成本无效'].sum())
            summary['有效数据条数'] = int(len(self.all_df) - summary['无效数据条数'])
        else:
            summary['无效数据条数'] = 0
            summary['有效数据条数'] = int(len(self.all_df))

        # 计算异常数据数量（数据异常但不成本无效）
        if '数据异常' in self.all_df.columns:
            if '成本无效' in self.all_df.columns:
                # 异常数据 = 数据异常且不成本无效的数据
                summary['异常数据条数'] = int((self.all_df['数据异常'] & ~self.all_df['成本无效']).sum())
            else:
                summary['异常数据条数'] = int(self.all_df['数据异常'].sum())
        else:
            summary['异常数据条数'] = 0

        # 参与分析的数据（有效数据 + 异常数据）
        summary['参与分析数据条数'] = int(summary['有效数据条数'] + summary['异常数据条数'])

        # 检查一致性：参与分析数据条数应该等于总数据条数 - 无效数据条数
        expected_analysis_data = summary['总数据条数'] - summary['无效数据条数']

        # 确保所有值都是标量而不是Series
        if isinstance(summary['参与分析数据条数'], pd.Series):
            summary['参与分析数据条数'] = int(
                summary['参与分析数据条数'].iloc[0] if len(summary['参与分析数据条数']) > 0 else 0)

        # 现在进行安全比较
        if summary['参与分析数据条数'] != expected_analysis_data:
            logger.warning(
                f"数据统计不一致！总数据{summary['总数据条数']} - 无效数据{summary['无效数据条数']} = 预期参与分析{expected_analysis_data}，但实际统计为{summary['参与分析数据条数']}")
            # 修正数据
            summary['参与分析数据条数'] = expected_analysis_data

        if summary['总数据条数'] > 0:
            summary['有效数据比例(%)'] = f"{(summary['有效数据条数'] / summary['总数据条数'] * 100):.2f}%"
            summary['无效数据比例(%)'] = f"{(summary['无效数据条数'] / summary['总数据条数'] * 100):.2f}%"
            summary['异常数据比例(%)'] = f"{(summary['异常数据条数'] / summary['总数据条数'] * 100):.2f}%"
            summary['参与分析数据比例(%)'] = f"{(summary['参与分析数据条数'] / summary['总数据条数'] * 100):.2f}%"

        # ========== 详细的无效数据原因分类统计 ==========
        if '成本无效' in self.all_df.columns:
            invalid_df = self.all_df[self.all_df['成本无效']]
            if not invalid_df.empty and '成本无效原因' in invalid_df.columns:
                # 统计各种无效原因的数量
                invalid_reasons = {}

                # 按两种情况分类统计
                # 1. 成本为0或缺失
                if '成本无效原因' in invalid_df.columns:
                    cost_zero_mask = invalid_df['成本无效原因'].str.contains('成本为0或缺失', na=False)
                    if cost_zero_mask.sum() > 0:
                        invalid_reasons['成本为0或缺失'] = int(cost_zero_mask.sum())

                    # 2. 数据不全
                    data_incomplete_mask = invalid_df['成本无效原因'].str.contains('数据不全', na=False)
                    if data_incomplete_mask.sum() > 0:
                        invalid_reasons['数据不全'] = int(data_incomplete_mask.sum())

                    # 其他原因
                    other_mask = ~cost_zero_mask & ~data_incomplete_mask
                    if other_mask.sum() > 0:
                        other_reasons = invalid_df.loc[other_mask, '成本无效原因'].value_counts().to_dict()
                        invalid_reasons.update(other_reasons)

                summary['无效数据原因分布'] = invalid_reasons

                # 计算各种无效原因的成本
                if '成本' in invalid_df.columns:
                    invalid_cost_by_reason = {}

                    # 成本为0或缺失的成本
                    if 'cost_zero_mask' in locals() and cost_zero_mask.sum() > 0:
                        invalid_cost_by_reason['成本为0或缺失'] = round(invalid_df.loc[cost_zero_mask, '成本'].sum(), 2)

                    # 数据不全的成本
                    if 'data_incomplete_mask' in locals() and data_incomplete_mask.sum() > 0:
                        invalid_cost_by_reason['数据不全'] = round(invalid_df.loc[data_incomplete_mask, '成本'].sum(),
                                                                   2)

                    # 其他原因的成本
                    if 'other_mask' in locals() and other_mask.sum() > 0:
                        other_cost = round(invalid_df.loc[other_mask, '成本'].sum(), 2)
                        invalid_cost_by_reason['其他原因'] = other_cost

                    if invalid_cost_by_reason:
                        summary['无效数据成本分布(元)'] = invalid_cost_by_reason

                    summary['无效数据总成本(元)'] = round(invalid_df['成本'].sum(), 2)

        # ========== 详细的异常数据原因分类统计 ==========
        if '数据异常' in self.all_df.columns:
            abnormal_df = self.all_df[self.all_df['数据异常']]
            if not abnormal_df.empty and '数据异常原因' in abnormal_df.columns:
                abnormal_reasons = abnormal_df['数据异常原因'].value_counts().to_dict()
                summary['异常数据原因分布'] = abnormal_reasons

                # 计算异常数据的成本
                if '成本' in abnormal_df.columns:
                    summary['异常数据总成本(元)'] = round(abnormal_df['成本'].sum(), 2)

        # ========== 原有统计逻辑（基于参与分析的数据：有效+异常） ==========
        # 获取参与分析的数据（排除无效数据）
        if '成本无效' in self.all_df.columns:
            analysis_df = self.all_df[~self.all_df['成本无效']].copy()
        else:
            analysis_df = self.all_df.copy()

        if not analysis_df.empty:
            summary['总媒介数'] = int(analysis_df['定档媒介'].nunique())
            summary['总达人数'] = int(analysis_df['达人昵称'].nunique())
            summary['总项目数'] = int(analysis_df['项目名称'].nunique())

            # 成本相关统计（基于参与分析的数据）
            summary['总成本'] = round(analysis_df['成本'].sum(), 2)
            summary['总报价'] = round(analysis_df['报价'].sum(), 2)
            if '返点金额' in analysis_df.columns:
                summary['总返点金额'] = round(analysis_df['返点金额'].sum(), 2)
            if '手续费' in analysis_df.columns:
                summary['总手续费'] = round(analysis_df['手续费'].sum(), 2)

            # 平均指标（基于参与分析的数据）
            if len(analysis_df) > 0:
                summary['平均成本'] = round(summary['总成本'] / len(analysis_df), 2)
                summary['平均报价'] = round(summary['总报价'] / len(analysis_df), 2)
                if '总返点金额' in summary:
                    summary['平均返点'] = round(summary['总返点金额'] / len(analysis_df), 2)

                # CPM和CPE
                if 'cpm' in analysis_df.columns:
                    valid_cpm = analysis_df['cpm'].dropna()
                    if len(valid_cpm) > 0:
                        summary['平均CPM'] = round(valid_cpm.mean(), 2)

                if 'cpe' in analysis_df.columns:
                    valid_cpe = analysis_df['cpe'].dropna()
                    if len(valid_cpe) > 0:
                        summary['平均CPE'] = round(valid_cpe.mean(), 2)

            # 比例指标（基于参与分析的数据）
            if '总报价' in summary and summary['总报价'] > 0:
                if '总返点金额' in summary:
                    summary['整体返点占报价比例(%)'] = f"{(summary['总返点金额'] / summary['总报价'] * 100):.2f}%"
                summary['整体成本占报价比例(%)'] = f"{(summary['总成本'] / summary['总报价'] * 100):.2f}%"

            if '总成本' in summary and summary['总成本'] > 0 and '总手续费' in summary:
                summary['整体手续费占总成本比例(%)'] = f"{(summary['总手续费'] / summary['总成本'] * 100):.2f}%"

        logger.info(f"整体成本汇总计算完成，总数据: {summary.get('总数据条数', 0)} 条")
        return summary

    def _calculate_media_detail(self) -> pd.DataFrame:
        """计算媒介成本明细（简化版）"""
        if self.valid_df.empty:
            return pd.DataFrame()

        # 按媒介分组统计
        media_detail = self.valid_df.groupby('定档媒介').agg(
            所属小组=('定档媒介小组', 'first'),
            数据条数=('成本', 'count'),
            总成本=('成本', 'sum'),
            总报价=('报价', 'sum'),
            总返点=('返点', 'sum'),
            总手续费=('手续费', 'sum'),
            平均成本=('成本', 'mean'),
            平均报价=('报价', 'mean'),
            返点比例=('返点比例', 'mean')
        ).reset_index()

        # 计算比例
        media_detail['返点占报价比例(%)'] = np.where(
            media_detail['总报价'] > 0,
            (media_detail['总返点'] / media_detail['总报价'] * 100).round(2),
            0.0
        )
        media_detail['返点占报价比例(%)'] = media_detail['返点占报价比例(%)'].apply(lambda x: f"{x:.2f}%")

        media_detail['成本占报价比例(%)'] = np.where(
            media_detail['总报价'] > 0,
            (media_detail['总成本'] / media_detail['总报价'] * 100).round(2),
            0.0
        )
        media_detail['成本占报价比例(%)'] = media_detail['成本占报价比例(%)'].apply(lambda x: f"{x:.2f}%")

        # 格式化返点比例
        if '返点比例' in media_detail.columns:
            media_detail['平均返点比例(%)'] = (media_detail['返点比例'] * 100).round(2).apply(lambda x: f"{x:.2f}%")
            media_detail = media_detail.drop('返点比例', axis=1)

        # 按所属小组顺序排序：数码媒介组 → 家居媒介组 → 快消媒介组 → 其他
        group_order = {'数码媒介组': 0, '家居媒介组': 1, '快消媒介组': 2}

        if '所属小组' in media_detail.columns:
            # 添加排序键
            media_detail['_排序键'] = media_detail['所属小组'].apply(
                lambda x: group_order.get(x, 99)  # 其他小组排在最后
            )
            # 按排序键和总成本排序
            media_detail = media_detail.sort_values(['_排序键', '总成本'], ascending=[True, False])
            # 删除临时排序键
            media_detail = media_detail.drop('_排序键', axis=1)

        return media_detail

    def _calculate_group_summary(self) -> pd.DataFrame:
        """计算小组成本汇总"""
        if self.valid_df.empty or '定档媒介小组' not in self.valid_df.columns:
            return pd.DataFrame()

        group_summary = self.valid_df.groupby('定档媒介小组').agg(
            媒介数量=('定档媒介', 'nunique'),
            数据条数=('成本', 'count'),
            总成本=('成本', 'sum'),
            总报价=('报价', 'sum'),
            总返点=('返点', 'sum'),
            平均成本=('成本', 'mean'),
            平均返点=('返点', 'mean')
        ).reset_index()

        # 按小组顺序排序：数码媒介组 → 家居媒介组 → 快消媒介组 → other组
        group_order = {'数码媒介组': 0, '家居媒介组': 1, '快消媒介组': 2}

        # 添加排序键
        group_summary['_排序键'] = group_summary['定档媒介小组'].apply(
            lambda x: group_order.get(x, 99)  # 其他组排在最后
        )
        # 按排序键和总成本排序
        group_summary = group_summary.sort_values(['_排序键', '总成本'], ascending=[True, False])
        # 删除临时排序键
        group_summary = group_summary.drop('_排序键', axis=1)

        return group_summary

    def _calculate_filtered_summary(self) -> Dict[str, Any]:
        """
        计算筛除数据汇总 - 简化版，因为不删除数据
        """
        logger.info("计算筛除数据汇总（简化版：无数据被筛除）")

        # 直接返回空的汇总，因为不删除任何数据
        return {
            "total_filtered_count": 0,
            "total_filtered_cost": 0,
            "main_reasons": {},
            "说明": "系统设置为不删除任何数据，所有数据均已保留"
        }

    # 在 _calculate_filtered_summary 方法后添加：
    def _calculate_invalid_data_detail(self) -> List[Dict]:
        """
        计算无效数据详情，按照三种情况分类
        1. 成本为0或缺失
        2. 报价<不含手续费的下单价
        3. 数据异常
        """
        logger.info("计算无效数据详情（按照三种情况分类）")

        if self.invalid_df.empty:
            logger.info("无无效数据")
            return []

        invalid_details = []

        # 重置索引
        self.invalid_df = self.invalid_df.reset_index(drop=True)

        for idx in self.invalid_df.index:
            row = self.invalid_df.iloc[idx]

            # 获取无效原因
            invalid_reason = str(row.get('成本无效原因', '未知原因')).strip()

            # 判断属于哪种情况
            invalid_type = '其他原因'
            if '成本为0或缺失' in invalid_reason:
                invalid_type = '成本为0或缺失'
            elif '报价<不含手续费的下单价' in invalid_reason:
                invalid_type = '报价<不含手续费的下单价'
            elif '数据异常' in invalid_reason:
                invalid_type = '数据异常'

            # ✅ 修复：安全地获取数值字段，处理转换错误
            def safe_float(value, default=0.0):
                try:
                    if pd.isna(value):
                        return default
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        return float(value)
                    # 如果是字符串，尝试转换
                    if isinstance(value, str):
                        # 如果是字段名，返回默认值
                        if value in ['order_amount', '报价', '下单价', '成本', '返点']:
                            return default
                        # 尝试转换为浮点数
                        try:
                            return float(value)
                        except:
                            return default
                    return float(value)
                except (ValueError, TypeError):
                    return default

            detail = {
                '记录序号': idx + 1,
                '达人昵称': str(row.get('达人昵称', '未知')).strip(),
                '项目名称': str(row.get('项目名称', '未知')).strip(),
                '定档媒介': str(row.get('定档媒介', '未知')).strip(),
                '成本': safe_float(row.get('成本', 0)),
                '报价': safe_float(row.get('报价', 0)),
                '下单价': safe_float(row.get('下单价', 0)),
                '返点': safe_float(row.get('返点', 0)),
                '不含手续费的下单价': str(row.get('不含手续费的下单价', '')).strip(),
                '成本无效原因': invalid_reason,
                '无效类型': invalid_type,
                '是否被筛除': bool(row.get('被筛除标志', False))
            }

            # 如果被筛除，添加筛除原因
            if row.get('被筛除标志', False):
                detail['筛除原因'] = str(row.get('筛除原因', '未知原因')).strip()
            else:
                detail['筛除原因'] = ''

            # 优先显示筛除原因（如果有）
            if detail['筛除原因'] and detail['筛除原因'] != '正常':
                detail['显示原因'] = detail['筛除原因']
            else:
                detail['显示原因'] = detail['成本无效原因']

            # 添加更多有用的字段
            if '笔记类型(图文/视频)' in row:
                detail['笔记类型'] = str(row['笔记类型(图文/视频)']).strip()

            if '互动量' in row:
                detail['互动量'] = safe_float(row['互动量'])

            if '阅读量' in row:
                detail['阅读量'] = safe_float(row['阅读量'])

            invalid_details.append(detail)

        logger.info(f"生成无效数据详情: {len(invalid_details)} 条")

        # 按无效类型分组统计
        type_counts = {}
        for detail in invalid_details:
            invalid_type = detail.get('无效类型', '其他原因')
            type_counts[invalid_type] = type_counts.get(invalid_type, 0) + 1

        logger.info(f"无效数据类型分布: {type_counts}")

        return invalid_details

    def _calculate_abnormal_data_detail(self) -> List[Dict]:
        """
        计算异常数据详情，按照四种情况分类
        1. 报价异常（报价<不含手续费的下单价）
        2. 数据异常（无法判断手续费情况）
        3. 返点异常（返点比例异常）
        4. 其他异常（筛除原因异常等）
        """
        logger.info("计算异常数据详情（参与分析但标记异常）")

        # 获取所有异常数据
        if '数据异常' in self.all_df.columns:
            abnormal_df = self.all_df[self.all_df['数据异常']].copy()
        else:
            abnormal_df = pd.DataFrame()

        if abnormal_df.empty:
            logger.info("无异常数据")
            return []

        abnormal_details = []

        # 重置索引
        abnormal_df = abnormal_df.reset_index(drop=True)

        for idx in abnormal_df.index:
            row = abnormal_df.iloc[idx]

            # 获取异常原因
            abnormal_reason = str(row.get('数据异常原因', '未知异常')).strip()

            # ✅ 修改：统一判断逻辑
            abnormal_type = '其他异常'
            if '报价或下单价异常' in abnormal_reason:
                abnormal_type = '报价或下单价异常'
            elif '无法判断' in abnormal_reason:
                abnormal_type = '数据异常'
            elif '返点比例' in abnormal_reason:
                abnormal_type = '返点异常'
            else:
                # 检查是否有旧的报价异常格式，也统一为"报价或下单价异常"
                if any(keyword in abnormal_reason for keyword in ['报价<', '报价(', '下单价']):
                    abnormal_type = '报价或下单价异常'
                    abnormal_reason = '报价或下单价异常'  # 统一原因
                elif row.get('筛除原因', '') and row['筛除原因'] != '正常':
                    abnormal_type = '筛除异常'
                    abnormal_reason = row['筛除原因']

            # ✅ 修复：安全地获取数值字段
            def safe_float(value, default=0.0):
                try:
                    if pd.isna(value):
                        return default
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        return float(value)
                    # 如果是字符串，尝试转换
                    if isinstance(value, str):
                        # 如果是字段名，返回默认值
                        if value in ['order_amount', '报价', '下单价', '成本', '返点', '返点比例']:
                            return default
                        # 尝试转换为浮点数
                        try:
                            return float(value)
                        except:
                            return default
                    return float(value)
                except (ValueError, TypeError):
                    return default

            detail = {
                '记录序号': idx + 1,
                '达人昵称': str(row.get('达人昵称', '未知')).strip(),
                '项目名称': str(row.get('项目名称', '未知')).strip(),
                '定档媒介': str(row.get('定档媒介', '未知')).strip(),
                '成本': safe_float(row.get('成本', 0)),
                '报价': safe_float(row.get('报价', 0)),
                '下单价': safe_float(row.get('下单价', 0)),
                '返点': safe_float(row.get('返点', 0)),
                '返点比例': safe_float(row.get('返点比例', 0)) * 100 if '返点比例' in row else 0,
                '不含手续费的下单价': str(row.get('不含手续费的下单价', '')).strip(),
                '数据异常原因': abnormal_reason,
                '异常类型': abnormal_type,
                '是否参与分析': True,  # 异常数据参与分析
                '参与分析标识': '异常数据'
            }

            # 添加更多有用的字段
            if '笔记类型(图文/视频)' in row:
                detail['笔记类型'] = str(row['笔记类型(图文/视频)']).strip()

            if '互动量' in row:
                detail['互动量'] = safe_float(row['互动量'])

            if '阅读量' in row:
                detail['阅读量'] = safe_float(row['阅读量'])

            abnormal_details.append(detail)

        logger.info(f"生成异常数据详情: {len(abnormal_details)} 条")

        # 按异常类型分组统计
        type_counts = {}
        for detail in abnormal_details:
            abnormal_type = detail.get('异常类型', '其他异常')
            type_counts[abnormal_type] = type_counts.get(abnormal_type, 0) + 1

        logger.info(f"异常数据类型分布: {type_counts}")

        return abnormal_details

    def _generate_cost_efficiency_ranking(self, top_n: int) -> pd.DataFrame:
        """生成成本效益排名"""
        if self.valid_df.empty:
            return pd.DataFrame()

        # 计算媒介成本效益指标
        media_efficiency = self.valid_df.groupby('定档媒介').agg(
            所属小组=('定档媒介小组', 'first'),
            总成本=('成本', 'sum'),
            总返点=('返点', 'sum'),
            总报价=('报价', 'sum'),
            数据条数=('成本', 'count')
        ).reset_index()

        # 计算效益指标
        media_efficiency['返点比例'] = np.where(
            media_efficiency['总报价'] > 0,
            (media_efficiency['总返点'] / media_efficiency['总报价'] * 100).round(2),
            0.0
        )

        media_efficiency['成本比例'] = np.where(
            media_efficiency['总报价'] > 0,
            (media_efficiency['总成本'] / media_efficiency['总报价'] * 100).round(2),
            0.0
        )

        # 按所属小组顺序排序：数码媒介组 → 家居媒介组 → 快消媒介组 → 其他
        group_order = {'数码媒介组': 0, '家居媒介组': 1, '快消媒介组': 2}

        # 添加排序键
        media_efficiency['_排序键'] = media_efficiency['所属小组'].apply(
            lambda x: group_order.get(x, 99)  # 其他小组排在最后
        )
        # 按返点比例降序、成本比例升序排序
        media_efficiency = media_efficiency.sort_values(
            ['_排序键', '返点比例', '成本比例'],
            ascending=[True, False, True]
        ).head(top_n)

        # 添加排名
        media_efficiency['排名'] = range(1, len(media_efficiency) + 1)

        # 删除临时排序键
        media_efficiency = media_efficiency.drop('_排序键', axis=1)

        return media_efficiency[['排名', '定档媒介', '所属小组', '返点比例', '成本比例', '总成本', '总返点']]

    def get_fixed_media_cost_analysis(self) -> pd.DataFrame:
        """获取定档媒介成本分析结果"""
        return self.result.get("fixed_media_cost", pd.DataFrame())

    def get_fixed_media_rebate_analysis(self) -> pd.DataFrame:
        """获取定档媒介返点分析结果"""
        return self.result.get("fixed_media_rebate", pd.DataFrame())

    def get_fixed_media_performance_analysis(self) -> pd.DataFrame:
        """获取定档媒介效果分析结果"""
        return self.result.get("fixed_media_performance", pd.DataFrame())

    def get_fixed_media_level_analysis(self) -> pd.DataFrame:
        """获取定档媒介达人量级分析结果"""
        return self.result.get("fixed_media_level", pd.DataFrame())

    def get_fixed_media_comprehensive_analysis(self) -> pd.DataFrame:
        """获取定档媒介综合分析结果"""
        return self.result.get("fixed_media_comprehensive", pd.DataFrame())