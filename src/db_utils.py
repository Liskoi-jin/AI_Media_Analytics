# src/db_utils.py
"""数据库工具函数 - 独立于原有逻辑，负责数据库连接和数据查询"""
import pandas as pd
import pymysql
from typing import Dict, Optional
import decimal
from src.utils import logger, normalize_media_name, NAME_TO_GROUP_MAPPING, FLOWER_TO_NAME_MAPPING, ID_TO_NAME_MAPPING
from config import DB_CONFIG  # 复用已有配置


def create_db_connection() -> Optional[pymysql.connections.Connection]:
    """创建数据库连接（复用config.py中的DB_CONFIG配置）"""
    try:
        conn = pymysql.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database'],
            charset=DB_CONFIG['charset'],
            cursorclass=pymysql.cursors.DictCursor
        )
        logger.info("数据库连接成功")
        return conn
    except Exception as e:
        logger.error(f"数据库连接失败: {str(e)}")
        return None


def get_media_group(media_name):
    """根据媒介姓名获取所属小组（通用函数）"""
    if pd.isna(media_name) or media_name == '未知' or not isinstance(media_name, str):
        return 'other组'

    media_name_str = str(media_name).strip()

    # 处理字符串形式的'None'和'n'
    if media_name_str.lower() in ['none', 'null', 'nan', '', 'n', '木子']:
        return 'other组'

    # 1. 直接查找
    if media_name_str in NAME_TO_GROUP_MAPPING:
        return NAME_TO_GROUP_MAPPING[media_name_str]

    # 2. 标准化后查找
    normalized_name = normalize_media_name(media_name_str)
    if normalized_name in NAME_TO_GROUP_MAPPING:
        return NAME_TO_GROUP_MAPPING[normalized_name]

    # 3. 尝试匹配花名
    for flower_name, real_name in FLOWER_TO_NAME_MAPPING.items():
        if media_name_str == flower_name or media_name_str == real_name:
            if real_name in NAME_TO_GROUP_MAPPING:
                return NAME_TO_GROUP_MAPPING[real_name]

    return 'other组'


def convert_decimal_to_float(value):
    """将Decimal类型转换为float"""
    if isinstance(value, (decimal.Decimal,)):
        return float(value)
    elif pd.isna(value):
        return 0.0
    else:
        return value


def map_media_to_real_name(media_name):
    """
    核心修复：将媒介花名映射到真实姓名
    使用全局的 ID_TO_NAME_MAPPING 和 FLOWER_TO_NAME_MAPPING
    """
    # 首先处理空值和None
    if pd.isna(media_name) or media_name == '未知' or not isinstance(media_name, str):
        return '未知'

    media_name_str = str(media_name).strip()

    # 处理字符串形式的'None'和无效值
    if media_name_str.lower() in ['none', 'null', 'nan', '', 'n']:
        logger.debug(f"跳过无效媒介名称: '{media_name_str}'")
        return '未知'

    # 特殊处理常见问题名称
    if media_name_str == '木子':
        logger.debug(f"特殊处理名称: '木子' -> '未知'")
        return '未知'

    # 1. 直接查找 ID_TO_NAME_MAPPING（这是最全的映射）
    if media_name_str in ID_TO_NAME_MAPPING:
        real_name = ID_TO_NAME_MAPPING[media_name_str]
        logger.debug(f"通过ID_TO_NAME_MAPPING映射: '{media_name_str}' -> '{real_name}'")
        return real_name

    # 2. 查找 FLOWER_TO_NAME_MAPPING（花名到真名）
    if media_name_str in FLOWER_TO_NAME_MAPPING:
        real_name = FLOWER_TO_NAME_MAPPING[media_name_str]
        logger.debug(f"通过FLOWER_TO_NAME_MAPPING映射: '{media_name_str}' -> '{real_name}'")
        return real_name

    # 3. 反向查找 FLOWER_TO_NAME_MAPPING（真名到花名）
    for flower, real in FLOWER_TO_NAME_MAPPING.items():
        if real == media_name_str:
            logger.debug(f"反向映射: '{media_name_str}' 是真实姓名，无需映射")
            return media_name_str

    # 4. 检查是否已经是真实姓名（在 NAME_TO_GROUP_MAPPING 中）
    if media_name_str in NAME_TO_GROUP_MAPPING:
        logger.debug(f"'{media_name_str}' 已在NAME_TO_GROUP_MAPPING中，视为真实姓名")
        return media_name_str

    # 5. 标准化后再次尝试
    normalized_name = normalize_media_name(media_name_str)
    if normalized_name in ID_TO_NAME_MAPPING:
        real_name = ID_TO_NAME_MAPPING[normalized_name]
        logger.debug(f"标准化后通过ID_TO_NAME_MAPPING映射: '{media_name_str}' -> '{normalized_name}' -> '{real_name}'")
        return real_name

    logger.warning(f"未找到媒介 '{media_name_str}' 的真实姓名映射，使用原值")
    return media_name_str


def calculate_missing_fields(df):
    """计算缺失的字段，确保字段名与cost_analyzer.py完全匹配"""
    df = df.copy()

    # 计算cpm（每千次曝光的成本）
    if '曝光量' in df.columns and '成本' in df.columns:
        df['cpm'] = df.apply(
            lambda row: (row['成本'] / row['曝光量'] * 1000) if row['曝光量'] > 0 else 0.0,
            axis=1
        )
    else:
        df['cpm'] = 0.0

    # 计算cpe（每次互动的成本）
    if '互动量' in df.columns and '成本' in df.columns:
        df['cpe'] = df.apply(
            lambda row: (row['成本'] / row['互动量']) if row['互动量'] > 0 else 0.0,
            axis=1
        )
    else:
        df['cpe'] = 0.0

    # 计算cpv（每次阅读的成本）
    if '阅读量' in df.columns and '成本' in df.columns:
        df['cpv'] = df.apply(
            lambda row: (row['成本'] / row['阅读量']) if row['阅读量'] > 0 else 0.0,
            axis=1
        )
    else:
        df['cpv'] = 0.0

    # ✅ 关键修复1：计算返点金额（返点金额 = 返点）
    if '返点' in df.columns:
        df['返点金额'] = df['返点']
    else:
        df['返点金额'] = 0.0

    # ✅ 关键修复2：计算返点比例（返点比例 = 返点金额 / 下单价）
    if '返点金额' in df.columns and '下单价' in df.columns:
        df['返点比例'] = df.apply(
            lambda row: (row['返点金额'] / row['下单价']) if row['下单价'] > 0 else 0.0,
            axis=1
        )
    else:
        df['返点比例'] = 0.0

    # ✅ 关键修复3：添加不含手续费的下单价（默认使用下单价）
    if '下单价' in df.columns:
        df['不含手续费的下单价'] = df['下单价']
    else:
        df['不含手续费的下单价'] = 0.0

    # ✅ 关键修复4：添加手续费（默认0）
    df['手续费'] = 0.0

    # ✅ 关键修复5：添加成本无效标记（默认False）
    df['成本无效'] = False

    # ✅ 关键修复6：添加筛除原因（默认空）
    df['筛除原因'] = ''

    # ✅ 关键修复7：添加手续费情况（默认未知）
    df['手续费情况'] = '未知'

    # ✅ 关键修复8：添加被筛除标志（默认False）
    df['被筛除标志'] = False

    # ✅ 关键修复9：添加数据异常相关字段
    df['数据异常'] = False
    df['数据异常原因'] = ''

    # ✅ 关键修复10：添加点赞收藏量字段（如果没有）
    if '点赞收藏量' not in df.columns:
        if '笔记点赞数' in df.columns and '笔记收藏数' in df.columns:
            df['点赞收藏量'] = df['笔记点赞数'] + df['笔记收藏数']
        else:
            df['点赞收藏量'] = 0

    # ✅ 关键修复11：添加互动量最大值、最小值等字段（cost_analyzer.py需要的）
    if '互动量' in df.columns:
        df['互动量最大值'] = df['互动量']
        df['互动量最小值'] = df['互动量']
        df['互动量标准差'] = 0.0
    else:
        df['互动量最大值'] = 0
        df['互动量最小值'] = 0
        df['互动量标准差'] = 0.0

    # ✅ 关键修复12：添加其他cost_analyzer.py需要的字段
    if '成本' in df.columns:
        df['成本最大值(元)'] = df['成本']
        df['成本最小值(元)'] = df['成本']
        df['成本中位数(元)'] = df['成本']
    else:
        df['成本最大值(元)'] = 0.0
        df['成本最小值(元)'] = 0.0
        df['成本中位数(元)'] = 0.0

    # ✅ 关键修复13：添加报价相关字段
    if '报价' in df.columns:
        df['报价最大值(元)'] = df['报价']
        df['报价最小值(元)'] = df['报价']
    else:
        df['报价最大值(元)'] = 0.0
        df['报价最小值(元)'] = 0.0

    # ✅ 关键修复14：添加返点金额相关字段
    if '返点金额' in df.columns:
        df['返点金额最大值(元)'] = df['返点金额']
        df['返点金额最小值(元)'] = df['返点金额']
        df['返点金额中位数(元)'] = df['返点金额']
    else:
        df['返点金额最大值(元)'] = 0.0
        df['返点金额最小值(元)'] = 0.0
        df['返点金额中位数(元)'] = 0.0

    # ✅ 关键修复15：添加返点比例相关字段
    if '返点比例' in df.columns:
        df['返点比例最大值(%)'] = df['返点比例'] * 100
        df['返点比例最小值(%)'] = df['返点比例'] * 100
        df['返点比例中位数(%)'] = df['返点比例'] * 100
    else:
        df['返点比例最大值(%)'] = 0.0
        df['返点比例最小值(%)'] = 0.0
        df['返点比例中位数(%)'] = 0.0

    # 确保数值格式
    float_columns = ['返点比例', 'cpm', 'cpe', 'cpv', '返点金额', '手续费', '成本', '报价', '下单价']
    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0.0)
            # 对于返点比例，保留更多小数位
            if col == '返点比例':
                df[col] = df[col].round(6)
            else:
                df[col] = df[col].round(2)

    # 调试：检查字段
    logger.info(f"✅ 已添加成本分析字段（共 {len(df.columns)} 列）:")
    cost_fields = ['返点金额', '返点比例', '不含手续费的下单价', '手续费',
                   '成本无效', '筛除原因', '手续费情况', '被筛除标志', '数据异常',
                   'cpm', 'cpe', 'cpv', '点赞收藏量']

    for field in cost_fields:
        if field in df.columns:
            if len(df) > 0:
                sample_val = df[field].iloc[0]
                non_zero = (df[field] != 0).sum()
                total = len(df)
                logger.info(
                    f"  {field}: 样本值={sample_val}, 非零值={non_zero}/{total} ({non_zero / total * 100:.1f}%)")
            else:
                logger.info(f"  {field}: 数据为空")

    return df


def clean_and_prepare_data(df):
    """清理和准备数据，处理空值和无效值"""
    if df.empty:
        return df

    df = df.copy()

    # 处理schedule_user_name字段
    if 'schedule_user_name' in df.columns:
        # 填充空值
        df['schedule_user_name'] = df['schedule_user_name'].fillna('未知')
        # 替换字符串形式的None/null/nan
        df['schedule_user_name'] = df['schedule_user_name'].replace(['None', 'null', 'nan', 'N/A', 'n'], '未知')
        # 去除空格
        df['schedule_user_name'] = df['schedule_user_name'].astype(str).str.strip()

    # 处理submit_media_user_name字段
    if 'submit_media_user_name' in df.columns:
        df['submit_media_user_name'] = df['submit_media_user_name'].fillna('未知')
        df['submit_media_user_name'] = df['submit_media_user_name'].replace(['None', 'null', 'nan', 'N/A', 'n'], '未知')
        df['submit_media_user_name'] = df['submit_media_user_name'].astype(str).str.strip()

    # 处理其他关键字段
    str_fields = ['influencer_nickname', 'project_name', 'state', 'kol_koc_type', 'note_type']
    for field in str_fields:
        if field in df.columns:
            df[field] = df[field].fillna('')
            df[field] = df[field].astype(str).str.strip()

    # 确保数值字段正确
    numeric_fields = ['follower_count', 'cooperation_quote', 'order_amount',
                      'rebate_amount', 'cost_amount', 'note_like_count',
                      'note_favorite_count', 'note_comment_count', 'interaction_count',
                      'read_count', 'exposure_count', 'read_uv_count']
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce')
            df[field] = df[field].fillna(0)

    return df


def query_workload_data(start_date: str, end_date: str) -> pd.DataFrame:
    """查询工作量分析数据"""
    conn = create_db_connection()
    if not conn:
        return pd.DataFrame()

    sql = """
    SELECT
        id,
        influencer_nickname,
        project_name,
        schedule_user_name,
        submit_media_user_name,
        state,
        kol_koc_type,
        note_type,
        follower_count,
        cooperation_quote,
        order_amount,
        rebate_amount,
        cost_amount,
        influencer_source,
        influencer_purpose,
        note_like_count,
        note_favorite_count,
        note_comment_count,
        interaction_count,
        read_count,
        exposure_count,
        read_uv_count,
        system_status,
        schedule_time,
        submit_time
    FROM
        lgc_project_influencer
    WHERE
        schedule_time >= %s
        AND schedule_time < %s
        AND influencer_source = 'INSIDE'
        AND (state = "CHAIN_RETURNED" OR state = "SCHEDULED")
        AND project_name NOT IN ('快消组达人库', '家居组达人', '数码组达人库','测试-250801')
    """
    try:
        # ✅ 修复：直接使用pymysql查询，避免pandas的read_sql问题
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(sql, [f"{start_date} 00:00:00", f"{end_date} 23:59:59"])
            results = cursor.fetchall()

            logger.info(f"查询到工作量数据 {len(results)} 条")

            if not results:
                logger.warning("查询返回空结果")
                return pd.DataFrame()

            # ✅ 核心修复：直接从查询结果构建DataFrame
            df = pd.DataFrame(results)

            # ✅ 修复：转换Decimal类型为float
            decimal_fields = ['cost_amount', 'cooperation_quote', 'order_amount', 'rebate_amount',
                              'note_like_count', 'note_favorite_count', 'note_comment_count',
                              'interaction_count', 'read_count', 'exposure_count', 'read_uv_count',
                              'follower_count']

            for field in decimal_fields:
                if field in df.columns:
                    df[field] = df[field].apply(convert_decimal_to_float)

            # ✅ 修复：转换时间字段为字符串，避免JSON序列化问题
            time_columns = ['schedule_time', 'submit_time']
            for col in time_columns:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else ""
                    )

            # 清理数据
            df = clean_and_prepare_data(df)

            logger.info(f"从查询结果构建DataFrame成功，形状: {df.shape}")

            # 🔍 添加详细调试信息
            logger.info(f"🔍 DataFrame原始列名: {list(df.columns)}")
            logger.info(f"🔍 DataFrame行数: {len(df)}")

            # 检查关键字段
            key_fields = ['schedule_user_name', 'submit_media_user_name', 'influencer_nickname',
                          'cost_amount', 'rebate_amount', 'order_amount', 'cooperation_quote']
            for field in key_fields:
                if field in df.columns:
                    sample_values = df[field].dropna().unique()[:3]
                    logger.info(f"🔍 字段 '{field}' 样本值: {list(sample_values)}")
                else:
                    logger.warning(f"⚠️ 字段 '{field}' 不存在")

        # ✅ 关键修复：确保字段正确映射，保留原始字段名供分析器使用
        # 1. 确保有定档媒介字段（使用schedule_user_name）
        if 'schedule_user_name' in df.columns:
            # 创建定档媒介字段
            df['定档媒介'] = df['schedule_user_name']
            logger.info(f"✅ 已设置定档媒介字段，唯一值: {df['定档媒介'].nunique()}")
        else:
            logger.warning("schedule_user_name字段不存在，创建默认值")
            df['schedule_user_name'] = '未知'
            df['定档媒介'] = '未知'

        # 2. 确保有提交媒介字段（使用submit_media_user_name）
        if 'submit_media_user_name' in df.columns:
            df['提交媒介'] = df['submit_media_user_name']
            logger.info(f"✅ 已设置提交媒介字段，唯一值: {df['提交媒介'].nunique()}")
        else:
            logger.warning("submit_media_user_name字段不存在，创建默认值")
            df['submit_media_user_name'] = '未知'
            df['提交媒介'] = '未知'

        # 3. ✅ 核心修复：添加媒介姓名和对应真名字段，进行真实姓名映射
        logger.info("开始媒介姓名映射...")

        # 定档媒介 -> 媒介姓名（映射到真实姓名）
        df['媒介姓名'] = df['定档媒介'].apply(map_media_to_real_name)

        # 提交媒介 -> 对应真名（映射到真实姓名）
        df['对应真名'] = df['提交媒介'].apply(map_media_to_real_name)

        # 记录映射结果
        if 'schedule_user_name' in df.columns and '媒介姓名' in df.columns:
            unique_combinations = df[['schedule_user_name', '媒介姓名']].drop_duplicates()
            logger.info(f"🔍 定档媒介到真实姓名映射示例（前10个）:")
            for _, row in unique_combinations.head(10).iterrows():
                logger.info(f"  '{row['schedule_user_name']}' -> '{row['媒介姓名']}'")

        # 4. 添加其他必要字段（分析器需要的字段）
        # ✅ 修复：使用正确的分组映射
        df['所属小组'] = df['媒介姓名'].apply(get_media_group)  # 使用真实姓名获取小组
        df['数据类型'] = '定档'

        # 重命名其他关键字段为中文（保持分析器兼容性）
        column_mapping = {
            'influencer_nickname': '达人昵称',
            'project_name': '项目名称',
            'state': '状态',
            'kol_koc_type': '达人量级',
            'note_type': '笔记类型(图文/视频)',
            'follower_count': '粉丝数',
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
            'system_status': 'system_status'
        }

        # 只重命名存在的列
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # ✅ 关键修复：保留原始字段名供分析器使用
        if '达人用途' in df.columns:
            df['influencer_purpose'] = df['达人用途']
        if '达人昵称' in df.columns:
            df['influencer_nickname'] = df['达人昵称']
        if '项目名称' in df.columns:
            df['project_name'] = df['项目名称']
        if 'schedule_user_name' not in df.columns:
            if '定档媒介' in df.columns:
                df['schedule_user_name'] = df['定档媒介']

        # ✅ ✅ ✅ 核心修复：确保schedule_user_name和submit_media_user_name字段包含映射后的真实姓名
        # 而不是原始的花名或ID
        if 'schedule_user_name' in df.columns:
            # 将schedule_user_name替换为映射后的真实姓名（媒介姓名）
            df['schedule_user_name'] = df['媒介姓名']
            logger.info(
                f"✅ 工作量数据：已更新schedule_user_name为真实姓名，样本值: {df['schedule_user_name'].iloc[:3].tolist()}")

        if 'submit_media_user_name' in df.columns:
            # 将submit_media_user_name替换为映射后的真实姓名（对应真名）
            df['submit_media_user_name'] = df['对应真名']
            logger.info(
                f"✅ 工作量数据：已更新submit_media_user_name为真实姓名，样本值: {df['submit_media_user_name'].iloc[:3].tolist()}")

        # 🔥 新增：确保定档媒介字段也使用真实姓名（如果前端使用这个字段）
        if '定档媒介' in df.columns:
            df['定档媒介'] = df['媒介姓名']
            logger.info(f"✅ 工作量数据：已更新定档媒介为真实姓名，样本值: {df['定档媒介'].iloc[:3].tolist()}")

        # 确保数值字段正确转换
        numeric_fields = ['粉丝数', '报价', '下单价', '返点', '成本', '笔记点赞数', '笔记收藏数',
                          '笔记评论数', '互动量', '阅读量', '曝光量', '阅读uv数']

        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
                df[field] = df[field].fillna(0)

        # 清理小组名称中的空格
        if '所属小组' in df.columns:
            df['所属小组'] = df['所属小组'].astype(str).str.strip()

        # ✅ ✅ ✅ 关键修复：计算所有cost_analyzer.py需要的字段
        logger.info("开始计算成本分析必需字段...")
        df = calculate_missing_fields(df)

        # 调试：打印处理后数据
        if not df.empty:
            logger.info(f"✅ 最终处理后数据样本（前3行）- 成本分析字段检查：")
            for i in range(min(3, len(df))):
                logger.info(f"行{i}: "
                            f"达人昵称='{df.iloc[i].get('达人昵称', 'N/A')}', "
                            f"定档媒介='{df.iloc[i].get('定档媒介', 'N/A')}', "
                            f"所属小组='{df.iloc[i].get('所属小组', 'N/A')}', "
                            f"成本={df.iloc[i].get('成本', 0):.2f}, "
                            f"返点={df.iloc[i].get('返点', 0):.2f}, "
                            f"返点金额={df.iloc[i].get('返点金额', 0):.2f}, "
                            f"下单价={df.iloc[i].get('下单价', 0):.2f}, "
                            f"返点比例={df.iloc[i].get('返点比例', 0):.4f}, "
                            f"cpm={df.iloc[i].get('cpm', 0):.2f}, "
                            f"cpe={df.iloc[i].get('cpe', 0):.2f}")

            # 打印字段检查
            logger.info(f"🔍 字段存在性检查（成本分析必需字段）:")
            required_fields = [
                '返点金额', '返点比例', '不含手续费的下单价', '手续费',
                '成本无效', '筛除原因', '手续费情况', '被筛除标志',
                'cpm', 'cpe', 'cpv', '数据异常', '数据异常原因'
            ]

            for field in required_fields:
                exists = field in df.columns
                if exists and len(df) > 0:
                    sample_val = df[field].iloc[0]
                    logger.info(f"  {field}: ✅ (样本值: {sample_val})")
                else:
                    logger.info(f"  {field}: {'✅' if exists else '❌'}")

            # 打印小组分布
            if '所属小组' in df.columns:
                group_dist = df['所属小组'].value_counts().to_dict()
                logger.info(f"✅ 工作量数据小组分布: {group_dist}")

        return df
    except Exception as e:
        logger.error(f"工作量数据查询失败: {str(e)}", exc_info=True)
        return pd.DataFrame()
    finally:
        conn.close()


def query_quality_data(start_date: str, end_date: str) -> pd.DataFrame:
    """查询工作质量分析数据"""
    conn = create_db_connection()
    if not conn:
        return pd.DataFrame()

    sql = """
    SELECT
        id,
        influencer_nickname,
        project_name,
        schedule_user_name,
        submit_media_user_name,
        state,
        kol_koc_type,
        note_type,
        follower_count,
        cooperation_quote,
        order_amount,
        rebate_amount,
        cost_amount,
        influencer_source,
        influencer_purpose,
        note_like_count,
        note_favorite_count,
        note_comment_count,
        interaction_count,
        read_count,
        exposure_count,
        read_uv_count,
        system_status,
        submit_time
    FROM
        lgc_project_influencer
    WHERE
        submit_time >= %s
        AND submit_time < %s
        AND (influencer_purpose = '高阅读达人' OR influencer_purpose = '优质达人')
        AND influencer_source = 'INSIDE'
        AND project_name NOT IN('快消组达人库', '家居组达人', '数码组达人库', '测试-250801')
    """
    try:
        # ✅ 修复：直接使用pymysql查询，避免pandas的read_sql问题
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(sql, [f"{start_date} 00:00:00", f"{end_date} 23:59:59"])
            results = cursor.fetchall()

            logger.info(f"查询到工作质量数据 {len(results)} 条")

            if not results:
                logger.warning("查询返回空结果")
                return pd.DataFrame()

            # ✅ 直接从查询结果构建DataFrame
            df = pd.DataFrame(results)

            # ✅ 修复：转换Decimal类型为float
            decimal_fields = ['cost_amount', 'cooperation_quote', 'order_amount', 'rebate_amount',
                              'note_like_count', 'note_favorite_count', 'note_comment_count',
                              'interaction_count', 'read_count', 'exposure_count', 'read_uv_count',
                              'follower_count']

            for field in decimal_fields:
                if field in df.columns:
                    df[field] = df[field].apply(convert_decimal_to_float)

            # ✅ 修复：转换时间字段为字符串，避免JSON序列化问题
            time_columns = ['submit_time']
            for col in time_columns:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else ""
                    )

            # 清理数据
            df = clean_and_prepare_data(df)

            logger.info(f"从查询结果构建DataFrame成功，形状: {df.shape}")

            # 🔍 添加调试信息
            if 'schedule_user_name' in df.columns:
                unique_values = df['schedule_user_name'].dropna().unique()
                logger.info(f"🔍 schedule_user_name唯一值数量: {len(unique_values)}")
                logger.info(f"🔍 schedule_user_name前5个值: {list(unique_values)[:5]}")

            # 验证数据
            if not df.empty:
                logger.info("工作质量数据验证（前3行）:")
                for i in range(min(3, len(df))):
                    nickname = df.iloc[i].get('influencer_nickname', 'N/A')
                    schedule_name = df.iloc[i].get('schedule_user_name', 'N/A')
                    purpose = df.iloc[i].get('influencer_purpose', 'N/A')

                    logger.info(f"行{i}: influencer_nickname='{nickname}', "
                                f"schedule_user_name='{schedule_name}', "
                                f"influencer_purpose='{purpose}'")

        # ✅ 修复：确保字段正确映射
        # 1. 确保有schedule_user_name字段
        if 'schedule_user_name' in df.columns:
            df['定档媒介'] = df['schedule_user_name']
        else:
            df['schedule_user_name'] = '未知'
            df['定档媒介'] = '未知'

        # 2. 确保有submit_media_user_name字段
        if 'submit_media_user_name' in df.columns:
            df['提交媒介'] = df['submit_media_user_name']
        else:
            df['submit_media_user_name'] = '未知'
            df['提交媒介'] = '未知'

        # 3. ✅ 核心修复：添加媒介姓名和对应真名字段，进行真实姓名映射
        logger.info("开始工作质量分析的媒介姓名映射...")

        # 定档媒介 -> 媒介姓名（映射到真实姓名）
        df['媒介姓名'] = df['定档媒介'].apply(map_media_to_real_name)

        # 提交媒介 -> 对应真名（映射到真实姓名）
        df['对应真名'] = df['提交媒介'].apply(map_media_to_real_name)

        # 4. 添加其他必要字段
        # ✅ 修复：使用正确的分组映射
        df['所属小组'] = df['媒介姓名'].apply(get_media_group)  # 使用真实姓名获取小组
        df['数据类型'] = '提报'

        # 重命名字段为中文
        column_mapping = {
            'influencer_nickname': '达人昵称',
            'project_name': '项目名称',
            'state': '状态',
            'kol_koc_type': '达人量级',
            'note_type': '笔记类型(图文/视频)',
            'follower_count': '粉丝数',
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
            'system_status': 'system_status'
        }

        # 只重命名存在的列
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # ✅ 关键修复：确保有达人用途字段
        if '达人用途' not in df.columns:
            logger.warning("达人用途字段不存在，从influencer_purpose获取")
            if 'influencer_purpose' in df.columns:
                df['达人用途'] = df['influencer_purpose']
            else:
                df['达人用途'] = '优质达人'  # 默认值

        # ✅ ✅ ✅ 核心修复：保留原始字段名供分析器使用
        # 分析器需要这些字段名，但我们已经重命名为中文
        if '达人用途' in df.columns:
            df['influencer_purpose'] = df['达人用途']
        if '达人昵称' in df.columns:
            df['influencer_nickname'] = df['达人昵称']
        if '项目名称' in df.columns:
            df['project_name'] = df['项目名称']
        # 确保schedule_user_name存在（分析器需要）
        if 'schedule_user_name' not in df.columns:
            if '定档媒介' in df.columns:
                df['schedule_user_name'] = df['定档媒介']
            elif '媒介姓名' in df.columns:
                df['schedule_user_name'] = df['媒介姓名']
            else:
                df['schedule_user_name'] = '未知'

        # ✅ ✅ ✅ 核心修复：确保schedule_user_name和submit_media_user_name字段包含映射后的真实姓名
        # 而不是原始的花名或ID
        if 'schedule_user_name' in df.columns:
            # 将schedule_user_name替换为映射后的真实姓名（媒介姓名）
            df['schedule_user_name'] = df['媒介姓名']
            logger.info(
                f"✅ 质量数据：已更新schedule_user_name为真实姓名，样本值: {df['schedule_user_name'].iloc[:3].tolist()}")

        if 'submit_media_user_name' in df.columns:
            # 将submit_media_user_name替换为映射后的真实姓名（对应真名）
            df['submit_media_user_name'] = df['对应真名']
            logger.info(
                f"✅ 质量数据：已更新submit_media_user_name为真实姓名，样本值: {df['submit_media_user_name'].iloc[:3].tolist()}")

        # 🔥 新增：确保定档媒介字段也使用真实姓名（如果前端使用这个字段）
        if '定档媒介' in df.columns:
            df['定档媒介'] = df['媒介姓名']
            logger.info(f"✅ 质量数据：已更新定档媒介为真实姓名，样本值: {df['定档媒介'].iloc[:3].tolist()}")

        # 确保数值字段正确转换
        numeric_fields = ['粉丝数', '报价', '下单价', '返点', '成本', '笔记点赞数', '笔记收藏数',
                          '笔记评论数', '互动量', '阅读量', '曝光量', '阅读uv数']

        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
                df[field] = df[field].fillna(0)

        # 清理小组名称中的空格
        if '所属小组' in df.columns:
            df['所属小组'] = df['所属小组'].astype(str).str.strip()

        # ✅ ✅ ✅ 关键修复：计算所有cost_analyzer.py需要的字段
        logger.info("开始计算成本分析必需字段...")
        df = calculate_missing_fields(df)

        # 调试：打印处理后数据
        if not df.empty:
            logger.info(f"✅ 工作质量处理后数据样本（前3行）- 成本分析字段检查：")
            for i in range(min(3, len(df))):
                logger.info(f"行{i}: "
                            f"达人昵称='{df.iloc[i].get('达人昵称', 'N/A')}', "
                            f"定档媒介='{df.iloc[i].get('定档媒介', 'N/A')}', "
                            f"所属小组='{df.iloc[i].get('所属小组', 'N/A')}', "
                            f"成本={df.iloc[i].get('成本', 0):.2f}, "
                            f"返点={df.iloc[i].get('返点', 0):.2f}, "
                            f"返点金额={df.iloc[i].get('返点金额', 0):.2f}, "
                            f"下单价={df.iloc[i].get('下单价', 0):.2f}, "
                            f"返点比例={df.iloc[i].get('返点比例', 0):.4f}, "
                            f"cpm={df.iloc[i].get('cpm', 0):.2f}, "
                            f"cpe={df.iloc[i].get('cpe', 0):.2f}")

            # 打印小组分布
            if '所属小组' in df.columns:
                group_dist = df['所属小组'].value_counts().to_dict()
                logger.info(f"✅ 工作质量数据小组分布: {group_dist}")

        return df
    except Exception as e:
        logger.error(f"工作质量数据查询失败: {str(e)}", exc_info=True)
        return pd.DataFrame()
    finally:
        conn.close()


def query_cost_data(start_date: str, end_date: str) -> pd.DataFrame:
    """查询成本效益分析数据"""
    conn = create_db_connection()
    if not conn:
        return pd.DataFrame()

    sql = """
    SELECT
        id,
        influencer_nickname,
        project_name,
        schedule_user_name,
        submit_media_user_name,
        state,
        kol_koc_type,
        note_type,
        follower_count,
        cooperation_quote,
        order_amount,
        rebate_amount,
        cost_amount,
        influencer_source,
        influencer_purpose,
        note_like_count,
        note_favorite_count,
        note_comment_count,
        interaction_count,
        read_count,
        exposure_count,
        read_uv_count,
        system_status,
        schedule_time
    FROM
        lgc_project_influencer
    WHERE
        schedule_time >= %s
        AND schedule_time < %s
        AND influencer_purpose = '优质达人'
        AND influencer_source = 'INSIDE'
        AND (state = "CHAIN_RETURNED" OR state = "SCHEDULED")
        AND project_name NOT IN ('快消组达人库', '家居组达人', '数码组达人库','测试-250801')
    """
    try:
        # ✅ 修复：直接使用pymysql查询，避免pandas的read_sql问题
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(sql, [f"{start_date} 00:00:00", f"{end_date} 23:59:59"])
            results = cursor.fetchall()

            logger.info(f"查询到成本效益数据 {len(results)} 条")

            if not results:
                logger.warning("查询返回空结果")
                return pd.DataFrame()

            # ✅ 直接从查询结果构建DataFrame
            df = pd.DataFrame(results)

            # ✅ 修复：转换Decimal类型为float
            decimal_fields = ['cost_amount', 'cooperation_quote', 'order_amount', 'rebate_amount',
                              'note_like_count', 'note_favorite_count', 'note_comment_count',
                              'interaction_count', 'read_count', 'exposure_count', 'read_uv_count',
                              'follower_count']

            for field in decimal_fields:
                if field in df.columns:
                    df[field] = df[field].apply(convert_decimal_to_float)

            # ✅ 修复：转换时间字段为字符串，避免JSON序列化问题
            time_columns = ['schedule_time']
            for col in time_columns:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else ""
                    )

            # 清理数据
            df = clean_and_prepare_data(df)

            logger.info(f"从查询结果构建DataFrame成功，形状: {df.shape}")

            # 验证数据
            if not df.empty:
                logger.info("成本数据验证（前3行）:")
                for i in range(min(3, len(df))):
                    nickname = df.iloc[i].get('influencer_nickname', 'N/A')
                    schedule_name = df.iloc[i].get('schedule_user_name', 'N/A')
                    cost = df.iloc[i].get('cost_amount', 0)
                    purpose = df.iloc[i].get('influencer_purpose', 'N/A')

                    logger.info(f"行{i}: influencer_nickname='{nickname}', "
                                f"schedule_user_name='{schedule_name}', "
                                f"influencer_purpose='{purpose}', "
                                f"cost_amount={cost}")

        # ✅ 修复：确保字段正确映射
        # 1. 确保有schedule_user_name字段
        if 'schedule_user_name' in df.columns:
            df['定档媒介'] = df['schedule_user_name']
        else:
            df['schedule_user_name'] = '未知'
            df['定档媒介'] = '未知'

        # 2. 确保有submit_media_user_name字段
        if 'submit_media_user_name' in df.columns:
            df['提交媒介'] = df['submit_media_user_name']
        else:
            df['submit_media_user_name'] = '未知'
            df['提交媒介'] = '未知'

        # 3. ✅ 核心修复：添加媒介姓名和对应真名字段，进行真实姓名映射
        logger.info("开始成本分析的媒介姓名映射...")

        # 定档媒介 -> 媒介姓名（映射到真实姓名）
        df['媒介姓名'] = df['定档媒介'].apply(map_media_to_real_name)

        # 提交媒介 -> 对应真名（映射到真实姓名）
        df['对应真名'] = df['提交媒介'].apply(map_media_to_real_name)

        # 4. 添加其他必要字段
        # ✅ 关键修复：使用正确的分组映射，而不是硬编码'默认组'
        df['所属小组'] = df['媒介姓名'].apply(get_media_group)  # 使用真实姓名获取小组
        df['数据类型'] = '定档'

        # 重命名字段为中文
        column_mapping = {
            'influencer_nickname': '达人昵称',
            'project_name': '项目名称',
            'state': '状态',
            'kol_koc_type': '达人量级',
            'note_type': '笔记类型(图文/视频)',
            'follower_count': '粉丝数',
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
            'system_status': 'system_status'
        }

        # 只重命名存在的列
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # ✅ 关键修复：保留原始字段名供分析器使用
        if '达人用途' in df.columns:
            df['influencer_purpose'] = df['达人用途']
        if '达人昵称' in df.columns:
            df['influencer_nickname'] = df['达人昵称']
        if '项目名称' in df.columns:
            df['project_name'] = df['项目名称']

        # ✅ ✅ ✅ 核心修复：确保schedule_user_name和submit_media_user_name字段包含映射后的真实姓名
        # 而不是原始的花名或ID
        if 'schedule_user_name' in df.columns:
            # 将schedule_user_name替换为映射后的真实姓名（媒介姓名）
            df['schedule_user_name'] = df['媒介姓名']
            logger.info(
                f"✅ 成本数据：已更新schedule_user_name为真实姓名，样本值: {df['schedule_user_name'].iloc[:3].tolist()}")

        if 'submit_media_user_name' in df.columns:
            # 将submit_media_user_name替换为映射后的真实姓名（对应真名）
            df['submit_media_user_name'] = df['对应真名']
            logger.info(
                f"✅ 成本数据：已更新submit_media_user_name为真实姓名，样本值: {df['submit_media_user_name'].iloc[:3].tolist()}")

        # 🔥 新增：确保定档媒介字段也使用真实姓名（如果前端使用这个字段）
        if '定档媒介' in df.columns:
            df['定档媒介'] = df['媒介姓名']
            logger.info(f"✅ 成本数据：已更新定档媒介为真实姓名，样本值: {df['定档媒介'].iloc[:3].tolist()}")

        # 确保数值字段正确转换
        numeric_fields = ['粉丝数', '报价', '下单价', '返点', '成本', '笔记点赞数', '笔记收藏数',
                          '笔记评论数', '互动量', '阅读量', '曝光量', '阅读uv数']

        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
                df[field] = df[field].fillna(0)

        # 清理小组名称中的空格
        if '所属小组' in df.columns:
            df['所属小组'] = df['所属小组'].astype(str).str.strip()

        # ✅ ✅ ✅ 关键修复：计算所有cost_analyzer.py需要的字段
        logger.info("开始计算成本分析必需字段...")
        df = calculate_missing_fields(df)

        # 调试：打印处理后数据
        if not df.empty:
            logger.info(f"✅ 成本处理后数据样本（前3行）- 成本分析字段检查：")
            for i in range(min(3, len(df))):
                logger.info(f"行{i}: "
                            f"达人昵称='{df.iloc[i].get('达人昵称', 'N/A')}', "
                            f"定档媒介='{df.iloc[i].get('定档媒介', 'N/A')}', "
                            f"所属小组='{df.iloc[i].get('所属小组', 'N/A')}', "
                            f"成本={df.iloc[i].get('成本', 0):.2f}, "
                            f"返点={df.iloc[i].get('返点', 0):.2f}, "
                            f"返点金额={df.iloc[i].get('返点金额', 0):.2f}, "
                            f"下单价={df.iloc[i].get('下单价', 0):.2f}, "
                            f"返点比例={df.iloc[i].get('返点比例', 0):.4f}, "
                            f"cpm={df.iloc[i].get('cpm', 0):.2f}, "
                            f"cpe={df.iloc[i].get('cpe', 0):.2f}")

            # 打印小组分布
            if '所属小组' in df.columns:
                group_dist = df['所属小组'].value_counts().to_dict()
                logger.info(f"✅ 成本数据小组分布: {group_dist}")

        return df
    except Exception as e:
        logger.error(f"成本效益数据查询失败: {str(e)}", exc_info=True)
        return pd.DataFrame()
    finally:
        conn.close()