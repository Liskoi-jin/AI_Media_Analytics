# src/utils.py
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Union, Any
import hashlib
import re
from pathlib import Path
from flask import flash

def create_dir_if_not_exist(dir_path):
    """创建目录（如果不存在）"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"创建目录：{dir_path}")

def init_logger():
    """初始化日志配置"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    create_dir_if_not_exist(log_dir)
    log_file = os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('AI_Media_Analytics')

# 全局日志实例
logger = init_logger()

# 配置日志
def setup_logger(name: str, log_file: str = None, level: str = 'INFO') -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    # 清除现有的处理器
    if logger.handlers:
        logger.handlers.clear()
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # 文件处理器（如果有日志文件）
    if log_file:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

# 初始化日志
logger = setup_logger(__name__)

# 媒介映射表（从原始代码中提取完整映射）
ID_TO_NAME_MAPPING = {
    # 家居媒介组11人
     511: "周月",  # 风风
    633: "王珂雯",  # 眼眼/喂喂
    567: "成思锦",  # 饼饼
    583: "周鑫悦",  # 桃桃
    591: "高何笑",  # 岁岁
    589: "蔡紫萱",  # 笙笙
    578: "吴松华",  # 杰瑞
    635: "肖国楠",  # smoggy/小楠
    640: "张诗晨",  # 四季
    643: "覃钰芝",  # 实/007
    582: "卢舒乐",
    654: "张佳欢",
    660: "谭果",
    666: "陈安妮",
    667: "石硕华",
    # 快消媒介组12人
    545: "郭玉",  # 脆脆鲨
    563: "宋金鑫",  # 金鑫
    579: "潘书美",  # 树莓
    590: "杨雅冰",  # 小饼
    557: "郝佳红",  # 郝郝
    543: "范志毅",  # 小范
    558: "姜钰",  # 肉肉
    552: "许佳",  # 茶茶
    565: "王鑫杰",  # 王守义
    553: "闫婉莹",  # 颜盐盐
    564: "李佳盈",  # 佳佳
    631: "申佳怡",
    561: "倪瑗鸿",
    # 数码媒介组
    514: "黄婧媛",  # 黄婧媛
    624: "陈梦麒",  # 琪琪
    622: "胡景勋",  # 勋勋
    576: "倪文琪",  # 又又
    577: "于欣欣",  # 欣欣
    584: "贾文娇",  # 饺子
    592: "蒋方强",  # 小白
    638: "刘瑞洁",  # 坚果
    623: "李紫怡",  # 晏宴
    574: "王雅瑞",  # 筱筱
    639: "李莎莎",  # 莎莎
    594: "黄涛",
    593: "潘龙雨",
    658: "江棋",  # 小酱
    659:"苏辰雨",
    663:'董洁',
    652:'胡敏',
    661:'武紫娟',
    # 素材组4人
    575: "罗凤婷",  # 凤婷
    588: "陈凤丽",  # 菠萝
    637: "谢梅",  # eswl
    # 其他
    642: "蔡晓晴",  # 菜菜
    627: "陈小迪",
    636: "陈熙妍",
    492: "陈可欣",
    598: "文琪",
    536: "岑雨婷",
    566: "田兰馨",
    531: "白琴琴",
    571: "江倩怡",
    537: "林文晶",
    641: "胡胡",
    572: "李依蒙",
    612: "依蒙",
    647: "李谨",
    427: "灵感管理员",
    625: "刘文娟",
    465: "张三",
    581: "张怡潇",
    538: "黄敏佳",
    569: "听音",
    570: "鸡腿",
    585: "王天园",
    644: "于洋",
    634: "朱舒琪",
    521: "杨婉婷",
    526: "杨馥清",
    648: "杨可心",
    528: "姚思敏",
    464: "严佳淇",
    503: "周冰冰",
    541: "宗原",
    499: "傅嘉棋",
    530: "郑卓越",
    562: "唐飞雪",
    573: "孙倩倩",
    485: "宋丹",
    568: "科文-1",
    535: "科文-2",
    621: "王科文",
    587: "王科美",
    649: "王琪",
    480: "王斯阳",
    580: "王茜",
    559: "许翠楠",
    619: "潘婷婷",
    472: "余思佳",
    507: "来蕾",
    626: "外部媒介测试",
    632: "邵淑彤",
    603: "莓莓",
    604: "树莓",
}

# 花名到真实名字的映射表（完整映射）
FLOWER_TO_NAME_MAPPING = {
    # 家居媒介组
    "风风": "周月",
    "喂喂": "王珂雯",
    "饼饼": "成思锦",
    "桃桃": "周鑫悦",
    "岁岁": "⾼何笑",
    "笙笙": "蔡紫萱",
    "杰瑞": "吴松华",
    "smoggy": "肖国楠",
    "小楠": "肖国楠",
    "四季": "张诗晨",
    "007": "覃钰芝",
    "实": "覃钰芝",
    "巧克力": "卢舒乐",
    '脏包':'张佳欢',
    '果果': '谭果',
    '小象': '陈安妮',
    '八月': '石硕华',
    # 快消媒介组
    "脆脆鲨": "郭玉",
    "金鑫": "宋金鑫",
    "树莓": "潘书美",
    "小饼": "杨雅冰",
    "郝郝": "郝佳红",
    "佳红": "郝佳红",
    "jiahong": "郝佳红",
    "小范": "范志毅",
    "肉肉": "姜钰",
    "茶茶": "许佳",
    "王守义": "王鑫杰",
    "颜盐盐": "闫婉莹",
    "盐颜颜": "闫婉莹",
    "佳佳": "李佳盈",
    "瓜瓜": "倪瑗鸿",
    "佳怡": "申佳怡",
    # 数码媒介组
    "黄婧媛": "黄婧媛",
    "琪琪": "陈梦麒",
    "勋勋": "胡景勋",
    "又又": "倪文琪",
    "欣欣": "于欣欣",
    "饺子": "贾文娇",
    "小白": "蒋方强",
    "坚果": "刘瑞洁",
    "晏晏": "李紫怡",
    "宴宴": "李紫怡",
    "筱筱": "王雅瑞",
    "莎莎": "李莎莎",
    "莓莓": "潘龙雨",
    "小米": "黄涛",
    "小酱": "江棋",
    "酸菜鱼":"苏辰雨",
    "三三": "董洁",
    "胡敏": "胡敏",
    "小满": "武紫娟",
    # 素材组
    "凤婷": "罗凤婷",
    "菠萝": "陈凤丽",
    "碗碗": "李宛霞",
    "eswl": "谢梅",
    # 其他
    "小迪": "陈小迪", "快快": "陈熙妍", "wenqi": "文琪", "雨婷": "岑雨婷",
    "懒懒": "田兰馨", "qianyi": "江倩怡", "wenjing": "林文晶", "胡胡": "胡胡",
    "yimeng": "李依蒙", "小谨": "李谨", "科文": "灵感管理员", "文娟": "刘文娟",
    "潇潇": "张怡潇", "泡泡": "黄敏佳", "听音": "听音", "鸡腿": "鸡腿",
    "天园": "王天园", "蛋挞": "于洋", "七七": "朱舒琪", "婉婷": "杨婉婷",
    "杨杨": "杨可心", "酥酥": "姚思敏", "可颂": "周冰冰", "zongyuan": "宗原",
    "doki": "傅嘉棋", "雪雪": "唐飞雪", "QQ": "孙倩倩", "蛋蛋": "宋丹",
    "技术 1": "科文-1", "技术": "科文-2", "kewen": "王科文", "KEMEI": "王科美",
    "77": "王琪", "茜子": "王茜", "cuinan": "许翠楠", "tingting": "潘婷婷",
    "waibu": "外部媒介测试", "淑彤": "邵淑彤", "菜菜": "蔡晓晴"
}

# 真实名字到小组的映射表（完整映射）
NAME_TO_GROUP_MAPPING = {
    # 家居媒介组
    "周月": "家居媒介组",
    "王珂雯": "家居媒介组",
    "成思锦": "家居媒介组",
    "周鑫悦": "家居媒介组",
    "高何笑": "家居媒介组",
    "蔡紫萱": "家居媒介组",
    "吴松华": "家居媒介组",
    "肖国楠": "家居媒介组",
    "张诗晨": "家居媒介组",
    "覃钰芝": "家居媒介组",
    "卢舒乐": "家居媒介组",
    "张佳欢":"家居媒介组",
    "谭果": "家居媒介组",
    "陈安妮": "家居媒介组",
    "石硕华": "家居媒介组",
    # 快消媒介组
    "郭玉": "快消媒介组",
    "宋金鑫": "快消媒介组",
    "潘书美": "快消媒介组",
    "杨雅冰": "快消媒介组",
    "郝佳红": "快消媒介组",
    "范志毅": "快消媒介组",
    "姜钰": "快消媒介组",
    "许佳": "快消媒介组",
    "王鑫杰": "快消媒介组",
    "闫婉莹": "快消媒介组",
    "李佳盈": "快消媒介组",
    "申佳怡": "快消媒介组",
    "倪瑗鸿": "快消媒介组",
    # 数码媒介组
    "黄婧媛": "数码媒介组",
    "陈梦麒": "数码媒介组",
    "胡景勋": "数码媒介组",
    "倪文琪": "数码媒介组",
    "于欣欣": "数码媒介组",
    "贾文娇": "数码媒介组",
    "蒋方强": "数码媒介组",
    "刘瑞洁": "数码媒介组",
    "李紫怡": "数码媒介组",
    "王雅瑞": "数码媒介组",
    "李莎莎": "数码媒介组",
    "黄涛": "数码媒介组",
    "潘龙雨": "数码媒介组",
    "江棋": "数码媒介组",
    "苏辰雨":"数码媒介组",
    "董洁": "数码媒介组",
    "胡敏": "数码媒介组",
    "武紫娟": "数码媒介组",
    # 素材组
    "罗凤婷": "素材组",
    "陈凤丽": "素材组",
    "李宛霞": "素材组",
    "谢梅": "素材组",
    "叶艳红": "素材组",
    # 其他
    "蔡晓晴": "其他组", "陈小迪": "其他组", "陈熙妍": "other组", "陈可欣": "other组",
    "文琪": "other组", "岑雨婷": "other组", "田兰馨": "other组", "白琴琴": "other组",
    "江倩怡": "other组", "林文晶": "other组", "胡胡": "other组", "李依蒙": "other组",
    "依蒙": "other组", "李谨": "other组", "灵感管理员": "other组", "刘文娟": "other组",
    "张三": "other组", "张怡潇": "other组", "黄敏佳": "other组", "听音": "other组",
    "鸡腿": "other组", "王天园": "other组", "于洋": "other组", "朱舒琪": "other组",
    "杨婉婷": "other组", "杨馥清": "other组", "杨可心": "other组", "姚思敏": "other组",
    "严佳淇": "other组", "周冰冰": "other组", "宗原": "other组", "傅嘉棋": "other组",
    "郑卓越": "other组", "唐飞雪": "other组", "孙倩倩": "other组", "宋丹": "other组",
    "科文-1": "other组", "科文-2": "other组", "王科文": "other组", "王科美": "other组",
    "王琪": "other组", "王斯阳": "other组", "王茜": "other组", "许翠楠": "other组",
    "潘婷婷": "other组", "余思佳": "other组", "来蕾": "other组", "外部媒介测试": "other组",
    "邵淑彤": "other组"
}

def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """检查文件扩展名是否允许"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in allowed_extensions

def generate_filename(prefix: str, extension: str = 'xlsx',
                      timestamp_format: str = '%Y%m%d_%H%M%S') -> str:
    """生成带时间戳的文件名"""
    timestamp = datetime.now().strftime(timestamp_format)
    return f"{prefix}_{timestamp}.{extension}"

def safe_read_csv(file_path: str, encodings: List[str] = None) -> pd.DataFrame:
    """安全读取CSV文件，尝试多种编码"""
    if encodings is None:
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'utf-8-sig']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, engine='python')
            logger.info(f"使用 {encoding} 编码成功读取文件: {os.path.basename(file_path)}")
            return df
        except Exception as e:
            logger.debug(f"编码 {encoding} 读取失败: {str(e)[:100]}")
            continue
    # 如果所有编码都失败，尝试不带编码参数
    try:
        df = pd.read_csv(file_path, engine='python')
        logger.info(f"使用默认编码成功读取文件: {os.path.basename(file_path)}")
        return df
    except Exception as e:
        logger.error(f"所有编码都读取失败: {str(e)}")
        raise ValueError(f"无法读取CSV文件: {os.path.basename(file_path)}")

def safe_read_excel(file_path: str, engines: List[str] = None) -> pd.DataFrame:
    """安全读取Excel文件，尝试多种引擎"""
    if engines is None:
        engines = ['openpyxl', 'xlrd']
    for engine in engines:
        try:
            df = pd.read_excel(file_path, engine=engine)
            logger.info(f"使用 {engine} 引擎成功读取Excel文件: {os.path.basename(file_path)}")
            return df
        except Exception as e:
            logger.debug(f"引擎 {engine} 读取失败: {str(e)[:100]}")
            continue
    logger.error(f"所有引擎都读取Excel失败: {os.path.basename(file_path)}")
    raise ValueError(f"无法读取Excel文件: {os.path.basename(file_path)}")

def read_data_file(file_path: str, data_type: str = None) -> pd.DataFrame:
    """
    智能读取数据文件（支持CSV和Excel），并添加数据类型标记
    :param file_path: 文件路径
    :param data_type: 数据类型（"定档" / "提报" / None）
    :return: 带数据类型标记的DataFrame
    """
    ext = file_path.rsplit('.', 1)[1].lower() if '.' in file_path else ''
    if ext in ['csv']:
        df = safe_read_csv(file_path)
    elif ext in ['xlsx', 'xls']:
        df = safe_read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    # 自动识别并添加数据类型标记（优先手动指定，其次文件名识别）
    if data_type:
        df['数据类型'] = data_type
    else:
        file_name = os.path.basename(file_path).lower()
        if '定档' in file_name:
            df['数据类型'] = '定档'
        elif '提报' in file_name:
            df['数据类型'] = '提报'
        else:
            df['数据类型'] = '未知'
    logger.info(f"文件 {os.path.basename(file_path)} 已标记为 {df['数据类型'].iloc[0]} 数据")
    return df

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """清理列名：去除空格、特殊字符"""
    df = df.copy()
    # 创建列名映射
    rename_dict = {}
    for col in df.columns:
        if isinstance(col, str):
            # 去除空格
            new_col = col.strip()
            # 替换中文括号为英文括号
            new_col = new_col.replace('（', '(').replace('）', ')')
            # 去除特殊字符（保留字母、数字、中文、下划线、括号）
            new_col = re.sub(r'[^\w\u4e00-\u9fff()（）]', '_', new_col)
            # 去除开头和结尾的下划线
            new_col = new_col.strip('_')
            if new_col != col:
                rename_dict[col] = new_col
    if rename_dict:
        df = df.rename(columns=rename_dict)
        logger.debug(f"清理列名: {list(rename_dict.values())[:10]}...")
    return df

def normalize_media_name(name: str) -> str:
    """标准化媒介名称（兼容花名、ID、真名）"""
    if pd.isna(name) or not isinstance(name, str):
        return '未知'
    name = str(name).strip()
    # 尝试直接匹配花名
    if name in FLOWER_TO_NAME_MAPPING:
        return FLOWER_TO_NAME_MAPPING[name]
    # 尝试大小写不敏感匹配
    name_lower = name.lower()
    for key, value in FLOWER_TO_NAME_MAPPING.items():
        if key.lower() == name_lower:
            return value
    # 尝试ID匹配（兼容.0后缀）
    if name.isdigit() or (name.endswith('.0') and name[:-2].isdigit()):
        clean_id = name.replace('.0', '') if name.endswith('.0') else name
        if clean_id in ID_TO_NAME_MAPPING:
            return ID_TO_NAME_MAPPING[clean_id]
    # 尝试部分匹配
    for key, value in FLOWER_TO_NAME_MAPPING.items():
        if name in key or key in name:
            return value
    # 检查是否已经是真名
    if name in NAME_TO_GROUP_MAPPING:
        return name
    return name

def get_media_group(media_name: str) -> str:
    """根据媒介名字获取所属小组"""
    if pd.isna(media_name) or media_name == '未知' or not isinstance(media_name, str):
        return '未知'
    media_name = str(media_name).strip()
    # 直接查找
    if media_name in NAME_TO_GROUP_MAPPING:
        return NAME_TO_GROUP_MAPPING[media_name]
    # 标准化后查找
    normalized_name = normalize_media_name(media_name)
    if normalized_name in NAME_TO_GROUP_MAPPING:
        return NAME_TO_GROUP_MAPPING[normalized_name]
    return 'other组'

def calculate_percentage(numerator: float, denominator: float,
                         default: float = 0.0) -> float:
    """安全计算百分比"""
    if denominator == 0 or pd.isna(denominator):
        return default
    return (numerator / denominator) * 100

def format_number(value: Union[int, float],
                  decimals: int = 2,
                  as_percentage: bool = False) -> str:
    """格式化数字显示"""
    if pd.isna(value):
        return 'N/A'
    if as_percentage:
        return f"{value:.{decimals}f}%"
    elif isinstance(value, float):
        return f"{value:.{decimals}f}"
    else:
        return f"{value:,}"

def save_dataframe_to_excel(df: pd.DataFrame, file_path: str,
                            sheet_name: str = '数据') -> bool:
    """将DataFrame保存到Excel文件"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 保存数据
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
        logger.info(f"数据已保存到Excel: {os.path.basename(file_path)}")
        return True
    except Exception as e:
        logger.error(f"保存Excel失败: {str(e)}")
        return False

def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """验证DataFrame是否包含必需的列"""
    if df.empty:
        logger.warning("DataFrame为空")
        return False
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"缺少必需的列: {missing_columns}")
            return False
    return True

def deduplicate_dataframe(df: pd.DataFrame,
                          subset: List[str] = None,
                          keep: str = 'first') -> pd.DataFrame:
    """去重DataFrame"""
    if subset:
        return df.drop_duplicates(subset=subset, keep=keep)
    else:
        return df.drop_duplicates(keep=keep)

def export_to_html(df: pd.DataFrame, title: str = "数据表格") -> str:
    """将DataFrame转换为HTML表格"""
    html = f'<h3>{title}</h3>\n'
    html += df.to_html(index=False, classes='table table-striped table-hover')
    return html

def get_file_hash(file_path: str) -> str:
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def show_flash_message(message: str, type: str = 'info'):
    """显示Flash消息"""
    categories = {
        'info': 'alert-info',
        'success': 'alert-success',
        'warning': 'alert-warning',
        'error': 'alert-danger'
    }
    flash(f'<div class="alert {categories.get(type, "alert-info")}">{message}</div>', 'message')

def calculate_summary_statistics(df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
    """计算数值列的汇总统计"""
    summary = {}
    for col in numeric_columns:
        if col in df.columns:
            try:
                series = pd.to_numeric(df[col], errors='coerce')
                valid_values = series.dropna()
                if len(valid_values) > 0:
                    summary[f'{col}_mean'] = valid_values.mean()
                    summary[f'{col}_median'] = valid_values.median()
                    summary[f'{col}_min'] = valid_values.min()
                    summary[f'{col}_max'] = valid_values.max()
                    summary[f'{col}_sum'] = valid_values.sum()
                    summary[f'{col}_count'] = len(valid_values)
                else:
                    summary[f'{col}_count'] = 0
            except Exception as e:
                logger.warning(f"计算列 {col} 的统计信息失败: {str(e)}")
    return summary

def merge_dataframes(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """合并多个DataFrame"""
    if not df_list:
        return pd.DataFrame()
    try:
        # 合并所有DataFrame
        merged_df = pd.concat(df_list, ignore_index=True)
        logger.info(f"合并了 {len(df_list)} 个DataFrame，总行数: {len(merged_df)}")
        return merged_df
    except Exception as e:
        logger.error(f"合并DataFrame失败: {str(e)}")
        raise