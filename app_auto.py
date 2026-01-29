"""
媒介自动化审计分析系统 - Flask主应用
整合工作量、工作质量、成本三大真实分析模块
提供Web交互界面（兼容中文文件名+编码自动适配+去重上传+完整异常兜底）
"""

# app_auto.py 最顶部
import logging
import os


# ========== 【终极修复】一次性的日志配置 ==========
def setup_logging():
    """
    统一日志配置，确保只配置一次，避免重复打印
    """
    # 1. 获取根日志器和当前模块日志器
    root_logger = logging.getLogger()
    app_logger = logging.getLogger(__name__)

    # 2. 清除所有现有处理器（避免重复）
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    if app_logger.handlers:
        for handler in app_logger.handlers[:]:
            app_logger.removeHandler(handler)

    # 3. 设置日志级别
    root_logger.setLevel(logging.INFO)
    app_logger.setLevel(logging.INFO)

    # 4. 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 5. 只创建一个控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 6. 防止重复添加处理器
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)

    # 7. 防止向上传播到根日志器（避免重复打印）
    app_logger.propagate = False

    # 8. 确保 app_logger 也有处理器（如果不传播到根日志器）
    if not app_logger.handlers:
        app_logger.addHandler(console_handler)

    return app_logger


# 立即配置日志
logger = setup_logging()
logger.info("✅ 日志系统配置完成（单次配置，无重复）")
import os
import re
import json
import unicodedata
import traceback
import logging  # 先导入logging模块
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Any

import pandas as pd
import numpy as np
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, send_file, jsonify, g, make_response, send_from_directory
)
from werkzeug.utils import secure_filename

from auth import auth_bp, init_db as init_auth_db
from flask import session, redirect, url_for
from auth.utils import get_current_user, login_required

from src.db_utils import query_workload_data, query_quality_data, query_cost_data
from datetime import datetime



# ------------------------------ 初始化基础logger ------------------------------
# 先创建基础logger，确保在任何情况下都有logger可用
def setup_default_logger():
    """创建和配置默认logger"""
    logger = logging.getLogger(__name__)
    # 避免重复添加handler
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
            ]
        )
    return logger

# 立即创建基础logger
logger = setup_default_logger()

# ------------------------------ 引入真实分析模块 & 工具类 ------------------------------
# 兼容模块导入失败的兜底处理（避免应用启动报错）
ID_TO_NAME_MAPPING = {}
NAME_TO_GROUP_MAPPING = {}

try:
    from src.data_processor import DataProcessor
    from src.workload_analyzer import WorkloadAnalyzer
    from src.quality_analyzer import QualityAnalyzer  # 工作质量分析器
    from src.cost_analyzer import CostAnalyzer        # 成本分析器
    from src.report_generator import ReportGenerator  # 报告生成器
    from src.utils import ID_TO_NAME_MAPPING as SRC_ID_MAPPING, NAME_TO_GROUP_MAPPING as SRC_NAME_MAPPING

    # 如果成功导入，使用src模块中的映射
    ID_TO_NAME_MAPPING = SRC_ID_MAPPING
    NAME_TO_GROUP_MAPPING = SRC_NAME_MAPPING
    logger.info("✅ 成功导入src所有分析模块")

except ImportError as e:
    logger.warning(f"⚠️ 部分模块导入失败：{e}，已启用兜底模拟类，不影响基础运行")

    # 模拟DataProcessor类
    class DataProcessor:
        def process_for_media_analysis(self, file_paths, category):
            return {"processed_data": pd.DataFrame(), "filtered_data": pd.DataFrame(), "stats": {}}

        def process_for_cost_analysis(self, file_paths, category):
            return {"processed_data": pd.DataFrame(), "filtered_data": pd.DataFrame(), "stats": {}}

    # 模拟WorkloadAnalyzer类
    class WorkloadAnalyzer:
        def __init__(self, df, known_id_name_mapping=None, config=None):
            self.df = df
            self.config = config or {}
        def analyze(self, top_n=10):
            detail_df = self.df.reset_index(drop=False).fillna("") if not self.df.empty else pd.DataFrame()
            group_df = self.df.groupby("小组名称").sum().reset_index(drop=False).fillna("") if not self.df.empty else pd.DataFrame()
            return {"detail": detail_df, "summary": {}, "group_summary": group_df, "top_media_ranking": detail_df}

    # 模拟QualityAnalyzer类
    class QualityAnalyzer:
        def __init__(self, df, known_id_name_mapping=None, config=None):
            self.df = df
            self.config = config or {}
        def analyze(self, use_original_state=False):
            detail_df = self.df.reset_index(drop=False).fillna("") if not self.df.empty else pd.DataFrame()
            group_df = self.df.groupby("小组名称").sum().reset_index(drop=False).fillna("") if not self.df.empty else pd.DataFrame()
            return {"detail": detail_df, "summary": {}, "group_summary": group_df, "quality_distribution": detail_df}

    # 模拟CostAnalyzer类
    class CostAnalyzer:
        def __init__(self, processed_df, filtered_df):
            self.processed_df = processed_df
            self.filtered_df = filtered_df
        def analyze(self, top_n=10):
            media_detail = self.processed_df.reset_index(drop=False).fillna("") if not self.processed_df.empty else pd.DataFrame()
            group_summary = self.processed_df.groupby("小组名称").sum().reset_index(drop=False).fillna("") if not self.processed_df.empty else pd.DataFrame()
            return {
                "overall_summary": {'整体平均成本':0.0,'整体返点占报价比例(%)':'0%','总成本':0.0},
                "media_detail": media_detail,
                "group_summary": group_summary,
                "filtered_summary": {'筛除总成本':0,'筛除成本占比':0,'筛除达人数量':0,'筛除发布数量':0},
                "cost_efficiency_ranking": media_detail
            }

    # ========== 核心修复：重写真实的ReportGenerator 不再是模拟空表 ==========
    class ReportGenerator:
        def __init__(self, analysis_results=None, output_dir="./outputs"):
            self.analysis_results = analysis_results if analysis_results is not None else {}
            self.output_dir = output_dir
            # 确保报告输出目录存在
            os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'excel'), exist_ok=True)

        # ========== ✅ 修复：修改generate_excel_report方法，移除analysis_id参数 ==========
        def generate_excel_report(self, analysis_mode='full'):
            """生成Excel报告 - 真实写入多sheet数据+修复索引错乱+空数据写入表头"""
            try:
                # ✅ 修复：移除analysis_id参数，使用时间戳生成文件名
                time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                excel_filename = f"Media_Analysis_Report_{time_str}.xlsx"
                excel_file_path = os.path.join(self.output_dir, 'excel', excel_filename)

                # 获取分析结果数据
                workload_detail = self.analysis_results.get('workload', {}).get('result', pd.DataFrame())
                quality_detail = self.analysis_results.get('quality', {}).get('result', pd.DataFrame())
                cost_detail = self.analysis_results.get('cost', {}).get('media_detail', pd.DataFrame())
                cost_ranking = self.analysis_results.get('cost', {}).get('cost_efficiency_ranking', pd.DataFrame())

                # ✅ 核心修复：reset_index(drop=True) 丢弃索引，只保留业务数据，解决索引错乱
                # ✅ 空数据也写入表头，fillna(0) 数字列填充0，字符串列填充空字符串
                with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
                    # 工作量分析sheet - 必写，空数据也写入表头
                    if not workload_detail.empty:
                        workload_df = workload_detail.reset_index(drop=True).fillna(
                            {"小组名称": "", "媒介名称": ""}).fillna(0)
                        workload_df.to_excel(writer, sheet_name="媒介工作量分析", index=False)
                    else:
                        pd.DataFrame({"提示": ["无工作量分析数据"]}).to_excel(writer, sheet_name="媒介工作量分析", index=False)

                    # 质量分析sheet - 必写
                    if not quality_detail.empty:
                        quality_df = quality_detail.reset_index(drop=True).fillna({"小组名称": "", "媒介名称": ""}).fillna(0)
                        quality_df.to_excel(writer, sheet_name="媒介质量分析", index=False)
                    else:
                        pd.DataFrame({"提示": ["无质量分析数据"]}).to_excel(writer, sheet_name="媒介质量分析", index=False)

                    # 成本明细sheet - 必写
                    if not cost_detail.empty:
                        cost_df = cost_detail.reset_index(drop=True).fillna({"小组名称": "", "媒介名称": ""}).fillna(0)
                        cost_df.to_excel(writer, sheet_name="媒介成本明细", index=False)
                    else:
                        pd.DataFrame({"提示": ["无成本分析数据"]}).to_excel(writer, sheet_name="媒介成本明细", index=False)

                    # 成本效率排名sheet - 必写
                    if not cost_ranking.empty:
                        cost_rank_df = cost_ranking.reset_index(drop=True).fillna({"小组名称": "", "媒介名称": ""}).fillna(0)
                        cost_rank_df.to_excel(writer, sheet_name="成本效率排名", index=False)
                    else:
                        pd.DataFrame({"提示": ["无成本效率排名数据"]}).to_excel(writer, sheet_name="成本效率排名", index=False)

                    # 汇总sheet - 必写，展示各模块数据量
                    summary_df = pd.DataFrame({
                        '分析类型': ['工作量分析', '质量分析', '成本分析'],
                        '有效数据量': [len(workload_detail), len(quality_detail), len(cost_detail)],
                        '生成时间': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * 3
                    })
                    summary_df.to_excel(writer, sheet_name="分析汇总", index=False)

                logger.info(f"✅ Excel报告生成成功：{excel_file_path}")
                return excel_file_path
            except Exception as e:
                logger.error(f"生成Excel报告失败：{e}")
                return ""

        def generate_all_reports(self, analysis_mode='full'):
            """生成所有格式报告"""
            return {"excel_report": self.generate_excel_report(analysis_mode), "html_report": None}

    DataProcessor = DataProcessor
    WorkloadAnalyzer = WorkloadAnalyzer
    QualityAnalyzer = QualityAnalyzer
    CostAnalyzer = CostAnalyzer
    ReportGenerator = ReportGenerator

# ------------------------------ 初始化配置 ------------------------------
app = Flask(__name__)

# ========== 新增：数据库配置 ==========
# 直接设置数据库配置
DB_CONFIG = {
    'host': 'rm-cn-2104msjne000170o.rwlb.rds.aliyuncs.com',
    'port': 3306,
    'user': 'root',  # 改为你的数据库用户名
    'password': 'Lj041213',  # 改为你的数据库密码
    'database': 'ai_media_db',  # 改为你的数据库名
    'charset': 'utf8mb4'
}

# 构建数据库连接URI
app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?charset={DB_CONFIG['charset']}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = False

# 初始化权限模块数据库（独立初始化，不影响原有系统）
app.config['SECRET_KEY'] = app.config.get('SECRET_KEY', 'ai_media_auth_2025_secure')  # 用于会话加密
init_auth_db(app)  # 初始化用户表（若已存在则不创建）

# 注册权限模块蓝图（路由前缀/auth，与原有系统隔离）
app.register_blueprint(auth_bp)

# 生产环境推荐：从环境变量读取秘钥，本地开发用默认值
app.secret_key = os.getenv('SECRET_KEY', 'media-audit-2025-secure-key-@#$%^&*')

# 配置上传文件夹（使用绝对路径，避免权限问题）
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB文件上传限制

# 基础配置
app.config['DEBUG'] = True
app.config['OUTPUT_DIR'] = os.path.join(BASE_DIR, 'outputs')
app.config['LOG_FILE'] = os.path.join(app.config['OUTPUT_DIR'], 'logs', 'media_audit.log')
app.config['LOG_LEVEL'] = 'INFO'

# ========================== 核心修复 1/5：注册全局 format_number 过滤器 + 新增 safe_min 过滤器 ==========================
@app.template_filter('format_number')
def format_number_filter(value, decimal_places=2):
    """
    Jinja2过滤器：格式化数字，保留指定小数位，兼容空值/非数字/NaN/None，适配成本分析金额展示
    :param value: 要格式化的值（支持数字/字符串/空值）
    :param decimal_places: 保留小数位数，默认2位
    :return: 格式化后的字符串，空值返回 0.00
    """
    try:
        # 【修复】前置判断是否为数值类型，避免pd.isna处理非pd对象报错
        if isinstance(value, (str, int, float)) is False or value is None or value == '' or str(value).strip() == '-':
            return f"0.{0 * decimal_places}"
        # 处理空值、None、NaN等异常情况
        if pd.isna(value):
            return f"0.{0 * decimal_places}"
        # 转为浮点数后格式化
        num = float(value)
        return f"{num:.{decimal_places}f}"
    except (ValueError, TypeError, Exception):
        # 非数字类型直接返回原值，避免报错
        return f"0.{0 * decimal_places}"

@app.template_filter('safe_min')
def safe_min_filter(value, min_val):
    """
    【核心修复】解决 Jinja2 原生 min 过滤器报错：TypeError: 'float' object is not iterable
    专为 单个数值 和 对比值 设计的安全最小值过滤器，用于百分比/宽度限制场景
    """
    try:
        val = float(value) if value else 0.0
        minv = float(min_val) if min_val else 0.0
        return min(val, minv)
    except:
        return min_val

# 新增format_percentage过滤器，用于quality_analysis.html中的百分比格式化
@app.template_filter('format_percentage')
def format_percentage_filter(value, default='0.00%'):
    """
    格式化百分比，兼容字符串和数值类型
    """
    try:
        if value is None or pd.isna(value):
            return default
        if isinstance(value, (int, float)):
            return f"{value:.2f}%"
        if isinstance(value, str):
            # 如果已经是百分比格式，直接返回
            if '%' in value:
                return value
            # 否则尝试转换
            try:
                num = float(value)
                return f"{num:.2f}%"
            except:
                return default
        return default
    except:
        return default

# ------------------------------ 确保目录存在 ------------------------------
def create_dir_with_permission(dir_path):
    """创建目录并处理权限问题，带兜底方案"""
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, mode=0o755, exist_ok=True)
            logger.info(f"📂 目录创建成功：{dir_path}")
        except PermissionError as e:
            logger.error(f"❌ 创建目录失败，权限不足：{dir_path}，错误：{e}")
            fallback_dir = os.path.join(os.path.expanduser("~"), "media_audit", os.path.basename(dir_path))
            os.makedirs(fallback_dir, mode=0o755, exist_ok=True)
            logger.warning(f"⚠️ 自动创建兜底目录：{fallback_dir}")
            return fallback_dir
    return dir_path

# 创建所有必要目录
app.config['UPLOAD_FOLDER'] = create_dir_with_permission(app.config['UPLOAD_FOLDER'])
app.config['OUTPUT_DIR'] = create_dir_with_permission(app.config['OUTPUT_DIR'])
create_dir_with_permission(os.path.join(app.config['OUTPUT_DIR'], 'analysis_results'))
create_dir_with_permission(os.path.join(app.config['OUTPUT_DIR'], 'logs'))
create_dir_with_permission(os.path.join(app.config['OUTPUT_DIR'], 'reports'))
create_dir_with_permission(os.path.join(app.config['OUTPUT_DIR'], 'excel'))

# 重新配置日志（使用已创建的目录）
try:
    file_handler = logging.FileHandler(app.config['LOG_FILE'], encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(getattr(logging, app.config['LOG_LEVEL'].upper()))
    logger.info("📝 日志系统配置完成，日志文件已生效")
except Exception as e:
    logger.error(f"❌ 配置日志文件失败：{e}，将继续使用控制台日志")

# ------------------------------ 核心工具函数 ------------------------------
def secure_filename_cn(filename):
    """兼容中文的安全文件名处理，彻底解决中文乱码/非法字符问题"""
    if not filename:
        return 'unnamed_file'
    filename = unicodedata.normalize('NFKC', filename)
    illegal_chars = r'[\\/:*?"<>|]'
    filename = re.sub(illegal_chars, '_', filename)
    filename = filename.strip()
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:200] + ext
    return filename if filename else 'unnamed_file'

def save_file_with_duplicate_check(file, force_cover=False):
    """保存文件+双层去重：①同请求内去重 ②文件已存在去重，彻底解决重复上传"""
    if not file or file.filename.strip() == '':
        return ""
    original_filename = secure_filename_cn(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)

    # 同请求内去重：避免一次上传多个相同文件
    if not hasattr(g, 'uploaded_files'):
        g.uploaded_files = set()
    if original_filename in g.uploaded_files:
        logger.info(f"⏭️ 同请求重复文件，跳过：{original_filename}")
        return ""
    g.uploaded_files.add(original_filename)

    # 文件已存在去重
    if not os.path.exists(save_path) or force_cover:
        try:
            file.save(save_path)
            logger.info(f"✅ 文件保存成功：{original_filename}")
        except Exception as e:
            logger.error(f"❌ 文件保存失败：{e}")
            save_path = ""
    else:
        logger.info(f"⏭️ 文件已存在，跳过上传：{original_filename}")
    return save_path

def read_file_with_auto_encoding(file_path):
    """自动识别编码读取Excel/CSV，兼容所有常见编码，兜底空DataFrame"""
    if not os.path.exists(file_path):
        logger.error(f"❌ 文件不存在：{file_path}")
        return pd.DataFrame()
    file_ext = os.path.splitext(file_path)[1].lower()
    try:
        if file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, engine='openpyxl' if file_ext == '.xlsx' else 'xlrd')
        elif file_ext == '.csv':
            encoding_list = ['utf-8-sig', 'gbk', 'gb2312', 'latin-1', 'utf-8']
            for encoding in encoding_list:
                try:
                    return pd.read_csv(file_path, encoding=encoding)
                except (UnicodeDecodeError, Exception):
                    continue
            raise Exception(f"编码不兼容：{os.path.basename(file_path)}")
        else:
            logger.warning(f"⚠️ 不支持的文件格式：{file_ext}，仅支持xlsx/xls/csv")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"❌ 读取文件失败：{file_path}，错误：{e}")
        return pd.DataFrame()


# ========== 新增：数据库字段映射函数 ==========
def map_database_fields(df):
    """
    简化数据库字段映射 - 因为db_utils.py已经返回正确的字段名
    """
    if df.empty:
        return df

    df_copy = df.copy()

    logger.info(f"数据库字段映射，原始列名: {list(df_copy.columns)}")
    logger.info(f"数据示例: {df_copy.iloc[0].to_dict() if len(df_copy) > 0 else '空数据'}")

    # 确保有必要的字段
    required_fields = ['媒介姓名', '对应真名', '所属小组', '数据类型', '定档媒介', '提交媒介']
    for field in required_fields:
        if field not in df_copy.columns:
            logger.warning(f"缺少必要字段: {field}")
            if field == '所属小组':
                df_copy[field] = '默认组'
            elif field == '数据类型':
                df_copy[field] = '提报'  # 默认值
            elif field in ['媒介姓名', '定档媒介'] and 'schedule_user_name' in df_copy.columns:
                df_copy[field] = df_copy['schedule_user_name']
            elif field in ['对应真名', '提交媒介'] and 'submit_media_user_name' in df_copy.columns:
                df_copy[field] = df_copy['submit_media_user_name']
            else:
                df_copy[field] = '未知'

    return df_copy

# ========== 替换你当前的 convert_pandas_types_to_python 函数 ==========
def convert_pandas_types_to_python(data):
    """
    核心修复：递归转换Pandas/numpy特殊类型为Python原生类型
    彻底解决「Object of type int64/float64 is not JSON serializable」报错
    ✅ 新增：空DataFrame返回带表头的字典列表、NaN数字填充0、字符串填充空值
    """
    if isinstance(data, pd.DataFrame):
        if data.empty:
            # ✅ 空DataFrame返回空列表，模板遍历无数据时显示"暂无数据"
            return []
        # ✅ 关键修复：fillna(0) 处理所有数值列，确保不会出现NaN
        try:
            # 先复制数据
            df_copy = data.copy()
            # 获取数值列
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            # 数值列填充0
            if len(numeric_cols) > 0:
                df_copy[numeric_cols] = df_copy[numeric_cols].fillna(0)
            # 对象列填充空字符串
            object_cols = df_copy.select_dtypes(include=['object']).columns
            if len(object_cols) > 0:
                df_copy[object_cols] = df_copy[object_cols].fillna('')

            return df_copy.reset_index(drop=True).to_dict('records')
        except Exception as e:
            logger.error(f"转换DataFrame失败: {e}")
            return []
    elif isinstance(data, pd.Series):
        try:
            return data.reset_index(drop=True).fillna(0).to_dict()
        except:
            return {}
    elif isinstance(data, dict):
        return {key: convert_pandas_types_to_python(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [convert_pandas_types_to_python(item) for item in data]
    elif isinstance(data, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(data)
    elif pd.api.types.is_integer_dtype(type(data)):
        return int(data) if pd.notna(data) else 0
    elif isinstance(data, (np.floating, np.float16, np.float32, np.float64)):
        return float(data)
    elif pd.api.types.is_float_dtype(type(data)):
        return float(data) if pd.notna(data) else 0.0
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (pd.Timestamp, datetime)):
        return data.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(data) else ""
    elif pd.isna(data):
        return 0 if isinstance(data, (int, float)) else ""
    return data

def preprocess_percent_str_to_float(percent_str):
    """预处理百分数字符串转浮点型，模板渲染专用，异常值兜底0.0"""
    if not percent_str:
        return 0.0
    if not isinstance(percent_str, str):
        try:
            return float(percent_str)
        except (ValueError, TypeError):
            return 0.0
    try:
        num_str = percent_str.replace('%', '').strip()
        return float(num_str) if num_str else 0.0
    except (ValueError, AttributeError):
        return 0.0

# ------------------------------ 新增核心修复函数 - 补全小组数据字段 ------------------------------
def fill_group_data_fields(group_list):
    """
    修复核心问题：为小组数据补全【总定档数、总提报数】字段，解决模板Undefined报错
    所有缺失字段默认赋值0，完美适配workload_analysis.html的{{ group_data|map(attribute='总定档数')|max }}语法
    """
    filled_group = []
    for group in group_list:
        if isinstance(group, dict):
            group['总定档数'] = group.get('总定档数', 0) or 0
            group['总提报数'] = group.get('总提报数', 0) or 0
            group['定档数'] = group.get('定档数', 0) or 0
            group['提报数'] = group.get('提报数', 0) or 0
            group['小组名称'] = group.get('小组名称', '未知小组') or '未知小组'
        filled_group.append(group)
    return filled_group

# ------------------------------ 【新增核心修复】补全成本数据所有缺失字段 ------------------------------
def fill_cost_data_fields(cost_data_list):
    """
    修复核心报错：dict object has no attribute '筛除总成本'
    为成本分析的每条数据补全所有前端模板用到的中文key字段，默认值0，彻底解决字段不存在报错
    """
    filled_cost = []
    # 前端模板用到的所有成本相关字段，全部兜底
    cost_fields = [
        '筛除总成本', '筛除成本占比', '筛除达人数量', '筛除发布数量',
        '总成本', '平均成本', '总返点金额', '返点占比',
        '媒介名称', '小组名称', '总发布数', '总达人数',
        '有效发布数', '有效达人数', '成本发挥率'
    ]
    for row in cost_data_list:
        if isinstance(row, dict):
            for field in cost_fields:
                if field not in row or row[field] is None or pd.isna(row[field]):
                    row[field] = 0
        filled_cost.append(row)
    return filled_cost

# ------------------------------ 全局变量 ------------------------------
analysis_results = {}  # 内存存储分析结果
# 初始化核心模块（全局单例，避免重复初始化）
data_processor = DataProcessor()

# ------------------------------ 上下文处理器 ------------------------------
def has_endpoint(endpoint_name):
    return endpoint_name in app.view_functions


# ------------------------------ 新增：简化分析函数 ------------------------------
def create_simple_workload_analysis(df):
    """简化工作量分析 - 使用前端模板期望的字段名"""
    result = {
        "result": [],
        "summary": {},
        "group_summary": [],
        "top_media_ranking": []
    }

    try:
        if df.empty:
            return result

        logger.info(f"执行简化工作量分析，数据行数: {len(df)}")
        logger.info(f"数据列名: {list(df.columns)}")

        # 确保有必要的字段
        required_fields = ['媒介姓名', '对应真名', '所属小组']
        missing_fields = [f for f in required_fields if f not in df.columns]
        if missing_fields:
            logger.warning(f"缺少必要字段: {missing_fields}")
            # 尝试创建缺失字段
            if '媒介姓名' not in df.columns:
                if '定档媒介' in df.columns:
                    df['媒介姓名'] = df['定档媒介']
                else:
                    df['媒介姓名'] = '未知媒介'

            if '对应真名' not in df.columns:
                df['对应真名'] = df['媒介姓名']  # 使用媒介姓名作为对应真名

            if '所属小组' not in df.columns:
                df['所属小组'] = '默认组'

        # 统计每个媒介的工作量
        media_summary = df.groupby('媒介姓名').agg({
            '达人昵称': 'count'  # 统计处理达人数量
        }).reset_index()

        media_summary.columns = ['媒介姓名', '总处理量']

        # 获取每个媒介的其他信息
        # 1. 对应真名
        if '对应真名' in df.columns:
            media_realname = df.groupby('媒介姓名')['对应真名'].first().reset_index()
            media_summary = pd.merge(media_summary, media_realname, on='媒介姓名', how='left')
        else:
            media_summary['对应真名'] = media_summary['媒介姓名']

        # 2. 所属小组
        if '所属小组' in df.columns:
            media_group = df.groupby('媒介姓名')['所属小组'].first().reset_index()
            media_summary = pd.merge(media_summary, media_group, on='媒介姓名', how='left')
        else:
            media_summary['所属小组'] = '默认组'

        # 3. 计算定档量（这里简化处理，假设有成本或下单价的数据就是定档数据）
        if '成本' in df.columns:
            media_cost = df[df['成本'] > 0].groupby('媒介姓名').size().reset_index()
            media_cost.columns = ['媒介姓名', '定档量']
            media_summary = pd.merge(media_summary, media_cost, on='媒介姓名', how='left')
            media_summary['定档量'] = media_summary['定档量'].fillna(0)
        else:
            media_summary['定档量'] = media_summary['总处理量']  # 如果没有成本数据，假设所有都是定档

        # 4. 计算定档率
        media_summary['定档率(%)'] = media_summary.apply(
            lambda row: f"{(row['定档量'] / row['总处理量'] * 100):.2f}%" if row['总处理量'] > 0 else "0.00%",
            axis=1
        )

        # 5. 添加评估字段（前端模板需要）
        media_summary['定档率评估'] = media_summary['定档率(%)'].apply(
            lambda x: '优秀' if float(x.replace('%', '')) > 80 else '良好' if float(x.replace('%', '')) > 60 else '一般'
        )

        media_summary['产量评估'] = media_summary['总处理量'].apply(
            lambda x: '高产' if x > 50 else '中产' if x > 20 else '低产'
        )

        media_summary['综合评估'] = media_summary.apply(
            lambda row: 'S级' if row['总处理量'] > 100 and float(row['定档率(%)'].replace('%', '')) > 90
            else 'A级' if row['总处理量'] > 50 and float(row['定档率(%)'].replace('%', '')) > 80
            else 'B级' if row['总处理量'] > 20 and float(row['定档率(%)'].replace('%', '')) > 70
            else 'C级' if row['总处理量'] > 10 else 'D级',
            axis=1
        )

        # 6. 添加其他字段（前端模板需要）
        media_summary['已发布数'] = 0
        media_summary['未发布数'] = 0
        media_summary['其他状态数'] = 0

        # 7. 排序并添加排名
        media_summary = media_summary.sort_values('总处理量', ascending=False)
        media_summary['排名'] = range(1, len(media_summary) + 1)

        # 构建结果
        result = {
            "result": media_summary.to_dict('records'),
            "summary": {
                '总处理量': len(df),
                '总定档量': int(media_summary['定档量'].sum()),
                '媒介总数': len(media_summary),
                '整体定档率': f"{(media_summary['定档量'].sum() / len(df) * 100):.2f}%" if len(df) > 0 else "0.00%"
            },
            "group_summary": [],
            "top_media_ranking": media_summary.head(10).to_dict('records')
        }

        logger.info(f"简化工作量分析完成，涉及媒介数: {len(media_summary)}")
        logger.info(f"总处理量: {len(df)}, 总定档量: {int(media_summary['定档量'].sum())}")

    except Exception as e:
        logger.error(f"简化工作量分析失败: {e}", exc_info=True)

    return result


def create_simple_quality_analysis(df):
    """简化工作质量分析"""
    result = {
        "result": [],
        "summary": {},
        "group_summary": [],
        "quality_distribution": [],
        "premium_detail": [],
        "high_read_detail": []
    }

    try:
        if df.empty:
            return result

        logger.info(f"执行简化工作质量分析，数据行数: {len(df)}")

        # 确保有关键字段
        if '达人用途' not in df.columns:
            logger.warning("数据中无'达人用途'字段，创建默认值")
            df['达人用途'] = '普通达人'

        # 按达人用途统计
        purpose_distribution = []
        if '达人用途' in df.columns and len(df) > 0:
            purpose_counts = df['达人用途'].value_counts().reset_index()
            purpose_counts.columns = ['达人用途', '数量']
            purpose_distribution = purpose_counts.to_dict('records')

        # 提取优质达人数据
        premium_detail = []
        if '达人用途' in df.columns:
            premium_df = df[df['达人用途'].str.contains('优质达人', na=False)]
            premium_detail = premium_df.head(100).to_dict('records') if not premium_df.empty else []

        # 提取高阅读达人数据
        high_read_detail = []
        if '达人用途' in df.columns:
            high_read_df = df[df['达人用途'].str.contains('高阅读达人', na=False)]
            high_read_detail = high_read_df.head(100).to_dict('records') if not high_read_df.empty else []

        # 汇总统计
        summary = {
            '总提报数': len(df),
            '达人用途分布': purpose_distribution,
            '涉及项目数': df['项目名称'].nunique() if '项目名称' in df.columns else 0,
            '优质达人数量': len(premium_detail),
            '高阅读达人数量': len(high_read_detail),
            '备注': '简化工作质量分析（不依赖数据类型）'
        }

        # 取前100条作为明细
        detail_data = df.head(100).to_dict('records') if len(df) > 0 else []

        # 小组汇总（如果有小组信息）
        group_summary = []
        if '小组名称' in df.columns and len(df) > 0:
            group_df = df.groupby('小组名称').agg({
                '达人昵称': 'count'
            }).reset_index()
            group_df.columns = ['小组名称', '提报数量']
            group_summary = group_df.to_dict('records')

        result = {
            "result": detail_data,
            "summary": summary,
            "group_summary": group_summary,
            "quality_distribution": purpose_distribution,
            "premium_detail": premium_detail,
            "high_read_detail": high_read_detail
        }

        logger.info(f"简化质量分析完成，总记录数: {len(df)}")

    except Exception as e:
        logger.error(f"简化质量分析失败: {e}", exc_info=True)

    return result


def create_simple_cost_analysis(df):
    """简化成本分析 - 修复版"""
    result = {
        "result": [],
        "summary": {},
        "overall_summary": {},
        "invalid_data_stats": {},
        "media_detail": [],
        "group_summary": [],
        "filtered_summary": {'筛除总成本': 0, '筛除成本占比': 0},
        "cost_efficiency_ranking": [],
        "detailed_data": []
    }

    try:
        if df.empty:
            return result

        logger.info(f"执行简化成本分析，数据行数: {len(df)}")

        # ========== 修复成本字段处理 ==========
        df_copy = df.copy()

        # 检查成本字段是否存在
        cost_field_name = None
        for field in ['成本', '下单价', '报价', 'cost_amount', 'order_amount', 'cooperation_quote']:
            if field in df_copy.columns:
                cost_field_name = field
                break

        if cost_field_name:
            logger.info(f"使用成本字段: {cost_field_name}")

            # 确保成本字段是数值类型
            try:
                # 转换为字符串，然后清理
                df_copy['成本_数值'] = df_copy[cost_field_name].astype(str)
                df_copy['成本_数值'] = df_copy['成本_数值'].str.replace(',', '').str.strip()
                # 转换为数值
                df_copy['成本_数值'] = pd.to_numeric(df_copy['成本_数值'], errors='coerce')
                # 填充NaN
                df_copy['成本_数值'] = df_copy['成本_数值'].fillna(0.0)

                # 统计有效成本数据
                valid_cost = df_copy['成本_数值'] > 0
                logger.info(f"有效成本数据: {valid_cost.sum()}/{len(df_copy)}")

            except Exception as e:
                logger.error(f"转换成本字段失败: {e}")
                df_copy['成本_数值'] = 0.0
        else:
            logger.warning("未找到成本字段，全部设为0")
            df_copy['成本_数值'] = 0.0

        # ========== 计算统计信息 ==========
        total_cost = df_copy['成本_数值'].sum()
        avg_cost = df_copy['成本_数值'].mean() if len(df_copy) > 0 else 0

        # 返点比例计算
        rebate_ratio = 0
        rebate_total = 0
        quote_total = 0

        # 检查返点字段
        rebate_field = None
        for field in ['返点', 'rebate_amount']:
            if field in df_copy.columns:
                rebate_field = field
                break

        if rebate_field:
            try:
                # 转换返点字段
                df_copy['返点_数值'] = df_copy[rebate_field].astype(str)
                df_copy['返点_数值'] = df_copy['返点_数值'].str.replace(',', '').str.strip()
                df_copy['返点_数值'] = pd.to_numeric(df_copy['返点_数值'], errors='coerce').fillna(0.0)
                rebate_total = df_copy['返点_数值'].sum()
            except:
                rebate_total = 0

        # 检查报价字段
        quote_field = None
        for field in ['报价', 'cooperation_quote']:
            if field in df_copy.columns:
                quote_field = field
                break

        if quote_field:
            try:
                # 转换报价字段
                df_copy['报价_数值'] = df_copy[quote_field].astype(str)
                df_copy['报价_数值'] = df_copy['报价_数值'].str.replace(',', '').str.strip()
                df_copy['报价_数值'] = pd.to_numeric(df_copy['报价_数值'], errors='coerce').fillna(0.0)
                quote_total = df_copy['报价_数值'].sum()
            except:
                quote_total = 0

        if quote_total > 0:
            rebate_ratio = (rebate_total / quote_total * 100)

        # ========== 按媒介统计 ==========
        media_detail = []
        if '定档媒介' in df_copy.columns and len(df_copy) > 0:
            # 确保有有效的定档媒介字段
            df_copy['定档媒介_清洗'] = df_copy['定档媒介'].fillna('未知媒介').astype(str)

            # 按媒介分组统计
            media_stats = df_copy.groupby('定档媒介_清洗').agg({
                '成本_数值': 'sum',
                '达人昵称': 'count'
            }).reset_index()

            media_stats.columns = ['定档媒介', '总成本', '处理达人数量']

            # 计算平均成本（避免除以0）
            media_stats['平均成本'] = media_stats.apply(
                lambda row: row['总成本'] / row['处理达人数量'] if row['处理达人数量'] > 0 else 0,
                axis=1
            )

            # 如果有返点，添加返点信息
            if '返点_数值' in df_copy.columns:
                media_rebate = df_copy.groupby('定档媒介_清洗')['返点_数值'].sum().reset_index()
                media_rebate.columns = ['定档媒介', '总返点']
                media_stats = pd.merge(media_stats, media_rebate, on='定档媒介', how='left')

            media_detail = media_stats.to_dict('records')
            logger.info(f"媒介统计完成，涉及媒介数: {len(media_detail)}")

        # ========== 构建结果 ==========
        summary = {
            '总数据条数': len(df_copy),
            '总成本': total_cost,
            '整体平均成本': avg_cost,
            '整体返点占报价比例(%)': f"{rebate_ratio:.2f}%" if rebate_ratio > 0 else "0%",
            '有效数据条数': len(df_copy),
            '无效数据条数': 0,
            '备注': '简化成本分析（修复版）'
        }

        invalid_stats = {
            '总数据条数': len(df_copy),
            '有效数据条数': len(df_copy),
            '无效数据条数': 0,
            '有效数据比例(%)': '100%',
            '无效数据比例(%)': '0%',
            '无效数据原因分布': {},
            '无效数据总成本(元)': 0
        }

        # 成本效率排名（按平均成本升序）
        cost_efficiency_ranking = []
        if media_detail:
            ranking_df = pd.DataFrame(media_detail)
            if '平均成本' in ranking_df.columns:
                ranking_df = ranking_df[ranking_df['处理达人数量'] > 0]  # 只取有数据的媒介
                ranking_df = ranking_df.sort_values('平均成本', ascending=True).head(10)
                cost_efficiency_ranking = ranking_df.to_dict('records')

        # 详细数据（取前100条）
        detailed_data = []
        if len(df_copy) > 0:
            # 选择关键字段
            key_columns = []
            for col in ['达人昵称', '项目名称', '定档媒介', '成本_数值', '报价_数值', '返点_数值']:
                if col in df_copy.columns:
                    key_columns.append(col)

            if key_columns:
                detailed_df = df_copy[key_columns].head(100)
                # 重命名列
                column_mapping = {
                    '成本_数值': '成本',
                    '报价_数值': '报价',
                    '返点_数值': '返点'
                }
                detailed_df = detailed_df.rename(columns=column_mapping)
                detailed_data = detailed_df.to_dict('records')

        result = {
            "result": media_detail,
            "summary": summary,
            "overall_summary": summary,
            "invalid_data_stats": invalid_stats,
            "media_detail": media_detail,
            "group_summary": [],
            "filtered_summary": {'筛除总成本': 0, '筛除成本占比': 0},
            "cost_efficiency_ranking": cost_efficiency_ranking,
            "detailed_data": detailed_data,
            "media_group_workload": [],
            "fixed_media_workload": media_detail,
            "fixed_media_cost": media_detail,
            "fixed_media_rebate": [],
            "fixed_media_performance": [],
            "fixed_media_level": [],
            "fixed_media_comprehensive": []
        }

        logger.info(f"简化成本分析完成，总成本: {total_cost}")

    except Exception as e:
        logger.error(f"简化成本分析失败: {e}", exc_info=True)
        # 返回最基本的结果
        result = {
            "result": [],
            "summary": {"总数据条数": len(df) if not df.empty else 0, "备注": f"分析失败: {str(e)[:100]}"},
            "overall_summary": {},
            "invalid_data_stats": {},
            "media_detail": [],
            "group_summary": [],
            "filtered_summary": {'筛除总成本': 0, '筛除成本占比': 0},
            "cost_efficiency_ranking": [],
            "detailed_data": []
        }

    return result


def convert_list_to_dataframe(data_list, default_columns=None):
    """将列表转换为DataFrame，确保ReportGenerator能正确处理"""
    if not data_list:
        return pd.DataFrame()

    try:
        if isinstance(data_list, list):
            if len(data_list) == 0:
                return pd.DataFrame()

            # 检查第一个元素是否为字典
            if isinstance(data_list[0], dict):
                return pd.DataFrame(data_list)
            else:
                # 如果不是字典，创建简单DataFrame
                if default_columns:
                    return pd.DataFrame(data_list, columns=default_columns)
                else:
                    return pd.DataFrame({'数据': data_list})
        else:
            # 如果不是列表，尝试直接转换
            return pd.DataFrame(data_list)
    except Exception as e:
        logger.warning(f"转换列表到DataFrame失败: {e}")
        return pd.DataFrame()

@app.context_processor
def inject_common_variables():
    """注入全局通用变量到所有模板，无需手动传参"""
    now = datetime.now()
    return {
        'current_year': now.year,
        'current_date': now.strftime('%Y-%m-%d'),
        'current_datetime': now.strftime('%Y-%m-%d %H:%M:%S'),
        'app': app,
        'has_endpoint': has_endpoint,
        'view_functions': app.view_functions
    }

# ========== 核心修复：优化 load_analysis_result 函数，确保数据正确加载 ==========
def load_analysis_result(analysis_id):
    """
    核心优化：从内存/本地文件加载分析结果，自动完成类型转换+数据兜底
    所有返回数据均为Python原生类型，模板渲染绝对无报错
    """

    # 优先从内存读取（速度快）
    if analysis_id in analysis_results:
        analysis_data = analysis_results[analysis_id].copy()

        # ✅ 核心修复：确保full_result存在且包含各模块数据
        if 'full_result' not in analysis_data:
            # 如果full_result不存在，从原始数据构建
            analysis_data['full_result'] = {
                'workload': analysis_data.get('workload', {}),
                'quality': analysis_data.get('quality', {}),
                'cost': analysis_data.get('cost', {})
            }

        full_result = analysis_data.get('full_result', {})

        # ✅ 核心修复：确保数据结构正确
        if not isinstance(full_result, dict):
            full_result = {}

        # 转换数据类型
        full_result = convert_pandas_types_to_python(full_result)

        # ========================== 核心修复：统一数据键名 ==========================
        # 将 workload 中的 'detail' 转换为 'result'
        if 'workload' in full_result:
            workload_data = full_result['workload']
            if isinstance(workload_data, dict):
                # 确保有result字段
                if 'detail' in workload_data and 'result' not in workload_data:
                    workload_data['result'] = workload_data.pop('detail', [])
                elif 'detail_df' in workload_data and 'result' not in workload_data:
                    workload_data['result'] = workload_data.pop('detail_df', [])
                elif 'result' not in workload_data:
                    workload_data['result'] = []

                # 确保其他字段存在
                workload_data['summary'] = workload_data.get('summary', {})
                workload_data['group_summary'] = workload_data.get('group_summary', [])
                workload_data['top_media_ranking'] = workload_data.get('top_media_ranking', [])

        # 将 quality 中的 'detail' 转换为 'result'
        if 'quality' in full_result:
            quality_data = full_result['quality']
            if isinstance(quality_data, dict):
                if 'detail' in quality_data and 'result' not in quality_data:
                    quality_data['result'] = quality_data.pop('detail', [])
                elif 'detail_df' in quality_data and 'result' not in quality_data:
                    quality_data['result'] = quality_data.pop('detail_df', [])
                elif 'result' not in quality_data:
                    quality_data['result'] = []

                # ✅ 关键修复：确保分类数据字段存在且是列表
                quality_data['premium_detail'] = quality_data.get('premium_detail', [])
                quality_data['high_read_detail'] = quality_data.get('high_read_detail', [])

                # 确保其他字段存在
                quality_data['summary'] = quality_data.get('summary', {})
                quality_data['group_summary'] = quality_data.get('group_summary', [])
                quality_data['quality_distribution'] = quality_data.get('quality_distribution', [])

                # ✅ 修复：确保group_summary是列表
                if not isinstance(quality_data['group_summary'], list):
                    quality_data['group_summary'] = []
                if not isinstance(quality_data['quality_distribution'], list):
                    quality_data['quality_distribution'] = []
                if not isinstance(quality_data['premium_detail'], list):
                    quality_data['premium_detail'] = []
                if not isinstance(quality_data['high_read_detail'], list):
                    quality_data['high_read_detail'] = []

        # ========================== ✅ 新增核心修复：确保 cost 数据包含无效数据统计 ==========================
        if 'cost' in full_result:
            cost_data = full_result['cost']
            if not isinstance(cost_data, dict):
                cost_data = {}
                full_result['cost'] = cost_data

            # ✅ 核心修复：确保 invalid_data_detail 和 invalid_data_stats 字段存在
            if 'invalid_data_detail' not in cost_data:
                cost_data['invalid_data_detail'] = []

            # ✅ 从 overall_summary 提取无效数据统计
            overall_summary = cost_data.get('overall_summary', {})
            if not isinstance(overall_summary, dict):
                overall_summary = {}
                cost_data['overall_summary'] = overall_summary

            # ✅ 确保 overall_summary 包含无效数据统计字段
            if '总数据条数' not in overall_summary:
                overall_summary['总数据条数'] = 0
            if '有效数据条数' not in overall_summary:
                overall_summary['有效数据条数'] = 0
            if '无效数据条数' not in overall_summary:
                overall_summary['无效数据条数'] = 0
            if '有效数据比例(%)' not in overall_summary:
                overall_summary['有效数据比例(%)'] = '0%'
            if '无效数据比例(%)' not in overall_summary:
                overall_summary['无效数据比例(%)'] = '0%'
            if '无效数据原因分布' not in overall_summary:
                overall_summary['无效数据原因分布'] = {}
            if '无效数据总成本(元)' not in overall_summary:
                overall_summary['无效数据总成本(元)'] = 0

            # ✅ 创建独立的 invalid_data_stats 字段
            invalid_data_stats = {
                '总数据条数': overall_summary.get('总数据条数', 0),
                '有效数据条数': overall_summary.get('有效数据条数', 0),
                '无效数据条数': overall_summary.get('无效数据条数', 0),
                '有效数据比例(%)': overall_summary.get('有效数据比例(%)', '0%'),
                '无效数据比例(%)': overall_summary.get('无效数据比例(%)', '0%'),
                '无效数据原因分布': overall_summary.get('无效数据原因分布', {}),
                '无效数据总成本(元)': overall_summary.get('无效数据总成本(元)', 0)
            }

            cost_data['invalid_data_stats'] = invalid_data_stats
            # ========================== ✅ 新增核心修复：确保异常数据相关字段存在 ==========================
            # 在 load_analysis_result 函数中查找这部分代码
            if 'cost' in full_result:
                cost_data = full_result['cost']

                # ✅ 确保异常数据详情字段存在
                if 'abnormal_data_detail' not in cost_data:
                    cost_data['abnormal_data_detail'] = []

                # ✅ 确保异常数据统计字段存在
                if 'overall_summary' in cost_data:
                    overall_summary = cost_data['overall_summary']

                    # 确保 overall_summary 包含异常数据统计字段
                    if '异常数据条数' not in overall_summary:
                        overall_summary['异常数据条数'] = 0
                    if '异常数据比例(%)' not in overall_summary:
                        overall_summary['异常数据比例(%)'] = '0%'
                    if '异常数据原因分布' not in overall_summary:
                        overall_summary['异常数据原因分布'] = {}
                    if '异常数据总成本(元)' not in overall_summary:
                        overall_summary['异常数据总成本(元)'] = 0
                    if '参与分析数据条数' not in overall_summary:
                        overall_summary['参与分析数据条数'] = overall_summary.get('总数据条数',
                                                                                  0) - overall_summary.get(
                            '无效数据条数', 0)
                    if '参与分析数据比例(%)' not in overall_summary:
                        overall_summary['参与分析数据比例(%)'] = '100%'

                    # ✅ 创建独立的 abnormal_data_stats 字段
                    abnormal_data_stats = {
                        '异常数据条数': overall_summary.get('异常数据条数', 0),
                        '异常数据比例(%)': overall_summary.get('异常数据比例(%)', '0%'),
                        '异常数据原因分布': overall_summary.get('异常数据原因分布', {}),
                        '异常数据总成本(元)': overall_summary.get('异常数据总成本(元)', 0),
                        '参与分析数据条数': overall_summary.get('参与分析数据条数', 0),
                        '参与分析数据比例(%)': overall_summary.get('参与分析数据比例(%)', '100%')
                    }

                    cost_data['abnormal_data_stats'] = abnormal_data_stats

            if 'cost' in full_result:
                cost_data = full_result['cost']

                # ✅ 确保异常数据详情字段存在
                if 'abnormal_data_detail' not in cost_data:
                    cost_data['abnormal_data_detail'] = []

                # ✅ 从 overall_summary 提取异常数据统计
                if 'overall_summary' in cost_data:
                    overall_summary = cost_data['overall_summary']

                    # 确保 overall_summary 包含异常数据统计字段
                    if '异常数据条数' not in overall_summary:
                        overall_summary['异常数据条数'] = 0
                    if '异常数据比例(%)' not in overall_summary:
                        overall_summary['异常数据比例(%)'] = '0%'
                    if '异常数据原因分布' not in overall_summary:
                        overall_summary['异常数据原因分布'] = {}
                    if '异常数据总成本(元)' not in overall_summary:
                        overall_summary['异常数据总成本(元)'] = 0
                    if '参与分析数据条数' not in overall_summary:
                        overall_summary['参与分析数据条数'] = overall_summary.get('总数据条数',
                                                                                  0) - overall_summary.get(
                            '无效数据条数', 0)
                    if '参与分析数据比例(%)' not in overall_summary:
                        overall_summary['参与分析数据比例(%)'] = '100%'

                    # ✅ 创建独立的 abnormal_data_stats 字段
                    abnormal_data_stats = {
                        '异常数据条数': overall_summary.get('异常数据条数', 0),
                        '异常数据比例(%)': overall_summary.get('异常数据比例(%)', '0%'),
                        '异常数据原因分布': overall_summary.get('异常数据原因分布', {}),
                        '异常数据总成本(元)': overall_summary.get('异常数据总成本(元)', 0),
                        '参与分析数据条数': overall_summary.get('参与分析数据条数', 0),
                        '参与分析数据比例(%)': overall_summary.get('参与分析数据比例(%)', '100%')
                    }

                    cost_data['abnormal_data_stats'] = abnormal_data_stats

            # ✅ 确保其他成本字段存在
            cost_data['result'] = cost_data.get('result', [])
            cost_data['cleaned_data'] = cost_data.get('cleaned_data', [])
            cost_data['filtered_data'] = cost_data.get('filtered_data', [])
            cost_data['summary'] = overall_summary
            cost_data['overall_summary'] = overall_summary
            cost_data['detail_data'] = cost_data.get('detail_data', [])
            cost_data['media_detail'] = cost_data.get('media_detail', [])
            cost_data['group_summary'] = cost_data.get('group_summary', [])
            cost_data['filtered_summary'] = cost_data.get('filtered_summary', {})
            cost_data['cost_efficiency_ranking'] = cost_data.get('cost_efficiency_ranking', [])

            # ✅ 确保成本发挥分析的所有工作表字段存在
            cost_data['media_group_workload'] = cost_data.get('media_group_workload', [])
            cost_data['fixed_media_workload'] = cost_data.get('fixed_media_workload', [])
            cost_data['fixed_media_cost'] = cost_data.get('fixed_media_cost', [])
            cost_data['fixed_media_rebate'] = cost_data.get('fixed_media_rebate', [])
            cost_data['fixed_media_performance'] = cost_data.get('fixed_media_performance', [])
            cost_data['fixed_media_level'] = cost_data.get('fixed_media_level', [])
            cost_data['fixed_media_comprehensive'] = cost_data.get('fixed_media_comprehensive', [])
            cost_data['detailed_data'] = cost_data.get('detailed_data', [])

        # ========================== 核心修复 2/5：补全小组数据+质量分布数据 并做空值兜底 ==========================
        # 工作量小组数据兜底 + ✅ 关键修复：补全总定档数/总提报数字段
        workload_group = full_result.get('workload', {}).get('group_summary', [])
        if isinstance(workload_group, list):
            workload_group = fill_group_data_fields(workload_group)
            full_result['workload']['group_summary'] = workload_group
        else:
            full_result['workload']['group_summary'] = []

        # 质量分布数据兜底
        quality_dist = full_result.get('quality', {}).get('quality_distribution', [])
        if not isinstance(quality_dist, list):
            quality_dist = []
        full_result['quality']['quality_distribution'] = quality_dist

        # 质量小组数据兜底
        quality_group = full_result.get('quality', {}).get('group_summary', [])
        if isinstance(quality_group, list):
            quality_group = fill_group_data_fields(quality_group)
            full_result['quality']['group_summary'] = quality_group
        else:
            full_result['quality']['group_summary'] = []

        # ✅ 关键修复：补全分类数据
        premium_detail = full_result.get('quality', {}).get('premium_detail', [])
        if not isinstance(premium_detail, list):
            premium_detail = []
        full_result['quality']['premium_detail'] = premium_detail

        high_read_detail = full_result.get('quality', {}).get('high_read_detail', [])
        if not isinstance(high_read_detail, list):
            high_read_detail = []
        full_result['quality']['high_read_detail'] = high_read_detail

        # 数据兜底：确保所有核心字段是列表/字典，避免模板遍历报错
        workload_top = full_result.get('workload', {}).get('result', [])
        if not isinstance(workload_top, list):
            workload_top = []
        full_result['workload']['result'] = workload_top

        quality_top = full_result.get('quality', {}).get('result', [])
        if not isinstance(quality_top, list):
            quality_top = []
        full_result['quality']['result'] = quality_top

        cost_efficiency_ranking = full_result.get('cost', {}).get('cost_efficiency_ranking', [])
        cost_result = full_result.get('cost', {}).get('result', [])
        if not isinstance(cost_result, list):
            cost_result = []
        full_result['cost']['result'] = cost_efficiency_ranking if cost_efficiency_ranking else cost_result

        # ✅ 核心修复：成本数据 双层兜底 - 解决overall_summary未定义报错
        cost_overall = full_result.get('cost', {}).get('overall_summary', {}) or {}
        cost_summary = full_result.get('cost', {}).get('summary', {}) or {}
        # 合并两个汇总，优先overall_summary，彻底解决模板{{ overall_summary.get('整体平均成本',0) }}报错
        full_cost_summary = {**cost_summary, **cost_overall}
        full_result['cost']['overall_summary'] = full_cost_summary
        full_result['cost']['summary'] = full_cost_summary

        # ========================== 核心修复 3/5：成本数据字段补全 ==========================
        # 补全媒介明细/过滤数据/排名数据的所有缺失字段，解决筛除总成本报错
        full_result['cost']['media_detail'] = fill_cost_data_fields(full_result.get('cost', {}).get('media_detail', []))
        full_result['cost']['filtered_data'] = fill_cost_data_fields(
            full_result.get('cost', {}).get('filtered_data', []))
        full_result['cost']['cost_efficiency_ranking'] = fill_cost_data_fields(
            full_result.get('cost', {}).get('cost_efficiency_ranking', []))
        full_result['cost']['result'] = fill_cost_data_fields(full_result.get('cost', {}).get('result', []))

        # ✅ 核心修复：确保detail_data字段存在
        full_result['cost']['detail_data'] = full_result['cost']['media_detail']

        # 过滤汇总字段兜底
        if not full_result['cost']['filtered_summary'] or not isinstance(full_result['cost']['filtered_summary'], dict):
            full_result['cost']['filtered_summary'] = {
                '筛除总成本': 0, '筛除成本占比': 0, '筛除达人数量': 0, '筛除发布数量': 0
            }
        else:
            # 为过滤汇总补全缺失字段
            fs = full_result['cost']['filtered_summary']
            fs['筛除总成本'] = fs.get('筛除总成本', 0) or 0
            fs['筛除成本占比'] = fs.get('筛除成本占比', 0) or 0
            fs['筛除达人数量'] = fs.get('筛除达人数量', 0) or 0
            fs['筛除发布数量'] = fs.get('筛除发布数量', 0) or 0

        # ========================== ✅ 致命修复 核心根因：新增 detail_data 兜底赋值 ==========================
        # 解决 cost_analysis.html 第768行 const detailData = {{ detail_data|tojson|safe }}; 报Undefined序列化错误
        full_result['cost']['detail_data'] = full_result['cost']['media_detail']

        # 预处理成本百分数字段
        if isinstance(full_cost_summary, dict):
            rebate_key = '整体返点占报价比例(%)'
            full_cost_summary[rebate_key + '_num'] = preprocess_percent_str_to_float(
                full_cost_summary.get(rebate_key, '0%'))
            cost_keys = list(full_cost_summary.keys())
            for key in cost_keys:
                if '%' in str(key) and f'{key}_num' not in full_cost_summary:
                    full_cost_summary[f'{key}_num'] = preprocess_percent_str_to_float(full_cost_summary[key])

        full_result['cost']['summary'] = full_cost_summary
        analysis_data['full_result'] = full_result

        # ✅ 确保analysis_data包含所有必要字段
        analysis_data['category'] = analysis_data.get('category', '未知类目')
        analysis_data['timestamp'] = analysis_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        analysis_data['selected_groups'] = analysis_data.get('selected_groups', [])
        analysis_data['reports'] = analysis_data.get('reports', {
            "workload": {"excel": ""},
            "quality": {"excel": ""},
            "cost": {"excel": ""},
            "full": {"full_excel": ""}
        })

        return analysis_data

    # 内存无数据，从本地JSON读取
    result_file = os.path.join(app.config['OUTPUT_DIR'], 'analysis_results', f'{analysis_id}.json')
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # ========================== 核心修复：转换本地存储的键名 ==========================
            # 转换 workload 数据
            workload_data = data.get('workload', {})
            if 'detail' in workload_data and 'result' not in workload_data:
                workload_data['result'] = workload_data.pop('detail', [])

            # 转换 quality 数据
            quality_data = data.get('quality', {})
            if 'detail' in quality_data and 'result' not in quality_data:
                quality_data['result'] = quality_data.pop('detail', [])

            # ✅ 关键修复：读取分类数据并确保是列表
            premium_detail = quality_data.get('premium_detail', [])
            if not isinstance(premium_detail, list):
                premium_detail = []

            high_read_detail = quality_data.get('high_read_detail', [])
            if not isinstance(high_read_detail, list):
                high_read_detail = []

            quality_distribution = quality_data.get('quality_distribution', [])
            if not isinstance(quality_distribution, list):
                quality_distribution = []

            group_summary = quality_data.get('group_summary', [])
            if not isinstance(group_summary, list):
                group_summary = []

            # ✅ 核心修复：读取 cost 数据，确保包含无效数据统计
            cost_data = data.get('cost', {})
            if not isinstance(cost_data, dict):
                cost_data = {}

            # ✅ 确保 overall_summary 存在并包含无效数据统计
            overall_summary = cost_data.get('overall_summary', {})
            if not isinstance(overall_summary, dict):
                overall_summary = {}

            # ✅ 确保 overall_summary 包含无效数据统计字段
            overall_summary['总数据条数'] = overall_summary.get('总数据条数', data.get('总记录数', 0))
            overall_summary['有效数据条数'] = overall_summary.get('有效数据条数',
                                                                  overall_summary.get('有效数据条数', 0))
            overall_summary['无效数据条数'] = overall_summary.get('无效数据条数',
                                                                  overall_summary.get('无效数据条数', 0))

            if overall_summary['总数据条数'] > 0:
                overall_summary['有效数据比例(%)'] = overall_summary.get('有效数据比例(%)',
                                                                         f"{(overall_summary['有效数据条数'] / overall_summary['总数据条数'] * 100):.2f}%")
                overall_summary['无效数据比例(%)'] = overall_summary.get('无效数据比例(%)',
                                                                         f"{(overall_summary['无效数据条数'] / overall_summary['总数据条数'] * 100):.2f}%")
            else:
                overall_summary['有效数据比例(%)'] = '0%'
                overall_summary['无效数据比例(%)'] = '0%'

            overall_summary['无效数据原因分布'] = overall_summary.get('无效数据原因分布', {})
            overall_summary['无效数据总成本(元)'] = overall_summary.get('无效数据总成本(元)', 0)

            # ✅ 创建 invalid_data_stats
            invalid_data_stats = {
                '总数据条数': overall_summary.get('总数据条数', 0),
                '有效数据条数': overall_summary.get('有效数据条数', 0),
                '无效数据条数': overall_summary.get('无效数据条数', 0),
                '有效数据比例(%)': overall_summary.get('有效数据比例(%)', '0%'),
                '无效数据比例(%)': overall_summary.get('无效数据比例(%)', '0%'),
                '无效数据原因分布': overall_summary.get('无效数据原因分布', {}),
                '无效数据总成本(元)': overall_summary.get('无效数据总成本(元)', 0)
            }

            cost_summary = data.get('cost_summary', {})
            full_cost_summary = {**cost_summary, **overall_summary}

            if isinstance(full_cost_summary, dict):
                rebate_key = '整体返点占报价比例(%)'
                full_cost_summary[rebate_key + '_num'] = preprocess_percent_str_to_float(
                    full_cost_summary.get(rebate_key, '0%'))
                cost_keys = list(full_cost_summary.keys())
                for key in cost_keys:
                    if '%' in str(key) and f'{key}_num' not in full_cost_summary:
                        full_cost_summary[f'{key}_num'] = preprocess_percent_str_to_float(full_cost_summary[key])

            # 读取时也补全小组字段+成本字段
            workload_group = fill_group_data_fields(workload_data.get('group_summary', []))
            cost_media_detail = fill_cost_data_fields(cost_data.get('media_detail', []))

            # ✅ 读取 invalid_data_detail
            invalid_data_detail = cost_data.get('invalid_data_detail', [])
            if not isinstance(invalid_data_detail, list):
                invalid_data_detail = []

            # ✅ 确保 cost_data 包含所有工作表
            media_group_workload = cost_data.get('media_group_workload', [])
            fixed_media_workload = cost_data.get('fixed_media_workload', [])
            fixed_media_cost = cost_data.get('fixed_media_cost', [])
            fixed_media_rebate = cost_data.get('fixed_media_rebate', [])
            fixed_media_performance = cost_data.get('fixed_media_performance', [])
            fixed_media_level = cost_data.get('fixed_media_level', [])
            fixed_media_comprehensive = cost_data.get('fixed_media_comprehensive', [])

            result = {
                'analysis_id': analysis_id,
                'timestamp': data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                'category': data.get('category', '未知类目'),
                'selected_groups': data.get('selected_groups', []),
                'full_result': {
                    'workload': {
                        'result': workload_data.get('result', []),
                        'summary': workload_data.get('summary', {}),
                        'group_summary': workload_group
                    },
                    'quality': {
                        'result': quality_data.get('result', []),
                        'summary': quality_data.get('summary', {}),
                        'group_summary': group_summary,
                        'quality_distribution': quality_distribution,
                        'premium_detail': premium_detail,  # ✅ 新增：优质达人数据
                        'high_read_detail': high_read_detail  # ✅ 新增：高阅读达人数据
                    },
                    'cost': {
                        'result': cost_media_detail,
                        'cleaned_data': cost_data.get('cleaned_data', []),
                        'filtered_data': cost_data.get('filtered_data', []),
                        'summary': full_cost_summary,
                        'overall_summary': full_cost_summary,
                        'detail_data': cost_media_detail,
                        'media_detail': cost_media_detail,
                        'group_summary': cost_data.get('group_summary', []),
                        'filtered_summary': cost_data.get('filtered_summary', {'筛除总成本': 0, '筛除成本占比': 0}),
                        'cost_efficiency_ranking': cost_data.get('cost_efficiency_ranking', []),
                        # ✅ 新增：无效数据相关字段
                        'invalid_data_detail': invalid_data_detail,
                        'invalid_data_stats': invalid_data_stats,
                        # ✅ 新增：成本发挥分析所有工作表
                        'media_group_workload': media_group_workload,
                        'fixed_media_workload': fixed_media_workload,
                        'fixed_media_cost': fixed_media_cost,
                        'fixed_media_rebate': fixed_media_rebate,
                        'fixed_media_performance': fixed_media_performance,
                        'fixed_media_level': fixed_media_level,
                        'fixed_media_comprehensive': fixed_media_comprehensive,
                        'detailed_data': cost_data.get('detailed_data', [])
                    }
                },
                'reports': {
                    'workload': {'excel': data.get('report_files', {}).get('workload', '')},
                    'quality': {'excel': data.get('report_files', {}).get('quality', '')},
                    'cost': {'excel': data.get('report_files', {}).get('cost', '')},
                    'full': {'full_excel': data.get('report_files', {}).get('full', '')}
                }
            }

            return result

        except Exception as e:
            logger.error(f"❌ 读取本地分析结果失败：{result_file}，错误：{e}")
            return None
    else:
        logger.warning(f"⚠️ 分析结果 {analysis_id} 不存在")
        return None

# ========== 核心修复：优化 dashboard 路由 ==========
@app.route('/dashboard/<analysis_id>')
@login_required  # 新增：登录验证
def dashboard(analysis_id=None):
    """仪表盘：展示三大分析结果的概览和详情，兼容无参数访问"""
    analysis_id = analysis_id or request.args.get('analysis_id', 'latest')
    upload_success = request.args.get('upload_success', '0')
    analysis_data = None

    if analysis_id == 'latest':
        if analysis_results:
            latest_id = sorted(analysis_results.keys())[-1]
            analysis_data = load_analysis_result(latest_id)
            analysis_id = latest_id
        else:
            flash('⚠️ 暂无分析结果，请先上传文件进行分析', 'info')
            return redirect(url_for('index'))
    else:
        analysis_data = load_analysis_result(analysis_id)
        if not analysis_data:
            flash(f"❌ 分析结果 {analysis_id} 不存在", 'error')
            return redirect(url_for('index'))

    all_groups = sorted(list(set([v for v in NAME_TO_GROUP_MAPPING.values() if v != 'other组' and v is not None])))
    return render_template('dashboard.html',
                          analysis_id=analysis_id,
                          analysis_data=analysis_data,
                          upload_success=upload_success,
                          all_groups=all_groups)

# ------------------------------ 核心路由 ------------------------------
@app.route('/file-upload', methods=['GET', 'POST'])
@login_required  # 新增：登录验证
def index():
    # 新增：登录验证
    if not session.get('user_id'):
        return redirect(url_for('auth.login', next=url_for('index')))

    """首页：文件上传 + 分析参数配置，核心逻辑无改动，仅优化异常处理"""
    if request.method == 'POST':
        try:
            g.uploaded_files = set()
            # 获取表单参数
            category = request.form.get('category', '默认类目').strip()
            selected_groups = request.form.getlist('selected_groups[]')
            use_original_state = request.form.get('use_original_state', 'false') == 'true'
            cpm_good = float(request.form.get('cpm_good', 50.0))
            cpm_medium = float(request.form.get('cpm_medium', 100.0))
            cpe_good = float(request.form.get('cpe_good', 5.0))
            cpe_medium = float(request.form.get('cpe_medium', 10.0))

            # 获取上传文件
            workload_files = request.files.getlist('workload_files[]')
            quality_files = request.files.getlist('quality_files[]')
            cost_files = request.files.getlist('cost_files[]')

            # 验证是否有有效文件
            has_valid_file = any([file and file.filename.strip() for file_list in [workload_files, quality_files, cost_files] for file in file_list])
            if not has_valid_file:
                flash('⚠️ 请至少上传一个非空的Excel/CSV文件', 'warning')
                return redirect(url_for('index'))

            # 保存文件
            workload_file_paths = [save_file_with_duplicate_check(f, True) for f in workload_files if f and f.filename.strip()]
            workload_file_paths = [p for p in workload_file_paths if p]

            quality_file_paths = [save_file_with_duplicate_check(f, True) for f in quality_files if f and f.filename.strip()]
            quality_file_paths = [p for p in quality_file_paths if p]

            cost_file_paths = [save_file_with_duplicate_check(f, True) for f in cost_files if f and f.filename.strip()]
            cost_file_paths = [p for p in cost_file_paths if p]

            # ========== 工作量分析 - 兼容真实DataProcessor的两种返回格式 ✅核心修复 ==========
            workload_df = pd.DataFrame()
            workload_result = {"result": pd.DataFrame(), "summary": {}, "group_summary": pd.DataFrame(), "top_media_ranking": pd.DataFrame()}
            if workload_file_paths:
                process_result = data_processor.process_for_media_analysis(workload_file_paths, category)
                # 兼容：真实模块返回字典/直接返回df 两种格式
                if isinstance(process_result, dict):
                    workload_df = process_result.get('processed_data', pd.DataFrame())
                elif isinstance(process_result, pd.DataFrame):
                    workload_df = process_result
                else:
                    workload_df = pd.DataFrame()

                if not workload_df.empty:
                    workload_analyzer = WorkloadAnalyzer(
                        df=workload_df,
                        known_id_name_mapping=ID_TO_NAME_MAPPING,
                        config={"FLOWER_TO_NAME_MAPPING": {}}
                    )
                    workload_analysis = workload_analyzer.analyze(top_n=10)

                    # ✅ 核心修复：统一键名，确保前端模板使用'result'而不是'detail'
                    workload_result = {
                        "result": workload_analysis.get('detail', pd.DataFrame()),
                        "summary": workload_analysis.get('summary', {}),
                        "group_summary": workload_analysis.get('group_summary', pd.DataFrame()),
                        "top_media_ranking": workload_analysis.get('top_media_ranking', pd.DataFrame())
                    }

                    logger.info(f"📊 工作量分析完成，明细数据行数: {len(workload_result['result'])}")

            # ========== 工作质量分析 ✅ 核心BUG修复+兼容真实模块返回格式 ==========
            # ========== 工作质量分析 ✅ 核心BUG修复+兼容真实模块返回格式 ==========
            quality_df = pd.DataFrame()
            quality_result = {"result": pd.DataFrame(), "summary": {}, "group_summary": pd.DataFrame(),
                              "quality_distribution": pd.DataFrame(),
                              "premium_detail": pd.DataFrame(),  # ✅ 新增：优质达人数据
                              "high_read_detail": pd.DataFrame()}  # ✅ 新增：高阅读达人数据
            if quality_file_paths:
                process_result = data_processor.process_for_media_analysis(quality_file_paths, category)
                # 兼容：真实模块返回字典/直接返回df 两种格式
                if isinstance(process_result, dict):
                    quality_df = process_result.get('processed_data', pd.DataFrame())
                elif isinstance(process_result, pd.DataFrame):
                    quality_df = process_result
                else:
                    quality_df = pd.DataFrame()

                if not quality_df.empty:
                    quality_analyzer = QualityAnalyzer(
                        df=quality_df,
                        known_id_name_mapping=ID_TO_NAME_MAPPING,
                        config={"FLOWER_TO_NAME_MAPPING": {}}
                    )
                    # ✅ 修改：移除sort_by_quality参数，始终按小组排序
                    quality_analysis = quality_analyzer.analyze(use_original_state=use_original_state)

                    # ✅ 核心修复：统一键名，确保前端模板使用'result'而不是'detail'，并包含分类数据
                    quality_result = {
                        "result": quality_analysis.get('detail', pd.DataFrame()),
                        "summary": quality_analysis.get('summary', {}),
                        "group_summary": quality_analysis.get('group_summary', pd.DataFrame()),
                        "quality_distribution": quality_analysis.get('quality_distribution', pd.DataFrame()),
                        "premium_detail": quality_analysis.get('premium_detail', pd.DataFrame()),  # ✅ 新增
                        "high_read_detail": quality_analysis.get('high_read_detail', pd.DataFrame())  # ✅ 新增
                    }

                    logger.info(f"📊 质量分析完成，明细数据行数: {len(quality_result['result'])}")

            # ========== 成本分析 ✅ 兼容真实模块返回格式+使用所有数据 ==========
            cost_result = {
                "result": pd.DataFrame(), "cleaned_data": pd.DataFrame(), "filtered_data": pd.DataFrame(),
                "summary": {},
                "overall_summary": {}, "detail_data": pd.DataFrame(),
                "media_detail": pd.DataFrame(), "group_summary": pd.DataFrame(),
                "filtered_summary": {'筛除总成本': 0, '筛除成本占比': 0},
                "cost_efficiency_ranking": pd.DataFrame(),
                # 新增：无效数据相关字段
                "invalid_data_detail": [],
                "invalid_data_stats": {}
            }
            if cost_file_paths:
                process_result = data_processor.process_for_cost_analysis(cost_file_paths, category)
                # 兼容：真实模块返回字典/直接返回df 两种格式
                if isinstance(process_result, dict):
                    cost_raw_df = process_result.get('processed_data', pd.DataFrame())
                    cost_filtered_df = process_result.get('filtered_data', pd.DataFrame())
                elif isinstance(process_result, pd.DataFrame):
                    cost_raw_df = process_result
                    cost_filtered_df = pd.DataFrame()  # 不返回被筛除数据
                else:
                    cost_raw_df = pd.DataFrame()
                    cost_filtered_df = pd.DataFrame()

                if not cost_raw_df.empty:
                    cost_analyzer = CostAnalyzer(cost_raw_df, cost_filtered_df)
                    try:
                        cost_analysis = cost_analyzer.analyze(top_n=10)
                    except AttributeError as e:
                        logger.warning(f"成本分析出现属性错误，进行修复: {e}")
                        # 创建一个基本的成本分析结果
                        cost_analysis = {
                            'overall_summary': {'总数据条数': len(cost_raw_df)},
                            'media_detail': cost_raw_df,
                            'group_summary': pd.DataFrame(),
                            'filtered_summary': {'筛除总成本': 0, '筛除成本占比': 0},
                            'cost_efficiency_ranking': pd.DataFrame(),
                            'invalid_data_detail': [],
                            'media_group_workload': pd.DataFrame(),
                            'fixed_media_workload': pd.DataFrame(),
                            'fixed_media_cost': pd.DataFrame(),
                            'fixed_media_rebate': pd.DataFrame(),
                            'fixed_media_performance': pd.DataFrame(),
                            'fixed_media_level': pd.DataFrame(),
                            'fixed_media_comprehensive': pd.DataFrame(),
                            'detailed_data': cost_raw_df
                        }

                    # ✅ 核心修复：使用所有数据，包括无效数据
                    cost_summary = cost_analysis.get('overall_summary', cost_analysis.get('summary', {}))
                    cost_media_detail = cost_analysis.get('media_detail',
                                                          cost_analysis.get('detail_df', pd.DataFrame()))

                    # 获取无效数据详情
                    invalid_data_detail = cost_analysis.get('invalid_data_detail', [])
                    invalid_data_stats = {
                        '总数据条数': len(cost_raw_df),
                        '有效数据条数': len(cost_raw_df) - (
                            cost_raw_df['成本无效'].sum() if '成本无效' in cost_raw_df.columns else 0),
                        '无效数据条数': cost_raw_df['成本无效'].sum() if '成本无效' in cost_raw_df.columns else 0,
                        '有效数据比例(%)': f"{(len(cost_raw_df) - (cost_raw_df['成本无效'].sum() if '成本无效' in cost_raw_df.columns else 0)) / len(cost_raw_df) * 100:.2f}%" if len(
                            cost_raw_df) > 0 else '0%',
                        '无效数据比例(%)': f"{(cost_raw_df['成本无效'].sum() if '成本无效' in cost_raw_df.columns else 0) / len(cost_raw_df) * 100:.2f}%" if len(
                            cost_raw_df) > 0 else '0%',
                        '无效数据原因分布': {},
                        '无效数据总成本(元)': cost_raw_df.loc[
                            cost_raw_df['成本无效'] == True, '成本'].sum() if '成本无效' in cost_raw_df.columns else 0
                    }

                    cost_result = {
                        "result": cost_media_detail,
                        "cleaned_data": cost_raw_df,
                        "filtered_data": cost_filtered_df,
                        "summary": cost_summary,
                        "overall_summary": cost_summary,
                        "detail_data": cost_media_detail,
                        "media_detail": cost_media_detail,
                        "group_summary": cost_analysis.get('group_summary', pd.DataFrame()),
                        "filtered_summary": cost_analysis.get('filtered_summary', {'筛除总成本': 0, '筛除成本占比': 0}),
                        "cost_efficiency_ranking": cost_analysis.get('cost_efficiency_ranking', pd.DataFrame()),
                        # ✅ 新增：无效数据相关
                        "invalid_data_detail": invalid_data_detail,
                        "invalid_data_stats": invalid_data_stats,
                        # ✅ 新增：成本发挥分析的所有工作表数据
                        "media_group_workload": cost_analysis.get('media_group_workload', pd.DataFrame()),
                        "fixed_media_workload": cost_analysis.get('fixed_media_workload', pd.DataFrame()),
                        "fixed_media_cost": cost_analysis.get('fixed_media_cost', pd.DataFrame()),
                        "fixed_media_rebate": cost_analysis.get('fixed_media_rebate', pd.DataFrame()),
                        "fixed_media_performance": cost_analysis.get('fixed_media_performance', pd.DataFrame()),
                        "fixed_media_level": cost_analysis.get('fixed_media_level', pd.DataFrame()),
                        "fixed_media_comprehensive": cost_analysis.get('fixed_media_comprehensive', pd.DataFrame()),
                        "detailed_data": cost_analysis.get('detailed_data', pd.DataFrame())
                    }

                    logger.info(
                        f"📊 成本分析完成，总数据: {len(cost_raw_df)} 条, 无效数据: {invalid_data_stats['无效数据条数']} 条")

            # ========== 在生成报告之前先创建analysis_id ==========
            analysis_id = datetime.now().strftime('%Y%m%d%H%M%S')
            # 创建报告生成器实例 - 传入真实分析结果
            report_generator = ReportGenerator(
                analysis_results={
                    'workload': workload_result,
                    'quality': quality_result,
                    'cost': cost_result
                },
                output_dir=app.config['OUTPUT_DIR']
            )

            # 生成报告
            reports = {
                "workload": {"excel": ""},
                "quality": {"excel": ""},
                "cost": {"excel": ""},
                "full": {"full_excel": ""}
            }

            # 生成Excel报告（根据分析模式）
            analysis_mode = 'full'
            # ✅ 修复：移除analysis_id参数
            excel_report_path = report_generator.generate_excel_report(analysis_mode)
            if excel_report_path:
                reports["full"]["full_excel"] = excel_report_path

            # ✅ 核心修复：转换数据格式并统一键名
            workload_for_storage = {
                "result": convert_pandas_types_to_python(workload_result.get("result", [])),
                "summary": convert_pandas_types_to_python(workload_result.get("summary", {})),
                "group_summary": convert_pandas_types_to_python(workload_result.get("group_summary", [])),
                "top_media_ranking": convert_pandas_types_to_python(workload_result.get("top_media_ranking", []))
            }

            quality_for_storage = {
                "result": convert_pandas_types_to_python(quality_result.get("result", [])),
                "summary": convert_pandas_types_to_python(quality_result.get("summary", {})),
                "group_summary": convert_pandas_types_to_python(quality_result.get("group_summary", [])),
                "quality_distribution": convert_pandas_types_to_python(quality_result.get("quality_distribution", [])),
                "premium_detail": convert_pandas_types_to_python(quality_result.get("premium_detail", [])),
                "high_read_detail": convert_pandas_types_to_python(quality_result.get("high_read_detail", []))
            }

            # 在 index 函数中，找到存储成本分析结果的部分，修改为：
            # 在存储成本分析结果的地方，确保包含 invalid_data_detail
            cost_for_storage = {
                "result": convert_pandas_types_to_python(cost_result.get("result", [])),
                "summary": convert_pandas_types_to_python(cost_result.get("summary", {})),
                "overall_summary": convert_pandas_types_to_python(cost_result.get("overall_summary", {})),
                "media_detail": convert_pandas_types_to_python(cost_result.get("media_detail", [])),
                "group_summary": convert_pandas_types_to_python(cost_result.get("group_summary", [])),
                "filtered_summary": convert_pandas_types_to_python(cost_result.get("filtered_summary", {})),
                "cost_efficiency_ranking": convert_pandas_types_to_python(
                    cost_result.get("cost_efficiency_ranking", [])),
                # ✅ 新增：无效数据详情
                "invalid_data_detail": convert_pandas_types_to_python(cost_result.get("invalid_data_detail", [])),
                "invalid_data_stats": convert_pandas_types_to_python(cost_result.get("invalid_data_stats", {})),
                # ✅ 新增：成本发挥分析所有工作表
                "media_group_workload": convert_pandas_types_to_python(cost_result.get("media_group_workload", [])),
                "fixed_media_workload": convert_pandas_types_to_python(cost_result.get("fixed_media_workload", [])),
                "fixed_media_cost": convert_pandas_types_to_python(cost_result.get("fixed_media_cost", [])),
                "fixed_media_rebate": convert_pandas_types_to_python(cost_result.get("fixed_media_rebate", [])),
                "fixed_media_performance": convert_pandas_types_to_python(
                    cost_result.get("fixed_media_performance", [])),
                "fixed_media_level": convert_pandas_types_to_python(cost_result.get("fixed_media_level", [])),
                "fixed_media_comprehensive": convert_pandas_types_to_python(
                    cost_result.get("fixed_media_comprehensive", [])),
                "detailed_data": convert_pandas_types_to_python(cost_result.get("detailed_data", []))
            }

            analysis_data_full = {
                "analysis_id": analysis_id,
                "full_result": {
                    "workload": workload_for_storage,
                    "quality": quality_for_storage,
                    "cost": cost_for_storage
                },
                "reports": reports,
                "category": category,
                "selected_groups": selected_groups,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # ✅ 核心修复：确保数据存储到内存
            analysis_results[analysis_id] = analysis_data_full

            # 持久化到JSON
            analysis_data_serializable = convert_pandas_types_to_python({
                "analysis_id": analysis_id,
                "category": category,
                "selected_groups": selected_groups,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "workload": workload_for_storage,
                "quality": quality_for_storage,
                "cost": {  # 关键修复：改为完整的 cost 结构
                    "result": cost_for_storage.get("result", []),
                    "summary": cost_for_storage.get("summary", {}),
                    "overall_summary": cost_for_storage.get("overall_summary", {}),
                    "media_detail": cost_for_storage.get("media_detail", []),
                    "group_summary": cost_for_storage.get("group_summary", []),
                    "filtered_summary": cost_for_storage.get("filtered_summary", {}),
                    "cost_efficiency_ranking": cost_for_storage.get("cost_efficiency_ranking", []),
                    # ✅ 新增：成本发挥分析的所有工作表数据
                    "media_group_workload": convert_pandas_types_to_python(
                        cost_result.get("media_group_workload", pd.DataFrame())),
                    "fixed_media_workload": convert_pandas_types_to_python(
                        cost_result.get("fixed_media_workload", pd.DataFrame())),
                    "fixed_media_cost": convert_pandas_types_to_python(
                        cost_result.get("fixed_media_cost", pd.DataFrame())),
                    "fixed_media_rebate": convert_pandas_types_to_python(
                        cost_result.get("fixed_media_rebate", pd.DataFrame())),
                    "fixed_media_performance": convert_pandas_types_to_python(
                        cost_result.get("fixed_media_performance", pd.DataFrame())),
                    "fixed_media_level": convert_pandas_types_to_python(
                        cost_result.get("fixed_media_level", pd.DataFrame())),
                    "fixed_media_comprehensive": convert_pandas_types_to_python(
                        cost_result.get("fixed_media_comprehensive", pd.DataFrame())),
                    "detailed_data": convert_pandas_types_to_python(cost_result.get("detailed_data", pd.DataFrame()))
                },
                "report_files": {
                    "workload": reports["workload"]["excel"],
                    "quality": reports["quality"]["excel"],
                    "cost": reports["cost"]["excel"],
                    "full": reports["full"]["full_excel"]
                }
            })

            result_file_path = os.path.join(app.config['OUTPUT_DIR'], 'analysis_results', f'{analysis_id}.json')
            os.makedirs(os.path.dirname(result_file_path), exist_ok=True)

            with open(result_file_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data_serializable, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ 分析完成，分析ID：{analysis_id}")
            flash('✅ 文件上传成功，分析已完成！', 'success')
            return redirect(url_for('dashboard', analysis_id=analysis_id, upload_success=1))

        except Exception as e:
            error_msg = f"❌ 分析失败：{str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            flash(error_msg, 'error')
            return redirect(url_for('index'))

    # GET请求渲染首页
    all_groups = sorted(list(set([v for v in NAME_TO_GROUP_MAPPING.values() if v != 'other组' and v is not None])))
    return render_template('index.html', all_groups=all_groups)

# ========== 新增测试路由 ==========
@app.route('/test_data/<analysis_id>')
@login_required  # 新增：登录验证
def test_data(analysis_id):
    """测试路由：查看内存中的数据"""
    if analysis_id in analysis_results:
        data = analysis_results[analysis_id]
        # 返回JSON格式的数据以便检查
        return jsonify({
            'success': True,
            'analysis_id': analysis_id,
            'keys': list(data.keys()),
            'full_result_keys': list(data.get('full_result', {}).keys()) if data.get('full_result') else [],
            'workload_summary': data.get('full_result', {}).get('workload', {}).get('summary', {}),
            'timestamp': data.get('timestamp', ''),
            'category': data.get('category', '')
        })
    else:
        # 检查本地文件
        result_file = os.path.join(app.config['OUTPUT_DIR'], 'analysis_results', f'{analysis_id}.json')
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify({
                'success': True,
                'source': 'file',
                'data_keys': list(data.keys())
            })
        return jsonify({'success': False, 'error': '数据不存在'})

# ------------------------------ 各分析报告详情页 ------------------------------
@app.route('/report/workload/<analysis_id>')
@login_required  # 新增：登录验证
def workload_report(analysis_id):
    """工作量分析报告详情页，修复变量名匹配问题"""
    if analysis_id == 'latest':
        results_dir = os.path.join(app.config['OUTPUT_DIR'], 'analysis_results')
        try:
            result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
            if not result_files:
                return render_template('workload_analysis.html',
                                       analysis_id='latest',
                                       analysis_data={"category": "暂无类目",
                                                      "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                                       detail_data=[],
                                       workload_summary={},
                                       group_summary=[],
                                       top_ranking=[],
                                       report={"excel": ""})
            result_files.sort(reverse=True)
            latest_file = result_files[0]
            analysis_id = latest_file.replace('.json', '')
        except Exception as e:
            logger.error(f"获取最新分析结果失败: {e}")
            return render_template('workload_analysis.html',
                                   analysis_id='latest',
                                   analysis_data={"category": "暂无类目",
                                                  "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                                   detail_data=[],
                                   workload_summary={},
                                   group_summary=[],
                                   top_ranking=[],
                                   report={"excel": ""})

    # 加载分析结果
    analysis_data = load_analysis_result(analysis_id)
    if not analysis_data:
        category = "暂无类目"
        detail_data = []
        workload_summary = {}
        group_summary = []
        top_ranking = []
        report = {"excel": ""}
        analysis_data_info = {"category": category,
                              "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    else:
        # 从full_result.workload中获取数据
        full_result = analysis_data.get("full_result", {})
        workload_data = full_result.get("workload", {})
        category = analysis_data.get("category", "暂无类目")

        # 获取数据，使用模板期望的变量名
        detail_data = workload_data.get("result", [])
        workload_summary = workload_data.get("summary", {})
        group_summary = workload_data.get("group_summary", [])
        top_ranking = workload_data.get("top_media_ranking", [])

        # 如果result为空但有detail，尝试从detail获取
        if not detail_data and "detail" in workload_data:
            detail_data = workload_data.get("detail", [])

        # 确保top_ranking有数据（如果没有单独的top_ranking，使用detail_data前10条）
        if not top_ranking and detail_data:
            # 按综合评估排序后取前10
            try:
                if detail_data and len(detail_data) > 0:
                    # 尝试按综合评估排序
                    sorted_data = sorted(detail_data, key=lambda x: x.get('综合评估', '') if isinstance(x, dict) else '')
                    top_ranking = sorted_data[:10]
                else:
                    top_ranking = detail_data[:10] if len(detail_data) > 10 else detail_data
            except:
                top_ranking = detail_data[:10] if len(detail_data) > 10 else detail_data

        report = analysis_data.get("reports", {}).get("workload", {"excel": ""})
        analysis_data_info = {
            "category": category,
            "timestamp": analysis_data.get("timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        }

        # 修复：确保workload_summary包含必要的字段
        if not workload_summary:
            workload_summary = {
                '总定档量': 0,
                '总CHAIN_RETURNED数': 0,
                '整体定档率': '0%',
                '总处理量': 0,
                '媒介总数': len(detail_data) if detail_data else 0
            }
        else:
            # 确保有必要的字段
            workload_summary['总定档量'] = workload_summary.get('总定档量', 0) or 0
            workload_summary['总CHAIN_RETURNED数'] = workload_summary.get('总CHAIN_RETURNED数', 0) or 0
            workload_summary['整体定档率'] = workload_summary.get('整体定档率', '0%') or '0%'
            workload_summary['总处理量'] = workload_summary.get('总处理量', 0) or 0
            workload_summary['媒介总数'] = len(detail_data) if detail_data else 0

    # 关键修复：传递正确的变量名给模板
    return render_template('workload_analysis.html',
                           analysis_id=analysis_id,
                           analysis_data=analysis_data_info,
                           detail_data=detail_data,
                           workload_summary=workload_summary,
                           group_summary=group_summary,
                           top_ranking=top_ranking,
                           report=report)


@app.route('/')
def root_redirect():
    """根路径重定向到数据来源选择页"""
    return redirect(url_for('data_source_selector'))

# app_auto.py 新增路由
@app.route('/data-source')
@login_required
def data_source_selector():
    """数据来源选择页（登录后默认首页）"""
    return render_template('data_source_selector.html')


@app.route('/db-analysis')
@login_required
def db_analysis_index():
    """数据库分析配置页"""
    return render_template('db_analysis_index.html')

@app.route('/db-analysis/submit', methods=['POST'])
@login_required
def db_analysis_submit():
    """数据库分析提交处理（完整修复版）"""
    try:
        # 获取表单参数
        start_date = request.form.get('start_date').strip()
        end_date = request.form.get('end_date').strip()
        analysis_modules = request.form.getlist('analysis_modules')
        use_original_state = request.form.get('use_original_state', 'false') == 'true'
        cpm_good = float(request.form.get('cpm_good', 50.0))
        cpm_medium = float(request.form.get('cpm_medium', 100.0))
        cpe_good = float(request.form.get('cpe_good', 5.0))
        cpe_medium = float(request.form.get('cpe_medium', 10.0))
        category = f"数据库分析_{start_date}_{end_date}"

        # 验证参数
        if not start_date or not end_date:
            flash('⚠️ 请选择完整的时间段', 'warning')
            return redirect(url_for('db_analysis_index'))

        # ========== 数据库读取数据 ==========
        workload_df = pd.DataFrame()
        quality_df = pd.DataFrame()
        cost_df = pd.DataFrame()

        if 'workload' in analysis_modules:
            workload_df = query_workload_data(start_date, end_date)
            if not workload_df.empty:
                logger.info(f"📊 查询到工作量数据 {len(workload_df)} 条")

        if 'quality' in analysis_modules:
            quality_df = query_quality_data(start_date, end_date)
            if not quality_df.empty:
                logger.info(f"📈 查询到工作质量数据 {len(quality_df)} 条")

        if 'cost' in analysis_modules:
            cost_df = query_cost_data(start_date, end_date)
            if not cost_df.empty:
                logger.info(f"💰 查询到成本数据 {len(cost_df)} 条")
                # ✅ 新增：检查成本数据字段
                logger.info(f"✅ 成本数据列名: {list(cost_df.columns)}")
                logger.info(f"✅ 成本数据样本（前3行）:")
                for i in range(min(3, len(cost_df))):
                    logger.info(f"行{i}: {cost_df.iloc[i].to_dict()}")

        # 验证是否有有效数据
        has_valid_data = not (workload_df.empty and quality_df.empty and cost_df.empty)
        if not has_valid_data:
            flash('⚠️ 所选时间段内无有效数据，请调整日期范围', 'warning')
            return redirect(url_for('db_analysis_index'))

        # ========== 数据清洗和预处理 ==========
        def clean_dataframe(df):
            """数据清洗函数"""
            if df.empty:
                return df

            df_copy = df.copy()

            # 1. 清理字符串字段空格
            str_cols = df_copy.select_dtypes(include=['object']).columns
            for col in str_cols:
                df_copy[col] = df_copy[col].astype(str).str.strip()

            # 2. 处理小组名称中的空格和特殊字符
            if '所属小组' in df_copy.columns:
                df_copy['所属小组'] = df_copy['所属小组'].str.replace(r'\s+', '', regex=True)
                # 统一小组名称
                group_mapping = {
                    '家居媒介组': '家居媒介组',
                    ' 家居媒介组': '家居媒介组',
                    '家居媒介组 ': '家居媒介组',
                    '数码媒介组': '数码媒介组',
                    ' 数码媒介组': '数码媒介组',
                    '数码媒介组 ': '数码媒介组',
                    '快消媒介组': '快消媒介组',
                    ' 快消媒介组': '快消媒介组',
                    '快消媒介组 ': '快消媒介组',
                    'other组': 'other组',
                    ' other组': 'other组',
                    'other组 ': 'other组',
                    '默认组': 'other组'
                }
                df_copy['所属小组'] = df_copy['所属小组'].replace(group_mapping)

            # 3. 清理媒介名称空格
            media_fields = ['定档媒介', '提交媒介', '媒介姓名', '对应真名', '媒介真名',
                            'schedule_user_name', 'submit_media_user_name']
            for field in media_fields:
                if field in df_copy.columns:
                    df_copy[field] = df_copy[field].astype(str).str.strip()

            # 4. 修复None值
            df_copy = df_copy.replace({'None': '未知', 'none': '未知', 'nan': '未知', 'NaN': '未知'})
            df_copy = df_copy.fillna({
                '定档媒介': '未知媒介',
                '所属小组': 'other组',
                '媒介真名': '未知',
                'schedule_user_name': '未知'
            })

            return df_copy

        # 应用数据清洗
        if not workload_df.empty:
            workload_df = clean_dataframe(workload_df)
            logger.info(f"工作量数据清洗完成，行数: {len(workload_df)}")

        if not quality_df.empty:
            quality_df = clean_dataframe(quality_df)
            logger.info(f"工作质量数据清洗完成，行数: {len(quality_df)}")

        if not cost_df.empty:
            cost_df = clean_dataframe(cost_df)
            logger.info(f"成本数据清洗完成，行数: {len(cost_df)}")

        # ========== 应用字段映射 ==========
        logger.info("开始应用数据库字段映射...")
        if not workload_df.empty:
            # 工作量数据 - 确保是定档数据
            workload_df['数据类型'] = '定档'
            logger.info(f"工作量数据准备完成，行数: {len(workload_df)}")
            logger.info(f"字段示例: {list(workload_df.columns[:10])}")

        if not quality_df.empty:
            # 工作质量数据 - 确保是提报数据
            quality_df['数据类型'] = '提报'
            logger.info(f"工作质量数据准备完成，行数: {len(quality_df)}")

        if not cost_df.empty:
            # 成本数据 - 确保是定档数据
            cost_df['数据类型'] = '定档'
            logger.info(f"成本数据准备完成，行数: {len(cost_df)}")

            # ✅ 关键修复：确保成本数据有所有必要字段
            required_cost_fields = ['定档媒介', '所属小组', '定档媒介小组', '成本', '报价', '下单价', '返点']
            for field in required_cost_fields:
                if field not in cost_df.columns:
                    logger.warning(f"成本数据缺少字段: {field}，正在创建默认值")
                    if field == '定档媒介小组' and '所属小组' in cost_df.columns:
                        cost_df['定档媒介小组'] = cost_df['所属小组']
                    elif field == '定档媒介':
                        cost_df['定档媒介'] = cost_df.get('schedule_user_name', '未知媒介')
                    elif field in ['成本', '报价', '下单价', '返点']:
                        cost_df[field] = 0.0
                    else:
                        cost_df[field] = '未知'

            # ✅ 新增：检查异常数据字段
            if '数据异常' not in cost_df.columns:
                logger.warning("成本数据缺少'数据异常'字段，正在创建")
                cost_df['数据异常'] = False

            if '数据异常原因' not in cost_df.columns:
                cost_df['数据异常原因'] = ''

            if '筛除原因' not in cost_df.columns:
                cost_df['筛除原因'] = ''

            # ✅ 新增：识别异常数据
            logger.info("开始识别成本异常数据...")

            # 1. 报价异常（报价 < 下单价）
            if '报价' in cost_df.columns and '下单价' in cost_df.columns:
                mask_price_abnormal = (cost_df['报价'].notna()) & (cost_df['下单价'].notna()) & (
                            cost_df['报价'] < cost_df['下单价'])
                if mask_price_abnormal.sum() > 0:
                    cost_df.loc[mask_price_abnormal, '数据异常'] = True
                    cost_df.loc[mask_price_abnormal, '数据异常原因'] = '报价或下单价异常'
                    cost_df.loc[mask_price_abnormal, '筛除原因'] = '报价或下单价异常'
                    logger.info(f"✅ 识别到报价异常数据: {mask_price_abnormal.sum()} 条")

            # 2. 返点比例异常
            if '返点比例' in cost_df.columns:
                mask_rebate_abnormal = (cost_df['返点比例'].notna()) & (
                            (cost_df['返点比例'] > 1.0) | (cost_df['返点比例'] < -0.5))
                if mask_rebate_abnormal.sum() > 0:
                    cost_df.loc[mask_rebate_abnormal, '数据异常'] = True

                    def format_rebate_reason(ratio):
                        return f"返点比例异常({ratio * 100:.1f}%)"

                    cost_df.loc[mask_rebate_abnormal, '数据异常原因'] = cost_df.loc[
                        mask_rebate_abnormal, '返点比例'].apply(format_rebate_reason)
                    logger.info(f"✅ 识别到返点比例异常数据: {mask_rebate_abnormal.sum()} 条")

            # 3. 成本为0或缺失
            if '成本' in cost_df.columns:
                mask_zero_cost = (cost_df['成本'] == 0) | (cost_df['成本'].isna())
                if mask_zero_cost.sum() > 0:
                    cost_df.loc[mask_zero_cost, '数据异常'] = True
                    cost_df.loc[mask_zero_cost, '数据异常原因'] = '成本为0或缺失'
                    logger.info(f"✅ 识别到成本为0或缺失数据: {mask_zero_cost.sum()} 条")

            # 4. 数据异常标记
            if '数据异常' in cost_df.columns:
                abnormal_count = cost_df['数据异常'].sum()
                logger.info(f"✅ 总异常数据统计: {abnormal_count} 条")
                if abnormal_count > 0:
                    logger.info(
                        f"✅ 异常原因分布: {cost_df[cost_df['数据异常']]['数据异常原因'].value_counts().to_dict()}")

        # ========== 工作量分析 - 简化版 ==========
        workload_result = {"result": [], "summary": {}, "group_summary": [], "top_media_ranking": []}
        if not workload_df.empty:
            try:
                logger.info(f"📊 开始简化工作量分析，共 {len(workload_df)} 条数据")

                # 尝试使用原有的分析器
                try:
                    workload_analyzer = WorkloadAnalyzer(
                        df=workload_df,
                        known_id_name_mapping=ID_TO_NAME_MAPPING,
                        config={"FLOWER_TO_NAME_MAPPING": {}}
                    )
                    workload_analysis = workload_analyzer.analyze(top_n=10)

                    workload_result = {
                        "result": convert_pandas_types_to_python(workload_analysis.get('detail', pd.DataFrame())),
                        "summary": convert_pandas_types_to_python(workload_analysis.get('summary', {})),
                        "group_summary": convert_pandas_types_to_python(
                            workload_analysis.get('group_summary', pd.DataFrame())),
                        "top_media_ranking": convert_pandas_types_to_python(
                            workload_analysis.get('top_media_ranking', pd.DataFrame()))
                    }

                    logger.info(f"✅ 工作量分析成功，明细数据行数: {len(workload_result['result'])}")

                except Exception as e:
                    logger.warning(f"工作量标准分析失败，使用简化分析: {e}")
                    # 使用简化分析
                    workload_result = create_simple_workload_analysis(workload_df)

            except Exception as e:
                logger.error(f"工作量分析异常: {e}")
                # 最终兜底：创建最基本的结果
                workload_result = {
                    "result": [],
                    "summary": {"总数据条数": len(workload_df), "备注": "分析过程出现异常"},
                    "group_summary": [],
                    "top_media_ranking": []
                }

        # ========== 工作质量分析 - 简化版 ==========
        quality_result = {"result": [], "summary": {}, "group_summary": [],
                          "quality_distribution": [], "premium_detail": [], "high_read_detail": []}
        if not quality_df.empty:
            try:
                logger.info(f"📈 开始简化工作质量分析，共 {len(quality_df)} 条数据")

                # 尝试使用原有的分析器
                try:
                    quality_analyzer = QualityAnalyzer(
                        df=quality_df,
                        known_id_name_mapping=ID_TO_NAME_MAPPING,
                        config={"FLOWER_TO_NAME_MAPPING": {}}
                    )
                    quality_analysis = quality_analyzer.analyze(use_original_state=use_original_state)

                    quality_result = {
                        "result": convert_pandas_types_to_python(quality_analysis.get('detail', pd.DataFrame())),
                        "summary": convert_pandas_types_to_python(quality_analysis.get('summary', {})),
                        "group_summary": convert_pandas_types_to_python(
                            quality_analysis.get('group_summary', pd.DataFrame())),
                        "quality_distribution": convert_pandas_types_to_python(
                            quality_analysis.get('quality_distribution', pd.DataFrame())),
                        "premium_detail": convert_pandas_types_to_python(
                            quality_analysis.get('premium_detail', pd.DataFrame())),
                        "high_read_detail": convert_pandas_types_to_python(
                            quality_analysis.get('high_read_detail', pd.DataFrame()))
                    }

                    logger.info(f"✅ 工作质量分析成功，明细数据行数: {len(quality_result['result'])}")

                except Exception as e:
                    logger.warning(f"工作质量标准分析失败，使用简化分析: {e}")
                    # 使用简化分析
                    quality_result = create_simple_quality_analysis(quality_df)

            except Exception as e:
                logger.error(f"工作质量分析异常: {e}")
                # 最终兜底
                quality_result = {
                    "result": [],
                    "summary": {"总数据条数": len(quality_df), "备注": "分析过程出现异常"},
                    "group_summary": [],
                    "quality_distribution": [],
                    "premium_detail": [],
                    "high_read_detail": []
                }

        # ========== 成本分析 - 完整修复版 ==========
        cost_result = {
            "result": [], "cleaned_data": [], "filtered_data": [],
            "summary": {},
            "overall_summary": {}, "detail_data": [],
            "media_detail": [], "group_summary": [],
            "filtered_summary": {'筛除总成本': 0, '筛除成本占比': 0},
            "cost_efficiency_ranking": [],
            "invalid_data_detail": [],
            "invalid_data_stats": {},
            "abnormal_data_detail": [],
            "abnormal_data_stats": {},
            "media_group_workload": [],
            "fixed_media_workload": [],
            "fixed_media_cost": [],
            "fixed_media_rebate": [],
            "fixed_media_performance": [],
            "fixed_media_level": [],
            "fixed_media_comprehensive": [],
            "detailed_data": []
        }

        if not cost_df.empty:
            try:
                logger.info(f"💰 开始完整成本分析，共 {len(cost_df)} 条数据")

                # ✅ 关键修复：确保数据有所有必要字段
                cost_df_copy = cost_df.copy()

                # 检查成本字段是否存在
                cost_field_name = None
                for field in ['成本', 'cost_amount']:
                    if field in cost_df_copy.columns:
                        cost_field_name = field
                        break

                if cost_field_name:
                    # 确保成本字段是数值类型
                    cost_df_copy['成本'] = pd.to_numeric(cost_df_copy[cost_field_name], errors='coerce').fillna(0.0)
                    logger.info(
                        f"✅ 成本字段处理完成，有效成本数据: {(cost_df_copy['成本'] > 0).sum()}/{len(cost_df_copy)}")
                else:
                    logger.warning("未找到成本字段，创建默认成本字段")
                    cost_df_copy['成本'] = 0.0

                # 检查报价字段
                quote_field = None
                for field in ['报价', 'cooperation_quote']:
                    if field in cost_df_copy.columns:
                        quote_field = field
                        break

                if quote_field:
                    cost_df_copy['报价'] = pd.to_numeric(cost_df_copy[quote_field], errors='coerce').fillna(0.0)
                else:
                    cost_df_copy['报价'] = 0.0

                # 检查返点字段
                rebate_field = None
                for field in ['返点', 'rebate_amount']:
                    if field in cost_df_copy.columns:
                        rebate_field = field
                        break

                if rebate_field:
                    cost_df_copy['返点'] = pd.to_numeric(cost_df_copy[rebate_field], errors='coerce').fillna(0.0)
                else:
                    cost_df_copy['返点'] = 0.0

                # 检查下单价字段
                order_field = None
                for field in ['下单价', 'order_amount']:
                    if field in cost_df_copy.columns:
                        order_field = field
                        break

                if order_field:
                    cost_df_copy['下单价'] = pd.to_numeric(cost_df_copy[order_field], errors='coerce').fillna(0.0)
                else:
                    cost_df_copy['下单价'] = 0.0

                # ✅ 新增：计算返点比例
                if '返点' in cost_df_copy.columns and '报价' in cost_df_copy.columns:
                    cost_df_copy['返点比例'] = cost_df_copy.apply(
                        lambda row: row['返点'] / row['报价'] if row['报价'] > 0 else 0.0,
                        axis=1
                    )
                else:
                    cost_df_copy['返点比例'] = 0.0

                # ✅ 关键修复：添加成本无效标记
                cost_df_copy['成本无效'] = (cost_df_copy['成本'] == 0) | (cost_df_copy['成本'].isna())
                invalid_count = cost_df_copy['成本无效'].sum()
                logger.info(f"✅ 成本无效数据: {invalid_count} 条")

                # ✅ 关键修复：添加数据异常标记（如果尚未标记）
                if '数据异常' not in cost_df_copy.columns:
                    cost_df_copy['数据异常'] = False

                if '数据异常原因' not in cost_df_copy.columns:
                    cost_df_copy['数据异常原因'] = ''

                # 尝试使用原有的成本分析器
                try:
                    cost_filtered_df = pd.DataFrame()
                    cost_analyzer = CostAnalyzer(cost_df_copy, cost_filtered_df)
                    cost_analysis = cost_analyzer.analyze(top_n=10)

                    # ✅ 核心修复：从成本分析结果中提取所有必要数据
                    cost_summary = cost_analysis.get('overall_summary', cost_analysis.get('summary', {}))
                    cost_media_detail = cost_analysis.get('media_detail', pd.DataFrame())

                    # 提取无效数据详情
                    invalid_data_detail = cost_analysis.get('invalid_data_detail', [])

                    # 提取异常数据详情
                    abnormal_data_detail = cost_analysis.get('abnormal_data_detail', [])

                    # 提取无效数据统计
                    invalid_data_stats = cost_analysis.get('invalid_data_stats', {})

                    # 提取异常数据统计
                    abnormal_data_stats = cost_analysis.get('abnormal_data_stats', {})

                    # 构建完整的成本结果
                    cost_result = {
                        "result": convert_pandas_types_to_python(cost_media_detail),
                        "cleaned_data": convert_pandas_types_to_python(cost_df_copy),
                        "filtered_data": convert_pandas_types_to_python(cost_filtered_df),
                        "summary": convert_pandas_types_to_python(cost_summary),
                        "overall_summary": convert_pandas_types_to_python(cost_summary),
                        "detail_data": convert_pandas_types_to_python(cost_media_detail),
                        "media_detail": convert_pandas_types_to_python(cost_media_detail),
                        "group_summary": convert_pandas_types_to_python(
                            cost_analysis.get('group_summary', pd.DataFrame())),
                        "filtered_summary": convert_pandas_types_to_python(
                            cost_analysis.get('filtered_summary', {'筛除总成本': 0, '筛除成本占比': 0})),
                        "cost_efficiency_ranking": convert_pandas_types_to_python(
                            cost_analysis.get('cost_efficiency_ranking', pd.DataFrame())),
                        # ✅ 核心修复：无效数据相关
                        "invalid_data_detail": convert_pandas_types_to_python(invalid_data_detail),
                        "invalid_data_stats": convert_pandas_types_to_python(invalid_data_stats),
                        # ✅ 核心修复：异常数据相关
                        "abnormal_data_detail": convert_pandas_types_to_python(abnormal_data_detail),
                        "abnormal_data_stats": convert_pandas_types_to_python(abnormal_data_stats),
                        # ✅ 核心修复：成本发挥分析所有工作表
                        "media_group_workload": convert_pandas_types_to_python(
                            cost_analysis.get('media_group_workload', pd.DataFrame())),
                        "fixed_media_workload": convert_pandas_types_to_python(
                            cost_analysis.get('fixed_media_workload', pd.DataFrame())),
                        "fixed_media_cost": convert_pandas_types_to_python(
                            cost_analysis.get('fixed_media_cost', pd.DataFrame())),
                        "fixed_media_rebate": convert_pandas_types_to_python(
                            cost_analysis.get('fixed_media_rebate', pd.DataFrame())),
                        "fixed_media_performance": convert_pandas_types_to_python(
                            cost_analysis.get('fixed_media_performance', pd.DataFrame())),
                        "fixed_media_level": convert_pandas_types_to_python(
                            cost_analysis.get('fixed_media_level', pd.DataFrame())),
                        "fixed_media_comprehensive": convert_pandas_types_to_python(
                            cost_analysis.get('fixed_media_comprehensive', pd.DataFrame())),
                        "detailed_data": convert_pandas_types_to_python(
                            cost_analysis.get('detailed_data', pd.DataFrame()))
                    }

                    logger.info(f"✅ 完整成本分析成功，明细数据行数: {len(cost_result['result'])}")
                    logger.info(f"✅ 无效数据详情: {len(invalid_data_detail)} 条")
                    logger.info(f"✅ 异常数据详情: {len(abnormal_data_detail)} 条")

                except Exception as e:
                    logger.warning(f"成本标准分析失败，使用简化分析: {e}")
                    # 使用简化分析
                    cost_result = create_simple_cost_analysis(cost_df_copy)
                    logger.info("✅ 已使用简化成本分析")

            except Exception as e:
                logger.error(f"成本分析异常: {e}", exc_info=True)
                # 最终兜底
                cost_result = create_simple_cost_analysis(cost_df_copy if 'cost_df_copy' in locals() else cost_df)

        # ========== 生成报告 ==========
        analysis_id = datetime.now().strftime('%Y%m%d%H%M%S') + '_DB'
        report_generator = ReportGenerator(
            analysis_results={
                'workload': workload_result,
                'quality': quality_result,
                'cost': cost_result
            },
            output_dir=app.config['OUTPUT_DIR']
        )

        reports = {
            "workload": {"excel": ""},
            "quality": {"excel": ""},
            "cost": {"excel": ""},
            "full": {"full_excel": ""}
        }

        # ✅ 修复：移除analysis_id参数
        excel_report_path = report_generator.generate_excel_report('full')
        if excel_report_path:
            reports["full"]["full_excel"] = excel_report_path

        # ========== 数据格式转换和存储 ==========
        workload_for_storage = {
            "result": workload_result.get("result", []),
            "summary": workload_result.get("summary", {}),
            "group_summary": workload_result.get("group_summary", []),
            "top_media_ranking": workload_result.get("top_media_ranking", [])
        }

        quality_for_storage = {
            "result": quality_result.get("result", []),
            "summary": quality_result.get("summary", {}),
            "group_summary": quality_result.get("group_summary", []),
            "quality_distribution": quality_result.get("quality_distribution", []),
            "premium_detail": quality_result.get("premium_detail", []),
            "high_read_detail": quality_result.get("high_read_detail", [])
        }

        cost_for_storage = {
            "result": cost_result.get("result", []),
            "summary": cost_result.get("summary", {}),
            "overall_summary": cost_result.get("overall_summary", {}),
            "media_detail": cost_result.get("media_detail", []),
            "group_summary": cost_result.get("group_summary", []),
            "filtered_summary": cost_result.get("filtered_summary", {}),
            "cost_efficiency_ranking": cost_result.get("cost_efficiency_ranking", []),
            "invalid_data_detail": cost_result.get("invalid_data_detail", []),
            "invalid_data_stats": cost_result.get("invalid_data_stats", {}),
            "abnormal_data_detail": cost_result.get("abnormal_data_detail", []),
            "abnormal_data_stats": cost_result.get("abnormal_data_stats", {}),
            "media_group_workload": cost_result.get("media_group_workload", []),
            "fixed_media_workload": cost_result.get("fixed_media_workload", []),
            "fixed_media_cost": cost_result.get("fixed_media_cost", []),
            "fixed_media_rebate": cost_result.get("fixed_media_rebate", []),
            "fixed_media_performance": cost_result.get("fixed_media_performance", []),
            "fixed_media_level": cost_result.get("fixed_media_level", []),
            "fixed_media_comprehensive": cost_result.get("fixed_media_comprehensive", []),
            "detailed_data": cost_result.get("detailed_data", [])
        }

        # 存储分析结果
        analysis_data_full = {
            "analysis_id": analysis_id,
            "full_result": {
                "workload": workload_for_storage,
                "quality": quality_for_storage,
                "cost": cost_for_storage
            },
            "reports": reports,
            "category": category,
            "selected_groups": [],
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "data_source": "database"
        }

        analysis_results[analysis_id] = analysis_data_full

        # 持久化到JSON
        analysis_data_serializable = {
            "analysis_id": analysis_id,
            "category": category,
            "selected_groups": [],
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "data_source": "database",
            "workload": workload_for_storage,
            "quality": quality_for_storage,
            "cost": cost_for_storage,
            "report_files": {
                "workload": reports["workload"]["excel"],
                "quality": reports["quality"]["excel"],
                "cost": reports["cost"]["excel"],
                "full": reports["full"]["full_excel"]
            }
        }

        result_file_path = os.path.join(app.config['OUTPUT_DIR'], 'analysis_results', f'{analysis_id}.json')
        os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data_serializable, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 数据库分析完成，分析ID：{analysis_id}")
        logger.info(f"✅ 工作量数据：{len(workload_result.get('result', []))} 条")
        logger.info(f"✅ 工作质量数据：{len(quality_result.get('result', []))} 条")
        logger.info(f"✅ 成本数据：{len(cost_result.get('result', []))} 条")
        logger.info(f"✅ 无效数据详情：{len(cost_result.get('invalid_data_detail', []))} 条")
        logger.info(f"✅ 异常数据详情：{len(cost_result.get('abnormal_data_detail', []))} 条")

        flash('✅ 数据库数据读取成功，分析已完成！', 'success')
        return redirect(url_for('dashboard', analysis_id=analysis_id, upload_success=1))

    except Exception as e:
        error_msg = f"❌ 数据库分析失败：{str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        flash(error_msg, 'error')
        return redirect(url_for('db_analysis_index'))

@app.route('/report/quality/<analysis_id>')
@login_required  # 新增：登录验证
def quality_report(analysis_id):
    """工作质量分析报告详情页，修复变量名匹配问题"""
    if analysis_id == 'latest':
        results_dir = os.path.join(app.config['OUTPUT_DIR'], 'analysis_results')
        try:
            result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
            if not result_files:
                # 返回空数据
                return render_template('quality_analysis.html',
                                       analysis_id='latest',
                                       analysis_data={"category": "暂无类目",
                                                      "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                                       detail_data=[],
                                       summary={},
                                       group_summary=[],
                                       quality_distribution=[],
                                       premium_detail=[],
                                       high_read_detail=[],
                                       report={"excel": ""})
            result_files.sort(reverse=True)
            latest_file = result_files[0]
            analysis_id = latest_file.replace('.json', '')
        except Exception as e:
            logger.error(f"获取最新分析结果失败: {e}")
            return render_template('quality_analysis.html',
                                   analysis_id='latest',
                                   analysis_data={"category": "暂无类目",
                                                  "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                                   detail_data=[],
                                   summary={},
                                   group_summary=[],
                                   quality_distribution=[],
                                   premium_detail=[],
                                   high_read_detail=[],
                                   report={"excel": ""})

    # 加载分析结果
    analysis_data = load_analysis_result(analysis_id)
    if not analysis_data:
        # 返回空数据
        return render_template('quality_analysis.html',
                               analysis_id=analysis_id,
                               analysis_data={"category": "暂无类目",
                                              "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                               detail_data=[],
                               summary={},
                               group_summary=[],
                               quality_distribution=[],
                               premium_detail=[],
                               high_read_detail=[],
                               report={"excel": ""})

    # 从full_result.quality中获取数据
    full_result = analysis_data.get("full_result", {})
    quality_data = full_result.get("quality", {})

    # ✅ 关键修复：获取所有必需的数据字段
    detail_data = quality_data.get("result", [])
    summary = quality_data.get("summary", {})
    group_summary = quality_data.get("group_summary", [])
    quality_distribution = quality_data.get("quality_distribution", [])
    premium_detail = quality_data.get("premium_detail", [])
    high_read_detail = quality_data.get("high_read_detail", [])

    # 如果result为空但有detail，尝试从detail获取
    if not detail_data and "detail" in quality_data:
        detail_data = quality_data.get("detail", [])

    # ✅ 关键修复：确保所有数据都是正确的类型
    if not isinstance(detail_data, list):
        detail_data = []
    if not isinstance(group_summary, list):
        group_summary = []
    if not isinstance(quality_distribution, list):
        quality_distribution = []
    if not isinstance(premium_detail, list):
        premium_detail = []
    if not isinstance(high_read_detail, list):
        high_read_detail = []

    report = analysis_data.get("reports", {}).get("quality", {"excel": ""})
    analysis_data_info = {
        "category": analysis_data.get("category", "暂无类目"),
        "timestamp": analysis_data.get("timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    }

    # ✅ 关键修复：传递正确的变量名给模板
    return render_template('quality_analysis.html',
                           analysis_id=analysis_id,
                           analysis_data=analysis_data_info,
                           detail_data=detail_data,
                           summary=summary,
                           group_summary=group_summary,
                           quality_distribution=quality_distribution,
                           premium_detail=premium_detail,
                           high_read_detail=high_read_detail,
                           report=report)


@app.route('/report/cost/<analysis_id>')
@login_required  # 新增：登录验证
def cost_report(analysis_id):
    """成本分析报告详情页，修复变量名匹配问题"""
    # 处理 'latest' 情况
    if analysis_id == 'latest':
        results_dir = os.path.join(app.config['OUTPUT_DIR'], 'analysis_results')
        try:
            result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
            if not result_files:
                return render_template('cost_analysis.html',
                                       analysis_id='latest',
                                       analysis_data={"category": "暂无类目",
                                                      "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                                       overall_summary={},
                                       invalid_data_stats={
                                           '总数据条数': 0,
                                           '有效数据条数': 0,
                                           '无效数据条数': 0,
                                           '有效数据比例(%)': '0%',
                                           '无效数据比例(%)': '0%',
                                           '无效数据原因分布': {},
                                           '无效数据总成本(元)': 0
                                       },
                                       invalid_data_detail_count=0,
                                       abnormal_data_stats={
                                           '异常数据条数': 0,
                                           '异常数据比例(%)': '0%',
                                           '异常数据原因分布': {},
                                           '异常数据总成本(元)': 0,
                                           '参与分析数据条数': 0,
                                           '参与分析数据比例(%)': '0%'
                                       },
                                       abnormal_data_detail_count=0,
                                       media_group_workload=[],
                                       fixed_media_workload=[],
                                       fixed_media_cost=[],
                                       fixed_media_rebate=[],
                                       fixed_media_performance=[],
                                       fixed_media_level=[],
                                       fixed_media_comprehensive=[],
                                       report={"excel": ""})
            result_files.sort(reverse=True)
            latest_file = result_files[0]
            analysis_id = latest_file.replace('.json', '')
        except Exception as e:
            logger.error(f"获取最新分析结果失败: {e}")
            return render_template('cost_analysis.html',
                                   analysis_id='latest',
                                   analysis_data={"category": "暂无类目",
                                                  "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                                   overall_summary={},
                                   invalid_data_stats={
                                       '总数据条数': 0,
                                       '有效数据条数': 0,
                                       '无效数据条数': 0,
                                       '有效数据比例(%)': '0%',
                                       '无效数据比例(%)': '0%',
                                       '无效数据原因分布': {},
                                       '无效数据总成本(元)': 0
                                   },
                                   invalid_data_detail_count=0,
                                   abnormal_data_stats={
                                       '异常数据条数': 0,
                                       '异常数据比例(%)': '0%',
                                       '异常数据原因分布': {},
                                       '异常数据总成本(元)': 0,
                                       '参与分析数据条数': 0,
                                       '参与分析数据比例(%)': '0%'
                                   },
                                   abnormal_data_detail_count=0,
                                   media_group_workload=[],
                                   fixed_media_workload=[],
                                       fixed_media_cost=[],
                                       fixed_media_rebate=[],
                                       fixed_media_performance=[],
                                       fixed_media_level=[],
                                       fixed_media_comprehensive=[],
                                       report={"excel": ""})

    # 加载分析结果
    analysis_data = load_analysis_result(analysis_id)
    if not analysis_data:
        # 返回空数据，包含无效数据统计的默认值
        overall_summary = {}
        invalid_data_stats = {
            '总数据条数': 0,
            '有效数据条数': 0,
            '无效数据条数': 0,
            '有效数据比例(%)': '0%',
            '无效数据比例(%)': '0%',
            '无效数据原因分布': {},
            '无效数据总成本(元)': 0
        }
        abnormal_data_stats = {
            '异常数据条数': 0,
            '异常数据比例(%)': '0%',
            '异常数据原因分布': {},
            '异常数据总成本(元)': 0,
            '参与分析数据条数': 0,
            '参与分析数据比例(%)': '0%'
        }
        invalid_data_detail_count = 0
        abnormal_data_detail_count = 0
        media_group_workload = []
        fixed_media_workload = []
        fixed_media_cost = []
        fixed_media_rebate = []
        fixed_media_performance = []
        fixed_media_level = []
        fixed_media_comprehensive = []
        report = {"excel": ""}
        analysis_data_info = {"category": "暂无类目", "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    else:
        # 从full_result.cost中获取数据
        full_result = analysis_data.get("full_result", {})
        cost_data = full_result.get("cost", {})

        # ✅ 关键修复：确保获取所有成本分析工作表数据
        overall_summary = cost_data.get("overall_summary", {})

        # ✅ 核心修复：确保 overall_summary 包含无效数据统计
        if not isinstance(overall_summary, dict):
            overall_summary = {}

        # ✅ 确保 overall_summary 有基本的无效数据统计字段
        overall_summary['总数据条数'] = overall_summary.get('总数据条数', 0)
        overall_summary['有效数据条数'] = overall_summary.get('有效数据条数', 0)
        overall_summary['无效数据条数'] = overall_summary.get('无效数据条数', 0)
        overall_summary['异常数据条数'] = overall_summary.get('异常数据条数', 0)
        overall_summary['参与分析数据条数'] = overall_summary.get('参与分析数据条数', 0)

        if overall_summary['总数据条数'] > 0:
            if '有效数据比例(%)' not in overall_summary:
                overall_summary[
                    '有效数据比例(%)'] = f"{(overall_summary['有效数据条数'] / overall_summary['总数据条数'] * 100):.2f}%"
            if '无效数据比例(%)' not in overall_summary:
                overall_summary[
                    '无效数据比例(%)'] = f"{(overall_summary['无效数据条数'] / overall_summary['总数据条数'] * 100):.2f}%"
            if '异常数据比例(%)' not in overall_summary:
                overall_summary[
                    '异常数据比例(%)'] = f"{(overall_summary['异常数据条数'] / overall_summary['总数据条数'] * 100):.2f}%"
            if '参与分析数据比例(%)' not in overall_summary:
                overall_summary[
                    '参与分析数据比例(%)'] = f"{(overall_summary['参与分析数据条数'] / overall_summary['总数据条数'] * 100):.2f}%"
        else:
            overall_summary['有效数据比例(%)'] = overall_summary.get('有效数据比例(%)', '0%')
            overall_summary['无效数据比例(%)'] = overall_summary.get('无效数据比例(%)', '0%')
            overall_summary['异常数据比例(%)'] = overall_summary.get('异常数据比例(%)', '0%')
            overall_summary['参与分析数据比例(%)'] = overall_summary.get('参与分析数据比例(%)', '0%')

        overall_summary['无效数据原因分布'] = overall_summary.get('无效数据原因分布', {})
        overall_summary['异常数据原因分布'] = overall_summary.get('异常数据原因分布', {})
        overall_summary['无效数据总成本(元)'] = overall_summary.get('无效数据总成本(元)', 0)
        overall_summary['异常数据总成本(元)'] = overall_summary.get('异常数据总成本(元)', 0)

        # ✅ 获取 invalid_data_stats（优先从 cost_data 获取，否则从 overall_summary 生成）
        invalid_data_stats = cost_data.get("invalid_data_stats", {})
        if not invalid_data_stats:
            invalid_data_stats = {
                '总数据条数': overall_summary.get('总数据条数', 0),
                '有效数据条数': overall_summary.get('有效数据条数', 0),
                '无效数据条数': overall_summary.get('无效数据条数', 0),
                '有效数据比例(%)': overall_summary.get('有效数据比例(%)', '0%'),
                '无效数据比例(%)': overall_summary.get('无效数据比例(%)', '0%'),
                '无效数据原因分布': overall_summary.get('无效数据原因分布', {}),
                '无效数据总成本(元)': overall_summary.get('无效数据总成本(元)', 0)
            }

        # ✅ 获取 abnormal_data_stats（优先从 cost_data 获取，否则从 overall_summary 生成）
        abnormal_data_stats = cost_data.get("abnormal_data_stats", {})
        if not abnormal_data_stats:
            abnormal_data_stats = {
                '异常数据条数': overall_summary.get('异常数据条数', 0),
                '异常数据比例(%)': overall_summary.get('异常数据比例(%)', '0%'),
                '异常数据原因分布': overall_summary.get('异常数据原因分布', {}),
                '异常数据总成本(元)': overall_summary.get('异常数据总成本(元)', 0),
                '参与分析数据条数': overall_summary.get('参与分析数据条数', 0),
                '参与分析数据比例(%)': overall_summary.get('参与分析数据比例(%)', '0%')
            }

        # ✅ 获取 invalid_data_detail 和计数
        invalid_data_detail = cost_data.get("invalid_data_detail", [])
        invalid_data_detail_count = len(invalid_data_detail) if isinstance(invalid_data_detail, list) else 0

        # ✅ 获取 abnormal_data_detail 和计数
        abnormal_data_detail = cost_data.get("abnormal_data_detail", [])
        abnormal_data_detail_count = len(abnormal_data_detail) if isinstance(abnormal_data_detail, list) else 0

        # ✅ 获取所有成本工作表数据
        media_group_workload = cost_data.get("media_group_workload", [])
        if not isinstance(media_group_workload, list):
            media_group_workload = []

        fixed_media_workload = cost_data.get("fixed_media_workload", [])
        if not isinstance(fixed_media_workload, list):
            fixed_media_workload = []

        fixed_media_cost = cost_data.get("fixed_media_cost", [])
        if not isinstance(fixed_media_cost, list):
            fixed_media_cost = []

        fixed_media_rebate = cost_data.get("fixed_media_rebate", [])
        if not isinstance(fixed_media_rebate, list):
            fixed_media_rebate = []

        fixed_media_performance = cost_data.get("fixed_media_performance", [])
        if not isinstance(fixed_media_performance, list):
            fixed_media_performance = []

        fixed_media_level = cost_data.get("fixed_media_level", [])
        if not isinstance(fixed_media_level, list):
            fixed_media_level = []

        fixed_media_comprehensive = cost_data.get("fixed_media_comprehensive", [])
        if not isinstance(fixed_media_comprehensive, list):
            fixed_media_comprehensive = []

        report = analysis_data.get("reports", {}).get("cost", {"excel": ""})
        analysis_data_info = {
            "category": analysis_data.get("category", "暂无类目"),
            "timestamp": analysis_data.get("timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        }

    # ✅ 关键修复：传递正确的变量名给模板，匹配 cost_analysis.html 中的变量名
    return render_template('cost_analysis.html',
                           analysis_id=analysis_id,
                           analysis_data=analysis_data_info,
                           overall_summary=overall_summary,
                           invalid_data_stats=invalid_data_stats,  # ✅ 新增：无效数据统计
                           invalid_data_detail_count=invalid_data_detail_count,  # ✅ 新增：无效数据详情数量
                           abnormal_data_stats=abnormal_data_stats,  # ✅ 新增：异常数据统计
                           abnormal_data_detail_count=abnormal_data_detail_count,  # ✅ 新增：异常数据详情数量
                           media_group_workload=media_group_workload,
                           fixed_media_workload=fixed_media_workload,
                           fixed_media_cost=fixed_media_cost,
                           fixed_media_rebate=fixed_media_rebate,
                           fixed_media_performance=fixed_media_performance,
                           fixed_media_level=fixed_media_level,
                           fixed_media_comprehensive=fixed_media_comprehensive,
                           report=report)


@app.route('/report/cost/invalid_data/<analysis_id>')
def cost_invalid_data_report(analysis_id):
    """成本分析无效数据详情页 - 修复版本"""
    # 处理 'latest' 情况
    if analysis_id == 'latest':
        results_dir = os.path.join(app.config['OUTPUT_DIR'], 'analysis_results')
        try:
            result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
            if not result_files:
                return render_template('cost_invalid_data.html',
                                       analysis_id='latest',
                                       analysis_data={"category": "暂无类目",
                                                      "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                                       invalid_data_detail=[],
                                       invalid_data_stats={})
            result_files.sort(reverse=True)
            latest_file = result_files[0]
            analysis_id = latest_file.replace('.json', '')
        except Exception as e:
            logger.error(f"获取最新分析结果失败: {e}")
            return render_template('cost_invalid_data.html',
                                   analysis_id='latest',
                                   analysis_data={"category": "暂无类目",
                                                  "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                                   invalid_data_detail=[],
                                   invalid_data_stats={})

    # 加载分析结果
    analysis_data = load_analysis_result(analysis_id)
    if not analysis_data:
        return render_template('cost_invalid_data.html',
                               analysis_id=analysis_id,
                               analysis_data={"category": "暂无类目",
                                              "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                               invalid_data_detail=[],
                               invalid_data_stats={})

    # 获取无效数据详情和统计
    full_result = analysis_data.get("full_result", {})
    cost_data = full_result.get("cost", {})

    # ✅ 核心修复：直接从 cost_data 获取 invalid_data_detail
    invalid_data_detail = cost_data.get("invalid_data_detail", [])

    # ✅ 关键修复：如果 invalid_data_detail 为空，从 detailed_data 中筛选
    if not invalid_data_detail or len(invalid_data_detail) == 0:
        detailed_data = cost_data.get("detailed_data", [])
        if detailed_data and isinstance(detailed_data, list) and len(detailed_data) > 0:
            logger.info(f"从 detailed_data 中筛选无效数据，总数据: {len(detailed_data)}")

            invalid_data_detail = []
            for item in detailed_data:
                if isinstance(item, dict):
                    # 检查是否为无效数据
                    cost_invalid = item.get('成本无效', False)

                    # 如果是无效数据（成本=0或成本缺失）
                    if cost_invalid:
                        # 构建无效数据详情格式
                        detail = {
                            '记录序号': item.get('记录序号', 0),
                            '达人昵称': item.get('达人昵称', '未知'),
                            '项目名称': item.get('项目名称', '未知'),
                            '定档媒介': item.get('定档媒介', '未知'),
                            '成本': item.get('成本', 0),
                            '报价': item.get('报价', 0),
                            '下单价': item.get('下单价', 0),
                            '返点': item.get('返点', 0),
                            '返点比例': item.get('返点比例', 0) * 100 if item.get('返点比例') else 0,
                            '不含手续费的下单价': item.get('不含手续费的下单价', ''),
                            '成本无效原因': item.get('成本无效原因', '成本为0或缺失'),
                            '是否被筛除': True,  # 无效数据默认被筛除
                            '筛除原因': item.get('成本无效原因', '成本为0或缺失'),
                            '无效类型': '成本为0或缺失'
                        }

                        # 判断无效类型
                        if detail['成本'] == 0:
                            detail['成本无效原因'] = '成本为0'
                        elif pd.isna(detail['成本']):
                            detail['成本无效原因'] = '成本缺失'
                        elif '成本无效' in str(item):
                            detail['成本无效原因'] = '成本无效'
                        else:
                            detail['成本无效原因'] = '未知原因'

                        invalid_data_detail.append(detail)

            logger.info(f"筛选到无效数据: {len(invalid_data_detail)} 条")

    # ✅ 确保 invalid_data_detail 是列表
    if not isinstance(invalid_data_detail, list):
        logger.warning(f"invalid_data_detail 不是列表类型: {type(invalid_data_detail)}")
        # 尝试转换
        if isinstance(invalid_data_detail, pd.DataFrame):
            invalid_data_detail = invalid_data_detail.to_dict('records')
        elif isinstance(invalid_data_detail, dict):
            invalid_data_detail = [invalid_data_detail]
        else:
            invalid_data_detail = []

    # ✅ 获取无效数据统计
    invalid_data_stats = cost_data.get("invalid_data_stats", {})

    # 如果统计信息不存在或为空，根据详情计算
    if not invalid_data_stats or not isinstance(invalid_data_stats, dict):
        invalid_data_stats = {}

    # 确保统计信息包含必要字段
    if '无效数据条数' not in invalid_data_stats:
        invalid_data_stats['无效数据条数'] = len(invalid_data_detail)

    if '无效数据比例(%)' not in invalid_data_stats:
        overall_summary = cost_data.get("overall_summary", {})
        total_count = overall_summary.get('总数据条数', 0)
        if total_count > 0:
            invalid_data_stats['无效数据比例(%)'] = f"{(len(invalid_data_detail) / total_count * 100):.2f}%"
        else:
            invalid_data_stats['无效数据比例(%)'] = '0%'

    if '无效数据原因分布' not in invalid_data_stats:
        # 统计无效原因分布
        reason_dist = {}
        total_cost = 0
        for detail in invalid_data_detail:
            if isinstance(detail, dict):
                reason = detail.get('成本无效原因', '未知原因')
                reason_dist[reason] = reason_dist.get(reason, 0) + 1
                total_cost += detail.get('成本', 0)

        invalid_data_stats['无效数据原因分布'] = reason_dist
        invalid_data_stats['无效数据总成本(元)'] = total_cost

    # 确保其他统计字段存在
    if '无效数据总成本(元)' not in invalid_data_stats:
        invalid_data_stats['无效数据总成本(元)'] = 0

    if '有效数据条数' not in invalid_data_stats:
        overall_summary = cost_data.get("overall_summary", {})
        total_count = overall_summary.get('总数据条数', 0)
        invalid_data_stats['有效数据条数'] = total_count - len(invalid_data_detail) if total_count > 0 else 0

    if '有效数据比例(%)' not in invalid_data_stats:
        overall_summary = cost_data.get("overall_summary", {})
        total_count = overall_summary.get('总数据条数', 0)
        if total_count > 0:
            invalid_data_stats[
                '有效数据比例(%)'] = f"{((total_count - len(invalid_data_detail)) / total_count * 100):.2f}%"
        else:
            invalid_data_stats['有效数据比例(%)'] = '0%'

    if '总数据条数' not in invalid_data_stats:
        overall_summary = cost_data.get("overall_summary", {})
        invalid_data_stats['总数据条数'] = overall_summary.get('总数据条数', 0)

    logger.info(f"渲染无效数据页面: analysis_id={analysis_id}, 无效数据条数={len(invalid_data_detail)}")
    logger.info(f"无效数据统计: {invalid_data_stats}")

    analysis_data_info = {
        "category": analysis_data.get("category", "暂无类目"),
        "timestamp": analysis_data.get("timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    }

    return render_template('cost_invalid_data.html',
                           analysis_id=analysis_id,
                           analysis_data=analysis_data_info,
                           invalid_data_detail=invalid_data_detail,
                           invalid_data_stats=invalid_data_stats)

@app.route('/report/cost/abnormal_data/<analysis_id>')
@login_required
def cost_abnormal_data_report(analysis_id):
    """成本分析异常数据详情页 - 完整修复版本"""
    # 处理 'latest' 情况
    if analysis_id == 'latest':
        results_dir = os.path.join(app.config['OUTPUT_DIR'], 'analysis_results')
        try:
            result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
            if not result_files:
                return render_template('cost_abnormal_data.html',
                                       analysis_id='latest',
                                       analysis_data={"category": "暂无类目",
                                                      "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                                       abnormal_data_detail=[],
                                       abnormal_data_stats={})
            result_files.sort(reverse=True)
            latest_file = result_files[0]
            analysis_id = latest_file.replace('.json', '')
        except Exception as e:
            logger.error(f"获取最新分析结果失败: {e}")
            return render_template('cost_abnormal_data.html',
                                   analysis_id='latest',
                                   analysis_data={"category": "暂无类目",
                                                  "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                                   abnormal_data_detail=[],
                                   abnormal_data_stats={})

    # 加载分析结果
    analysis_data = load_analysis_result(analysis_id)
    if not analysis_data:
        return render_template('cost_abnormal_data.html',
                               analysis_id=analysis_id,
                               analysis_data={"category": "暂无类目",
                                              "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                               abnormal_data_detail=[],
                               abnormal_data_stats={})

    # 获取异常数据详情和统计
    full_result = analysis_data.get("full_result", {})
    cost_data = full_result.get("cost", {})

    # ✅ 核心修复：直接从 cost_data 获取 abnormal_data_detail
    abnormal_data_detail = cost_data.get("abnormal_data_detail", [])

    # ✅ 关键修复：如果 abnormal_data_detail 为空，从 detailed_data 中筛选
    if not abnormal_data_detail or len(abnormal_data_detail) == 0:
        detailed_data = cost_data.get("detailed_data", [])
        if detailed_data and isinstance(detailed_data, list) and len(detailed_data) > 0:
            logger.info(f"从 detailed_data 中筛选异常数据，总数据: {len(detailed_data)}")

            abnormal_data_detail = []
            for item in detailed_data:
                if isinstance(item, dict):
                    # 检查是否为异常数据
                    data_abnormal = item.get('数据异常', False)
                    cost_invalid = item.get('成本无效', False)

                    # 如果是异常数据（参与分析但标记异常）
                    if data_abnormal and not cost_invalid:
                        # 构建异常数据详情格式
                        detail = {
                            '记录序号': item.get('记录序号', 0),
                            '达人昵称': item.get('达人昵称', '未知'),
                            '项目名称': item.get('项目名称', '未知'),
                            '定档媒介': item.get('定档媒介', '未知'),
                            '成本': item.get('成本', 0),
                            '报价': item.get('报价', 0),
                            '下单价': item.get('下单价', 0),
                            '返点': item.get('返点', 0),
                            '返点比例': item.get('返点比例', 0) * 100 if item.get('返点比例') else 0,
                            '不含手续费的下单价': item.get('不含手续费的下单价', ''),
                            '数据异常原因': item.get('数据异常原因', '未知异常'),
                            '异常类型': '数据异常',
                            '是否参与分析': True,
                            '参与分析标识': '异常数据'
                        }

                        # 判断异常类型
                        reason = detail['数据异常原因']
                        if '报价<' in reason:
                            detail['异常类型'] = '报价异常'
                        elif '无法判断' in reason:
                            detail['异常类型'] = '数据异常'
                        elif '返点比例' in reason:
                            detail['异常类型'] = '返点异常'
                        elif '筛除' in reason or reason in ['数据异常', '成本为0', '成本缺失', '数据不全']:
                            detail['异常类型'] = '筛除异常'

                        abnormal_data_detail.append(detail)

            logger.info(f"筛选到异常数据: {len(abnormal_data_detail)} 条")

    # ✅ 确保 abnormal_data_detail 是列表
    if not isinstance(abnormal_data_detail, list):
        logger.warning(f"abnormal_data_detail 不是列表类型: {type(abnormal_data_detail)}")
        # 尝试转换
        if isinstance(abnormal_data_detail, pd.DataFrame):
            abnormal_data_detail = abnormal_data_detail.to_dict('records')
        elif isinstance(abnormal_data_detail, dict):
            abnormal_data_detail = [abnormal_data_detail]
        else:
            abnormal_data_detail = []

    # ✅ 获取异常数据统计
    abnormal_data_stats = cost_data.get("abnormal_data_stats", {})

    # 如果统计信息不存在或为空，根据详情计算
    if not abnormal_data_stats or not isinstance(abnormal_data_stats, dict):
        abnormal_data_stats = {}

    # 确保统计信息包含必要字段
    if '异常数据条数' not in abnormal_data_stats:
        abnormal_data_stats['异常数据条数'] = len(abnormal_data_detail)

    if '异常数据比例(%)' not in abnormal_data_stats:
        overall_summary = cost_data.get("overall_summary", {})
        total_count = overall_summary.get('总数据条数', 0)
        if total_count > 0:
            abnormal_data_stats['异常数据比例(%)'] = f"{(len(abnormal_data_detail) / total_count * 100):.2f}%"
        else:
            abnormal_data_stats['异常数据比例(%)'] = '0%'

    if '异常数据原因分布' not in abnormal_data_stats:
        # 统计异常原因分布
        reason_dist = {}
        total_cost = 0
        for detail in abnormal_data_detail:
            if isinstance(detail, dict):
                reason = detail.get('数据异常原因', '未知原因')
                reason_dist[reason] = reason_dist.get(reason, 0) + 1
                total_cost += detail.get('成本', 0)

        abnormal_data_stats['异常数据原因分布'] = reason_dist
        abnormal_data_stats['异常数据总成本(元)'] = total_cost

    # 确保其他统计字段存在
    if '异常数据总成本(元)' not in abnormal_data_stats:
        abnormal_data_stats['异常数据总成本(元)'] = 0

    if '参与分析数据条数' not in abnormal_data_stats:
        overall_summary = cost_data.get("overall_summary", {})
        total_count = overall_summary.get('总数据条数', 0)
        invalid_count = overall_summary.get('无效数据条数', 0)
        abnormal_data_stats['参与分析数据条数'] = total_count - invalid_count if total_count > 0 else 0

    if '参与分析数据比例(%)' not in abnormal_data_stats:
        overall_summary = cost_data.get("overall_summary", {})
        total_count = overall_summary.get('总数据条数', 0)
        invalid_count = overall_summary.get('无效数据条数', 0)
        if total_count > 0:
            abnormal_data_stats['参与分析数据比例(%)'] = f"{((total_count - invalid_count) / total_count * 100):.2f}%"
        else:
            abnormal_data_stats['参与分析数据比例(%)'] = '0%'

    logger.info(f"渲染异常数据页面: analysis_id={analysis_id}, 异常数据条数={len(abnormal_data_detail)}")

    analysis_data_info = {
        "category": analysis_data.get("category", "暂无类目"),
        "timestamp": analysis_data.get("timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    }

    return render_template('cost_abnormal_data.html',
                           analysis_id=analysis_id,
                           analysis_data=analysis_data_info,
                           abnormal_data_detail=abnormal_data_detail,
                           abnormal_data_stats=abnormal_data_stats)

# ========================== ✅ 新增缺失的路由定义 ==========================
@app.route('/download/table/<string:table_type>/<string:analysis_id>')
def download_table(table_type, analysis_id):
    """下载单个表格数据"""
    try:
        # 加载分析结果
        analysis_data = load_analysis_result(analysis_id)
        if not analysis_data:
            return jsonify({"error": "分析结果不存在"}), 404

        # 获取数据
        full_result = analysis_data.get('full_result', {})
        workload_data = full_result.get('workload', {})

        # 根据表格类型获取数据
        if table_type == 'workload_detail':
            data = workload_data.get('result', [])
            sheet_name = '工作量明细'
        elif table_type == 'workload_group':
            data = workload_data.get('group_summary', [])
            sheet_name = '工作量小组汇总'
        elif table_type == 'workload_top':
            data = workload_data.get('top_media_ranking', [])
            sheet_name = '工作量TOP排名'
        else:
            return jsonify({"error": "不支持的表格类型"}), 400

        if not data:
            return jsonify({"error": "表格数据为空"}), 404

        # 创建DataFrame
        df = pd.DataFrame(data)

        # 创建Excel文件
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        output.seek(0)

        # 返回文件
        filename = f"{sheet_name}_{analysis_id}.xlsx"
        return send_file(
            output,
            download_name=filename,
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        logger.error(f"下载表格失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ✅ 正确的路由（与 JavaScript 匹配）
@app.route('/download_cost_sheet/<string:sheet_key>/<string:analysis_id>')
def download_cost_sheet(sheet_key, analysis_id):
    """
    下载成本分析单个工作表
    """
    logger.info(f"下载工作表: sheet_key={sheet_key}, analysis_id={analysis_id}")

    try:
        # 如果是latest，获取最新的analysis_id
        if analysis_id == 'latest':
            # 获取最新分析结果
            if not analysis_results:
                return "没有可用的分析数据，请先进行分析", 404

            latest_id = sorted(analysis_results.keys())[-1]
            analysis_id = latest_id

        # 加载分析结果
        analysis_data = load_analysis_result(analysis_id)
        if not analysis_data:
            return "分析结果不存在", 404

        # 从分析结果中获取数据
        full_result = analysis_data.get('full_result', {})
        cost_data = full_result.get('cost', {})

        # 工作表映射
        sheet_mapping = {
            'media_group_workload': '媒介小组工作量分析',
            'fixed_media_workload': '定档媒介工作量分析',
            'fixed_media_cost': '定档媒介成本分析',
            'fixed_media_rebate': '定档媒介返点分析',
            'fixed_media_performance': '定档媒介效果分析',
            'fixed_media_level': '定档媒介达人量级分析',
            'fixed_media_comprehensive': '定档媒介综合分析'
        }

        # 根据 sheet_key 获取数据
        data = None
        sheet_name = sheet_mapping.get(sheet_key, sheet_key)

        if sheet_key == 'media_group_workload':
            data = cost_data.get("media_group_workload", [])
        elif sheet_key == 'fixed_media_workload':
            data = cost_data.get("fixed_media_workload", [])
        elif sheet_key == 'fixed_media_cost':
            data = cost_data.get("fixed_media_cost", [])
        elif sheet_key == 'fixed_media_rebate':
            data = cost_data.get("fixed_media_rebate", [])
        elif sheet_key == 'fixed_media_performance':
            data = cost_data.get("fixed_media_performance", [])
        elif sheet_key == 'fixed_media_level':
            data = cost_data.get("fixed_media_level", [])
        elif sheet_key == 'fixed_media_comprehensive':
            data = cost_data.get("fixed_media_comprehensive", [])
        else:
            return f"不支持的工作表类型: {sheet_key}", 400

        if not data:
            return f"工作表数据为空: {sheet_key}", 404

        # 将数据转换为DataFrame
        df = pd.DataFrame(data)

        # 创建 Excel 文件
        output = BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        output.seek(0)

        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"{sheet_name}_{analysis_id}_{timestamp}.xlsx"

        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logger.error(f"下载失败: {e}", exc_info=True)
        return f"下载失败: {str(e)}", 500

# ========== 下载Excel报告路由 ==========
@app.route('/download/excel/<analysis_id>')
def download_excel_report(analysis_id):
    """根据分析ID精准下载Excel报告"""
    excel_dir = os.path.join(app.config['OUTPUT_DIR'], 'excel')

    # 查找匹配的Excel文件
    excel_filename = None
    for file in os.listdir(excel_dir):
        if file.endswith('.xlsx') and analysis_id in file:
            excel_filename = file
            break

    if not excel_filename:
        # 尝试查找最新的文件
        excel_files = [f for f in os.listdir(excel_dir) if f.endswith('.xlsx')]
        if excel_files:
            excel_files.sort(reverse=True)
            excel_filename = excel_files[0]

    if not excel_filename:
        flash('❌ 未找到该分析ID对应的Excel报告', 'error')
        return redirect(url_for('dashboard', analysis_id=analysis_id))

    try:
        return send_from_directory(
            excel_dir,
            excel_filename,
            as_attachment=True,
            download_name=excel_filename
        )
    except Exception as e:
        logger.error(f"❌ 下载失败：{str(e)}")
        flash(f"❌ 下载失败：{str(e)}", 'error')
        return redirect(url_for('dashboard', analysis_id=analysis_id))


@app.route('/download/<analysis_id>/<report_type>')
@app.route('/download/<analysis_id>/<report_type>')
def download_report(analysis_id, report_type):
    """下载报告 - 根据报告类型导出完整数据"""
    try:
        # 加载分析结果
        analysis_data = load_analysis_result(analysis_id)
        if not analysis_data:
            flash('❌ 分析结果不存在', 'error')
            return redirect(url_for('dashboard', analysis_id=analysis_id))

        # 获取完整数据
        full_result = analysis_data.get('full_result', {})

        # 创建Excel文件
        output = BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            if report_type == 'workload':
                # 工作量分析的所有表格
                workload_data = full_result.get('workload', {})

                # 1. 工作量明细
                if workload_data.get('result'):
                    df_detail = pd.DataFrame(workload_data['result'])
                    df_detail.to_excel(writer, sheet_name='工作量明细', index=False)

                # 2. 小组汇总
                if workload_data.get('group_summary'):
                    df_group = pd.DataFrame(workload_data['group_summary'])
                    df_group.to_excel(writer, sheet_name='工作量小组汇总', index=False)

                # 3. TOP排名
                if workload_data.get('top_media_ranking'):
                    df_top = pd.DataFrame(workload_data['top_media_ranking'])
                    df_top.to_excel(writer, sheet_name='工作量TOP排名', index=False)

                # 4. 工作量汇总统计
                if workload_data.get('summary'):
                    df_summary = pd.DataFrame([workload_data['summary']])
                    df_summary.to_excel(writer, sheet_name='工作量汇总', index=False)

            elif report_type == 'quality':
                # 质量分析的所有表格
                quality_data = full_result.get('quality', {})

                # 1. 质量明细
                if quality_data.get('result'):
                    df_detail = pd.DataFrame(quality_data['result'])
                    df_detail.to_excel(writer, sheet_name='质量明细', index=False)

                # 2. 小组汇总
                if quality_data.get('group_summary'):
                    df_group = pd.DataFrame(quality_data['group_summary'])
                    df_group.to_excel(writer, sheet_name='质量小组汇总', index=False)

                # 3. 质量分布
                if quality_data.get('quality_distribution'):
                    df_dist = pd.DataFrame(quality_data['quality_distribution'])
                    df_dist.to_excel(writer, sheet_name='质量分布', index=False)

                # 4. 优质达人明细
                if quality_data.get('premium_detail'):
                    df_premium = pd.DataFrame(quality_data['premium_detail'])
                    df_premium.to_excel(writer, sheet_name='优质达人质量明细', index=False)

                # 5. 高阅读达人明细
                if quality_data.get('high_read_detail'):
                    df_high_read = pd.DataFrame(quality_data['high_read_detail'])
                    df_high_read.to_excel(writer, sheet_name='高阅读达人质量明细', index=False)

            elif report_type == 'cost':
                # 成本分析的所有表格
                cost_data = full_result.get('cost', {})

                # 所有工作表
                sheets = [
                    ('media_group_workload', '媒介小组工作量分析'),
                    ('fixed_media_workload', '定档媒介工作量分析'),
                    ('fixed_media_cost', '定档媒介成本分析'),
                    ('fixed_media_rebate', '定档媒介返点分析'),
                    ('fixed_media_performance', '定档媒介效果分析'),
                    ('fixed_media_level', '定档媒介达人量级分析'),
                    ('fixed_media_comprehensive', '定档媒介综合分析'),
                    ('media_detail', '详细数据'),
                    ('group_summary', '小组汇总'),
                    ('cost_efficiency_ranking', '成本效率排名')
                ]

                for sheet_key, sheet_name in sheets:
                    if cost_data.get(sheet_key):
                        df_sheet = pd.DataFrame(cost_data[sheet_key])
                        df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

                        # 确保至少有一个sheet不是隐藏的
                        workbook = writer.book
                        if workbook:
                            sheet = workbook[sheet_name]
                            sheet.sheet_state = 'visible'

            else:
                # 完整报告（所有分析）- 不调用report_generator，直接创建
                workbook_data = full_result

                # 确保写入至少一个可见的工作表
                write_at_least_one = False

                # 工作量数据
                if 'workload' in workbook_data and workbook_data['workload'].get('result'):
                    df_workload = pd.DataFrame(workload_data['workload']['result'])
                    df_workload.to_excel(writer, sheet_name='工作量分析', index=False)
                    write_at_least_one = True

                # 质量数据
                if 'quality' in workbook_data and workbook_data['quality'].get('result'):
                    df_quality = pd.DataFrame(workload_data['quality']['result'])
                    df_quality.to_excel(writer, sheet_name='质量分析', index=False)
                    write_at_least_one = True

                # 成本数据
                if 'cost' in workbook_data and workbook_data['cost'].get('media_detail'):
                    df_cost = pd.DataFrame(workload_data['cost']['media_detail'])
                    df_cost.to_excel(writer, sheet_name='成本分析', index=False)
                    write_at_least_one = True

                if not write_at_least_one:
                    # 如果没有任何数据，至少创建一个空的工作表
                    pd.DataFrame({'提示': ['无分析数据']}).to_excel(writer, sheet_name='报告汇总', index=False)

        output.seek(0)

        # 设置文件名
        filename = f"媒介分析报告_{report_type}_{analysis_id}.xlsx"

        return send_file(
            output,
            download_name=filename,
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        logger.error(f"导出报告失败: {e}", exc_info=True)
        flash(f'❌ 导出报告失败: {str(e)}', 'error')
        return redirect(url_for('dashboard', analysis_id=analysis_id))


# 在 download_report 路由后添加：
@app.route('/export/invalid_data/<analysis_id>')
def export_invalid_data(analysis_id):
    """导出无效数据为Excel"""
    try:
        # 加载分析结果
        analysis_data = load_analysis_result(analysis_id)
        if not analysis_data:
            flash('❌ 分析结果不存在', 'error')
            return redirect(url_for('cost_report', analysis_id=analysis_id))

        # 获取无效数据详情
        full_result = analysis_data.get('full_result', {})
        cost_data = full_result.get('cost', {})
        invalid_data_detail = cost_data.get('invalid_data_detail', [])
        invalid_data_stats = cost_data.get('invalid_data_stats', {})

        if not invalid_data_detail:
            flash('⚠️ 无无效数据可导出', 'info')
            return redirect(url_for('cost_report', analysis_id=analysis_id))

        # 创建DataFrame
        df = pd.DataFrame(invalid_data_detail)

        # 创建Excel文件
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 写入无效数据明细
            df.to_excel(writer, sheet_name='无效数据明细', index=False)

            # 写入统计信息
            stats_data = []
            if invalid_data_stats:
                stats_data = [
                    ['总数据条数', invalid_data_stats.get('总数据条数', 0)],
                    ['有效数据条数', invalid_data_stats.get('有效数据条数', 0)],
                    ['无效数据条数', invalid_data_stats.get('无效数据条数', 0)],
                    ['有效数据比例', invalid_data_stats.get('有效数据比例(%)', '0%')],
                    ['无效数据比例', invalid_data_stats.get('无效数据比例(%)', '0%')],
                    ['无效数据总成本(元)', invalid_data_stats.get('无效数据总成本(元)', 0)]
                ]

            # 写入无效原因分布
            reason_dist = invalid_data_stats.get('无效数据原因分布', {})
            if reason_dist:
                stats_data.append(['', ''])
                stats_data.append(['无效原因分布', '数量'])
                for reason, count in reason_dist.items():
                    stats_data.append([reason, count])

            if stats_data:
                stats_df = pd.DataFrame(stats_data, columns=['项目', '数值'])
                stats_df.to_excel(writer, sheet_name='数据统计', index=False)

        output.seek(0)

        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"无效数据明细_{analysis_id}_{timestamp}.xlsx"

        # 返回文件
        return send_file(
            output,
            download_name=filename,
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        logger.error(f"❌ 导出无效数据失败: {e}", exc_info=True)
        flash(f'❌ 导出失败: {str(e)}', 'error')
        return redirect(url_for('cost_report', analysis_id=analysis_id))


@app.route('/export/abnormal_data/<analysis_id>')
@login_required
def export_abnormal_data(analysis_id):
    """导出异常数据为Excel - 完整修复版本"""
    try:
        # 加载分析结果
        analysis_data = load_analysis_result(analysis_id)
        if not analysis_data:
            flash('❌ 分析结果不存在', 'error')
            return redirect(url_for('cost_report', analysis_id=analysis_id))

        # 获取异常数据详情
        full_result = analysis_data.get('full_result', {})
        cost_data = full_result.get('cost', {})
        abnormal_data_detail = cost_data.get('abnormal_data_detail', [])

        # ✅ 关键修复：如果 abnormal_data_detail 为空，尝试从 detailed_data 中筛选
        if not abnormal_data_detail:
            detailed_data = cost_data.get("detailed_data", [])
            if detailed_data and isinstance(detailed_data, list) and len(detailed_data) > 0:
                logger.info(f"导出时从 detailed_data 中筛选异常数据，总数据: {len(detailed_data)}")

                abnormal_data_detail = []
                for item in detailed_data:
                    if isinstance(item, dict):
                        # 检查是否为异常数据
                        data_abnormal = item.get('数据异常', False)
                        cost_invalid = item.get('成本无效', False)

                        # 如果是异常数据（参与分析但标记异常）
                        if data_abnormal and not cost_invalid:
                            # 构建异常数据详情格式
                            detail = {
                                '记录序号': item.get('记录序号', 0),
                                '达人昵称': item.get('达人昵称', '未知'),
                                '项目名称': item.get('项目名称', '未知'),
                                '定档媒介': item.get('定档媒介', '未知'),
                                '成本': item.get('成本', 0),
                                '报价': item.get('报价', 0),
                                '下单价': item.get('下单价', 0),
                                '返点': item.get('返点', 0),
                                '返点比例': item.get('返点比例', 0) * 100 if item.get('返点比例') else 0,
                                '不含手续费的下单价': item.get('不含手续费的下单价', ''),
                                '数据异常原因': item.get('数据异常原因', '未知异常'),
                                '异常类型': '数据异常',
                                '是否参与分析': True
                            }

                            # 判断异常类型
                            reason = detail['数据异常原因']
                            if '报价<' in reason:
                                detail['异常类型'] = '报价异常'
                            elif '无法判断' in reason:
                                detail['异常类型'] = '数据异常'
                            elif '返点比例' in reason:
                                detail['异常类型'] = '返点异常'
                            elif '筛除' in reason or reason in ['数据异常', '成本为0', '成本缺失', '数据不全']:
                                detail['异常类型'] = '筛除异常'

                            abnormal_data_detail.append(detail)

                logger.info(f"导出时筛选到异常数据: {len(abnormal_data_detail)} 条")

        abnormal_data_stats = cost_data.get('abnormal_data_stats', {})

        if not abnormal_data_detail:
            flash('⚠️ 无异常数据可导出', 'info')
            return redirect(url_for('cost_abnormal_data_report', analysis_id=analysis_id))

        # 创建DataFrame
        df = pd.DataFrame(abnormal_data_detail)

        # 添加必要的中文列名
        column_mapping = {
            '记录序号': '序号',
            '达人昵称': '达人昵称',
            '项目名称': '项目名称',
            '定档媒介': '定档媒介',
            '成本': '成本(元)',
            '报价': '报价(元)',
            '下单价': '下单价(元)',
            '返点': '返点(元)',
            '返点比例': '返点比例(%)',
            '不含手续费的下单价': '不含手续费下单价',
            '数据异常原因': '异常原因',
            '异常类型': '异常类型',
            '是否参与分析': '是否参与分析'
        }

        # 重命名列
        df = df.rename(columns=column_mapping)

        # 创建Excel文件
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 写入异常数据明细
            df.to_excel(writer, sheet_name='异常数据明细', index=False)

            # 写入统计信息
            stats_data = []
            if abnormal_data_stats:
                stats_data = [
                    ['异常数据条数', abnormal_data_stats.get('异常数据条数', len(abnormal_data_detail))],
                    ['异常数据比例', abnormal_data_stats.get('异常数据比例(%)',
                                                             f"{(len(abnormal_data_detail) / (abnormal_data_stats.get('总数据条数', 1)) * 100):.2f}%" if abnormal_data_stats.get(
                                                                 '总数据条数', 0) > 0 else '0%')],
                    ['参与分析数据条数', abnormal_data_stats.get('参与分析数据条数', 0)],
                    ['参与分析数据比例', abnormal_data_stats.get('参与分析数据比例(%)', '100%')],
                    ['异常数据总成本(元)', abnormal_data_stats.get('异常数据总成本(元)', 0)]
                ]

            # 写入异常原因分布
            reason_dist = abnormal_data_stats.get('异常数据原因分布', {})
            if reason_dist:
                stats_data.append(['', ''])
                stats_data.append(['异常原因分布', '数量'])
                for reason, count in reason_dist.items():
                    stats_data.append([reason, count])

            if stats_data:
                stats_df = pd.DataFrame(stats_data, columns=['项目', '数值'])
                stats_df.to_excel(writer, sheet_name='数据统计', index=False)

        output.seek(0)

        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"异常数据明细_{analysis_id}_{timestamp}.xlsx"

        # 返回文件
        return send_file(
            output,
            download_name=filename,
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        logger.error(f"❌ 导出异常数据失败: {e}", exc_info=True)
        flash(f'❌ 导出失败: {str(e)}', 'error')
        return redirect(url_for('cost_abnormal_data_report', analysis_id=analysis_id))

@app.route('/debug/user')
@login_required
def debug_user():
    """调试用户信息"""
    user = get_current_user()
    if user:
        return jsonify({
            'user_id': session.get('user_id'),
            'username': session.get('username'),
            'role': session.get('role'),
            'is_admin': user.is_admin() if user else False,
            'is_active': user.is_active() if user else False
        })
    else:
        return jsonify({'error': '用户未登录'})

# ------------------------------ 辅助路由 ------------------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    """兼容旧提交逻辑的重定向"""
    return redirect(url_for('index'), code=307)

@app.route('/clear_results')
def clear_results():
    """清除所有内存中的分析结果"""
    global analysis_results
    analysis_results.clear()
    flash('✅ 所有内存中的分析结果已清除', 'success')
    return redirect(url_for('index'))

# ------------------------------ 错误处理 ------------------------------
@app.errorhandler(404)
def page_not_found(e):
    """404页面处理"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """500服务器错误处理，记录完整堆栈信息"""
    error_msg = f"❌ 服务器内部错误：{str(e)}"
    logger.error(f"{error_msg}\n{traceback.format_exc()}")
    return render_template('500.html', error_message=error_msg), 500

@app.route('/favicon.ico')
def favicon():
    """屏蔽favicon.ico请求错误"""
    return '', 204

# ------------------------------ 应用入口 ------------------------------
if __name__ == '__main__':
    logger.info("="*50)
    logger.info("🚀 媒介自动化审计分析系统 - 真实数据模式 启动成功")
    logger.info(f"🌐 服务访问地址：http://0.0.0.0:5000")
    logger.info(f"📂 上传目录：{app.config['UPLOAD_FOLDER']}")
    logger.info(f"📤 输出目录：{app.config['OUTPUT_DIR']}")
    logger.info("="*50)
    # 优化启动参数，避免多进程冲突+支持并发
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG'],
        threaded=True,
        use_reloader=False
    )
