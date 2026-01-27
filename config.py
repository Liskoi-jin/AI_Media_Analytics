import os
from datetime import timedelta
from dotenv import load_dotenv
import os


class Config:
    """应用配置类"""
    # 密钥（生产环境请改为随机字符串）
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-production-secret-key'

    # 路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')

    # 文件限制
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json'}

    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_FILE = os.path.join(BASE_DIR, 'logs/app.log')

# 加载环境变量
load_dotenv()


class Config:
    """基础配置类"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    PERMANENT_SESSION_LIFETIME = timedelta(seconds=int(os.environ.get('PERMANENT_SESSION_LIFETIME', 86400)))

    # 文件配置
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 52428800))
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    ALLOWED_EXTENSIONS = set(os.environ.get('ALLOWED_EXTENSIONS', 'csv,xlsx,xls').split(','))

    # 输出目录
    OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'outputs')
    EXCEL_OUTPUT_DIR = os.environ.get('EXCEL_OUTPUT_DIR', 'outputs/excel')
    HTML_OUTPUT_DIR = os.environ.get('HTML_OUTPUT_DIR', 'outputs/html')
    TEMP_DIR = os.environ.get('TEMP_DIR', 'outputs/temp')

    # 日志配置
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'logs/app.log')

    # 媒介分组
    MEDIA_GROUPS = os.environ.get('MEDIA_GROUPS', '家居媒介组,快消媒介组,数码媒介组,素材组,其他组').split(',')

    # 邮件配置
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'True').lower() == 'true'
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER')
    MAIL_RECIPIENTS = os.environ.get('MAIL_RECIPIENTS', '').split(',')

    # 调度配置
    SCHEDULE_DAILY_TIME = os.environ.get('SCHEDULE_DAILY_TIME', '08:00')
    SCHEDULE_WEEKLY_DAY = os.environ.get('SCHEDULE_WEEKLY_DAY', 'Monday')
    SCHEDULE_WEEKLY_TIME = os.environ.get('SCHEDULE_WEEKLY_TIME', '09:00')


class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    FLASK_ENV = 'development'


class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    FLASK_ENV = 'production'


# 配置字典
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': ProductionConfig
}


# 确保目录存在
def create_directories():
    """创建必要的目录"""
    dirs = [
        Config.UPLOAD_FOLDER,
        Config.OUTPUT_DIR,
        Config.EXCEL_OUTPUT_DIR,
        Config.HTML_OUTPUT_DIR,
        Config.TEMP_DIR,
        os.path.dirname(Config.LOG_FILE)
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


# 初始化时创建目录
create_directories()

# ========== 新增：用户权限模块数据库配置（独立于原有系统） ==========

# MySQL数据库连接配置（对应Navicat创建的连接和数据库）
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',  # 你的MySQL用户名
    'password': 'root',  # 替换为你的MySQL密码
    'database': 'ai_media_db',  # 必须与创建的数据库名一致
    'charset': 'utf8mb4'
}
# 原有配置保持不变...
def create_directories():
    """创建必要的目录"""
# 会话配置（用于登录状态保持）
SECRET_KEY = os.getenv('AUTH_SECRET_KEY', 'ai_media_auth_2025_secure')  # 加密会话用
PERMANENT_SESSION_LIFETIME = 3600 * 24  # 会话有效期：24小时