import os
from datetime import timedelta
from dotenv import load_dotenv

# 加载环境变量（移到顶部，确保所有配置都能读取）
load_dotenv()


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


class BaseConfig(Config):
    """基础配置类（整合原有Config，避免重复）"""
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


class DevelopmentConfig(BaseConfig):
    """开发环境配置"""
    DEBUG = True
    FLASK_ENV = 'development'


class ProductionConfig(BaseConfig):
    """生产环境配置"""
    DEBUG = False
    FLASK_ENV = 'production'


# 配置字典
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': ProductionConfig
}


# 确保目录存在（修复原有重复定义问题，保留一份即可）
def create_directories():
    """创建必要的目录"""
    # 整合所有需要创建的目录
    dirs = [
        BaseConfig.UPLOAD_FOLDER,
        BaseConfig.OUTPUT_DIR,
        BaseConfig.EXCEL_OUTPUT_DIR,
        BaseConfig.HTML_OUTPUT_DIR,
        BaseConfig.TEMP_DIR,
        os.path.dirname(BaseConfig.LOG_FILE)
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


# 初始化时创建目录
create_directories()

# ========== 核心修改：替换为阿里云RDS数据库配置 ==========
# 替换下面5个参数为你的阿里云RDS信息，其他不动！
DB_CONFIG = {
    'host': 'rm-cn-2104msjne000170o.rwlb.rds.aliyuncs.com',  # 例：rm-cn-2104msjne00017.mysql.rds.aliyuncs.com
    'port': 3306,                    # 固定3306，不用改
    'user': 'root',                  # 你创建的阿里云高权限账号
    'password': 'Lj041213',          # 你设置的阿里云数据库密码
    'database': 'ai_media_db',       # 刚创建的阿里云数据库名
    'charset': 'utf8mb4'             # 固定utf8mb4，不用改
}

# 会话配置（用于登录状态保持）
SECRET_KEY = os.getenv('AUTH_SECRET_KEY', 'ai_media_auth_2025_secure')  # 加密会话用
PERMANENT_SESSION_LIFETIME = 3600 * 24  # 会话有效期：24小时
