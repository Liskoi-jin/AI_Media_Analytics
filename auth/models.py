# auth/models.py - 修复版本
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import generate_password_hash, check_password_hash
from datetime import datetime
import pymysql
from config import DB_CONFIG

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = 'sys_user'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True, autoincrement=True, comment='用户ID')
    username = db.Column(db.String(50), unique=True, nullable=False, comment='登录账号')

    # 将列名改为 _password_hash，用于存储加密后的密码
    _password_hash = db.Column('password', db.String(100), nullable=False, comment='加密密码')

    full_name = db.Column(db.String(50), default='', comment='用户姓名')
    email = db.Column(db.String(100), default='', comment='邮箱')
    phone = db.Column(db.String(20), default='', comment='手机号')
    role = db.Column(db.String(20), default='user', comment='角色')
    status = db.Column(db.SmallInteger, default=1, comment='状态：1启用/0禁用')
    create_time = db.Column(db.DateTime, default=datetime.now, comment='创建时间')
    update_time = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now, comment='更新时间')

    def __init__(self, username, password, full_name='', email='', phone='', role='user', status=1):
        self.username = username
        self.password = password  # 这会调用 setter 方法自动加密
        self.full_name = full_name
        self.email = email
        self.phone = phone
        self.role = role
        self.status = status

    @property
    def password(self):
        """密码不可读"""
        raise AttributeError('密码不可读')

    @password.setter
    def password(self, plain_password):
        """设置密码时自动加密"""
        self._password_hash = generate_password_hash(plain_password).decode('utf-8')

    def check_password(self, plain_password):
        """验证密码是否正确"""
        try:
            return check_password_hash(self._password_hash, plain_password)
        except Exception as e:
            print(f"密码验证错误: {e}, 用户: {self.username}")
            print(f"哈希值: {self._password_hash[:50] if self._password_hash else 'None'}")
            return False

    def is_admin(self):
        """判断是否为管理员"""
        return self.role == 'admin'

    def is_active(self):
        """判断账号是否启用"""
        return self.status == 1

    def __repr__(self):
        return f'<User {self.username}>'

# 初始化数据库（创建表结构，若表已存在则不执行）
def init_db(app):
    """将数据库模型与Flask应用绑定，初始化表结构"""
    db.init_app(app)
    # 绑定应用上下文，创建表（仅当表不存在时创建）
    with app.app_context():
        db.create_all()  # 不会覆盖已有表，安全执行
        print("数据库初始化完成：表结构已创建（若不存在）")