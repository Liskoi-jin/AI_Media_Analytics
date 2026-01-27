# auth/__init__.py
from auth.views import auth_bp
from auth.models import init_db

# 对外暴露蓝图和初始化函数
__all__ = ['auth_bp', 'init_db']