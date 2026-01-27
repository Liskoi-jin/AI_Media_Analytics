# auth/utils.py
from functools import wraps
from flask import redirect, url_for, session, flash, request
from auth.models import User

def login_required(f):
    """登录验证装饰器：未登录用户跳转到登录页"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 检查session中是否有登录标识
        if not session.get('user_id'):
            flash('请先登录后再操作', 'warning')
            return redirect(url_for('auth.login', next=request.url))  # 记录跳转前页面
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """管理员权限装饰器：仅管理员可访问"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user_id'):
            flash('请先登录后再操作', 'warning')
            return redirect(url_for('auth.login'))
        # 获取当前登录用户
        user = User.query.get(session['user_id'])
        if not user or not user.is_admin():
            flash('无权限访问该页面', 'danger')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

def get_current_user():
    """获取当前登录用户信息"""
    if session.get('user_id'):
        return User.query.get(session['user_id'])
    return None