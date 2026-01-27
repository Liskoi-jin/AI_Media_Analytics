# auth/views.py
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from flask_login import login_user, logout_user, login_required as flask_login_required
from auth.models import db, User
from auth.utils import login_required, admin_required, get_current_user
from config import SECRET_KEY

# 创建独立蓝图（与原有系统路由隔离）
auth_bp = Blueprint('auth', __name__, url_prefix='/auth', template_folder='../templates/auth')


# 登录页面
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """登录视图：GET请求返回登录页，POST请求验证登录"""
    if request.method == 'POST':
        # 获取表单数据
        username = request.form.get('username').strip()
        password = request.form.get('password').strip()

        # 验证表单
        if not username or not password:
            flash('账号和密码不能为空', 'danger')
            return render_template('auth/login.html', username=username)

        # 查询用户
        user = User.query.filter_by(username=username).first()

        # 验证用户是否存在、状态是否启用、密码是否正确
        if not user:
            flash('账号不存在', 'danger')
            return render_template('auth/login.html', username=username)
        if not user.is_active():
            flash('账号已被禁用，请联系管理员', 'danger')
            return render_template('auth/login.html', username=username)
        if not user.check_password(password):
            flash('密码错误', 'danger')
            return render_template('auth/login.html', username=username)

        # 登录成功：设置session（与原有系统隔离）
        session['user_id'] = user.id
        session['username'] = user.username
        session['role'] = user.role

        # 跳转到之前访问的页面，若无则跳转到系统首页
        next_page = request.args.get('next', url_for('data_source_selector'))  # index是原有系统首页路由
        flash(f'登录成功，欢迎回来，{user.full_name or user.username}', 'success')
        return redirect(next_page)

    # GET请求：返回登录页面
    return render_template('auth/login.html')


# 退出登录
@auth_bp.route('/logout')
def logout():
    """退出登录：清除session"""
    session.clear()  # 仅清除权限模块的session，不影响原有系统
    flash('已成功退出登录', 'info')
    return redirect(url_for('auth.login'))


# 账号管理页面（仅管理员可访问）
@auth_bp.route('/user-manage')
@admin_required  # 管理员权限验证
def user_manage():
    """账号列表管理"""
    users = User.query.order_by(User.create_time.desc()).all()
    return render_template('auth/user_manage.html', users=users)


# 开通新账号（仅管理员可访问）
@auth_bp.route('/add-user', methods=['GET', 'POST'])
@admin_required
def add_user():
    """新增用户账号"""
    if request.method == 'POST':
        # 获取表单数据
        username = request.form.get('username').strip()
        password = request.form.get('password').strip()
        full_name = request.form.get('full_name').strip()
        email = request.form.get('email').strip()
        phone = request.form.get('phone').strip()
        role = request.form.get('role', 'user')
        status = 1 if request.form.get('status') == 'on' else 0

        # 验证表单
        if not username or not password:
            flash('账号和密码不能为空', 'danger')
            return render_template('auth/add_user.html',
                                   username=username, full_name=full_name,
                                   email=email, phone=phone, role=role)

        # 检查账号是否已存在
        if User.query.filter_by(username=username).first():
            flash('该账号已存在，请更换账号名', 'danger')
            return render_template('auth/add_user.html',
                                   username=username, full_name=full_name,
                                   email=email, phone=phone, role=role)

        # 创建新用户
        try:
            new_user = User(
                username=username,
                password=password,
                full_name=full_name,
                email=email,
                phone=phone,
                role=role,
                status=status
            )
            db.session.add(new_user)
            db.session.commit()
            flash('账号开通成功', 'success')
            return redirect(url_for('auth.user_manage'))
        except Exception as e:
            db.session.rollback()
            flash(f'账号开通失败：{str(e)}', 'danger')
            return render_template('auth/add_user.html',
                                   username=username, full_name=full_name,
                                   email=email, phone=phone, role=role)

    # GET请求：返回新增账号页面
    return render_template('auth/add_user.html')


# 编辑用户（仅管理员可访问）
@auth_bp.route('/edit-user/<int:user_id>', methods=['GET', 'POST'])
@admin_required
def edit_user(user_id):
    """编辑已有用户"""
    user = User.query.get_or_404(user_id)

    if request.method == 'POST':
        # 获取表单数据
        full_name = request.form.get('full_name').strip()
        email = request.form.get('email').strip()
        phone = request.form.get('phone').strip()
        role = request.form.get('role', 'user')
        status = 1 if request.form.get('status') == 'on' else 0
        new_password = request.form.get('new_password').strip()

        # 更新用户信息
        try:
            user.full_name = full_name
            user.email = email
            user.phone = phone
            user.role = role
            user.status = status

            # 若填写了新密码，则更新密码
            if new_password:
                user.password = user._hash_password(new_password)

            db.session.commit()
            flash('用户信息更新成功', 'success')
            return redirect(url_for('auth.user_manage'))
        except Exception as e:
            db.session.rollback()
            flash(f'用户信息更新失败：{str(e)}', 'danger')
            return render_template('auth/edit_user.html', user=user)

    # GET请求：返回编辑页面
    return render_template('auth/edit_user.html', user=user)


# 删除用户（仅管理员可访问）
@auth_bp.route('/delete-user/<int:user_id>', methods=['POST'])
@admin_required
def delete_user(user_id):
    """删除用户（谨慎操作）"""
    user = User.query.get_or_404(user_id)

    # 禁止删除管理员账号
    if user.is_admin():
        flash('管理员账号不能删除', 'danger')
        return redirect(url_for('auth.user_manage'))

    try:
        db.session.delete(user)
        db.session.commit()
        flash('用户删除成功', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'用户删除失败：{str(e)}', 'danger')

    return redirect(url_for('auth.user_manage'))