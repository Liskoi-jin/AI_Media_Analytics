# reset_admin_password_fixed.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from auth.models import db, User
from flask_bcrypt import generate_password_hash


def reset_admin_password():
    # 配置数据库
    DB_CONFIG = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': 'root',
        'database': 'ai_media_db',
        'charset': 'utf8mb4'
    }

    # 创建Flask应用
    app = Flask(__name__)
    app.config[
        'SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?charset={DB_CONFIG['charset']}"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'ai_media_auth_2025_secure'

    db.init_app(app)

    with app.app_context():
        print("🔧 重置管理员密码")
        print("=" * 60)

        # 查找或创建管理员
        admin = User.query.filter_by(username='admin').first()

        if not admin:
            print("创建管理员账户...")
            admin = User(
                username='admin',
                password='Admin@2024',  # 这会自动加密
                full_name='系统管理员',
                email='admin@example.com',
                role='admin',
                status=1
            )
            db.session.add(admin)
            action = "创建"
        else:
            print("重置管理员密码...")
            print(f"当前密码哈希: {admin._password_hash[:30] if admin._password_hash else 'None'}...")

            # 询问操作
            print("\n请选择操作:")
            print("1. 重置密码为 'Admin@2024'")
            print("2. 自定义新密码")
            print("3. 查看当前信息")

            choice = input("\n请输入选择 (1/2/3): ").strip()

            if choice == '1':
                new_password = 'Admin@2024'
            elif choice == '2':
                new_password = input("请输入新密码: ").strip()
                if not new_password:
                    print("❌ 密码不能为空")
                    return
                if len(new_password) < 6:
                    print("⚠️ 密码太短，建议使用至少8位包含字母和数字的密码")
                    confirm = input("确认使用此密码? (y/n): ").strip().lower()
                    if confirm != 'y':
                        return
            else:
                # 查看信息
                print(f"\n管理员信息:")
                print(f"用户名: {admin.username}")
                print(f"姓名: {admin.full_name}")
                print(f"邮箱: {admin.email}")
                print(f"角色: {admin.role}")
                print(f"状态: {'启用' if admin.status == 1 else '禁用'}")
                print(f"创建时间: {admin.create_time}")

                # 直接从数据库获取哈希
                import pymysql
                conn = pymysql.connect(**DB_CONFIG)
                with conn.cursor() as cursor:
                    cursor.execute("SELECT password FROM sys_user WHERE username = 'admin'")
                    db_hash = cursor.fetchone()[0]
                    print(f"密码哈希: {db_hash[:50]}...")
                    print(f"哈希长度: {len(db_hash)}")
                    print(f"是否是 bcrypt: {db_hash.startswith('$2')}")
                conn.close()
                return

            # 设置新密码
            admin.password = new_password
            action = "重置"

        try:
            db.session.commit()

            # 从数据库直接获取哈希
            import pymysql
            conn = pymysql.connect(**DB_CONFIG)
            with conn.cursor() as cursor:
                cursor.execute("SELECT password FROM sys_user WHERE username = 'admin'")
                db_hash = cursor.fetchone()[0]
            conn.close()

            print(f"\n✅ {action}成功！")
            print(f"用户名: admin")
            print(f"密码: {new_password if 'new_password' in locals() else 'Admin@2024'}")
            print(f"密码哈希: {db_hash[:30]}...")
            print(f"哈希长度: {len(db_hash)}")
            print(f"是否是 bcrypt: {db_hash.startswith('$2')}")

            # 验证密码
            print(f"密码验证测试: {admin.check_password(new_password if 'new_password' in locals() else 'Admin@2024')}")

        except Exception as e:
            db.session.rollback()
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    reset_admin_password()