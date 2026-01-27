# src/__init__.py
"""
AI媒介自动化审计分析系统
一站式媒介工作质量、工作量和成本效益分析平台
整合了Media_Analysis.py和成本发挥分析.py的完整逻辑
"""

__version__ = '2.0.0'
__author__ = 'Media Analytics Team'

# 导出核心模块
from src.data_processor import DataProcessor
from src.analyzer import MediaAnalyzer
from src.report_generator import ReportGenerator
from src.visualizations import VisualizationGenerator
from src.workload_analyzer import WorkloadAnalyzer
from src.quality_analyzer import QualityAnalyzer
from src.cost_analyzer import CostAnalyzer
from src.utils import *
# 新增：导出数据库工具函数
from src.db_utils import create_db_connection, query_workload_data, query_quality_data, query_cost_data


# 初始化日志
from src.utils import setup_logger, logger

# 确保所有必要的目录都已创建
from config import create_directories
create_directories()

logger.info(f"媒介自动化审计分析系统 v{__version__} 初始化完成")
logger.info("已整合Media_Analysis.py和成本发挥分析.py的完整逻辑")