// static/js/charts.js

// 图表相关功能

class ChartManager {
    constructor() {
        this.charts = new Map();
        this.defaultOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.7)',
                    titleFont: {
                        size: 14
                    },
                    bodyFont: {
                        size: 13
                    },
                    padding: 10
                }
            }
        };
    }

    // 创建柱状图
    createBarChart(canvasId, data, options = {}) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        const chartOptions = {
            ...this.defaultOptions,
            ...options,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            }
        };

        const chart = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: data,
            options: chartOptions
        });

        this.charts.set(canvasId, chart);
        return chart;
    }

    // 创建折线图
    createLineChart(canvasId, data, options = {}) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        const chartOptions = {
            ...this.defaultOptions,
            ...options,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            },
            elements: {
                line: {
                    tension: 0.4
                }
            }
        };

        const chart = new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: data,
            options: chartOptions
        });

        this.charts.set(canvasId, chart);
        return chart;
    }

    // 创建饼图
    createPieChart(canvasId, data, options = {}) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        const chartOptions = {
            ...this.defaultOptions,
            ...options,
            plugins: {
                ...this.defaultOptions.plugins,
                legend: {
                    position: 'right'
                }
            }
        };

        const chart = new Chart(ctx.getContext('2d'), {
            type: 'pie',
            data: data,
            options: chartOptions
        });

        this.charts.set(canvasId, chart);
        return chart;
    }

    // 创建雷达图
    createRadarChart(canvasId, data, options = {}) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        const chartOptions = {
            ...this.defaultOptions,
            ...options,
            scales: {
                r: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            }
        };

        const chart = new Chart(ctx.getContext('2d'), {
            type: 'radar',
            data: data,
            options: chartOptions
        });

        this.charts.set(canvasId, chart);
        return chart;
    }

    // 创建散点图
    createScatterChart(canvasId, data, options = {}) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        const chartOptions = {
            ...this.defaultOptions,
            ...options,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                x: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            }
        };

        const chart = new Chart(ctx.getContext('2d'), {
            type: 'scatter',
            data: data,
            options: chartOptions
        });

        this.charts.set(canvasId, chart);
        return chart;
    }

    // 更新图表数据
    updateChart(canvasId, newData) {
        const chart = this.charts.get(canvasId);
        if (!chart) return;

        chart.data = newData;
        chart.update();
    }

    // 更新图表数据集
    updateChartData(canvasId, datasetIndex, newData) {
        const chart = this.charts.get(canvasId);
        if (!chart) return;

        chart.data.datasets[datasetIndex].data = newData;
        chart.update();
    }

    // 添加数据集
    addDataset(canvasId, dataset) {
        const chart = this.charts.get(canvasId);
        if (!chart) return;

        chart.data.datasets.push(dataset);
        chart.update();
    }

    // 移除数据集
    removeDataset(canvasId, datasetIndex) {
        const chart = this.charts.get(canvasId);
        if (!chart) return;

        chart.data.datasets.splice(datasetIndex, 1);
        chart.update();
    }

    // 销毁图表
    destroyChart(canvasId) {
        const chart = this.charts.get(canvasId);
        if (chart) {
            chart.destroy();
            this.charts.delete(canvasId);
        }
    }

    // 销毁所有图表
    destroyAllCharts() {
        this.charts.forEach((chart, canvasId) => {
            chart.destroy();
        });
        this.charts.clear();
    }

    // 调整图表大小
    resizeChart(canvasId) {
        const chart = this.charts.get(canvasId);
        if (chart) {
            chart.resize();
        }
    }

    // 调整所有图表大小
    resizeAllCharts() {
        this.charts.forEach(chart => {
            chart.resize();
        });
    }

    // 获取图表实例
    getChart(canvasId) {
        return this.charts.get(canvasId);
    }
}

// 工作量分析图表
class WorkloadCharts {
    constructor() {
        this.manager = new ChartManager();
    }

    createWorkloadOverview(data) {
        const chartData = {
            labels: data.labels || [],
            datasets: [{
                label: '总处理量',
                data: data.totalProcessing || [],
                backgroundColor: 'rgba(59, 130, 246, 0.8)',
                borderColor: 'rgb(59, 130, 246)',
                borderWidth: 1
            }, {
                label: '定档量',
                data: data.scheduledCount || [],
                backgroundColor: 'rgba(16, 185, 129, 0.8)',
                borderColor: 'rgb(16, 185, 129)',
                borderWidth: 1
            }]
        };

        return this.manager.createBarChart('workloadOverviewChart', chartData, {
            plugins: {
                title: {
                    display: true,
                    text: '工作量概览'
                }
            }
        });
    }

    createSchedulingRateChart(data) {
        const chartData = {
            labels: data.labels || [],
            datasets: [{
                label: '定档率 (%)',
                data: data.schedulingRates || [],
                backgroundColor: 'rgba(245, 158, 11, 0.2)',
                borderColor: 'rgb(245, 158, 11)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        };

        return this.manager.createLineChart('schedulingRateChart', chartData, {
            plugins: {
                title: {
                    display: true,
                    text: '定档率趋势'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        });
    }

    createGradeDistribution(data) {
        const chartData = {
            labels: data.labels || [],
            datasets: [{
                data: data.values || [],
                backgroundColor: [
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(59, 130, 246, 0.8)',
                    'rgba(245, 158, 11, 0.8)'
                ],
                borderColor: [
                    'rgb(239, 68, 68)',
                    'rgb(16, 185, 129)',
                    'rgb(59, 130, 246)',
                    'rgb(245, 158, 11)'
                ],
                borderWidth: 1
            }]
        };

        return this.manager.createPieChart('gradeDistributionChart', chartData, {
            plugins: {
                title: {
                    display: true,
                    text: '媒介综合评估分布'
                }
            }
        });
    }

    createGroupComparison(data) {
        const chartData = {
            labels: data.labels || [],
            datasets: [{
                label: '小组处理量',
                data: data.groupProcessing || [],
                backgroundColor: 'rgba(139, 92, 246, 0.8)',
                borderColor: 'rgb(139, 92, 246)',
                borderWidth: 1
            }]
        };

        return this.manager.createBarChart('groupComparisonChart', chartData, {
            indexAxis: 'y',
            plugins: {
                title: {
                    display: true,
                    text: '各小组工作量对比'
                }
            }
        });
    }
}

// 工作质量分析图表
class QualityCharts {
    constructor() {
        this.manager = new ChartManager();
    }

    createScreeningRateChart(data) {
        const chartData = {
            labels: data.labels || [],
            datasets: [{
                label: '过筛率 (%)',
                data: data.screeningRates || [],
                backgroundColor: 'rgba(16, 185, 129, 0.2)',
                borderColor: 'rgb(16, 185, 129)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        };

        return this.manager.createLineChart('screeningRateChart', chartData, {
            plugins: {
                title: {
                    display: true,
                    text: '过筛率趋势'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        });
    }

    createQualityAssessmentChart(data) {
        const chartData = {
            labels: data.labels || [],
            datasets: [{
                data: data.values || [],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(59, 130, 246, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(249, 115, 22, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderColor: [
                    'rgb(16, 185, 129)',
                    'rgb(59, 130, 246)',
                    'rgb(245, 158, 11)',
                    'rgb(249, 115, 22)',
                    'rgb(239, 68, 68)'
                ],
                borderWidth: 1
            }]
        };

        return this.manager.createPieChart('qualityAssessmentChart', chartData, {
            plugins: {
                title: {
                    display: true,
                    text: '工作质量评估分布'
                }
            }
        });
    }

    createStatusDistribution(data) {
        const chartData = {
            labels: data.labels || [],
            datasets: [{
                label: '状态分布',
                data: data.values || [],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.6)',
                    'rgba(59, 130, 246, 0.6)',
                    'rgba(245, 158, 11, 0.6)',
                    'rgba(139, 92, 246, 0.6)',
                    'rgba(239, 68, 68, 0.6)'
                ],
                borderColor: [
                    'rgb(16, 185, 129)',
                    'rgb(59, 130, 246)',
                    'rgb(245, 158, 11)',
                    'rgb(139, 92, 246)',
                    'rgb(239, 68, 68)'
                ],
                borderWidth: 1
            }]
        };

        return this.manager.createBarChart('statusDistributionChart', chartData, {
            plugins: {
                title: {
                    display: true,
                    text: '媒介工作状态分布'
                }
            }
        });
    }

    createGroupScreeningComparison(data) {
        const chartData = {
            labels: data.labels || [],
            datasets: [{
                label: '小组过筛率',
                data: data.groupScreeningRates || [],
                backgroundColor: 'rgba(59, 130, 246, 0.6)',
                borderColor: 'rgb(59, 130, 246)',
                borderWidth: 1
            }]
        };

        return this.manager.createBarChart('groupScreeningChart', chartData, {
            plugins: {
                title: {
                    display: true,
                    text: '各小组过筛率对比'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        });
    }
}

// 成本分析图表
class CostCharts {
    constructor() {
        this.manager = new ChartManager();
    }

    createCostTrendChart(data) {
        const chartData = {
            labels: data.labels || [],
            datasets: [{
                label: '平均CPM',
                data: data.cpmValues || [],
                backgroundColor: 'rgba(239, 68, 68, 0.2)',
                borderColor: 'rgb(239, 68, 68)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }, {
                label: '平均CPE',
                data: data.cpeValues || [],
                backgroundColor: 'rgba(59, 130, 246, 0.2)',
                borderColor: 'rgb(59, 130, 246)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        };

        return this.manager.createLineChart('costTrendChart', chartData, {
            plugins: {
                title: {
                    display: true,
                    text: '成本趋势分析'
                }
            }
        });
    }

    createRebateDistribution(data) {
        const chartData = {
            labels: data.labels || [],
            datasets: [{
                label: '返点比例分布',
                data: data.values || [],
                backgroundColor: 'rgba(245, 158, 11, 0.6)',
                borderColor: 'rgb(245, 158, 11)',
                borderWidth: 1
            }]
        };

        return this.manager.createBarChart('rebateDistributionChart', chartData, {
            plugins: {
                title: {
                    display: true,
                    text: '返点比例分布'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        });
    }

    createCostStructure(data) {
        const chartData = {
            labels: data.labels || [],
            datasets: [{
                data: data.values || [],
                backgroundColor: [
                    'rgba(59, 130, 246, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(139, 92, 246, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderColor: [
                    'rgb(59, 130, 246)',
                    'rgb(16, 185, 129)',
                    'rgb(245, 158, 11)',
                    'rgb(139, 92, 246)',
                    'rgb(239, 68, 68)'
                ],
                borderWidth: 1
            }]
        };

        return this.manager.createPieChart('costStructureChart', chartData, {
            plugins: {
                title: {
                    display: true,
                    text: '成本结构分布'
                }
            }
        });
    }

    createCostEfficiencyMatrix(data) {
        const chartData = {
            datasets: [{
                label: '媒介成本效益',
                data: data.points || [],
                backgroundColor: data.colors || 'rgba(59, 130, 246, 0.6)',
                borderColor: 'rgb(59, 130, 246)',
                borderWidth: 1,
                pointRadius: 8
            }]
        };

        return this.manager.createScatterChart('costEfficiencyMatrix', chartData, {
            plugins: {
                title: {
                    display: true,
                    text: '成本效益矩阵'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const point = context.raw;
                            return `${point.label}: CPM=${point.x}, CPE=${point.y}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'CPE (互动成本)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'CPM (千次曝光成本)'
                    }
                }
            }
        });
    }

    createGroupCostComparison(data) {
        const chartData = {
            labels: data.labels || [],
            datasets: [{
                label: '平均CPM',
                data: data.cpmValues || [],
                backgroundColor: 'rgba(239, 68, 68, 0.6)',
                borderColor: 'rgb(239, 68, 68)',
                borderWidth: 1
            }, {
                label: '平均CPE',
                data: data.cpeValues || [],
                backgroundColor: 'rgba(59, 130, 246, 0.6)',
                borderColor: 'rgb(59, 130, 246)',
                borderWidth: 1
            }]
        };

        return this.manager.createBarChart('groupCostChart', chartData, {
            plugins: {
                title: {
                    display: true,
                    text: '各小组成本指标对比'
                }
            }
        });
    }
}

// 全局图表管理器实例
const chartManager = new ChartManager();
const workloadCharts = new WorkloadCharts();
const qualityCharts = new QualityCharts();
const costCharts = new CostCharts();

// 初始化图表
function initializeCharts() {
    // 监听窗口大小变化，调整图表
    window.addEventListener('resize', debounce(() => {
        chartManager.resizeAllCharts();
    }, 250));

    console.log('图表系统初始化完成');
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', initializeCharts);

// 全局导出
window.Charts = {
    manager: chartManager,
    workload: workloadCharts,
    quality: qualityCharts,
    cost: costCharts
};