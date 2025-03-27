// chart_utils.js - 图表工具函数

// 生成随机颜色数组
function generateColors(count) {
    const colors = [];
    
    for (let i = 0; i < count; i++) {
        // 生成有足够对比度的颜色
        const hue = i * (360 / count);
        const saturation = 70 + Math.random() * 10; // 70-80%
        const lightness = 50 + Math.random() * 10;  // 50-60%
        
        colors.push(`hsl(${hue}, ${saturation}%, ${lightness}%)`);
    }
    
    return colors;
}

// 创建水平条形图
function createHorizontalBarChart(canvasId, labels, values, title, colorScheme) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // 生成背景色和边框色
    const backgroundColors = colorScheme || generateColors(labels.length);
    const borderColors = backgroundColors.map(color => color);
    
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: title,
                data: values,
                backgroundColor: backgroundColors,
                borderColor: borderColors,
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: title
                }
            }
        }
    });
    
    return chart;
}

// 创建折线图
function createLineChart(canvasId, labels, datasets, title) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // 为每个数据集生成颜色
    const colors = generateColors(datasets.length);
    
    // 格式化数据集
    const formattedDatasets = datasets.map((dataset, index) => {
        return {
            label: dataset.label,
            data: dataset.data,
            fill: false,
            borderColor: colors[index],
            tension: 0.1
        };
    });
    
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: formattedDatasets
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: title
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
    
    return chart;
}

// 创建饼图/环形图
function createPieChart(canvasId, labels, values, title, isDonut = false) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // 生成颜色
    const backgroundColors = generateColors(labels.length);
    
    const chart = new Chart(ctx, {
        type: isDonut ? 'doughnut' : 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: backgroundColors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: title
                },
                legend: {
                    position: 'right'
                }
            }
        }
    });
    
    return chart;
}

// 创建雷达图
function createRadarChart(canvasId, labels, datasets, title) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // 为每个数据集生成颜色
    const colors = generateColors(datasets.length);
    
    // 格式化数据集
    const formattedDatasets = datasets.map((dataset, index) => {
        return {
            label: dataset.label,
            data: dataset.data,
            fill: true,
            backgroundColor: `${colors[index]}40`, // 添加透明度
            borderColor: colors[index],
            pointBackgroundColor: colors[index],
            pointBorderColor: '#fff',
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: colors[index]
        };
    });
    
    const chart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: formattedDatasets
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: title
                }
            }
        }
    });
    
    return chart;
}

// 创建散点图
function createScatterChart(canvasId, datasets, title, xLabel, yLabel) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // 为每个数据集生成颜色
    const colors = generateColors(datasets.length);
    
    // 格式化数据集
    const formattedDatasets = datasets.map((dataset, index) => {
        return {
            label: dataset.label,
            data: dataset.data,
            backgroundColor: colors[index]
        };
    });
    
    const chart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: formattedDatasets
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: title
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: xLabel
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: yLabel
                    }
                }
            }
        }
    });
    
    return chart;
}

// 更新图表数据
function updateChartData(chart, labels, datasets) {
    // 更新标签
    chart.data.labels = labels;
    
    // 更新每个数据集
    datasets.forEach((dataset, index) => {
        if (chart.data.datasets[index]) {
            chart.data.datasets[index].data = dataset.data;
            
            // 更新其他属性（如果提供）
            if (dataset.label) {
                chart.data.datasets[index].label = dataset.label;
            }
        }
    });
    
    // 更新图表
    chart.update();
}

// 为词云数据格式化
function formatWordCloudData(words, counts) {
    if (!words || !counts || words.length !== counts.length) {
        console.error('无效的词云数据');
        return [];
    }
    
    // 返回格式化的数据
    return words.map((word, index) => {
        return {
            text: word,
            value: counts[index]
        };
    });
}

// 销毁图表
function destroyChart(chart) {
    if (chart && typeof chart.destroy === 'function') {
        chart.destroy();
    }
}
