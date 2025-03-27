// main.js - 主要的JavaScript功能

document.addEventListener("DOMContentLoaded", function() {
    // 初始化事件监听器
    initEventListeners();
    
    // 如果在历史页面，加载历史数据
    if (document.getElementById('history-table')) {
        loadHistory();
    }
    
    // 如果在特征页面，加载词云数据
    if (document.getElementById('wordcloud-container')) {
        loadWordCloudData();
    }
    
    // 如果在主页面，初始化模型指标图表
    if (document.getElementById('model-metrics-chart')) {
        loadModelMetrics();
    }
    
    // 如果在主页面，初始化漂移图表
    if (document.getElementById('drift-chart')) {
        initDriftChart();
    }
});

// 初始化所有事件监听器
function initEventListeners() {
    // 预测表单提交
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePredictionSubmit);
    }
    
    // 文件上传表单
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFileUpload);
    }
    
    // 历史记录搜索
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        searchInput.addEventListener('input', filterHistory);
    }
    
    // 模式切换
    const themeSwitcher = document.getElementById('theme-switcher');
    if (themeSwitcher) {
        themeSwitcher.addEventListener('click', toggleTheme);
        // 根据存储的主题设置初始状态，默认为light
        const currentTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-bs-theme', currentTheme);
        themeSwitcher.textContent = currentTheme === 'dark' ? '🌙' : '☀️';
        
        // 更新导航栏样式
        const navbar = document.querySelector('.navbar');
        if (navbar) {
            if (currentTheme === 'dark') {
                navbar.classList.remove('navbar-light', 'bg-light');
                navbar.classList.add('navbar-dark', 'bg-dark');
            } else {
                navbar.classList.remove('navbar-dark', 'bg-dark');
                navbar.classList.add('navbar-light', 'bg-light');
            }
        }
    }
}

// 处理预测表单提交
async function handlePredictionSubmit(event) {
    event.preventDefault();
    
    const form = event.target;
    const submitBtn = form.querySelector('button[type="submit"]');
    const resultContainer = document.getElementById('prediction-result');
    const loadingSpinner = document.getElementById('loading-spinner');
    
    // 显示加载动画
    loadingSpinner.classList.remove('d-none');
    submitBtn.disabled = true;
    resultContainer.classList.add('d-none');
    
    try {
        const formData = new FormData(form);
        
        // 发送请求到后端API
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || '预测请求失败');
        }
        
        const data = await response.json();
        
        // 处理结果
        displayPredictionResult(data);
        
        // 更新漂移图表
        updateDriftChart();
        
    } catch (error) {
        console.error('预测错误:', error);
        
        // 显示错误消息
        resultContainer.innerHTML = `
            <div class="alert alert-danger">
                <strong>错误:</strong> ${error.message}
            </div>
        `;
        resultContainer.classList.remove('d-none');
        
    } finally {
        // 隐藏加载动画，恢复提交按钮
        loadingSpinner.classList.add('d-none');
        submitBtn.disabled = false;
    }
}

// 显示预测结果
function displayPredictionResult(data) {
    const resultContainer = document.getElementById('prediction-result');
    
    // 设置结果类型（垃圾短信或正常短信）
    const resultType = data.prediction === '垃圾短信' ? 'danger' : 'success';
    const resultIcon = data.prediction === '垃圾短信' ? 
        '<i class="fas fa-exclamation-triangle"></i>' : 
        '<i class="fas fa-check-circle"></i>';
    
    // 设置置信度等级
    let confidenceLevel = '低';
    if (data.confidence > 0.8) confidenceLevel = '高';
    else if (data.confidence > 0.6) confidenceLevel = '中';
    
    // 格式化置信度和预测时间
    const confidencePercent = (data.confidence * 100).toFixed(2);
    const predTime = data.prediction_time.toFixed(3);
    
    // 构建结果HTML
    resultContainer.innerHTML = `
        <div class="card border-${resultType} mb-3">
            <div class="card-header bg-${resultType} text-white">
                ${resultIcon} 预测结果: <strong>${data.prediction}</strong>
            </div>
            <div class="card-body">
                <p class="card-text"><strong>输入文本:</strong> ${data.input_text}</p>
                <p class="card-text"><strong>置信度:</strong> ${confidencePercent}% (${confidenceLevel})</p>
                <p class="card-text"><strong>预测耗时:</strong> ${predTime} 秒</p>
            </div>
        </div>
    `;
    
    resultContainer.classList.remove('d-none');
}

// 处理文件上传
async function handleFileUpload(event) {
    event.preventDefault();
    
    const form = event.target;
    const submitBtn = form.querySelector('button[type="submit"]');
    const resultContainer = document.getElementById('upload-result');
    const loadingSpinner = document.getElementById('upload-spinner');
    
    // 显示加载动画
    loadingSpinner.classList.remove('d-none');
    submitBtn.disabled = true;
    resultContainer.classList.add('d-none');
    
    try {
        const formData = new FormData(form);
        
        // 发送请求到后端API
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('文件上传失败');
        }
        
        const data = await response.json();
        
        // 显示上传结果
        resultContainer.innerHTML = `
            <div class="alert alert-success">
                <strong>成功!</strong> 已处理 ${data.results.length} 条短信
            </div>
        `;
        
        // 显示处理的结果摘要
        if (data.results && data.results.length > 0) {
            let summaryHtml = '<div class="mt-3"><h5>处理结果摘要:</h5>';
            
            // 计算统计信息
            const spamCount = data.results.filter(r => r.prediction === '垃圾短信').length;
            const hamCount = data.results.length - spamCount;
            
            // 添加统计信息
            summaryHtml += `
                <p>共 ${data.results.length} 条短信:</p>
                <ul>
                    <li>垃圾短信: ${spamCount} 条 (${((spamCount / data.results.length) * 100).toFixed(1)}%)</li>
                    <li>正常短信: ${hamCount} 条 (${((hamCount / data.results.length) * 100).toFixed(1)}%)</li>
                </ul>
            `;
            
            // 添加前5条结果
            summaryHtml += '<h6>前5条预测结果:</h6><ul class="list-group">';
            
            for (let i = 0; i < Math.min(5, data.results.length); i++) {
                const result = data.results[i];
                const itemClass = result.prediction === '垃圾短信' ? 'list-group-item-danger' : 'list-group-item-success';
                
                summaryHtml += `
                    <li class="list-group-item ${itemClass}">
                        <strong>${result.prediction}</strong> (${(result.confidence * 100).toFixed(1)}%): ${result.text.substring(0, 50)}${result.text.length > 50 ? '...' : ''}
                    </li>
                `;
            }
            
            summaryHtml += '</ul></div>';
            resultContainer.innerHTML += summaryHtml;
        }
        
        resultContainer.classList.remove('d-none');
        
    } catch (error) {
        console.error('上传错误:', error);
        
        // 显示错误消息
        resultContainer.innerHTML = `
            <div class="alert alert-danger">
                <strong>错误:</strong> ${error.message}
            </div>
        `;
        resultContainer.classList.remove('d-none');
        
    } finally {
        // 隐藏加载动画，恢复提交按钮
        loadingSpinner.classList.add('d-none');
        submitBtn.disabled = false;
    }
}

// 加载预测历史数据
async function loadHistory() {
    const historyTable = document.getElementById('history-table');
    const tableBody = historyTable.querySelector('tbody');
    const loadingSpinner = document.getElementById('history-spinner');
    
    // 显示加载动画
    loadingSpinner.classList.remove('d-none');
    
    try {
        // 发送请求到后端API
        const response = await fetch('/get_history');
        
        if (!response.ok) {
            throw new Error('获取历史记录失败');
        }
        
        const data = await response.json();
        
        // 清空表格
        tableBody.innerHTML = '';
        
        // 填充表格
        if (data.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center">暂无记录</td>
                </tr>
            `;
        } else {
            data.forEach((item, index) => {
                const row = document.createElement('tr');
                
                // 设置行的类，根据预测结果上色
                row.className = item.prediction === '垃圾短信' ? 'table-danger' : 'table-success';
                
                // 设置行内容
                row.innerHTML = `
                    <td>${index + 1}</td>
                    <td>${item.text.substring(0, 30)}${item.text.length > 30 ? '...' : ''}</td>
                    <td>${item.send_freq}</td>
                    <td>${item.is_night}</td>
                    <td>${item.prediction}</td>
                    <td>${(item.confidence * 100).toFixed(1)}%</td>
                    <td>${item.model_type}</td>
                    <td>${item.timestamp}</td>
                `;
                
                tableBody.appendChild(row);
            });
        }
        
    } catch (error) {
        console.error('加载历史记录错误:', error);
        
        // 显示错误消息
        tableBody.innerHTML = `
            <tr>
                <td colspan="7" class="text-center text-danger">
                    加载失败: ${error.message}
                </td>
            </tr>
        `;
        
    } finally {
        // 隐藏加载动画
        loadingSpinner.classList.add('d-none');
    }
}

// 筛选历史记录
function filterHistory() {
    const searchInput = document.getElementById('search-input');
    const searchText = searchInput.value.toLowerCase();
    const historyTable = document.getElementById('history-table');
    const rows = historyTable.querySelectorAll('tbody tr');
    
    // 遍历所有行
    rows.forEach(row => {
        const text = row.cells[1].textContent.toLowerCase();
        const prediction = row.cells[4].textContent.toLowerCase();
        const model = row.cells[6].textContent.toLowerCase();
        
        // 如果任何一个字段包含搜索文本，显示该行
        if (text.includes(searchText) || prediction.includes(searchText) || model.includes(searchText)) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
}

// 加载词云数据
async function loadWordCloudData() {
    const spamContainer = document.getElementById('spam-wordcloud');
    const hamContainer = document.getElementById('ham-wordcloud');
    const loadingSpinner = document.getElementById('wordcloud-spinner');
    
    // 显示加载动画
    loadingSpinner.classList.remove('d-none');
    
    try {
        // 发送请求到后端API
        const response = await fetch('/get_features');
        
        if (!response.ok) {
            throw new Error('获取词云数据失败');
        }
        
        const data = await response.json();
        
        // 渲染词云
        if (data.spam_words && data.spam_words.length > 0) {
            renderWordCloud(spamContainer, data.spam_words, '#dc3545');
        } else {
            spamContainer.innerHTML = '<div class="alert alert-info">暂无足够数据生成垃圾短信词云</div>';
        }
        
        if (data.ham_words && data.ham_words.length > 0) {
            renderWordCloud(hamContainer, data.ham_words, '#28a745');
        } else {
            hamContainer.innerHTML = '<div class="alert alert-info">暂无足够数据生成正常短信词云</div>';
        }
        
    } catch (error) {
        console.error('加载词云数据错误:', error);
        
        // 显示错误消息
        spamContainer.innerHTML = `<div class="alert alert-danger">加载垃圾短信词云失败: ${error.message}</div>`;
        hamContainer.innerHTML = `<div class="alert alert-danger">加载正常短信词云失败: ${error.message}</div>`;
        
    } finally {
        // 隐藏加载动画
        loadingSpinner.classList.add('d-none');
    }
}

// 渲染词云
function renderWordCloud(container, words, color) {
    // 词云配置
    const options = {
        list: words,
        fontFamily: 'Pingfang SC, Source Sans Pro, Microsoft Yahei',
        fontWeight: 'bold',
        color: color,
        minSize: 12,
        weightFactor: 2,
        backgroundColor: 'transparent',
        gridSize: 8,
        drawOutOfBound: false,
        hover: function(item, dimension) {
            container.querySelector('.word-info').textContent = `"${item[0]}" 出现 ${item[1]} 次`;
        }
    };
    
    // 清空容器
    container.innerHTML = '';
    
    // 添加词云标题和信息显示区域
    container.innerHTML = '<div class="word-info text-center mb-2">&nbsp;</div>';
    
    // 创建词云canvas
    const canvas = document.createElement('canvas');
    canvas.width = container.offsetWidth;
    canvas.height = 300;
    container.appendChild(canvas);
    
    // 渲染词云
    WordCloud(canvas, options);
}

// 加载模型指标数据
async function loadModelMetrics() {
    const metricsContainer = document.getElementById('model-metrics-chart');
    const loadingSpinner = document.getElementById('metrics-spinner');
    
    // 显示加载动画
    loadingSpinner.classList.remove('d-none');
    
    try {
        // 发送请求到后端API
        const response = await fetch('/get_model_metrics');
        
        if (!response.ok) {
            throw new Error('获取模型指标失败');
        }
        
        const data = await response.json();
        
        // 渲染模型指标图表
        renderModelMetricsChart(metricsContainer, data);
        
    } catch (error) {
        console.error('加载模型指标错误:', error);
        
        // 显示错误消息
        metricsContainer.innerHTML = `
            <div class="alert alert-danger">
                加载模型指标失败: ${error.message}
            </div>
        `;
        
    } finally {
        // 隐藏加载动画
        loadingSpinner.classList.add('d-none');
    }
}

// 渲染模型指标图表
function renderModelMetricsChart(container, data) {
    // 提取数据
    const models = Object.keys(data);
    const metrics = ['accuracy', 'precision', 'recall', 'f1_score'];
    const metricNames = ['准确率', '精确率', '召回率', 'F1分数'];
    
    // 设置图表数据
    const chartData = {
        labels: models,
        datasets: metrics.map((metric, index) => {
            return {
                label: metricNames[index],
                data: models.map(model => data[model][metric]),
                backgroundColor: [
                    'rgba(255, 99, 132, 0.7)',
                    'rgba(54, 162, 235, 0.7)',
                    'rgba(255, 206, 86, 0.7)',
                    'rgba(75, 192, 192, 0.7)'
                ][index],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)'
                ][index],
                borderWidth: 1
            };
        })
    };
    
    // 创建canvas元素
    container.innerHTML = '';
    const canvas = document.createElement('canvas');
    container.appendChild(canvas);
    
    // 渲染图表
    new Chart(canvas, {
        type: 'bar',
        data: chartData,
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: '各模型性能指标对比'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${(context.raw * 100).toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    }
                }
            }
        }
    });
    
    // 创建模型数据表格
    const tableContainer = document.createElement('div');
    tableContainer.className = 'mt-4';
    tableContainer.innerHTML = `
        <h5 class="text-center">模型性能数据表</h5>
        <div class="table-responsive">
            <table class="table table-bordered table-hover">
                <thead class="table-dark">
                    <tr>
                        <th>模型</th>
                        <th>准确率</th>
                        <th>精确率</th>
                        <th>召回率</th>
                        <th>F1分数</th>
                        <th>样本数量</th>
                    </tr>
                </thead>
                <tbody>
                    ${models.map(model => `
                        <tr>
                            <td>${model}</td>
                            <td>${(data[model].accuracy * 100).toFixed(1)}%</td>
                            <td>${(data[model].precision * 100).toFixed(1)}%</td>
                            <td>${(data[model].recall * 100).toFixed(1)}%</td>
                            <td>${(data[model].f1_score * 100).toFixed(1)}%</td>
                            <td>${data[model].count}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
    container.appendChild(tableContainer);
}

// 初始化漂移图表
function initDriftChart() {
    const driftContainer = document.getElementById('drift-chart');
    const canvas = document.createElement('canvas');
    driftContainer.appendChild(canvas);
    
    // 初始数据
    const chartData = {
        labels: [],
        datasets: [{
            label: '语义漂移值',
            data: [],
            fill: false,
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1
        }]
    };
    
    // 创建图表
    const driftChart = new Chart(canvas, {
        type: 'line',
        data: chartData,
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: '短信语义漂移监测'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `漂移值: ${context.raw.toFixed(3)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: '漂移强度'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '时间'
                    }
                }
            }
        }
    });
    
    // 将图表对象存储为全局变量
    window.driftChart = driftChart;
    
    // 立即更新一次漂移图表
    updateDriftChart();
}

// 更新漂移图表
async function updateDriftChart() {
    if (!window.driftChart) return;
    
    try {
        // 获取当前选择的模型类型
        const modelSelect = document.getElementById('model-select');
        const modelType = modelSelect ? modelSelect.value : 'roberta';
        
        // 发送请求到后端API，包含当前选择的模型类型
        const response = await fetch(`/track_drift?model_type=${modelType}`);
        
        if (!response.ok) {
            console.error('获取漂移数据失败');
            return;
        }
        
        const data = await response.json();
        const chart = window.driftChart;
        
        // 更新图表数据
        chart.data.labels.push(data.timestamp);
        chart.data.datasets[0].data.push(data.drift_value);
        
        // 保持最多显示10个点
        if (chart.data.labels.length > 10) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        
        // 更新图表
        chart.update();
        
        // 更新漂移警告
        updateDriftWarning(data.drift_value, data.is_adapted, data.model_type);
        
    } catch (error) {
        console.error('更新漂移图表错误:', error);
    }
}

// 更新漂移警告
function updateDriftWarning(driftValue, isAdapted, modelType) {
    const warningContainer = document.getElementById('drift-warning');
    if (!warningContainer) return;
    
    // 清空容器
    warningContainer.innerHTML = '';
    
    // 微调信息
    const adaptationInfo = isAdapted 
        ? `<div class="mt-2 alert alert-info">
            <i class="fas fa-sync-alt"></i> 
            <strong>模型已自动微调!</strong> ${modelType} 模型已基于最新数据进行了自动微调。
           </div>`
        : '';
    
    // 根据漂移值显示不同警告
    if (driftValue > 0.5) {
        warningContainer.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle"></i> 
                <strong>高漂移警告!</strong> 当前漂移值: ${driftValue.toFixed(3)}
                <p class="mb-0">检测到显著的语义漂移，系统将尝试自动微调模型。</p>
            </div>
            ${adaptationInfo}
        `;
    } else if (driftValue > 0.3) {
        warningContainer.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-circle"></i>
                <strong>中等漂移!</strong> 当前漂移值: ${driftValue.toFixed(3)}
                <p class="mb-0">检测到中等程度的语义漂移，建议关注模型性能。</p>
            </div>
            ${adaptationInfo}
        `;
    } else {
        warningContainer.innerHTML = `
            <div class="alert alert-success">
                <i class="fas fa-check-circle"></i>
                <strong>稳定!</strong> 当前漂移值: ${driftValue.toFixed(3)}
                <p class="mb-0">未检测到明显语义漂移，模型表现稳定。</p>
            </div>
            ${adaptationInfo}
        `;
    }
}

// 切换明/暗模式
function toggleTheme() {
    const themeSwitcher = document.getElementById('theme-switcher');
    const currentTheme = document.documentElement.getAttribute('data-bs-theme');
    const navbar = document.querySelector('.navbar');
    
    if (currentTheme === 'dark') {
        // 切换到亮色模式
        document.documentElement.setAttribute('data-bs-theme', 'light');
        themeSwitcher.textContent = '☀️';
        localStorage.setItem('theme', 'light');
        
        // 更新导航栏样式
        if (navbar) {
            navbar.classList.remove('navbar-dark', 'bg-dark');
            navbar.classList.add('navbar-light', 'bg-light');
        }
    } else {
        // 切换到暗色模式
        document.documentElement.setAttribute('data-bs-theme', 'dark');
        themeSwitcher.textContent = '🌙';
        localStorage.setItem('theme', 'dark');
        
        // 更新导航栏样式
        if (navbar) {
            navbar.classList.remove('navbar-light', 'bg-light');
            navbar.classList.add('navbar-dark', 'bg-dark');
        }
    }
}
