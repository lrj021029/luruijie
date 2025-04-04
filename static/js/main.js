// main.js - 主要的JavaScript功能

document.addEventListener("DOMContentLoaded", function() {
    // 初始化全局状态管理
    if (typeof AppState !== 'undefined') {
        window.AppState = new AppState();
        window.AppState.init();
        console.log('全局状态管理已初始化');
    } else {
        console.warn('AppState未定义，无法初始化状态管理');
    }
    
    // 初始化事件监听器
    initEventListeners();
    
    // 如果在历史页面，加载历史数据
    if (document.getElementById('history-table')) {
        loadHistory();
    }
    
    // 如果在特征页面，加载词云数据
    if (document.getElementById('spam-wordcloud') && document.getElementById('ham-wordcloud')) {
        loadWordCloudData();
    }
    
    // 如果在主页面
    if (window.location.pathname === "/" || window.location.pathname === "/index") {
        // 加载已训练模型列表并填充下拉框
        loadTrainedModelsForPrediction();
        
        // 初始化模型指标图表
        if (document.getElementById('model-metrics-chart')) {
            loadModelMetrics();
        }
        
        // 初始化漂移图表
        if (document.getElementById('drift-chart')) {
            initDriftChart();
        }
        
        // 恢复页面状态（如果有）
        restorePageState('index');
    }
    
    // 如果有模型训练表单，设置相关事件
    if (document.getElementById('train-model-form')) {
        setupModelTraining();
    }
    
    // 如果有已保存模型列表，加载模型列表
    if (document.getElementById('saved-models-list')) {
        loadSavedModels();
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
        
        // 文件选择变化时读取CSV头部
        const fileInput = document.getElementById('file');
        if (fileInput) {
            fileInput.addEventListener('change', handleFileSelect);
        }
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
    const modelSelect = document.getElementById('model-select');
    
    // 检查是否选择了模型
    if (!modelSelect.value) {
        resultContainer.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                <strong>请先选择模型!</strong> 在进行预测前，请先训练并选择一个模型。
            </div>
        `;
        resultContainer.classList.remove('d-none');
        return;
    }
    
    // 显示加载动画
    loadingSpinner.classList.remove('d-none');
    submitBtn.disabled = true;
    resultContainer.classList.add('d-none');
    
    try {
        const formData = new FormData(form);
        
        // 添加模型路径信息，如果选项中有path数据属性
        const selectedOption = modelSelect.options[modelSelect.selectedIndex];
        if (selectedOption && selectedOption.dataset.path) {
            formData.append('model_path', selectedOption.dataset.path);
        }
        
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
        
        // 保存当前页面状态（选择的模型等）
        saveCurrentPageState('index');
        
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
    const fileInput = form.querySelector('input[type="file"]');
    const modelSelect = document.getElementById('upload-model-select');
    
    // 检查是否选择了模型
    if (modelSelect && !modelSelect.value) {
        resultContainer.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                <strong>请先选择模型!</strong> 在进行批量预测前，请先训练并选择一个模型。
            </div>
        `;
        resultContainer.classList.remove('d-none');
        return;
    }
    
    // 检查是否选择了文件
    if (!fileInput.files || fileInput.files.length === 0) {
        resultContainer.innerHTML = `
            <div class="alert alert-warning">
                <strong>请选择文件:</strong> 请选择一个CSV文件进行上传
            </div>
        `;
        resultContainer.classList.remove('d-none');
        return;
    }
    
    // 检查文件格式是否为CSV
    const fileName = fileInput.files[0].name;
    if (!fileName.toLowerCase().endsWith('.csv')) {
        resultContainer.innerHTML = `
            <div class="alert alert-warning">
                <strong>文件格式错误:</strong> 请上传CSV格式的文件
            </div>
        `;
        resultContainer.classList.remove('d-none');
        return;
    }
    
    // 显示加载动画
    loadingSpinner.classList.remove('d-none');
    submitBtn.disabled = true;
    resultContainer.classList.add('d-none');
    
    try {
        const formData = new FormData(form);
        
        // 添加模型路径信息，如果选项中有path数据属性
        if (modelSelect) {
            const selectedOption = modelSelect.options[modelSelect.selectedIndex];
            if (selectedOption && selectedOption.dataset.path) {
                formData.append('model_path', selectedOption.dataset.path);
            }
        }
        
        // 发送请求到后端API
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || '文件上传失败');
        }
        
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
        
        // 保存当前页面状态
        saveCurrentPageState('index');
        
    } catch (error) {
        console.error('上传错误:', error);
        
        // 显示错误消息
        resultContainer.innerHTML = `
            <div class="alert alert-danger">
                <strong>错误:</strong> ${error.message || '上传处理过程中发生错误'}
            </div>
            <div class="alert alert-info mt-2">
                <strong>提示:</strong> 请确保CSV文件至少包含一个名为"text"的列。其他列如"send_freq"和"is_night"是可选的。
            </div>
        `;
        resultContainer.classList.remove('d-none');
        
    } finally {
        // 隐藏加载动画，恢复提交按钮
        loadingSpinner.classList.add('d-none');
        submitBtn.disabled = false;
    }
}

// 显示历史记录数据到表格
function displayHistoryData(data) {
    const historyTable = document.getElementById('history-table');
    if (!historyTable) return;
    
    const tableBody = historyTable.querySelector('tbody');
    const selectAllCheckbox = document.getElementById('select-all-records');
    const deleteSelectedBtn = document.getElementById('delete-selected');
    
    // 清空表格
    tableBody.innerHTML = '';
    
    // 重置全选框
    if (selectAllCheckbox) selectAllCheckbox.checked = false;
    if (deleteSelectedBtn) deleteSelectedBtn.disabled = true;
    
    // 填充表格
    if (data.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="7" class="text-center">暂无历史记录</td>
            </tr>
        `;
        return;
    }
    
    // 创建表格行
    for (const item of data) {
        const row = document.createElement('tr');
        
        // 设置行的类
        if (item.prediction === '垃圾短信') {
            row.classList.add('table-danger');
        }
        
        // 生成单元格
        row.innerHTML = `
            <td>
                <div class="form-check">
                    <input class="form-check-input history-select-checkbox" type="checkbox" value="${item.id}" onchange="updateDeleteSelectedButton()">
                </div>
            </td>
            <td>${item.id}</td>
            <td class="text-truncate" style="max-width: 300px;" title="${item.text}">${item.text}</td>
            <td>${item.send_freq}</td>
            <td>${item.is_night ? '是' : '否'}</td>
            <td>${item.prediction === '垃圾短信' 
                ? '<span class="badge bg-danger">垃圾短信</span>' 
                : '<span class="badge bg-success">正常短信</span>'}
            </td>
            <td>${(item.confidence * 100).toFixed(1)}%</td>
            <td>${getModelDisplayName(item.model_type)}</td>
            <td>${new Date(item.timestamp).toLocaleString()}</td>
            <td>
                <button class="btn btn-sm btn-outline-danger" onclick="deleteRecord(${item.id})">
                    <i class="fa fa-trash"></i>
                </button>
            </td>
        `;
        
        tableBody.appendChild(row);
    }
    
    // 添加事件监听器
    addHistoryEventListeners();
}

// 加载预测历史数据
async function loadHistory(forceReload = false) {
    const historyTable = document.getElementById('history-table');
    if (!historyTable) return;
    
    const loadingSpinner = document.getElementById('history-spinner');
    
    // 检查是否有缓存数据
    if (window.cachedHistoryData && !forceReload) {
        console.log('使用缓存的历史记录数据');
        displayHistoryData(window.cachedHistoryData);
        return window.cachedHistoryData;
    }
    
    // 显示加载动画
    if (loadingSpinner) {
        loadingSpinner.classList.remove('d-none');
    }
    
    try {
        // 发送请求到后端API
        const response = await fetch('/get_history');
        
        if (!response.ok) {
            throw new Error('获取历史记录失败');
        }
        
        const data = await response.json();
        
        // 缓存历史记录数据
        window.cachedHistoryData = data;
        
        // 显示数据
        displayHistoryData(data);
        
        // 隐藏加载动画
        if (loadingSpinner) {
            loadingSpinner.classList.add('d-none');
        }
        
        return data;
    } catch (error) {
        console.error('加载历史记录错误:', error);
        
        // 显示错误消息
        if (historyTable) {
            const tableBody = historyTable.querySelector('tbody');
            if (tableBody) {
                tableBody.innerHTML = `
                    <tr>
                        <td colspan="7" class="text-center text-danger">
                            <i class="fas fa-exclamation-circle"></i> 加载历史记录失败: ${error.message}
                        </td>
                    </tr>
                `;
            }
        }
        
        // 隐藏加载动画
        if (loadingSpinner) {
            loadingSpinner.classList.add('d-none');
        }
        
        return [];
    }
}

// 添加历史记录表格的事件监听器
function addHistoryEventListeners() {
    // 全选/取消全选
    const selectAllCheckbox = document.getElementById('select-all-records');
    if (selectAllCheckbox) {
        selectAllCheckbox.addEventListener('change', function() {
            const checkboxes = document.querySelectorAll('.history-select-checkbox');
            checkboxes.forEach(checkbox => {
                checkbox.checked = this.checked;
            });
            updateDeleteSelectedButton();
        });
    }
}

// 更新删除选中按钮状态
function updateDeleteSelectedButton() {
    const checkboxes = document.querySelectorAll('.history-select-checkbox:checked');
    const deleteSelectedBtn = document.getElementById('delete-selected');
    if (deleteSelectedBtn) {
        deleteSelectedBtn.disabled = checkboxes.length === 0;
    }
}

// 删除单个记录
async function deleteRecord(recordId) {
    try {
        const response = await fetch(`/delete_record/${recordId}`, {
            method: 'DELETE',
        });
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.message || '删除记录失败');
        }
        
        // 重新加载历史记录，强制刷新数据
        await loadHistory(true);
        
        // 显示成功消息
        showToast('success', '删除成功', '记录已成功删除');
        
    } catch (error) {
        console.error('删除记录错误:', error);
        showToast('error', '删除失败', error.message);
    }
}

// 删除选中的记录
async function deleteSelectedRecords() {
    const checkboxes = document.querySelectorAll('.history-select-checkbox:checked');
    const ids = Array.from(checkboxes).map(cb => cb.value);
    
    if (ids.length === 0) {
        return;
    }
    
    if (!confirm(`确定要删除选中的 ${ids.length} 条记录吗？`)) {
        return;
    }
    
    try {
        const response = await fetch('/delete_records', {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ ids }),
        });
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.message || '删除记录失败');
        }
        
        // 重新加载历史记录，强制刷新数据
        await loadHistory(true);
        
        // 显示成功消息
        showToast('success', '批量删除成功', `已成功删除 ${ids.length} 条记录`);
        
    } catch (error) {
        console.error('批量删除记录错误:', error);
        showToast('error', '删除失败', error.message);
    }
}

// 删除所有记录
async function deleteAllRecords() {
    try {
        const response = await fetch('/delete_all_records', {
            method: 'DELETE',
        });
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.message || '删除记录失败');
        }
        
        // 关闭确认对话框
        const modal = bootstrap.Modal.getInstance(document.getElementById('confirmDeleteAllModal'));
        modal.hide();
        
        // 重新加载历史记录，强制刷新数据
        await loadHistory(true);
        
        // 显示成功消息
        showToast('success', '清空成功', '已成功删除所有记录');
        
    } catch (error) {
        console.error('删除所有记录错误:', error);
        showToast('error', '删除失败', error.message);
    }
}

// 显示提示消息
function showToast(type, title, message) {
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        // 如果没有toast容器，创建一个
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(container);
    }
    
    const toastId = 'toast-' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header ${type === 'error' ? 'bg-danger text-white' : 'bg-success text-white'}">
                <strong class="me-auto">${title}</strong>
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;
    
    document.getElementById('toast-container').insertAdjacentHTML('beforeend', toastHtml);
    
    // 初始化并显示toast
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, { delay: 5000 });
    toast.show();
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

// 处理文件选择，读取CSV头部并显示列选择界面
async function handleFileSelect(event) {
    const fileInput = event.target;
    const file = fileInput.files[0];
    const columnMapping = document.getElementById('column-mapping');
    const resultContainer = document.getElementById('upload-result');
    
    // 隐藏之前的结果
    resultContainer.classList.add('d-none');
    
    // 检查是否选择了文件
    if (!file) {
        columnMapping.style.display = 'none';
        return;
    }
    
    // 检查文件格式是否为CSV
    if (!file.name.toLowerCase().endsWith('.csv')) {
        columnMapping.style.display = 'none';
        resultContainer.innerHTML = `
            <div class="alert alert-warning">
                <strong>文件格式错误:</strong> 请上传CSV格式的文件
            </div>
        `;
        resultContainer.classList.remove('d-none');
        return;
    }
    
    try {
        // 读取CSV文件的前几行以获取列名
        const reader = new FileReader();
        
        reader.onload = function(e) {
            const content = e.target.result;
            const lines = content.split('\n');
            
            if (lines.length === 0) {
                throw new Error('文件为空');
            }
            
            // 获取列名（假设第一行是标题行）
            let headers = lines[0].split(',');
            
            // 如果使用引号包裹的CSV，去除引号
            headers = headers.map(header => {
                header = header.trim();
                if (header.startsWith('"') && header.endsWith('"')) {
                    return header.substring(1, header.length - 1);
                }
                return header;
            });
            
            // 设置隐藏的文件名字段
            document.getElementById('csv_filename').value = file.name;
            
            // 更新列选择下拉框
            updateColumnSelects(headers);
            
            // 显示列映射区域
            columnMapping.style.display = 'block';
            
            // 设置隐藏的映射模式字段为true
            document.getElementById('mapping_mode').value = 'true';
        };
        
        reader.onerror = function() {
            throw new Error('读取文件失败');
        };
        
        // 以文本形式读取文件
        reader.readAsText(file);
        
    } catch (error) {
        console.error('解析CSV头部错误:', error);
        
        columnMapping.style.display = 'none';
        resultContainer.innerHTML = `
            <div class="alert alert-danger">
                <strong>错误:</strong> ${error.message || '解析CSV文件失败'}
            </div>
        `;
        resultContainer.classList.remove('d-none');
    }
}

// 更新列选择下拉框
function updateColumnSelects(headers) {
    // 获取所有列选择器
    const textSelect = document.getElementById('text_column');
    const labelSelect = document.getElementById('label_column');
    const sendFreqSelect = document.getElementById('send_freq_column');
    const isNightSelect = document.getElementById('is_night_column');
    
    // 清空下拉框选项
    textSelect.innerHTML = '<option value="">-- 请选择 --</option>';
    labelSelect.innerHTML = '<option value="">-- 无 --</option>';
    sendFreqSelect.innerHTML = '<option value="">-- 无 --</option>';
    isNightSelect.innerHTML = '<option value="">-- 无 --</option>';
    
    // 添加各列选项
    headers.forEach(header => {
        if (!header) return; // 跳过空列名
        
        const textOption = document.createElement('option');
        textOption.value = header;
        textOption.textContent = header;
        
        const labelOption = document.createElement('option');
        labelOption.value = header;
        labelOption.textContent = header;
        
        const sendFreqOption = document.createElement('option');
        sendFreqOption.value = header;
        sendFreqOption.textContent = header;
        
        const isNightOption = document.createElement('option');
        isNightOption.value = header;
        isNightOption.textContent = header;
        
        textSelect.appendChild(textOption);
        labelSelect.appendChild(labelOption);
        sendFreqSelect.appendChild(sendFreqOption);
        isNightSelect.appendChild(isNightOption);
    });
    
    // 尝试智能匹配列名
    Array.from(textSelect.options).forEach(option => {
        const lowerValue = option.value.toLowerCase();
        if (lowerValue.includes('text') || lowerValue.includes('content') || lowerValue.includes('message') || 
            lowerValue.includes('短信') || lowerValue.includes('内容') || lowerValue.includes('文本')) {
            option.selected = true;
        }
    });
    
    Array.from(labelSelect.options).forEach(option => {
        const lowerValue = option.value.toLowerCase();
        if (lowerValue.includes('label') || lowerValue.includes('class') || lowerValue.includes('type') || 
            lowerValue.includes('标签') || lowerValue.includes('分类')) {
            option.selected = true;
        }
    });
    
    Array.from(sendFreqSelect.options).forEach(option => {
        const lowerValue = option.value.toLowerCase();
        if (lowerValue.includes('freq') || lowerValue.includes('frequency') || lowerValue.includes('rate') || 
            lowerValue.includes('频率') || lowerValue.includes('频次')) {
            option.selected = true;
        }
    });
    
    Array.from(isNightSelect.options).forEach(option => {
        const lowerValue = option.value.toLowerCase();
        if (lowerValue.includes('night') || lowerValue.includes('time') || lowerValue.includes('hour') || 
            lowerValue.includes('夜间') || lowerValue.includes('时间')) {
            option.selected = true;
        }
    });
}

// 全局词云数据缓存
window.cachedSpamWordCloud = null;
window.cachedHamWordCloud = null;
window.cachedModelMetrics = null;
window.cachedHistoryData = null;

// 加载词云数据
async function loadWordCloudData(forceReload = false) {
    const spamContainer = document.getElementById('spam-word-cloud') || document.getElementById('spam-wordcloud');
    const hamContainer = document.getElementById('ham-word-cloud') || document.getElementById('ham-wordcloud');
    const loadingSpinner = document.getElementById('wordcloud-spinner');
    
    if (!spamContainer || !hamContainer) {
        console.warn('词云容器不存在，跳过加载');
        return;
    }
    
    // 显示加载动画
    if (loadingSpinner) {
        loadingSpinner.classList.remove('d-none');
    }
    
    try {
        // 检查是否有缓存数据
        if (window.cachedSpamWordCloud && window.cachedHamWordCloud && !forceReload) {
            console.log('使用缓存的词云数据');
            
            // 渲染词云
            if (window.cachedSpamWordCloud.length > 0) {
                renderWordCloud(spamContainer, window.cachedSpamWordCloud, 'danger');
            } else {
                spamContainer.innerHTML = '<div class="alert alert-info">暂无足够数据生成垃圾短信词云</div>';
            }
            
            if (window.cachedHamWordCloud.length > 0) {
                renderWordCloud(hamContainer, window.cachedHamWordCloud, 'info');
            } else {
                hamContainer.innerHTML = '<div class="alert alert-info">暂无足够数据生成正常短信词云</div>';
            }
            
            // 如果有加载指示器则隐藏
            if (loadingSpinner) {
                loadingSpinner.classList.add('d-none');
            }
            
            return;
        }
        
        // 发送请求到后端API
        const response = await fetch('/get_features');
        
        if (!response.ok) {
            throw new Error('获取词云数据失败');
        }
        
        const data = await response.json();
        
        // 缓存词云数据
        if (data.spam_words) {
            window.cachedSpamWordCloud = data.spam_words;
            console.log('正在渲染词云，单词数量:', data.spam_words.length);
            console.log('词云数据示例:', data.spam_words.slice(0, 5));
        }
        
        if (data.ham_words) {
            window.cachedHamWordCloud = data.ham_words;
            console.log('正在渲染词云，单词数量:', data.ham_words.length);
            console.log('词云数据示例:', data.ham_words.slice(0, 5));
        }
        
        // 渲染词云
        if (data.spam_words && data.spam_words.length > 0) {
            renderWordCloud(spamContainer, data.spam_words, 'danger');
        } else {
            spamContainer.innerHTML = '<div class="alert alert-info">暂无足够数据生成垃圾短信词云</div>';
        }
        
        if (data.ham_words && data.ham_words.length > 0) {
            renderWordCloud(hamContainer, data.ham_words, 'info');
        } else {
            hamContainer.innerHTML = '<div class="alert alert-info">暂无足够数据生成正常短信词云</div>';
        }
        
    } catch (error) {
        console.error('加载词云数据错误:', error);
        
        // 显示错误消息
        if (spamContainer) {
            spamContainer.innerHTML = `<div class="alert alert-danger">加载垃圾短信词云失败: ${error.message}</div>`;
        }
        if (hamContainer) {
            hamContainer.innerHTML = `<div class="alert alert-danger">加载正常短信词云失败: ${error.message}</div>`;
        }
        
    } finally {
        // 隐藏加载动画
        loadingSpinner.classList.add('d-none');
    }
}

// 渲染词云
function renderWordCloud(container, words, color) {
    try {
        // 标准化数据格式为WordCloud2.js需要的格式 [[word, weight], [word, weight]]
        let wordList = [];
        
        // 检查数据格式并转换
        if (words.length > 0) {
            // 检查第一项是否是数组格式 [word, weight]
            if (Array.isArray(words[0])) {
                wordList = words;
            } 
            // 检查是否为对象格式 {word: string, value: number}
            else if (typeof words[0] === 'object' && words[0].hasOwnProperty('word') && words[0].hasOwnProperty('value')) {
                wordList = words.map(item => [item.word, item.value]);
            }
            // 检查是否为元组格式（可能来自Python后端） (word, value)
            else if (typeof words[0] === 'object' && words[0].length === 2) {
                wordList = words.map(item => [item[0], item[1]]);
            }
            else {
                console.error("无法识别词云数据格式:", words[0]);
                throw new Error('词云数据格式不支持');
            }
        }
        
        console.log("正在渲染词云，单词数量:", wordList.length);
        console.log("词云数据示例:", wordList.slice(0, 5));
        
        // 检查全局WordCloud对象是否存在
        if (typeof WordCloud === 'undefined') {
            throw new Error('WordCloud库未加载');
        }
        
        // 清空容器
        container.innerHTML = '';
        
        // 添加词云标题和信息显示区域
        const infoDiv = document.createElement('div');
        infoDiv.className = 'word-info text-center mb-2';
        infoDiv.innerHTML = '&nbsp;';
        container.appendChild(infoDiv);
        
        // 创建词云canvas
        const canvas = document.createElement('canvas');
        canvas.width = container.offsetWidth;
        canvas.height = 300;
        container.appendChild(canvas);
        
        // 确保容器和canvas可见且有实际宽度
        if (container.offsetWidth === 0) {
            console.error("词云容器宽度为0，无法正确渲染");
            throw new Error('容器宽度为0');
        }
        
        // 确保词云有数据
        if (wordList.length === 0) {
            console.error("词云数据为空");
            throw new Error('词云数据为空');
        }
        
        // 词云配置
        const options = {
            list: wordList,
            fontFamily: 'Pingfang SC, Source Sans Pro, Microsoft Yahei',
            fontWeight: 'bold',
            color: color == 'danger' ? 'rgba(220, 53, 69, 0.8)' : 'rgba(13, 202, 240, 0.8)',
            minSize: 14,
            weightFactor: 8,  // 增大权重因子使词云更明显
            backgroundColor: 'transparent',
            gridSize: 8,
            drawOutOfBound: false,
            shape: 'circle',
            rotateRatio: 0.5,
            shrinkToFit: true,
            hover: function(item, dimension) {
                try {
                    const infoElement = container.querySelector('.word-info');
                    if (infoElement && item && item[0]) {
                        infoElement.textContent = `"${item[0]}" 出现 ${item[1]} 次`;
                        infoElement.style.fontWeight = 'bold';
                    }
                } catch (hoverError) {
                    console.error("词云hover事件错误:", hoverError);
                }
            }
        };
        
        // 渲染词云
        WordCloud(canvas, options);
    } catch (error) {
        console.error("词云渲染失败:", error);
        container.innerHTML = `<div class="alert alert-danger">词云渲染失败: ${error.message}</div>`;
    }
}

// 加载已训练模型列表并填充预测表单下拉菜单
async function loadTrainedModelsForPrediction() {
    return new Promise(async (resolve, reject) => {
        try {
            const modelSelect = document.getElementById('model-select');
            const uploadModelSelect = document.getElementById('upload-model-select');
            const detectButton = document.getElementById('detect-button');
            const uploadButton = document.getElementById('upload-button');
            const noModelWarning = document.getElementById('no-model-warning');
            
            if (!modelSelect) {
                console.log('模型选择框未找到');
                return resolve(false);
            }
            
            // 获取已训练模型列表
            const response = await fetch('/get_models');
            
            if (!response.ok) {
                throw new Error('获取已训练模型列表失败');
            }
            
            const data = await response.json();
            
            // 清空现有选项，保留默认选项
            modelSelect.innerHTML = '<option value="" disabled selected>-- 请先训练模型 --</option>';
            if (uploadModelSelect) {
                uploadModelSelect.innerHTML = '<option value="" disabled selected>-- 请先训练模型 --</option>';
            }
            
            // 计算是否有已训练的模型
            let hasTrainedModels = false;
            let modelCount = 0;
            
            // 检查是否有训练好的模型
            if (data && data.success && data.models) {
                // 遍历所有模型类型
                Object.keys(data.models).forEach(modelType => {
                    const models = data.models[modelType];
                    
                    if (models && models.length > 0) {
                        hasTrainedModels = true;
                        
                        // 遍历当前类型的所有已训练模型
                        models.forEach(modelInfo => {
                            modelCount++;
                            const modelName = getModelDisplayName(modelType);
                            
                            // 创建选项
                            const option = document.createElement('option');
                            option.value = modelType;
                            option.textContent = `${modelName} (${modelInfo.date || modelInfo.timestamp})`;
                            option.dataset.path = modelInfo.filename;
                            
                            // 添加到单条预测下拉菜单
                            modelSelect.appendChild(option.cloneNode(true));
                            
                            // 添加到批量预测下拉菜单
                            if (uploadModelSelect) {
                                uploadModelSelect.appendChild(option.cloneNode(true));
                            }
                        });
                    }
                });
                
                // 隐藏警告
                if (hasTrainedModels) {
                    if (noModelWarning) noModelWarning.style.display = 'none';
                    
                    // 启用检测按钮
                    if (detectButton) detectButton.disabled = false;
                    if (uploadButton) uploadButton.disabled = false;
                    
                    // 如果URL中有模型类型参数，选中对应的选项
                    const urlParams = new URLSearchParams(window.location.search);
                    const modelTypeParam = urlParams.get('model_type');
                    
                    if (modelTypeParam) {
                        const modelOptions = modelSelect.querySelectorAll(`option[value="${modelTypeParam}"]`);
                        if (modelOptions && modelOptions.length > 0) {
                            modelOptions[0].selected = true;
                        }
                        
                        if (uploadModelSelect) {
                            const uploadModelOptions = uploadModelSelect.querySelectorAll(`option[value="${modelTypeParam}"]`);
                            if (uploadModelOptions && uploadModelOptions.length > 0) {
                                uploadModelOptions[0].selected = true;
                            }
                        }
                    }
                    
                    console.log(`已加载 ${modelCount} 个已训练模型供使用`);
                } else {
                    // 显示警告，没有训练好的模型
                    if (noModelWarning) noModelWarning.style.display = 'block';
                    
                    // 禁用检测按钮
                    if (detectButton) detectButton.disabled = true;
                    if (uploadButton) uploadButton.disabled = true;
                }
            } else {
                // 显示警告，没有训练好的模型
                if (noModelWarning) noModelWarning.style.display = 'block';
                
                // 禁用检测按钮
                if (detectButton) detectButton.disabled = true;
                if (uploadButton) uploadButton.disabled = true;
            }
        } catch (error) {
            console.error('加载已训练模型列表错误:', error);
            
            // 显示错误消息
            if (noModelWarning) {
                noModelWarning.style.display = 'block';
                noModelWarning.innerHTML = `
                    <div class="alert alert-warning">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        <strong>提示：</strong>请先训练模型才能进行预测
                    </div>
                `;
            }
            
            // 禁用检测按钮
            if (detectButton) detectButton.disabled = true;
            if (uploadButton) uploadButton.disabled = true;
            
            reject(error);
        }
        
        // 无论成功或失败，都返回结果
        return resolve(true);
    });
}

// 格式化时间戳为可读形式
function formatTimestamp(timestamp) {
    if (!timestamp) return '';
    
    // 将YYYYMMDDhhmmss格式转换为YYYY-MM-DD hh:mm:ss
    const year = timestamp.substring(0, 4);
    const month = timestamp.substring(4, 6);
    const day = timestamp.substring(6, 8);
    const hour = timestamp.substring(8, 10) || '00';
    const minute = timestamp.substring(10, 12) || '00';
    
    return `${year}-${month}-${day} ${hour}:${minute}`;
}

// 获取模型显示名称
function getModelDisplayName(modelType) {
    const modelNames = {
        'roberta': 'RoBERTa',
        'bert': 'BERT',
        'lstm': 'LSTM',
        'cnn': 'CNN',
        'xlnet': 'XLNet',
        'gpt': 'GPT',
        'attention_lstm': 'Attention LSTM',
        'svm': 'SVM',
        'naive_bayes': '朴素贝叶斯',
        'ensemble': 'Ensemble集成'
    };
    
    return modelNames[modelType] || modelType;
}

// 加载模型指标数据
async function loadModelMetrics(forceReload = false) {
    const metricsContainer = document.getElementById('model-metrics-chart') || document.getElementById('model-metrics-container');
    if (!metricsContainer) {
        console.warn('模型指标容器不存在，跳过加载');
        return;
    }
    
    const loadingSpinner = document.getElementById('metrics-spinner');
    
    // 显示加载动画
    if (loadingSpinner) {
        loadingSpinner.classList.remove('d-none');
    }
    
    try {
        // 检查是否有缓存数据
        if (window.cachedModelMetrics && !forceReload) {
            console.log('使用缓存的模型指标数据');
            renderModelMetricsChart(metricsContainer, window.cachedModelMetrics);
            
            // 如果有加载指示器则隐藏
            if (loadingSpinner) {
                loadingSpinner.classList.add('d-none');
            }
            
            return;
        }
        
        // 发送请求到后端API
        const response = await fetch('/get_model_metrics');
        
        if (!response.ok) {
            throw new Error('获取模型指标失败');
        }
        
        const data = await response.json();
        
        // 缓存模型指标数据
        window.cachedModelMetrics = data;
        
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
        if (loadingSpinner) {
            loadingSpinner.classList.add('d-none');
        }
    }
}

// 渲染模型指标图表
function renderModelMetricsChart(container, data) {
    // 转换模型类型显示名称
    const modelDisplayNames = {
        'naive_bayes': '朴素贝叶斯',
        'svm': 'SVM',
        'lstm': 'LSTM',
        'residual_attention_lstm': 'ResidAttn LSTM'
    };

    // 提取数据，仅保留我们关注的四个模型
    const models = Object.keys(data).filter(model => 
        ['naive_bayes', 'svm', 'lstm', 'residual_attention_lstm'].includes(model));
    
    // 显示友好的模型名称
    const modelLabels = models.map(model => modelDisplayNames[model] || model);
    
    // 指标和指标名称
    const metrics = ['accuracy', 'precision', 'recall', 'f1_score'];
    const metricNames = ['准确率', '精确率', '召回率', 'F1分数'];
    
    // 为每个指标创建一个图表
    // 使用图表容器作为父容器
    container.innerHTML = '<h6 class="text-center mb-4">模型性能指标对比</h6>';
    
    // 创建一个包含所有图表的容器
    const chartsContainer = document.createElement('div');
    chartsContainer.className = 'row';
    container.appendChild(chartsContainer);
    
    // 为每个指标创建一个图表
    metrics.forEach((metric, index) => {
        // 创建列和图表容器
        const colDiv = document.createElement('div');
        colDiv.className = 'col-md-6 mb-4';
        chartsContainer.appendChild(colDiv);
        
        const chartContainer = document.createElement('div');
        chartContainer.style.height = '250px';
        colDiv.appendChild(chartContainer);
        
        const canvas = document.createElement('canvas');
        chartContainer.appendChild(canvas);
        
        // 获取此指标的所有模型数据
        const metricData = models.map(model => data[model][metric]);
        
        // 根据指标选择不同颜色
        const colors = {
            'accuracy': ['rgba(75, 192, 192, 0.7)', 'rgba(75, 192, 192, 1)'],
            'precision': ['rgba(54, 162, 235, 0.7)', 'rgba(54, 162, 235, 1)'],
            'recall': ['rgba(255, 206, 86, 0.7)', 'rgba(255, 206, 86, 1)'],
            'f1_score': ['rgba(255, 99, 132, 0.7)', 'rgba(255, 99, 132, 1)']
        };
        
        // 创建图表
        new Chart(canvas, {
            type: 'bar',
            data: {
                labels: modelLabels,
                datasets: [{
                    label: metricNames[index],
                    data: metricData,
                    backgroundColor: colors[metric][0],
                    borderColor: colors[metric][1],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: metricNames[index]
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${metricNames[index]}: ${(context.raw * 100).toFixed(1)}%`;
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
    });
    
    // 添加一个模型计数表格
    const countTableDiv = document.createElement('div'); 
    countTableDiv.className = 'mt-3';
    container.appendChild(countTableDiv);
    
    // 创建表格HTML
    let tableHTML = `
    <div class="table-responsive">
        <table class="table table-sm table-bordered">
            <thead>
                <tr>
                    <th>模型</th>
                    <th class="text-center">已处理数据量</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    // 添加每个模型的数据计数
    models.forEach(model => {
        const count = data[model].count || 0;
        tableHTML += `
            <tr>
                <td>${modelDisplayNames[model] || model}</td>
                <td class="text-center">${count}</td>
            </tr>
        `;
    });
    
    tableHTML += `
            </tbody>
        </table>
    </div>
    `;
    
    countTableDiv.innerHTML = tableHTML;
    
    // 创建混淆矩阵容器
    const confusionMatrixContainer = document.createElement('div');
    confusionMatrixContainer.className = 'row mt-4';
    container.appendChild(confusionMatrixContainer);
    
    // 为每个模型创建混淆矩阵
    models.forEach(model => {
        if (!data[model].confusion_matrix) return;
        
        // 获取混淆矩阵数据
        const cm = data[model].confusion_matrix;
        
        // 创建列
        const colDiv = document.createElement('div');
        colDiv.className = 'col-md-6 mb-4';
        confusionMatrixContainer.appendChild(colDiv);
        
        // 创建卡片
        const cardDiv = document.createElement('div');
        cardDiv.className = 'card h-100';
        colDiv.appendChild(cardDiv);
        
        // 卡片标题
        const cardHeader = document.createElement('div');
        cardHeader.className = 'card-header';
        cardHeader.textContent = `${modelDisplayNames[model] || model} 混淆矩阵`;
        cardDiv.appendChild(cardHeader);
        
        // 卡片内容
        const cardBody = document.createElement('div');
        cardBody.className = 'card-body';
        cardDiv.appendChild(cardBody);
        
        // 计算总样本数和百分比
        const total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1];
        const getPercent = (value) => ((value / total) * 100).toFixed(1) + '%';
        
        // 创建混淆矩阵表
        const matrixTable = document.createElement('table');
        matrixTable.className = 'table table-bordered cm-table';
        matrixTable.innerHTML = `
            <thead>
                <tr>
                    <th></th>
                    <th class="text-center">预测: 正常</th>
                    <th class="text-center">预测: 垃圾</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <th>实际: 正常</th>
                    <td class="text-center true-negative">
                        <span class="cm-value">${cm[0][0]}</span>
                        <span class="cm-percent">${getPercent(cm[0][0])}</span>
                    </td>
                    <td class="text-center false-positive">
                        <span class="cm-value">${cm[0][1]}</span>
                        <span class="cm-percent">${getPercent(cm[0][1])}</span>
                    </td>
                </tr>
                <tr>
                    <th>实际: 垃圾</th>
                    <td class="text-center false-negative">
                        <span class="cm-value">${cm[1][0]}</span>
                        <span class="cm-percent">${getPercent(cm[1][0])}</span>
                    </td>
                    <td class="text-center true-positive">
                        <span class="cm-value">${cm[1][1]}</span>
                        <span class="cm-percent">${getPercent(cm[1][1])}</span>
                    </td>
                </tr>
            </tbody>
        `;
        cardBody.appendChild(matrixTable);
        
        // 添加指标摘要
        const metricSummary = document.createElement('div');
        metricSummary.className = 'mt-3 cm-summary';
        metricSummary.innerHTML = `
            <div class="row">
                <div class="col-6">
                    <div class="metric-item">
                        <span class="metric-name">准确率:</span>
                        <span class="metric-value">${(data[model].accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-name">精确率:</span>
                        <span class="metric-value">${(data[model].precision * 100).toFixed(1)}%</span>
                    </div>
                </div>
                <div class="col-6">
                    <div class="metric-item">
                        <span class="metric-name">召回率:</span>
                        <span class="metric-value">${(data[model].recall * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-name">F1分数:</span>
                        <span class="metric-value">${(data[model].f1_score * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
        `;
        cardBody.appendChild(metricSummary);
    });
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
        // 获取当前选择的模型类型和路径
        const modelSelect = document.getElementById('model-select');
        let modelType = '';
        let modelPath = '';
        
        if (modelSelect && modelSelect.selectedIndex > 0) {
            const selectedOption = modelSelect.options[modelSelect.selectedIndex];
            modelType = selectedOption.value;
            modelPath = selectedOption.dataset.path || '';
        }
        
        // 如果没有选择模型，则显示提示并返回
        if (!modelType) {
            const warningContainer = document.getElementById('drift-warning');
            if (warningContainer) {
                warningContainer.innerHTML = `
                    <div class="alert alert-warning">
                        <i class="fas fa-exclamation-circle"></i>
                        <strong>请先选择模型</strong>
                        <p class="mb-0">漂移检测需要先选择一个已训练的模型才能使用。</p>
                    </div>
                `;
            }
            return;
        }
        
        // 构建请求URL
        let url = `/track_drift?model_type=${modelType}`;
        if (modelPath) {
            url += `&model_path=${modelPath}`;
        }
        
        // 发送请求到后端API
        const response = await fetch(url);
        
        // 即使响应不是200，我们也尝试解析JSON
        const data = await response.json();
        
        // 检查是否有错误信息
        if (!response.ok || data.error) {
            console.error('获取漂移数据失败:', data.error || '未知错误');
            
            // 显示一个简单的错误提示，但不中断整个操作
            const warningContainer = document.getElementById('drift-warning');
            if (warningContainer) {
                warningContainer.innerHTML = `
                    <div class="alert alert-warning">
                        <i class="fas fa-exclamation-circle"></i>
                        <strong>漂移检测暂时不可用</strong>
                        <p class="mb-0">数据不足或系统处理中，请稍后再试。${data.error ? `<br>错误信息: ${data.error}` : ''}</p>
                    </div>
                `;
            }
            return;
        }
        
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
        // 显示一个友好的错误信息
        const warningContainer = document.getElementById('drift-warning');
        if (warningContainer) {
            warningContainer.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle"></i>
                    <strong>漂移检测异常</strong>
                    <p class="mb-0">服务暂时不可用，请稍后再试。</p>
                </div>
            `;
        }
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
// document.addEventListener的结束括号
