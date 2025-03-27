// 模型训练相关JavaScript功能

document.addEventListener("DOMContentLoaded", function() {
    // 如果有模型训练表单，设置相关事件
    if (document.getElementById('train-model-form')) {
        setupModelTraining();
    }
    
    // 如果有已保存模型容器，加载模型列表
    if (document.getElementById('saved-models-container')) {
        loadSavedModels();
    }
});

// 设置模型训练相关功能
function setupModelTraining() {
    const trainModelForm = document.getElementById('train-model-form');
    const trainButton = document.getElementById('train-button');
    const trainResultDiv = document.querySelector('.train-result');
    const trainResultContent = document.getElementById('train-result-content');
    
    // 添加模型训练表单提交事件
    trainModelForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        
        // 禁用按钮并显示加载状态
        trainButton.disabled = true;
        trainButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>训练中...';
        trainResultDiv.classList.add('d-none');
        
        try {
            const formData = new FormData(this);
            
            // 先发送预览请求，检查是否需要用户选择列
            formData.append('preview_data', 'true');
            
            // 发送请求到后端API
            let response = await fetch('/train_model', {
                method: 'POST',
                body: formData
            });
            
            let data = await response.json();
            
            // 如果需要用户选择列
            if (!response.ok && data.preview_needed) {
                // 显示列选择对话框
                const columnsModal = new bootstrap.Modal(document.getElementById('columnsModal') || createColumnsModal());
                
                // 填充列选择下拉框
                const textColumnSelect = document.getElementById('text-column-select');
                const labelColumnSelect = document.getElementById('label-column-select');
                
                // 清空之前的选项
                textColumnSelect.innerHTML = '';
                labelColumnSelect.innerHTML = '';
                
                // 添加列选项
                data.columns.forEach(column => {
                    textColumnSelect.innerHTML += `<option value="${column}">${column}</option>`;
                    labelColumnSelect.innerHTML += `<option value="${column}">${column}</option>`;
                });
                
                // 显示模态框
                columnsModal.show();
                
                // 处理列选择表单提交
                document.getElementById('columns-form').onsubmit = async function(e) {
                    e.preventDefault();
                    columnsModal.hide();
                    
                    // 获取所选列
                    const textColumn = textColumnSelect.value;
                    const labelColumn = labelColumnSelect.value;
                    
                    // 添加到原始表单数据
                    formData.delete('preview_data');
                    formData.append('text_column', textColumn);
                    formData.append('label_column', labelColumn);
                    
                    // 重新发送训练请求
                    response = await fetch('/train_model', {
                        method: 'POST',
                        body: formData
                    });
                    
                    data = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(data.error || '训练模型时发生错误');
                    }
                    
                    // 处理成功响应
                    handleTrainingSuccess(data);
                };
                
                // 恢复按钮状态
                trainButton.disabled = false;
                trainButton.innerHTML = '<i class="fas fa-graduation-cap me-2"></i>开始训练模型';
                
                return;
            }
            
            if (!response.ok) {
                throw new Error(data.error || '训练模型时发生错误');
            }
            
            // 处理成功响应
            handleTrainingSuccess(data);
            
        } catch (error) {
            console.error('训练模型错误:', error);
            
            // 显示错误消息
            trainResultDiv.querySelector('.alert').className = 'alert alert-danger';
            trainResultDiv.querySelector('.alert-heading').innerHTML = '<i class="fas fa-exclamation-circle me-2"></i>训练失败!';
            trainResultContent.innerHTML = `
                <p class="mb-0">${error.message}</p>
                <p class="mt-2 mb-0"><small>请检查上传的CSV文件格式，确保包含文本内容列和标签列。</small></p>
            `;
            trainResultDiv.classList.remove('d-none');
            
            // 重新启用按钮
            trainButton.disabled = false;
            trainButton.innerHTML = '<i class="fas fa-graduation-cap me-2"></i>开始训练模型';
        }
    });
    
    // 处理训练成功的函数
    function handleTrainingSuccess(data) {
        // 显示训练成功结果
        trainResultDiv.querySelector('.alert').className = 'alert alert-success';
        trainResultDiv.querySelector('.alert-heading').innerHTML = '<i class="fas fa-check-circle me-2"></i>训练成功!';
        
        // 格式化指标数据
        const metricsHtml = `
            <p><strong>模型类型:</strong> ${data.model_type}</p>
            <p><strong>数据量:</strong> ${data.data_size} 条</p>
            <p><strong>模型性能:</strong></p>
            <ul>
                <li>准确率 (Accuracy): ${(data.metrics.accuracy * 100).toFixed(2)}%</li>
                <li>精确率 (Precision): ${(data.metrics.precision * 100).toFixed(2)}%</li>
                <li>召回率 (Recall): ${(data.metrics.recall * 100).toFixed(2)}%</li>
                <li>F1值: ${(data.metrics.f1 * 100).toFixed(2)}%</li>
            </ul>
            <p class="mb-0"><small>已保存模型: ${data.model_path}</small></p>
            
            <div class="mt-3 btn-group">
                <button type="button" class="btn btn-primary" id="use-model-btn">
                    <i class="fas fa-check-circle me-1"></i> 立即使用此模型
                </button>
                <button type="button" class="btn btn-secondary" id="continue-training-btn">
                    <i class="fas fa-sync me-1"></i> 继续训练其他模型
                </button>
            </div>
        `;
        
        trainResultContent.innerHTML = metricsHtml;
        trainResultDiv.classList.remove('d-none');
        
        // 添加使用此模型按钮事件
        document.getElementById('use-model-btn').addEventListener('click', function() {
            // 先自动调用加载模型接口
            fetch('/load_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_path: data.model_path,
                    model_type: data.model_type
                })
            })
            .then(response => response.json())
            .then(result => {
                if (result.success) {
                    // 保存模型选择到localStorage
                    localStorage.setItem('selected_model_type', data.model_type);
                    
                    // 显示加载成功提示
                    showToast('success', '模型已加载', `${getModelName(data.model_type)}模型已成功加载并可以使用`);
                    
                    // 手动保存当前页面状态
                    if (typeof saveCurrentPageState === 'function') {
                        saveCurrentPageState('index');
                    }
                    
                    // 延迟跳转以确保toast可见
                    setTimeout(() => {
                        // 使用锚点触发软路由
                        window.location.href = `/?model_type=${data.model_type}&from=training`;
                    }, 500);
                } else {
                    showToast('danger', '模型加载失败', result.error || '未知错误');
                }
            })
            .catch(error => {
                console.error('加载模型错误:', error);
                showToast('danger', '模型加载失败', error.message);
            });
        });
        
        // 添加继续训练按钮事件
        document.getElementById('continue-training-btn').addEventListener('click', function() {
            trainResultDiv.classList.add('d-none');
            document.getElementById('train-model-form').reset();
            trainButton.disabled = false;
            trainButton.innerHTML = '<i class="fas fa-graduation-cap me-2"></i>开始训练模型';
        });
        
        // 刷新模型列表
        loadSavedModels();
        
        // 重新启用按钮，但保持训练结果可见
        trainButton.disabled = false;
        trainButton.innerHTML = '<i class="fas fa-graduation-cap me-2"></i>开始训练模型';
        
        // 显示提示消息
        showToast('success', '训练成功', '模型已保存，点击"立即使用此模型"使用，或继续训练其他模型');
    }
    
    // 添加刷新模型列表按钮事件
    const refreshModelsBtn = document.getElementById('refresh-models-btn');
    if (refreshModelsBtn) {
        refreshModelsBtn.addEventListener('click', function() {
            loadSavedModels();
        });
    }
}

// 加载已保存的模型列表
async function loadSavedModels() {
    const savedModelsContainer = document.getElementById('saved-models-container');
    const loadingElement = document.getElementById('models-loading');
    const savedModelsList = document.getElementById('saved-models-list');
    const noModelsMessage = document.getElementById('no-models-message');
    const savedModelsAccordion = document.getElementById('savedModelsAccordion');
    
    if (!savedModelsContainer) return;
    
    // 显示加载中状态
    loadingElement.classList.remove('d-none');
    savedModelsList.classList.add('d-none');
    noModelsMessage.classList.add('d-none');
    
    try {
        // 发送请求到后端API
        const response = await fetch('/get_models');
        
        if (!response.ok) {
            throw new Error('获取模型列表失败');
        }
        
        const data = await response.json();
        
        // 检查是否有模型
        let hasModels = false;
        for (const modelType in data.models) {
            if (data.models[modelType].length > 0) {
                hasModels = true;
                break;
            }
        }
        
        if (!hasModels) {
            // 显示无模型消息
            noModelsMessage.classList.remove('d-none');
            savedModelsList.classList.add('d-none');
        } else {
            // 清空并填充模型列表
            savedModelsAccordion.innerHTML = '';
            
            // 遍历每种模型类型
            for (const modelType in data.models) {
                if (data.models[modelType].length === 0) continue;
                
                // 创建模型类型的折叠面板
                const modelTypeItem = document.createElement('div');
                modelTypeItem.className = 'accordion-item';
                
                // 获取模型的友好名称
                const modelTypeName = getModelName(modelType);
                
                // 创建当前加载状态的显示
                const isCurrentlyLoaded = data.current_models[modelType];
                const currentBadge = isCurrentlyLoaded ? 
                    '<span class="badge bg-success ms-2">当前加载</span>' : '';
                
                modelTypeItem.innerHTML = `
                    <h2 class="accordion-header">
                        <button class="accordion-button collapsed" type="button" 
                                data-bs-toggle="collapse" data-bs-target="#collapse${modelType}">
                            <span class="model-badge badge-${modelType}">${modelTypeName}</span>
                            ${currentBadge}
                            <span class="ms-2 badge bg-primary">${data.models[modelType].length}</span>
                        </button>
                    </h2>
                    <div id="collapse${modelType}" class="accordion-collapse collapse" data-bs-parent="#savedModelsAccordion">
                        <div class="accordion-body p-2">
                            <div class="list-group list-group-flush">
                                ${data.models[modelType].map(model => `
                                    <div class="list-group-item">
                                        <div class="d-flex justify-content-between align-items-center">
                                            <div>
                                                <span class="badge bg-secondary me-2">${model.date}</span>
                                                <small class="text-muted">${model.filename}</small>
                                            </div>
                                            <div>
                                                <div class="btn-group btn-group-sm">
                                                    <button class="btn btn-outline-primary load-model-btn" 
                                                            data-model-path="${model.path}" data-model-type="${modelType}">
                                                        <i class="fas fa-upload"></i> 加载
                                                    </button>
                                                    <button class="btn btn-outline-danger delete-model-btn"
                                                            data-model-path="${model.path}" data-model-type="${modelType}">
                                                        <i class="fas fa-trash"></i>
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    </div>
                `;
                
                savedModelsAccordion.appendChild(modelTypeItem);
            }
            
            // 显示模型列表
            savedModelsList.classList.remove('d-none');
            
            // 添加加载模型按钮的事件处理
            document.querySelectorAll('.load-model-btn').forEach(btn => {
                btn.addEventListener('click', async function() {
                    const modelPath = this.getAttribute('data-model-path');
                    const modelType = this.getAttribute('data-model-type');
                    
                    // 显示加载状态
                    this.disabled = true;
                    this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>';
                    
                    try {
                        // 调用加载模型的API
                        const response = await fetch('/load_model', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                model_path: modelPath,
                                model_type: modelType,
                                redirect_home: true
                            })
                        });
                        
                        const data = await response.json();
                        
                        if (!response.ok) {
                            throw new Error(data.error || '加载模型失败');
                        }
                        
                        // 显示成功消息
                        showToast('success', '加载成功', `${getModelName(modelType)}模型已成功加载`);
                        
                        // 保存模型选择到localStorage
                        localStorage.setItem('selected_model_type', modelType);
                        
                        // 刷新列表以更新当前加载状态
                        await loadSavedModels();
                        
                        // 检查响应中是否有重定向属性
                        if (data.redirect) {
                            // 自动重定向到指定的URL
                            window.location.href = data.redirect;
                        }
                        
                    } catch (error) {
                        console.error('加载模型错误:', error);
                        showToast('danger', '加载失败', error.message);
                    } finally {
                        // 恢复按钮状态
                        this.disabled = false;
                        this.innerHTML = '<i class="fas fa-upload"></i> 加载';
                    }
                });
            });
            
            // 添加删除模型按钮的事件处理
            document.querySelectorAll('.delete-model-btn').forEach(btn => {
                btn.addEventListener('click', async function() {
                    const modelPath = this.getAttribute('data-model-path');
                    const modelType = this.getAttribute('data-model-type');
                    const modelItem = this.closest('.list-group-item');
                    
                    // 显示确认对话框
                    if (!confirm(`确定要删除此模型吗？\n模型类型: ${getModelName(modelType)}\n模型路径: ${modelPath}`)) {
                        return;  // 用户取消删除
                    }
                    
                    // 显示加载状态
                    this.disabled = true;
                    this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>';
                    
                    try {
                        // 调用后端API删除模型
                        const response = await fetch('/delete_model', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                model_path: modelPath
                            })
                        });
                        
                        const data = await response.json();
                        
                        if (!response.ok) {
                            throw new Error(data.error || '删除模型失败');
                        }
                        
                        // 删除成功，从DOM中移除该项
                        modelItem.remove();
                        
                        // 显示成功消息
                        showToast('success', '删除成功', '模型已成功删除');
                        
                        // 检查该模型类型是否还有其他模型，如果没有则隐藏该类型
                        const modelTypeContainer = document.querySelector(`#collapse${modelType}`);
                        if (modelTypeContainer && modelTypeContainer.querySelectorAll('.list-group-item').length === 0) {
                            const accordionItem = modelTypeContainer.closest('.accordion-item');
                            if (accordionItem) {
                                accordionItem.remove();
                            }
                        }
                        
                        // 检查是否还有任何模型，如果没有则显示"无模型"消息
                        const allModelItems = document.querySelectorAll('.accordion-item');
                        if (allModelItems.length === 0) {
                            savedModelsList.classList.add('d-none');
                            noModelsMessage.classList.remove('d-none');
                        }
                        
                    } catch (error) {
                        console.error('删除模型错误:', error);
                        showToast('danger', '删除失败', error.message);
                        
                        // 恢复按钮状态
                        this.disabled = false;
                        this.innerHTML = '<i class="fas fa-trash"></i>';
                    }
                });
            });
        }
        
    } catch (error) {
        console.error('获取模型列表错误:', error);
        noModelsMessage.innerHTML = `
            <i class="fas fa-exclamation-circle text-danger mb-2" style="font-size: 2rem;"></i>
            <p>获取模型列表失败: ${error.message}</p>
        `;
        noModelsMessage.classList.remove('d-none');
    } finally {
        // 隐藏加载状态
        loadingElement.classList.add('d-none');
    }
}

// 获取模型名称的友好显示
function getModelName(modelType) {
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
        'ensemble': 'Ensemble'
    };
    
    return modelNames[modelType] || modelType;
}

// 显示Toast消息（复用main.js中的函数）
// 创建列选择模态框
function createColumnsModal() {
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.id = 'columnsModal';
    modal.setAttribute('tabindex', '-1');
    modal.setAttribute('aria-labelledby', 'columnsModalLabel');
    modal.setAttribute('aria-hidden', 'true');
    
    modal.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="columnsModalLabel">选择数据列</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <p>请选择CSV文件中的文本列和标签列：</p>
                    <form id="columns-form">
                        <div class="mb-3">
                            <label for="text-column-select" class="form-label">文本列</label>
                            <select class="form-select" id="text-column-select" required>
                                <!-- 选项将由JavaScript动态生成 -->
                            </select>
                            <div class="form-text">包含短信文本内容的列</div>
                        </div>
                        <div class="mb-3">
                            <label for="label-column-select" class="form-label">标签列</label>
                            <select class="form-select" id="label-column-select" required>
                                <!-- 选项将由JavaScript动态生成 -->
                            </select>
                            <div class="form-text">包含标签（垃圾/正常）的列</div>
                        </div>
                        <div class="d-grid">
                            <button type="submit" class="btn btn-primary">确认并开始训练</button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    return modal;
}

function showToast(type, title, message) {
    let toastContainer = document.querySelector('.toast-container');
    
    if (!toastContainer) {
        // 如果没有Toast容器，创建一个
        const newContainer = document.createElement('div');
        newContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(newContainer);
        
        // 更新引用
        toastContainer = newContainer;
    }
    
    // 创建Toast元素
    const toastEl = document.createElement('div');
    toastEl.className = 'toast show';
    toastEl.setAttribute('role', 'alert');
    toastEl.setAttribute('aria-live', 'assertive');
    toastEl.setAttribute('aria-atomic', 'true');
    
    // 设置Toast内容
    let headerClass = '';
    let headerIcon = '';
    
    switch (type) {
        case 'success':
            headerClass = 'bg-success text-white';
            headerIcon = '✓ ';
            break;
        case 'danger':
        case 'error':
            headerClass = 'bg-danger text-white';
            headerIcon = '❌ ';
            break;
        case 'warning':
            headerClass = 'bg-warning';
            headerIcon = '⚠️ ';
            break;
        default:
            headerClass = 'bg-info text-white';
            headerIcon = 'ℹ️ ';
    }
    
    toastEl.innerHTML = `
        <div class="toast-header ${headerClass}">
            <strong class="me-auto">${headerIcon}${title}</strong>
            <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
        <div class="toast-body">
            ${message}
        </div>
    `;
    
    // 添加到容器
    toastContainer.appendChild(toastEl);
    
    // 自动消失
    setTimeout(() => {
        toastEl.remove();
    }, 5000);
    
    // 添加关闭按钮事件
    const closeBtn = toastEl.querySelector('.btn-close');
    closeBtn.addEventListener('click', () => {
        toastEl.remove();
    });
}