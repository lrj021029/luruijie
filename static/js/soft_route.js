/**
 * 软路由实现 - 在不同页面之间切换时保持状态
 * 使用app_state.js提供的全局状态管理系统
 */

// 初始化软路由
document.addEventListener("DOMContentLoaded", function() {
    console.log('初始化软路由...');
    
    // 拦截导航链接点击
    setupNavLinkInterception();
    
    // 从URL获取当前页面类型
    const currentPath = window.location.pathname;
    let currentPage = 'index';
    
    if (currentPath.includes('features')) {
        currentPage = 'features';
    } else if (currentPath.includes('history')) {
        currentPage = 'history';
    } else if (currentPath.includes('datasets')) {
        currentPage = 'datasets';
    }
    
    console.log('当前页面:', currentPage);
    
    // 从URL参数中获取信息
    const urlParams = new URLSearchParams(window.location.search);
    const modelType = urlParams.get('model_type');
    const fromTraining = urlParams.get('from') === 'training';
    
    // 如果是从训练页返回首页，加载最新的模型列表
    if (fromTraining && currentPage === 'index') {
        console.log('从训练页返回，将加载最新模型列表');
        // 延迟执行，确保DOM已完全加载
        setTimeout(() => {
            if (typeof loadTrainedModelsForPrediction === 'function') {
                loadTrainedModelsForPrediction().then(() => {
                    console.log('已加载最新模型列表');
                    // 然后从本地存储恢复其他状态
                    restorePageState(currentPage);
                });
            } else {
                // 如果函数不可用，仍然尝试恢复状态
                restorePageState(currentPage);
            }
        }, 100);
    } else {
        // 从本地存储恢复状态
        restorePageState(currentPage);
    }
    
    // 在页面离开时保存状态
    window.addEventListener('beforeunload', function() {
        saveCurrentPageState(currentPage);
    });
    
    // 从本地存储初始化模型选择（如果URL没有指定）
    if (!modelType) {
        const savedModelType = localStorage.getItem('selected_model_type');
        const modelSelect = document.getElementById('model-select');
        
        if (savedModelType && modelSelect) {
            // 延迟执行，确保下拉选项已加载
            setTimeout(() => {
                if (Array.from(modelSelect.options).some(opt => opt.value === savedModelType)) {
                    modelSelect.value = savedModelType;
                    console.log('已从localStorage恢复模型选择:', savedModelType);
                }
            }, 300);
        }
    }
});

// 设置导航链接拦截
function setupNavLinkInterception() {
    document.querySelectorAll('.navbar-nav .nav-link').forEach(link => {
        link.addEventListener('click', function(event) {
            // 获取当前页面和目标页面
            const currentPath = window.location.pathname;
            const targetPath = this.getAttribute('href');
            
            // 当前页面类型
            let currentPage = 'index';
            if (currentPath.includes('features')) {
                currentPage = 'features';
            } else if (currentPath.includes('history')) {
                currentPage = 'history';
            } else if (currentPath.includes('datasets')) {
                currentPage = 'datasets';
            }
            
            // 保存当前页面状态
            saveCurrentPageState(currentPage);
            
            // 正常导航到目标页面
            // 不阻止默认行为，让浏览器正常加载目标页面
        });
    });
}

// 保存当前页面状态
function saveCurrentPageState(pageType) {
    try {
        // 确保AppState已初始化
        if (!window.AppState) {
            console.warn('AppState未初始化，无法保存状态');
            return;
        }
        
        // A. 使用AppState存储数据
        // 根据页面类型收集和保存不同的数据
        switch (pageType) {
            case 'index':
                // 保存首页数据
                const textInput = document.getElementById('text-input');
                const sendFreqInput = document.getElementById('send-freq-input');
                const isNightInput = document.getElementById('is-night-input');
                const modelSelect = document.getElementById('model-select');
                const predictionResult = document.getElementById('prediction-result');
                const modelTrainingResult = document.querySelector('.train-result');
                const trainingContent = document.getElementById('train-result-content');
                
                // 批量更新首页状态
                const indexUpdates = {};
                
                if (textInput) indexUpdates.predictionText = textInput.value;
                if (sendFreqInput) indexUpdates.sendFreq = sendFreqInput.value;
                if (isNightInput) indexUpdates.isNight = isNightInput.checked ? 1 : 0;
                if (modelSelect) indexUpdates.modelType = modelSelect.value;
                
                if (predictionResult && !predictionResult.classList.contains('d-none')) {
                    indexUpdates.predictionResult = predictionResult.innerHTML;
                }
                
                // 保存模型训练结果
                if (modelTrainingResult && !modelTrainingResult.classList.contains('d-none') && trainingContent) {
                    indexUpdates.trainingResult = {
                        visible: true,
                        content: trainingContent.innerHTML
                    };
                } else {
                    indexUpdates.trainingResult = {
                        visible: false,
                        content: ''
                    };
                }
                
                // 批量更新状态
                window.AppState.updatePageState('index', indexUpdates);
                break;
                
            case 'features':
                // 特征页面数据
                const featuresUpdates = {
                    loaded: true,
                    scrollPosition: window.scrollY
                };
                
                // 尝试缓存词云数据
                if (window.cachedSpamWordCloud) {
                    featuresUpdates.spamWordCloud = window.cachedSpamWordCloud;
                }
                if (window.cachedHamWordCloud) {
                    featuresUpdates.hamWordCloud = window.cachedHamWordCloud;
                }
                
                // 尝试缓存模型性能指标
                if (window.cachedModelMetrics) {
                    featuresUpdates.modelMetrics = window.cachedModelMetrics;
                }
                
                window.AppState.updatePageState('features', featuresUpdates);
                break;
                
            case 'history':
                // 历史页面数据
                const historyUpdates = {
                    loaded: true,
                    scrollPosition: window.scrollY
                };
                
                const filterInput = document.getElementById('history-filter');
                if (filterInput) historyUpdates.filter = filterInput.value;
                
                // 保存选中的记录ID
                const checkboxes = document.querySelectorAll('.history-select-checkbox:checked');
                if (checkboxes.length > 0) {
                    historyUpdates.selectedIds = Array.from(checkboxes)
                        .map(cb => parseInt(cb.value));
                }
                
                // 保存记录数据
                if (window.cachedHistoryData) {
                    historyUpdates.records = window.cachedHistoryData;
                }
                
                // 保存排序状态
                if (window.currentSortColumn) {
                    historyUpdates.sortColumn = window.currentSortColumn;
                    historyUpdates.sortDirection = window.currentSortDirection || 'desc';
                }
                
                // 保存分页信息
                if (window.currentPage) {
                    historyUpdates.currentPage = window.currentPage;
                }
                
                window.AppState.updatePageState('history', historyUpdates);
                break;
                
            case 'datasets':
                // 数据集页面数据
                const datasetsUpdates = {
                    loaded: true,
                    scrollPosition: window.scrollY
                };
                
                // 保存表单状态
                const nameInput = document.getElementById('dataset-name');
                const descriptionInput = document.getElementById('dataset-description');
                
                if (nameInput || descriptionInput) {
                    datasetsUpdates.uploadForm = {
                        name: nameInput ? nameInput.value : '',
                        description: descriptionInput ? descriptionInput.value : ''
                    };
                }
                
                // 保存数据集列表数据
                if (window.cachedDatasetsData) {
                    datasetsUpdates.datasetsList = window.cachedDatasetsData;
                }
                
                window.AppState.updatePageState('datasets', datasetsUpdates);
                break;
        }
        
        // 显式保存到存储中以确保持久性
        window.AppState.saveToStorage();
        console.log(`已保存${pageType}页面状态`);
    } catch (error) {
        console.error('保存页面状态错误:', error);
    }
}

// 恢复页面状态
function restorePageState(pageType) {
    try {
        // 确保AppState已初始化
        if (!window.AppState) {
            console.warn('AppState未初始化，无法恢复状态');
            return;
        }
        
        // 从AppState获取页面状态
        const pageState = window.AppState.getPageState(pageType);
        if (!pageState) return;
        
        // 使用AppState中的数据
        
        // 根据页面类型恢复不同的数据
        switch (pageType) {
            case 'index':
                // 检查是否有URL参数
                const urlParams = new URLSearchParams(window.location.search);
                const fromTraining = urlParams.get('from') === 'training';
                const modelTypeParam = urlParams.get('model_type');
                
                // 恢复首页数据
                const textInput = document.getElementById('text-input');
                const sendFreqInput = document.getElementById('send-freq-input');
                const isNightInput = document.getElementById('is-night-input');
                const modelSelect = document.getElementById('model-select');
                const predictionResult = document.getElementById('prediction-result');
                const trainResultDiv = document.querySelector('.train-result');
                const trainResultContent = document.getElementById('train-result-content');
                
                // 先等待模型列表加载
                setTimeout(() => {
                    // 如果从训练页面返回，优先使用URL参数中的模型类型
                    if (fromTraining && modelTypeParam && modelSelect) {
                        // 刷新模型列表后再选择模型
                        if (typeof loadTrainedModelsForPrediction === 'function') {
                            loadTrainedModelsForPrediction().then(() => {
                                console.log('模型列表已刷新，尝试选择模型:', modelTypeParam);
                                // 等待模型列表加载完成后再选择
                                setTimeout(() => {
                                    if (Array.from(modelSelect.options).some(opt => opt.value === modelTypeParam)) {
                                        modelSelect.value = modelTypeParam;
                                        console.log('已选择模型:', modelTypeParam);
                                    }
                                }, 200);
                            });
                        }
                    } else if (modelSelect && pageState.modelType) {
                        // 恢复保存的模型选择
                        if (Array.from(modelSelect.options).some(opt => opt.value === pageState.modelType)) {
                            modelSelect.value = pageState.modelType;
                            console.log('已恢复模型选择:', pageState.modelType);
                        }
                    }
                }, 300);
                
                // 恢复其他表单字段
                if (textInput && pageState.predictionText) {
                    textInput.value = pageState.predictionText;
                }
                
                if (sendFreqInput && pageState.sendFreq !== null) {
                    sendFreqInput.value = pageState.sendFreq;
                }
                
                if (isNightInput && pageState.isNight !== null) {
                    isNightInput.checked = pageState.isNight === 1;
                }
                
                // 恢复预测结果
                if (predictionResult && pageState.predictionResult) {
                    predictionResult.innerHTML = pageState.predictionResult;
                    predictionResult.classList.remove('d-none');
                }
                
                // 恢复训练结果
                if (trainResultDiv && trainResultContent && pageState.trainingResult) {
                    if (pageState.trainingResult.visible && pageState.trainingResult.content) {
                        trainResultContent.innerHTML = pageState.trainingResult.content;
                        trainResultDiv.classList.remove('d-none');
                        
                        // 重新添加按钮事件
                        const useModelBtn = document.getElementById('use-model-btn');
                        const continueTrainingBtn = document.getElementById('continue-training-btn');
                        
                        if (useModelBtn) {
                            useModelBtn.addEventListener('click', function() {
                                if (typeof loadTrainedModelsForPrediction === 'function') {
                                    loadTrainedModelsForPrediction();
                                }
                                trainResultDiv.classList.add('d-none');
                            });
                        }
                        
                        if (continueTrainingBtn) {
                            continueTrainingBtn.addEventListener('click', function() {
                                trainResultDiv.classList.add('d-none');
                                const trainForm = document.getElementById('train-model-form');
                                if (trainForm) trainForm.reset();
                            });
                        }
                    } else {
                        trainResultDiv.classList.add('d-none');
                    }
                }
                break;
                
            case 'features':
                // 特征页面
                if (pageState.loaded) {
                    // 恢复缓存的数据
                    if (pageState.spamWordCloud) {
                        window.cachedSpamWordCloud = pageState.spamWordCloud;
                    }
                    if (pageState.hamWordCloud) {
                        window.cachedHamWordCloud = pageState.hamWordCloud;
                    }
                    if (pageState.modelMetrics) {
                        window.cachedModelMetrics = pageState.modelMetrics;
                    }
                    
                    // 判断是否需要重新加载，如果已有缓存数据则尝试直接使用
                    const hasWordCloudCache = window.cachedSpamWordCloud && window.cachedHamWordCloud;
                    
                    // 自动触发加载或使用缓存
                    setTimeout(() => {
                        if (typeof loadWordCloudData === 'function') {
                            // 如果有缓存，则直接渲染
                            if (hasWordCloudCache && typeof renderWordCloud === 'function') {
                                console.log('使用缓存的词云数据');
                                // 获取容器
                                const spamContainer = document.getElementById('spam-word-cloud');
                                const hamContainer = document.getElementById('ham-word-cloud');
                                
                                if (spamContainer && hamContainer) {
                                    // 清空容器
                                    spamContainer.innerHTML = '';
                                    hamContainer.innerHTML = '';
                                    
                                    // 渲染词云
                                    renderWordCloud(spamContainer, window.cachedSpamWordCloud, 'danger');
                                    renderWordCloud(hamContainer, window.cachedHamWordCloud, 'info');
                                } else {
                                    // 容器不存在，手动加载
                                    loadWordCloudData();
                                }
                            } else {
                                // 否则重新加载
                                loadWordCloudData();
                            }
                            
                            // 恢复模型性能指标显示
                            if (window.cachedModelMetrics && typeof renderModelMetricsChart === 'function') {
                                setTimeout(() => {
                                    const metricsContainer = document.getElementById('model-metrics-container');
                                    if (metricsContainer) {
                                        renderModelMetricsChart(metricsContainer, window.cachedModelMetrics);
                                    }
                                }, 200);
                            }
                        }
                        
                        // 恢复滚动位置
                        if (pageState.scrollPosition) {
                            setTimeout(() => {
                                window.scrollTo(0, pageState.scrollPosition);
                            }, 300);
                        }
                    }, 500);
                }
                break;
                
            case 'history':
                // 历史页面
                if (pageState.loaded) {
                    // 恢复过滤器值
                    const filterInput = document.getElementById('history-filter');
                    if (filterInput && pageState.filter) {
                        filterInput.value = pageState.filter;
                    }
                    
                    // 恢复缓存的记录数据
                    if (pageState.records) {
                        window.cachedHistoryData = pageState.records;
                    }
                    
                    // 恢复排序设置
                    if (pageState.sortColumn) {
                        window.currentSortColumn = pageState.sortColumn;
                        window.currentSortDirection = pageState.sortDirection;
                    }
                    
                    // 恢复分页设置
                    if (pageState.currentPage) {
                        window.currentPage = pageState.currentPage;
                    }
                    
                    // 恢复历史记录
                    setTimeout(() => {
                        if (typeof loadHistory === 'function') {
                            // 如果我们有缓存的数据
                            if (window.cachedHistoryData && typeof displayHistoryData === 'function') {
                                console.log('使用缓存的历史记录数据');
                                displayHistoryData(window.cachedHistoryData);
                                
                                // 恢复选中状态
                                if (pageState.selectedIds && pageState.selectedIds.length > 0) {
                                    setTimeout(() => {
                                        pageState.selectedIds.forEach(id => {
                                            const checkbox = document.querySelector(`.history-select-checkbox[value="${id}"]`);
                                            if (checkbox) {
                                                checkbox.checked = true;
                                            }
                                        });
                                        
                                        // 更新删除选中按钮状态
                                        if (typeof updateDeleteSelectedButton === 'function') {
                                            updateDeleteSelectedButton();
                                        }
                                    }, 200);
                                }
                            } else {
                                // 否则重新加载
                                loadHistory().then(() => {
                                    // 应用过滤器
                                    if (filterInput && filterInput.value && typeof filterHistory === 'function') {
                                        setTimeout(filterHistory, 300);
                                    }
                                });
                            }
                            
                            // 恢复滚动位置
                            if (pageState.scrollPosition) {
                                setTimeout(() => {
                                    window.scrollTo(0, pageState.scrollPosition);
                                }, 300);
                            }
                        }
                    }, 500);
                }
                break;
                
            case 'datasets':
                // 数据集页面
                if (pageState.loaded) {
                    // 恢复表单值
                    const nameInput = document.getElementById('dataset-name');
                    const descriptionInput = document.getElementById('dataset-description');
                    
                    if (nameInput && pageState.uploadForm && pageState.uploadForm.name) {
                        nameInput.value = pageState.uploadForm.name;
                    }
                    
                    if (descriptionInput && pageState.uploadForm && pageState.uploadForm.description) {
                        descriptionInput.value = pageState.uploadForm.description;
                    }
                    
                    // 恢复缓存的数据集列表
                    if (pageState.datasetsList) {
                        window.cachedDatasetsData = pageState.datasetsList;
                    }
                    
                    // 自动加载数据集列表
                    setTimeout(() => {
                        if (typeof loadDatasets === 'function') {
                            // 如果有缓存数据，可以考虑直接使用
                            if (window.cachedDatasetsData) {
                                // 这里可以添加使用缓存数据的逻辑
                                // 现在先直接重新加载数据
                                loadDatasets();
                            } else {
                                // 否则加载新数据
                                loadDatasets();
                            }
                            
                            // 恢复滚动位置
                            if (pageState.scrollPosition) {
                                setTimeout(() => {
                                    window.scrollTo(0, pageState.scrollPosition);
                                }, 300);
                            }
                        }
                    }, 200);
                }
                break;
        }
    } catch (error) {
        console.error('恢复页面状态错误:', error);
    }
}