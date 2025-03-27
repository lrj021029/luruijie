/**
 * 软路由实现 - 在不同页面之间切换时保持状态
 */

// 存储当前页面数据和状态
let currentPageData = {
    // 首页数据
    index: {
        predictionText: null,     // 预测文本内容
        sendFreq: 0,              // 发送频率
        isNight: 0,               // 是否夜间发送
        modelType: null,          // 选择的模型类型
        predictionResult: null,   // 预测结果
        trainingResult: {         // 训练结果
            visible: false,       
            content: ''
        }
    },
    // 特征页面数据
    features: {
        loaded: false             // 是否已加载特征数据
    },
    // 历史页面数据
    history: {
        loaded: false,            // 是否已加载历史数据
        filter: ''                // 筛选条件
    }
};

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
        // 根据页面类型保存不同的数据
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
                
                if (textInput) currentPageData.index.predictionText = textInput.value;
                if (sendFreqInput) currentPageData.index.sendFreq = sendFreqInput.value;
                if (isNightInput) currentPageData.index.isNight = isNightInput.checked ? 1 : 0;
                if (modelSelect) {
                    currentPageData.index.modelType = modelSelect.value;
                    // 同时保存模型选择到全局选择
                    localStorage.setItem('selected_model_type', modelSelect.value);
                }
                if (predictionResult && !predictionResult.classList.contains('d-none')) {
                    currentPageData.index.predictionResult = predictionResult.innerHTML;
                }
                
                // 保存模型训练结果
                if (modelTrainingResult && !modelTrainingResult.classList.contains('d-none') && trainingContent) {
                    currentPageData.index.trainingResult = {
                        visible: true,
                        content: trainingContent.innerHTML
                    };
                } else {
                    currentPageData.index.trainingResult = {
                        visible: false,
                        content: ''
                    };
                }
                break;
                
            case 'features':
                // 特征页面数据已加载标记
                currentPageData.features.loaded = true;
                break;
                
            case 'history':
                // 历史页面数据
                const filterInput = document.getElementById('history-filter');
                if (filterInput) currentPageData.history.filter = filterInput.value;
                currentPageData.history.loaded = true;
                break;
        }
        
        // 保存到localStorage
        localStorage.setItem('pageState', JSON.stringify(currentPageData));
        console.log(`已保存${pageType}页面状态到localStorage`);
    } catch (error) {
        console.error('保存页面状态错误:', error);
    }
}

// 恢复页面状态
function restorePageState(pageType) {
    try {
        // 从localStorage获取保存的状态
        const savedState = localStorage.getItem('pageState');
        if (!savedState) return;
        
        // 解析保存的状态
        const savedData = JSON.parse(savedState);
        
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
                    } else if (modelSelect && savedData.index.modelType) {
                        // 恢复保存的模型选择
                        if (Array.from(modelSelect.options).some(opt => opt.value === savedData.index.modelType)) {
                            modelSelect.value = savedData.index.modelType;
                            console.log('已恢复模型选择:', savedData.index.modelType);
                        }
                    }
                }, 300);
                
                // 恢复其他表单字段
                if (textInput && savedData.index.predictionText) {
                    textInput.value = savedData.index.predictionText;
                }
                
                if (sendFreqInput && savedData.index.sendFreq !== null) {
                    sendFreqInput.value = savedData.index.sendFreq;
                }
                
                if (isNightInput && savedData.index.isNight !== null) {
                    isNightInput.checked = savedData.index.isNight === 1;
                }
                
                // 恢复预测结果
                if (predictionResult && savedData.index.predictionResult) {
                    predictionResult.innerHTML = savedData.index.predictionResult;
                    predictionResult.classList.remove('d-none');
                }
                
                // 恢复训练结果
                if (trainResultDiv && trainResultContent && savedData.index.trainingResult) {
                    if (savedData.index.trainingResult.visible && savedData.index.trainingResult.content) {
                        trainResultContent.innerHTML = savedData.index.trainingResult.content;
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
                // 特征页面 - 如果之前已加载过，则自动重新加载
                if (savedData.features.loaded) {
                    // 自动触发加载
                    setTimeout(() => {
                        if (typeof loadWordCloudData === 'function') {
                            loadWordCloudData();
                        }
                    }, 500);
                }
                break;
                
            case 'history':
                // 历史页面
                const filterInput = document.getElementById('history-filter');
                if (filterInput && savedData.history.filter) {
                    filterInput.value = savedData.history.filter;
                }
                
                // 如果之前已加载过，则自动重新加载
                if (savedData.history.loaded) {
                    setTimeout(() => {
                        if (typeof loadHistory === 'function') {
                            loadHistory();
                            // 应用过滤器
                            if (filterInput && filterInput.value && typeof filterHistory === 'function') {
                                setTimeout(filterHistory, 300);
                            }
                        }
                    }, 500);
                }
                break;
        }
    } catch (error) {
        console.error('恢复页面状态错误:', error);
    }
}