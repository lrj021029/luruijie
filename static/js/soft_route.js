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
        predictionResult: null    // 预测结果
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
    
    // 从本地存储恢复状态
    restorePageState(currentPage);
    
    // 在页面离开时保存状态
    window.addEventListener('beforeunload', function() {
        saveCurrentPageState(currentPage);
    });
    
    // 从URL参数中获取modelType
    const urlParams = new URLSearchParams(window.location.search);
    const modelType = urlParams.get('model_type');
    
    // 如果URL中有模型类型参数，则使用它
    if (modelType && document.getElementById('model-select')) {
        const modelSelect = document.getElementById('model-select');
        if (Array.from(modelSelect.options).some(opt => opt.value === modelType)) {
            modelSelect.value = modelType;
            // 将模型类型保存到localStorage
            localStorage.setItem('selected_model_type', modelType);
        }
    } else {
        // 否则尝试从localStorage获取已选择的模型类型
        const savedModelType = localStorage.getItem('selected_model_type');
        if (savedModelType && document.getElementById('model-select')) {
            const modelSelect = document.getElementById('model-select');
            if (Array.from(modelSelect.options).some(opt => opt.value === savedModelType)) {
                modelSelect.value = savedModelType;
            }
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
                
                if (textInput) currentPageData.index.predictionText = textInput.value;
                if (sendFreqInput) currentPageData.index.sendFreq = sendFreqInput.value;
                if (isNightInput) currentPageData.index.isNight = isNightInput.checked ? 1 : 0;
                if (modelSelect) currentPageData.index.modelType = modelSelect.value;
                if (predictionResult && !predictionResult.classList.contains('d-none')) {
                    currentPageData.index.predictionResult = predictionResult.innerHTML;
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
                // 恢复首页数据
                const textInput = document.getElementById('text-input');
                const sendFreqInput = document.getElementById('send-freq-input');
                const isNightInput = document.getElementById('is-night-input');
                const modelSelect = document.getElementById('model-select');
                const predictionResult = document.getElementById('prediction-result');
                
                if (textInput && savedData.index.predictionText) {
                    textInput.value = savedData.index.predictionText;
                }
                
                if (sendFreqInput && savedData.index.sendFreq !== null) {
                    sendFreqInput.value = savedData.index.sendFreq;
                }
                
                if (isNightInput && savedData.index.isNight !== null) {
                    isNightInput.checked = savedData.index.isNight === 1;
                }
                
                if (modelSelect && savedData.index.modelType) {
                    // 检查该选项是否存在
                    if (Array.from(modelSelect.options).some(opt => opt.value === savedData.index.modelType)) {
                        modelSelect.value = savedData.index.modelType;
                    }
                }
                
                if (predictionResult && savedData.index.predictionResult) {
                    predictionResult.innerHTML = savedData.index.predictionResult;
                    predictionResult.classList.remove('d-none');
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