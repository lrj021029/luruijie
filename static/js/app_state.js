/**
 * 全局状态管理系统
 * 提供一个全局状态对象和相关API，用于管理应用程序的全局状态
 * 替代直接使用localStorage来实现更健壮的状态管理
 */

// 全局状态对象
const AppState = {
    // 存储状态的主对象
    state: {
        // 首页数据
        index: {
            predictionText: '',     // 预测文本内容
            sendFreq: 0,            // 发送频率
            isNight: 0,             // 是否夜间发送
            modelType: null,        // 选择的模型类型
            predictionResult: null, // 预测结果
            trainingResult: {       // 训练结果
                visible: false,       
                content: ''
            }
        },
        // 特征页面数据
        features: {
            loaded: false,          // 是否已加载特征数据
            spamWordCloud: null,    // 垃圾短信词云数据缓存
            hamWordCloud: null,     // 正常短信词云数据缓存
            modelMetrics: null,     // 模型性能指标数据缓存
            scrollPosition: 0       // 页面滚动位置
        },
        // 历史页面数据
        history: {
            loaded: false,          // 是否已加载历史数据
            filter: '',             // 筛选条件
            records: null,          // 历史记录数据缓存
            selectedIds: [],        // 选中的记录ID
            sortColumn: 'timestamp',// 排序列
            sortDirection: 'desc',  // 排序方向
            currentPage: 1,         // 当前页码
            scrollPosition: 0       // 页面滚动位置
        },
    },
    
    // 订阅者列表，按页面和属性路径分类
    subscribers: {
        index: {},
        features: {},
        history: {}
    },
    
    /**
     * 初始化状态管理系统
     * 从localStorage中加载之前保存的状态
     */
    init() {
        try {
            // 从localStorage获取保存的状态
            const savedState = localStorage.getItem('appState');
            if (savedState) {
                // 合并保存的状态
                const parsedState = JSON.parse(savedState);
                this.state = this._mergeObjects(this.state, parsedState);
                console.log('已从存储中恢复应用状态');
            }
        } catch (error) {
            console.error('初始化应用状态时出错:', error);
        }
        
        // 注册卸载前保存状态
        window.addEventListener('beforeunload', () => {
            this.saveToStorage();
        });
        
        return this;
    },
    
    /**
     * 获取指定页面的状态
     * @param {string} page - 页面名称('index', 'features', 'history')
     * @returns {object} 页面状态对象
     */
    getPageState(page) {
        return this.state[page] || {};
    },
    
    /**
     * 获取指定路径的状态值
     * @param {string} path - 状态路径，如'index.modelType'
     * @returns {any} 状态值
     */
    get(path) {
        const parts = path.split('.');
        let current = this.state;
        
        for (const part of parts) {
            if (current === undefined || current === null) {
                return undefined;
            }
            current = current[part];
        }
        
        return current;
    },
    
    /**
     * 设置指定路径的状态值
     * @param {string} path - 状态路径，如'index.modelType'
     * @param {any} value - 要设置的值
     */
    set(path, value) {
        const parts = path.split('.');
        const lastPart = parts.pop();
        let current = this.state;
        
        // 遍历路径确保对象存在
        for (const part of parts) {
            if (!current[part]) {
                current[part] = {};
            }
            current = current[part];
        }
        
        // 设置值
        const oldValue = current[lastPart];
        current[lastPart] = value;
        
        // 通知订阅者
        this._notifySubscribers(path, value, oldValue);
        
        // 如果是模型类型，自动保存到localStorage
        if (path === 'index.modelType' && value) {
            localStorage.setItem('selected_model_type', value);
        }
        
        return this;
    },
    
    /**
     * 批量更新状态
     * @param {string} pageName - 页面名称
     * @param {object} updates - 要更新的状态对象
     */
    updatePageState(pageName, updates) {
        if (!this.state[pageName]) {
            this.state[pageName] = {};
        }
        
        this.state[pageName] = this._mergeObjects(this.state[pageName], updates);
        
        // 通知页面级订阅者
        Object.keys(this.subscribers[pageName] || {}).forEach(path => {
            const fullPath = `${pageName}.${path}`;
            if (this.subscribers[pageName][path]) {
                this.subscribers[pageName][path].forEach(callback => {
                    try {
                        callback(this.get(fullPath));
                    } catch (error) {
                        console.error(`调用订阅回调出错 ${fullPath}:`, error);
                    }
                });
            }
        });
        
        return this;
    },
    
    /**
     * 订阅状态变化
     * @param {string} path - 状态路径，如'index.modelType'
     * @param {function} callback - 当状态改变时调用的回调函数
     * @returns {function} 取消订阅的函数
     */
    subscribe(path, callback) {
        const parts = path.split('.');
        const pageName = parts[0];
        const subPath = parts.slice(1).join('.');
        
        // 初始化订阅者对象
        if (!this.subscribers[pageName]) {
            this.subscribers[pageName] = {};
        }
        
        if (!this.subscribers[pageName][subPath]) {
            this.subscribers[pageName][subPath] = [];
        }
        
        // 添加订阅
        this.subscribers[pageName][subPath].push(callback);
        
        // 返回取消订阅的函数
        return () => {
            if (this.subscribers[pageName][subPath]) {
                this.subscribers[pageName][subPath] = this.subscribers[pageName][subPath]
                    .filter(cb => cb !== callback);
            }
        };
    },
    
    /**
     * 保存状态到localStorage
     */
    saveToStorage() {
        try {
            localStorage.setItem('appState', JSON.stringify(this.state));
            console.log('已保存应用状态到本地存储');
        } catch (error) {
            console.error('保存应用状态时出错:', error);
        }
    },
    
    /**
     * 清除指定页面的状态
     * @param {string} pageName - 页面名称
     */
    clearPageState(pageName) {
        if (this.state[pageName]) {
            // 保留页面对象，但重置其中的属性
            const resetObject = {
                index: {
                    predictionText: '',
                    sendFreq: 0,
                    isNight: 0,
                    modelType: this.state.index.modelType, // 保留模型选择
                    predictionResult: null,
                    trainingResult: { visible: false, content: '' }
                },
                features: {
                    loaded: false,
                    spamWordCloud: null,
                    hamWordCloud: null,
                    modelMetrics: null,
                    scrollPosition: 0
                },
                history: {
                    loaded: false,
                    filter: '',
                    records: null,
                    selectedIds: [],
                    sortColumn: 'timestamp',
                    sortDirection: 'desc',
                    currentPage: 1,
                    scrollPosition: 0
                }
            };
            
            this.state[pageName] = resetObject[pageName];
            
            // 触发页面级订阅者通知
            for (const path in this.subscribers[pageName]) {
                this._notifySubscribers(`${pageName}.${path}`, this.get(`${pageName}.${path}`), undefined);
            }
        }
    },
    
    /**
     * 合并两个对象，深度合并
     * @private
     * @param {object} target - 目标对象
     * @param {object} source - 源对象
     * @returns {object} 合并后的对象
     */
    _mergeObjects(target, source) {
        const output = Object.assign({}, target);
        
        if (this._isObject(target) && this._isObject(source)) {
            Object.keys(source).forEach(key => {
                if (this._isObject(source[key])) {
                    if (!(key in target)) {
                        Object.assign(output, { [key]: source[key] });
                    } else {
                        output[key] = this._mergeObjects(target[key], source[key]);
                    }
                } else {
                    Object.assign(output, { [key]: source[key] });
                }
            });
        }
        
        return output;
    },
    
    /**
     * 检查是否为对象
     * @private
     * @param {any} item - 要检查的项
     * @returns {boolean} 是否为对象
     */
    _isObject(item) {
        return (item && typeof item === 'object' && !Array.isArray(item));
    },
    
    /**
     * 通知订阅者状态变化
     * @private
     * @param {string} path - 状态路径
     * @param {any} newValue - 新值
     * @param {any} oldValue - 旧值
     */
    _notifySubscribers(path, newValue, oldValue) {
        const parts = path.split('.');
        const pageName = parts[0];
        const subPath = parts.slice(1).join('.');
        
        // 如果有直接订阅该路径的订阅者，通知它们
        if (this.subscribers[pageName] && this.subscribers[pageName][subPath]) {
            this.subscribers[pageName][subPath].forEach(callback => {
                try {
                    callback(newValue, oldValue);
                } catch (error) {
                    console.error(`调用订阅回调出错 ${path}:`, error);
                }
            });
        }
    }
};

// 初始化全局状态管理
document.addEventListener('DOMContentLoaded', () => {
    // 初始化状态管理器
    window.AppState = AppState.init();
    console.log('全局状态管理系统已初始化');
});