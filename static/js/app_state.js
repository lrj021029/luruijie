/**
 * 全局状态管理系统 (AppState)
 * 用于管理和持久化应用程序的状态
 */
class AppState {
    constructor() {
        // 初始化状态对象
        this.state = {
            // 首页状态
            index: {
                predictionText: '', // 预测文本
                sendFreq: 0,        // 发送频率
                isNight: 0,         // 是否夜间发送
                modelType: '',      // 选择的模型类型
                predictionResult: '', // 预测结果HTML
                trainingResult: {
                    visible: false,
                    content: ''
                }
            },
            
            // 特征页面状态
            features: {
                loaded: false,
                scrollPosition: 0,
                spamWordCloud: null,
                hamWordCloud: null,
                modelMetrics: null
            },
            
            // 历史页面状态
            history: {
                loaded: false,
                filter: '',
                scrollPosition: 0,
                selectedIds: [],
                records: null,
                sortColumn: 'timestamp',
                sortDirection: 'desc',
                currentPage: 1
            },
            
            // 数据集页面状态
            datasets: {
                loaded: false,
                scrollPosition: 0,
                datasetsList: null,
                uploadForm: {
                    name: '',
                    description: ''
                }
            }
        };
        
        // 订阅者映射表
        this.subscribers = {};
    }
    
    /**
     * 初始化状态管理系统
     * 从localStorage中加载之前保存的状态
     */
    init() {
        try {
            const savedState = localStorage.getItem('appState');
            if (savedState) {
                const parsedState = JSON.parse(savedState);
                this.state = this._mergeObjects(this.state, parsedState);
                console.log('从localStorage加载状态成功');
            }
        } catch (error) {
            console.error('从localStorage加载状态失败:', error);
        }
    }
    
    /**
     * 获取指定页面的状态
     * @param {string} page - 页面名称('index', 'features', 'history')
     * @returns {object} 页面状态对象
     */
    getPageState(page) {
        return this.state[page] || null;
    }
    
    /**
     * 获取指定路径的状态值
     * @param {string} path - 状态路径，如'index.modelType'
     * @returns {any} 状态值
     */
    get(path) {
        const parts = path.split('.');
        let current = this.state;
        
        for (const part of parts) {
            if (current === null || current === undefined || typeof current !== 'object') {
                return undefined;
            }
            current = current[part];
        }
        
        return current;
    }
    
    /**
     * 设置指定路径的状态值
     * @param {string} path - 状态路径，如'index.modelType'
     * @param {any} value - 要设置的值
     */
    set(path, value) {
        const parts = path.split('.');
        const lastPart = parts.pop();
        let current = this.state;
        
        // 定位到目标对象
        for (const part of parts) {
            if (current[part] === undefined) {
                current[part] = {};
            }
            current = current[part];
        }
        
        // 保存旧值以供通知
        const oldValue = current[lastPart];
        
        // 设置新值
        current[lastPart] = value;
        
        // 通知订阅者
        this._notifySubscribers(path, value, oldValue);
    }
    
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
        
        // 通知相关订阅者
        for (const key in updates) {
            const path = `${pageName}.${key}`;
            this._notifySubscribers(path, updates[key], this.get(path));
        }
    }
    
    /**
     * 订阅状态变化
     * @param {string} path - 状态路径，如'index.modelType'
     * @param {function} callback - 当状态改变时调用的回调函数
     * @returns {function} 取消订阅的函数
     */
    subscribe(path, callback) {
        if (!this.subscribers[path]) {
            this.subscribers[path] = [];
        }
        
        this.subscribers[path].push(callback);
        
        // 返回取消订阅的函数
        return () => {
            this.subscribers[path] = this.subscribers[path].filter(cb => cb !== callback);
        };
    }
    
    /**
     * 保存状态到localStorage
     */
    saveToStorage() {
        try {
            localStorage.setItem('appState', JSON.stringify(this.state));
        } catch (error) {
            console.error('保存状态到localStorage失败:', error);
        }
    }
    
    /**
     * 清除指定页面的状态
     * @param {string} pageName - 页面名称
     */
    clearPageState(pageName) {
        if (this.state[pageName]) {
            // 保存旧状态以便通知
            const oldState = {...this.state[pageName]};
            
            // 创建默认状态
            const defaultState = {
                // 首页状态默认值
                index: {
                    predictionText: '',
                    sendFreq: 0,
                    isNight: 0,
                    modelType: '',
                    predictionResult: '',
                    trainingResult: {
                        visible: false,
                        content: ''
                    }
                },
                // 特征页面默认值
                features: {
                    loaded: false,
                    scrollPosition: 0,
                    spamWordCloud: null,
                    hamWordCloud: null,
                    modelMetrics: null
                },
                // 历史页面默认值
                history: {
                    loaded: false,
                    filter: '',
                    scrollPosition: 0,
                    selectedIds: [],
                    records: null,
                    sortColumn: 'timestamp',
                    sortDirection: 'desc',
                    currentPage: 1
                }
            };
            
            // 重置状态为默认值
            this.state[pageName] = {...defaultState[pageName]};
            
            // 通知订阅者
            for (const key in oldState) {
                const path = `${pageName}.${key}`;
                this._notifySubscribers(path, this.state[pageName][key], oldState[key]);
            }
            
            // 保存到localStorage
            this.saveToStorage();
        }
    }
    
    /**
     * 合并两个对象，深度合并
     * @private
     * @param {object} target - 目标对象
     * @param {object} source - 源对象
     * @returns {object} 合并后的对象
     */
    _mergeObjects(target, source) {
        if (!source) return target;
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
    }
    
    /**
     * 检查是否为对象
     * @private
     * @param {any} item - 要检查的项
     * @returns {boolean} 是否为对象
     */
    _isObject(item) {
        return (item && typeof item === 'object' && !Array.isArray(item));
    }
    
    /**
     * 通知订阅者状态变化
     * @private
     * @param {string} path - 状态路径
     * @param {any} newValue - 新值
     * @param {any} oldValue - 旧值
     */
    _notifySubscribers(path, newValue, oldValue) {
        if (this.subscribers[path]) {
            this.subscribers[path].forEach(callback => {
                try {
                    callback(newValue, oldValue);
                } catch (error) {
                    console.error(`执行订阅者回调失败 (${path}):`, error);
                }
            });
        }
        
        // 同时通知通配符订阅者
        if (this.subscribers['*']) {
            this.subscribers['*'].forEach(callback => {
                try {
                    callback(path, newValue, oldValue);
                } catch (error) {
                    console.error('执行通配符订阅者回调失败:', error);
                }
            });
        }
    }
}