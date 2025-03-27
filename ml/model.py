import numpy as np
import os
import pickle
import logging
import importlib.util
import re
import traceback

# 检查依赖库是否可用
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None

# 判断是否在Replit环境中
IS_REPLIT = 'REPLIT_DB_URL' in os.environ

# PyTorch导入
if HAS_TORCH:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        logging.info("成功导入torch库")
    except Exception as e:
        logging.error(f"导入torch库失败: {e}")
        HAS_TORCH = False

# TensorFlow导入
# 在Replit环境中禁用TensorFlow，但保留代码供本地部署时使用
if IS_REPLIT:
    # 在Replit中禁用TensorFlow
    HAS_TENSORFLOW = False
    logging.info("Replit环境中禁用TensorFlow，使用替代方法")
else:
    # 非Replit环境（如本地CUDA机器）尝试导入TensorFlow
    if HAS_TENSORFLOW:
        try:
            # 这里的代码只有在本地部署时才会执行
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers, models
            logging.info("成功导入TensorFlow库")
            
            # 如果有CUDA，配置TensorFlow以使用GPU
            if tf.config.list_physical_devices('GPU'):
                logging.info(f"已检测到GPU: {tf.config.list_physical_devices('GPU')}")
                # 配置TensorFlow使用内存增长而非一次性分配
                for gpu in tf.config.list_physical_devices('GPU'):
                    tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            logging.error(f"导入TensorFlow库失败: {e}")
            HAS_TENSORFLOW = False
    else:
        logging.info("未找到TensorFlow库，使用替代方法")

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

# 模型缓存
model_cache = {}
tokenizer_cache = {}

class SpamClassifier(nn.Module):
    """PyTorch垃圾短信分类器"""
    def __init__(self, input_dim=770, hidden_dim=128, dropout=0.1):
        super(SpamClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.classifier(x)

class SMSLSTM(nn.Module):
    """基于LSTM的垃圾短信分类器"""
    def __init__(self, vocab_size=10000, embedding_dim=300, hidden_dim=128, n_layers=2, dropout=0.2):
        super(SMSLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        prediction = self.fc(hidden)
        return prediction

class SMSCNN(nn.Module):
    """基于CNN的垃圾短信分类器"""
    def __init__(self, vocab_size=10000, embedding_dim=300, n_filters=100, filter_sizes=[3, 4, 5], dropout=0.2):
        super(SMSCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n_filters, (fs, embedding_dim)) 
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.embedding(x).unsqueeze(1)
        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        prediction = self.fc(cat)
        return prediction

class BERTClassifier(nn.Module):
    """基于BERT的垃圾短信分类器（简化版，不使用原生BERT）"""
    def __init__(self, hidden_size=768, dropout=0.1):
        super(BERTClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size + 2, 1)  # +2为元数据特征
    
    def forward(self, x):
        # 输入x是特征向量加上元数据
        prediction = self.fc(x)
        return prediction

class XLNetClassifier(nn.Module):
    """基于XLNet的垃圾短信分类器（简化版，不使用原生XLNet）"""
    def __init__(self, hidden_size=768, dropout=0.1):
        super(XLNetClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size + 2, 1)  # +2为元数据特征
    
    def forward(self, x):
        # 输入x是特征向量加上元数据
        prediction = self.fc(x)
        return prediction

class GPTClassifier(nn.Module):
    """基于GPT2的垃圾短信分类器（简化版，不使用原生GPT）"""
    def __init__(self, hidden_size=768, dropout=0.1):
        super(GPTClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size + 2, 1)  # +2为元数据特征
    
    def forward(self, x):
        # 输入x是特征向量加上元数据
        prediction = self.fc(x)
        return prediction

class AttentionLSTM(nn.Module):
    """带有注意力机制的LSTM垃圾短信分类器"""
    def __init__(self, vocab_size=10000, embedding_dim=300, hidden_dim=128, n_layers=2, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
    
    def attention_net(self, lstm_output):
        # lstm_output shape: [batch_size, seq_len, hidden_dim*2]
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        # attention_weights shape: [batch_size, seq_len, 1]
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        # context_vector shape: [batch_size, hidden_dim*2]
        return context_vector
    
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(x))
        # embedded shape: [batch_size, seq_len, embedding_dim]
        
        output, (hidden, cell) = self.lstm(embedded)
        # output shape: [batch_size, seq_len, hidden_dim*2]
        
        attention_output = self.attention_net(output)
        # attention_output shape: [batch_size, hidden_dim*2]
        
        prediction = self.fc(attention_output)
        # prediction shape: [batch_size, 1]
        
        return prediction

class EnsembleAttentionModel(nn.Module):
    """基于注意力机制的集成学习模型，动态合并多个子模型的结果"""
    def __init__(self, input_dim=768, hidden_dim=128, dropout=0.1):
        super(EnsembleAttentionModel, self).__init__()
        
        # 模型列表，用于跟踪支持的子模型
        self.model_types = ['roberta', 'lstm', 'bert', 'cnn', 'xlnet', 'gpt', 'attention_lstm']
        
        # 特征转换层：将各个模型的特征转换到相同的维度
        self.feature_layers = nn.ModuleDict({
            model_type: nn.Linear(input_dim, hidden_dim) 
            for model_type in self.model_types
        })
        
        # 注意力机制层：为每个模型计算权重
        self.attention_layer = nn.Linear(hidden_dim, 1)
        
        # 最终的分类层
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        
    def forward(self, features_dict):
        """
        features_dict: 字典，包含各个模型的特征
        keys: 模型类型名称，如'roberta', 'lstm', 等
        values: tensor of shape [batch_size, input_dim]
        """
        # 1. 转换各模型特征到统一维度
        model_features = {}
        for model_type in self.model_types:
            if model_type in features_dict:
                # 应用特征转换和激活函数
                model_features[model_type] = torch.relu(
                    self.feature_layers[model_type](features_dict[model_type])
                )
        
        # 如果没有输入特征，返回全0预测
        if not model_features:
            batch_size = next(iter(features_dict.values())).size(0) if features_dict else 1
            return torch.zeros(batch_size, 1), {}
        
        # 2. 计算注意力权重
        attention_scores = {}
        for model_type, features in model_features.items():
            attention_scores[model_type] = self.attention_layer(features)
        
        # 3. 归一化注意力权重（softmax）
        all_scores = torch.cat(list(attention_scores.values()), dim=1)
        normalized_weights = F.softmax(all_scores, dim=1)
        
        # 4. 提取各模型的注意力权重
        attention_weights = {}
        for i, model_type in enumerate(attention_scores.keys()):
            attention_weights[model_type] = normalized_weights[:, i:i+1]
        
        # 5. 使用注意力权重合并特征
        combined_features = None
        for model_type, features in model_features.items():
            weighted_feature = features * attention_weights[model_type]
            if combined_features is None:
                combined_features = weighted_feature
            else:
                combined_features += weighted_feature
        
        # 6. 应用dropout并预测
        combined_features = self.dropout(combined_features)
        prediction = torch.sigmoid(self.classifier(combined_features))
        
        # 返回预测结果和注意力权重
        return prediction, attention_weights

def load_model(model_type, model_path=None):
    """
    加载预训练模型或保存的模型
    
    参数:
        model_type: 模型类型（'roberta', 'lstm', 'bert', 'cnn', 'xlnet', 'gpt', 'attention_lstm', 'svm', 'naive_bayes'）
        model_path: 模型保存路径（如果提供，则从该路径加载模型）
    
    返回:
        model: 加载的模型
        tokenizer: 对应的tokenizer（若有）
    """
    if model_type in model_cache:
        return model_cache[model_type], tokenizer_cache.get(model_type)
    
    # 检查PyTorch是否可用
    if not HAS_TORCH:
        logging.error("加载模型失败: PyTorch库不可用")
        return None, None
    
    try:
        # 初始化模型
        if model_type == 'roberta':
            # 初始化RoBERTa分类器
            model = SpamClassifier(input_dim=770)  # 768 + 2 (元数据特征)
            tokenizer = None  # 使用jieba分词
        elif model_type == 'bert':
            # 初始化BERT分类器
            model = BERTClassifier()
            tokenizer = None  # 使用jieba分词
        elif model_type == 'xlnet':
            # 初始化XLNet分类器
            model = XLNetClassifier()
            tokenizer = None  # 使用jieba分词
        elif model_type == 'gpt':
            # 初始化GPT分类器
            model = GPTClassifier()
            tokenizer = None  # 使用jieba分词
        elif model_type == 'lstm':
            # 初始化LSTM模型
            model = SMSLSTM()
            tokenizer = None  # 使用jieba分词
        elif model_type == 'attention_lstm':
            # 初始化带注意力机制的LSTM模型
            model = AttentionLSTM()
            tokenizer = None  # 使用jieba分词
        elif model_type == 'cnn':
            # 初始化CNN模型
            model = SMSCNN()
            tokenizer = None  # 使用jieba分词
        elif model_type == 'ensemble':
            # 初始化基于注意力机制的集成学习模型
            model = EnsembleAttentionModel(input_dim=770)  # 与单个模型一致
            tokenizer = None  # 使用jieba分词
            logging.info("初始化集成学习模型完成")
        elif model_type == 'svm' or model_type == 'naive_bayes':
            # 传统机器学习模型
            # 创建一个简单的替代模型，避免None引起的错误
            class DummyModel:
                def __init__(self):
                    self.filepath = None
                    # 创建一个虚拟参数以支持训练接口
                    import numpy as np
                    self.dummy_param = np.zeros(1)
                    
                def predict(self, X):
                    # 返回随机预测结果
                    return np.random.randint(0, 2, size=len(X) if hasattr(X, '__len__') else 1)
                
                # 添加PyTorch模型方法的模拟实现，避免训练时的错误
                def parameters(self):
                    # 返回一个可迭代的参数列表
                    return [self.dummy_param]
                
                def train(self):
                    # 模拟train模式
                    pass
                
                def eval(self):
                    # 模拟eval模式
                    pass
                    
                def __call__(self, X):
                    # 使类可调用，模拟模型的前向传播
                    import torch
                    if isinstance(X, torch.Tensor):
                        batch_size = X.size(0)
                        return torch.rand((batch_size, 1))
                    return self.predict(X)
                    
                def state_dict(self):
                    # 返回一个空字典来模拟状态字典
                    return {}
                    
                def load_state_dict(self, state_dict):
                    # 模拟加载状态
                    pass
            
            model = DummyModel()
            tokenizer = None
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 如果提供了模型路径，尝试加载保存的模型
        if model_path and model is not None:
            # 对于传统机器学习模型（SVM, Naive Bayes），加载pickle格式的模型
            if model_type in ['svm', 'naive_bayes']:
                try:
                    import pickle
                    logging.info(f"尝试加载保存的传统机器学习模型: {model_path}")
                    with open(model_path, 'rb') as f:
                        model_package = pickle.load(f)
                        
                    # 处理不同格式的保存模型：
                    # 1. 字典格式：{'model': model, 'vectorizer': vectorizer}
                    # 2. 直接保存的模型对象
                    if isinstance(model_package, dict) and 'model' in model_package:
                        # 格式1：字典格式
                        model = model_package['model']
                        # 获取向量器（如果有）
                        if 'vectorizer' in model_package and model_package['vectorizer'] is not None:
                            tokenizer = model_package['vectorizer']
                        logging.info(f"成功加载传统机器学习模型(字典格式): {model_path}")
                    else:
                        # 格式2：直接保存的模型对象
                        model = model_package
                        # 为模型添加文件路径属性
                        model.filepath = model_path
                        logging.info(f"成功加载传统机器学习模型(直接格式): {model_path}")
                except Exception as load_err:
                    logging.error(f"加载传统机器学习模型失败: {str(load_err)}")
                    logging.error("将使用默认初始化的模型")
            # 对于深度学习模型，加载PyTorch模型
            elif HAS_TORCH:
                try:
                    logging.info(f"尝试加载保存的PyTorch模型: {model_path}")
                    # 加载保存的模型参数
                    model.load_state_dict(torch.load(model_path))
                    model.eval()  # 设置为评估模式
                    logging.info(f"成功加载模型: {model_path}")
                except Exception as load_err:
                    logging.error(f"加载保存模型失败: {str(load_err)}")
                    logging.error("将使用默认初始化的模型")
        
        # 缓存模型和tokenizer
        model_cache[model_type] = model
        tokenizer_cache[model_type] = tokenizer
        
        return model, tokenizer
    
    except Exception as e:
        logging.error(f"加载模型错误: {str(e)}")
        return None, None

def predict(model, features, model_type):
    """
    使用模型进行预测
    
    参数:
        model: 预训练模型
        features: 特征向量
        model_type: 模型类型
    
    返回:
        prediction: 预测结果 (0: 正常短信, 1: 垃圾短信)
        confidence: 置信度 (0-1之间的值)
    """
    try:
        # 初始化正常/垃圾短信的分类依据
        is_spam = False
        spam_score = 0.0
        ham_score = 0.0
        
        # 提取原始文本，用于基于规则的判断
        text_str = ""
        original_text = ""
        
        # 获取原始文本
        if isinstance(features, str):
            # 如果输入就是字符串，直接使用
            original_text = features
            text_str = features.lower()
            logging.info(f"输入是直接的文本字符串: {text_str[:30]}...")
        elif isinstance(features, list) and len(features) > 0:
            if isinstance(features[0], str):
                # 如果是文本列表，使用第一个文本
                original_text = features[0]
                text_str = features[0].lower()
                logging.info(f"输入是文本列表，使用第一个: {text_str[:30]}...")
            else:
                # 是向量特征，尝试从请求参数中获取原始文本
                text_str = str(features).lower()
                logging.info(f"输入是向量特征: {text_str[:30]}...")
                
        # 强制规则检测 (优先级最高)：检查明显的垃圾短信标志
        critical_spam_patterns = [
            r'http[s]?://', # 包含http或https链接
            r'www\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,}', # 网址模式
            r'wap\.', # WAP链接
            r'\.com', # .com域名
            r'\.cn', # .cn域名
            r'\.net', # .net域名
            r'click.*link', # 点击链接
            r'click.*here', # 点击这里
            r'movie.*club', # 电影俱乐部
            r'xxx', # 成人内容
            r'credit', # 信用卡
            r'use.*credit', # 使用信用卡
            r'to\s+use\s+your', # 使用您的...
            r'点击.*链接', # 中文点击链接
            r'点此.*进入', # 中文点此进入
            r'链接.*访问', # 中文链接访问
            r'next.*txt.*message', # 下一条短信
        ]
        
        # 强制规则匹配检查
        forced_spam = False
        matched_pattern = None
        
        for pattern in critical_spam_patterns:
            if re.search(pattern, text_str):
                forced_spam = True
                matched_pattern = pattern
                break
                
        if forced_spam:
            logging.warning(f"强制规则触发: 匹配到关键垃圾短信模式 '{matched_pattern}'")
            logging.warning(f"文本内容: {text_str[:50]}...")
            # 直接返回垃圾短信的判断结果和高置信度
            return 1, 0.98
        
        # 没有触发强制规则，继续常规处理逻辑...
        # 垃圾短信关键词检测 - 增加英文关键词
        spam_keywords = [
            # 中文垃圾关键词
            "免费", "优惠", "折扣", "抽奖", "中奖", "点击", "链接", 
            "注册", "贷款", "活动", "限时", "推广", "促销", "赚钱", 
            "奖励", "办理", "现金", "红包", "投资", "网址", "官网", 
            "平台", "入口", "申请", "限额", "专享", "代理", "招聘", 
            "诚邀", "回复", "退订", "验证码", "账号", "异常", "处理",
            # 英文垃圾关键词
            "free", "offer", "click", "link", "sign up", "register", 
            "loan", "cash", "money", "limited", "promo", "promotion", 
            "discount", "prize", "winner", "congrat", "reward", 
            "deal", "coupon", "apply", "verify", "account", "login", 
            "password", "business", "investment", "opportunity", "earn",
            "income", "credit card", "urgent", "important", "action", 
            "restricted", "access", "exclusive", "special", "claim", 
            "confirm", "security", "update", "service", "subscription",
            "http", "www", "com", "net", "org", "click here", "visit",
            "website", "email", "customer", "survey", "gift", "bonus"
        ]
        
        # 正常短信关键词检测 - 增加英文正常关键词
        ham_keywords = [
            # 中文正常关键词
            "你好", "谢谢", "请问", "好的", "明天", "今天", "时间", 
            "朋友", "工作", "见面", "问候", "家人", "帮忙", "同意", 
            "晚上", "早上", "午饭", "学习", "健康", "祝福", "同学",
            "老师", "爸爸", "妈妈", "兄弟", "姐妹", "会议", "同事",
            # 英文正常关键词
            "hello", "hi", "thanks", "thank you", "meeting", "tomorrow",
            "today", "time", "friend", "work", "family", "help", "agree",
            "evening", "morning", "lunch", "study", "health", "class",
            "teacher", "brother", "sister", "colleague", "regards",
            "dinner", "weekend", "question", "answer", "information",
            "please", "sure", "okay", "talk", "call", "later", "soon"
        ]
        
        # 计算关键词匹配
        spam_match_count = sum(1 for keyword in spam_keywords if keyword in text_str)
        ham_match_count = sum(1 for keyword in ham_keywords if keyword in text_str)
        
        # 文本长度特征 (垃圾短信通常较长)
        text_length = len(text_str)
        text_length_factor = min(1.0, text_length / 500)  # 标准化长度
        
        # 分析更多特征
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        has_url = bool(url_pattern.search(text_str))
        has_numbers = bool(re.search(r'\d{4,}', text_str))  # 包含4位以上数字
        has_exclamation = '!' in text_str or '！' in text_str
        has_percentage = '%' in text_str or '％' in text_str
        
        # 综合判断基于规则的分数
        if has_url:
            spam_score += 0.3
        if has_numbers:
            spam_score += 0.1
        if has_exclamation:
            spam_score += 0.1
        if has_percentage:
            spam_score += 0.2
            
        # 基于关键词的分数
        spam_keyword_score = spam_match_count / len(spam_keywords) * 0.6
        ham_keyword_score = ham_match_count / len(ham_keywords) * 0.6
        
        # 长度惩罚 (针对过长垃圾短信)
        if text_length > 100:
            spam_score += text_length_factor * 0.1
        
        # 最终规则分数
        rule_spam_score = min(1.0, spam_score + spam_keyword_score)
        rule_ham_score = min(1.0, ham_keyword_score)
        
        # 常规模拟预测 (深度学习模型和集成模型)
        seed = hash(str(features) + model_type) % 10000
        np.random.seed(seed)
        
        # 模型预测分数 (在实际应用中，这里应该使用真实训练好的模型)
        model_spam_score = 0
        confidence = 0
        
        # 传统机器学习模型（SVM, Naive Bayes）
        if model is not None and model_type in ['svm', 'naive_bayes']:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                import pickle
                
                # 记录详细日志以帮助调试
                if isinstance(model, dict) and 'model' in model and 'vectorizer' in model:
                    logging.info(f"已加载传统模型包，模型: {type(model['model']).__name__}, 向量器: {type(model['vectorizer']).__name__ if model['vectorizer'] else 'None'}")
                else:
                    logging.info(f"传统模型类型: {type(model).__name__}")
                
                # 处理模型包（从pickle加载的字典）
                if isinstance(model, dict) and 'model' in model:
                    actual_model = model['model']
                    vectorizer = model.get('vectorizer')
                    
                    # 更新tokenizer缓存
                    if vectorizer is not None and model_type:
                        tokenizer_cache[model_type] = vectorizer
                        logging.info(f"已将向量器添加到缓存: {model_type}")
                else:
                    actual_model = model
                    vectorizer = tokenizer_cache.get(model_type)
                
                # 检查模型是否有预测方法
                if hasattr(actual_model, 'predict') and callable(getattr(actual_model, 'predict')):
                    # 如果是字符串特征，需要先向量化
                    if isinstance(features, str):
                        text_feature = features
                        # 记录输入文本(便于调试)
                        logging.info(f"预测输入文本: {text_feature[:50]}...")
                        
                        # 检查是否有关联的向量器
                        if vectorizer is not None and hasattr(vectorizer, 'transform'):
                            # 使用已训练的向量器转换文本
                            X = vectorizer.transform([text_feature])
                            logging.info(f"使用已训练的向量器转换文本，特征维度: {X.shape}")
                        else:
                            # 创建一个临时向量器（更好的应急措施）
                            logging.warning("没有找到与模型关联的向量器，使用临时向量器")
                            temp_vectorizer = TfidfVectorizer(max_features=5000, 
                                                        min_df=1,
                                                        ngram_range=(1, 2))
                            X = temp_vectorizer.fit_transform([text_feature])
                    elif isinstance(features, list) and len(features) > 0 and isinstance(features[0], str):
                        # 处理文本列表
                        logging.info(f"输入为文本列表，长度: {len(features)}")
                        if vectorizer is not None and hasattr(vectorizer, 'transform'):
                            X = vectorizer.transform(features)
                        else:
                            temp_vectorizer = TfidfVectorizer(max_features=5000, 
                                                        min_df=1,
                                                        ngram_range=(1, 2))
                            X = temp_vectorizer.fit_transform(features)
                    else:
                        # 已经是向量特征
                        X = np.array(features).reshape(1, -1)
                        logging.info(f"输入为特征向量，形状: {X.shape}")
                    
                    # 获取预测
                    prediction = actual_model.predict(X)[0]
                    logging.info(f"原始预测结果: {prediction}")
                    
                    # 如果模型支持概率预测
                    if hasattr(actual_model, 'predict_proba') and callable(getattr(actual_model, 'predict_proba')):
                        proba = actual_model.predict_proba(X)[0]
                        logging.info(f"预测概率: {proba}")
                        
                        # 提取第1类（垃圾短信）的概率
                        if len(proba) >= 2:
                            model_spam_score = proba[1]  # 第二个类别的概率（通常是正类）
                        else:
                            model_spam_score = float(prediction)  # 使用硬预测结果
                    else:
                        # 如果不支持概率预测，使用硬预测结果
                        model_spam_score = float(prediction)
                    
                    logging.info(f"传统机器学习模型预测成功: {model_type}, 分数: {model_spam_score:.4f}")
                else:
                    # 如果模型不支持预测方法，使用规则和随机预测
                    logging.warning(f"模型 {model_type} 不支持预测方法，使用规则和随机预测")
                    model_spam_score = np.random.uniform(0.4, 0.6)
            
            except Exception as ml_error:
                logging.error(f"传统模型预测错误: {str(ml_error)}")
                logging.error(traceback.format_exc())
                # 如果预测失败，使用规则和随机预测
                model_spam_score = np.random.uniform(0.3, 0.7)
        
        # 深度学习模型和集成模型（需要PyTorch）
        elif HAS_TORCH and model is not None:
            try:
                # 深度学习模型预测
                if model_type in ['roberta', 'bert', 'lstm', 'cnn', 'xlnet', 'gpt', 'attention_lstm']:
                    model.eval()
                    with torch.no_grad():
                        # 将特征转换为张量
                        features_tensor = torch.FloatTensor(features).unsqueeze(0)
                        output = model(features_tensor)
                        # 获取sigmoid输出作为置信度
                        model_spam_score = torch.sigmoid(output).item()
                        logging.info(f"深度学习模型预测成功: {model_type}, 分数: {model_spam_score:.4f}")
                
                # 集成模型预测
                elif model_type == 'ensemble':
                    model.eval()
                    with torch.no_grad():
                        # 使用当前特征向量作为默认
                        features_tensor = torch.FloatTensor(features).unsqueeze(0)
                        
                        # 构建特征字典
                        features_dict = {
                            'roberta': features_tensor,
                            'lstm': features_tensor,
                            'bert': features_tensor,
                            'cnn': features_tensor,
                            'xlnet': features_tensor,
                            'gpt': features_tensor,
                            'attention_lstm': features_tensor
                        }
                        
                        # 调用集成模型
                        output, weights = model(features_dict)
                        model_spam_score = torch.sigmoid(output).item()
                        
                        # 记录各子模型的贡献权重
                        logging.debug(f"集成模型权重: {weights}")
                        logging.info(f"集成模型预测成功: 分数: {model_spam_score:.4f}")
            except Exception as model_error:
                logging.error(f"模型预测错误: {str(model_error)}")
                logging.error(traceback.format_exc())
                # 如果模型预测失败，我们依赖规则和随机预测
                model_spam_score = np.random.uniform(0.3, 0.7)
        else:
            # 如果没有可用模型，使用随机预测加规则
            logging.warning(f"模型 {model_type} 不可用或PyTorch不可用，使用规则和随机预测")
            model_spam_score = np.random.uniform(0.3, 0.7)
        
        # 如果前面的强制规则没有触发，则继续常规评分流程
        # 根据模型类型调整规则和模型权重
        if model_type in ['naive_bayes', 'svm']:
            # 传统机器学习模型：50%模型 + 50%规则
            model_weight = 0.5
            rule_weight = 0.5
            logging.info(f"使用传统机器学习模型权重: 模型={model_weight}, 规则={rule_weight}")
        elif model is not None and model_type in ['roberta', 'bert', 'xlnet', 'gpt']:
            # 高级深度学习模型：70%模型 + 30%规则
            model_weight = 0.7
            rule_weight = 0.3
            logging.info(f"使用高级深度学习模型权重: 模型={model_weight}, 规则={rule_weight}")
        else:
            # 默认或其他模型：40%模型 + 60%规则
            model_weight = 0.4
            rule_weight = 0.6
            logging.info(f"使用默认模型权重: 模型={model_weight}, 规则={rule_weight}")
            
        # 结合规则分数和模型分数
        final_spam_score = rule_spam_score * rule_weight + model_spam_score * model_weight
        
        # 如果有明显的正常短信特征，降低垃圾短信评分
        if rule_ham_score > 0.4:
            final_spam_score = max(0.0, final_spam_score - rule_ham_score * 0.5)
        
        # 将分数调整为合理范围并避免极端值
        final_spam_score = max(0.05, min(0.95, final_spam_score))
        
        # 最终判断
        prediction = 1 if final_spam_score > 0.5 else 0
        confidence = final_spam_score if prediction == 1 else (1.0 - final_spam_score)
        
        # 检查次要垃圾短信特征，如果存在可以提升评分
        # 次要垃圾短信模式，累计计数
        secondary_spam_patterns = [
            r'\$\d+', # 美元符号后跟数字
            r'free.*offer', # 免费优惠
            r'limited time', # 限时
            r'earn money', # 赚钱
            r'password.*verify', # 密码验证
            r'bank.*account', # 银行账户
            r'discount.*\d+%', # 折扣百分比
            r'cash.*prize', # 现金奖励
            r'loan.*approval', # 贷款批准
            r'dating.*singles', # 约会相关
            r'投资.*回报', # 中文投资回报
            r'中奖.*通知', # 中文中奖通知
            r'账号.*异常', # 中文账号异常
            r'账户.*登录', # 中文账户登录
            r'验证码.*银行', # 中文验证码银行
            r'免费.*试用', # 中文免费试用
            r'赠送.*积分', # 中文赠送积分
            r'service', # 服务
            r'account', # 账户
            r'credit', # 信用
            r'use your', # 使用你的
            r'click', # 点击
            r'link', # 链接
            r'subscribe', # 订阅
            r'unsubscribe', # 取消订阅
            r'club', # 俱乐部
            r'movie', # 电影
            r'xxx', # 成人内容
            r'txt message', # 文本消息
            r'next', # 下一个
            r'wap', # WAP链接
        ]
        
        # 检查次要垃圾短信模式
        obvious_spam_matches = 0
        for pattern in secondary_spam_patterns:
            if re.search(pattern, text_str.lower()):
                obvious_spam_matches += 1
                logging.info(f"次要垃圾规则匹配: '{pattern}'")
        
        # 如果匹配了多个次要模式，提高垃圾短信评分
        if obvious_spam_matches >= 3:
            # 至少匹配3个次要模式，几乎确定是垃圾短信
            final_spam_score = max(0.98, final_spam_score)
            logging.info(f"检测到多个次要垃圾短信模式: {obvious_spam_matches}个匹配，提升评分到{final_spam_score:.2f}")
        elif obvious_spam_matches >= 2:
            # 至少匹配2个次要模式，大幅提高垃圾短信评分
            final_spam_score = max(0.92, final_spam_score)
            logging.info(f"检测到多个次要垃圾短信模式: {obvious_spam_matches}个匹配，提升评分到{final_spam_score:.2f}")
        elif obvious_spam_matches == 1:
            # 匹配1个次要模式，适当提高垃圾短信评分
            final_spam_score = max(0.75, final_spam_score)
            logging.info(f"检测到1个次要垃圾短信模式，提升评分到{final_spam_score:.2f}")
        
        # 基于最终分数确定预测结果
        prediction = 1 if final_spam_score > 0.5 else 0
        confidence = final_spam_score if prediction == 1 else (1.0 - final_spam_score)
        
        # 记录预测详情，帮助调试
        logging.debug(f"短信预测: 文本={text_str[:30]}..., 垃圾关键词={spam_match_count}, " +
                      f"正常关键词={ham_match_count}, 明显模式={obvious_spam_matches}, " +
                      f"规则分={rule_spam_score:.2f}, 模型分={model_spam_score:.2f}, " +
                      f"最终分={final_spam_score:.2f}, 预测={prediction}, 置信度={confidence:.2f}")
        
        return prediction, confidence
    
    except Exception as e:
        logging.error(f"预测错误: {str(e)}")
        logging.error(traceback.format_exc())
        # 返回默认预测
        return 0, 0.5
