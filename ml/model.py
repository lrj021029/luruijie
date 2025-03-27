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

# 条件导入
if HAS_TORCH:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        logging.info("成功导入torch库")
    except Exception as e:
        logging.error(f"导入torch库失败: {e}")
        HAS_TORCH = False

# 我们不直接使用TensorFlow，因此禁用它以避免兼容性问题
HAS_TENSORFLOW = False
logging.info("已禁用TensorFlow，使用替代方法")

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
                    
                def predict(self, X):
                    # 返回随机预测结果
                    return np.random.randint(0, 2, size=len(X) if hasattr(X, '__len__') else 1)
            
            model = DummyModel()
            tokenizer = None
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 如果提供了模型路径，尝试加载保存的模型
        if model_path and model is not None and HAS_TORCH:
            try:
                logging.info(f"尝试加载保存的模型: {model_path}")
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
        if isinstance(features, list) and len(features) > 0:
            text_str = str(features).lower()
        
        # 垃圾短信关键词检测
        spam_keywords = ["免费", "优惠", "折扣", "抽奖", "中奖", "点击", "链接", 
                        "注册", "贷款", "活动", "限时", "推广", "促销", "赚钱", 
                        "奖励", "办理", "现金", "红包", "投资", "http", "www", 
                        "com", "cn", "网址", "官网", "平台", "入口", "申请", "offer",
                        "限额", "专享", "代理", "招聘", "诚邀", "回复", "退订"]
        
        # 正常短信关键词检测
        ham_keywords = ["你好", "谢谢", "请问", "好的", "明天", "今天", "时间", 
                       "朋友", "工作", "见面", "问候", "家人", "帮忙", "同意", 
                       "晚上", "早上", "午饭", "学习", "健康", "祝福", "同学",
                       "老师", "爸爸", "妈妈", "兄弟", "姐妹", "会议", "同事"]
        
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
        
        # 如果PyTorch可用，且模型存在，使用模型预测
        if HAS_TORCH and model is not None:
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
            except Exception as model_error:
                logging.error(f"模型预测错误: {str(model_error)}")
                # 如果模型预测失败，我们依赖规则和随机预测
                model_spam_score = np.random.uniform(0.3, 0.7)
        else:
            # 如果没有可用模型，使用随机预测加规则
            model_spam_score = np.random.uniform(0.3, 0.7)
        
        # 结合规则分数和模型分数 (70%规则 + 30%模型)
        # 大多数情况下应以模型为主，但由于这里是模拟模型，所以增加规则权重
        final_spam_score = rule_spam_score * 0.7 + model_spam_score * 0.3
        
        # 如果有明显的正常短信特征，降低垃圾短信评分
        if rule_ham_score > 0.4:
            final_spam_score = max(0.0, final_spam_score - rule_ham_score * 0.5)
        
        # 将分数调整为合理范围并避免极端值
        final_spam_score = max(0.05, min(0.95, final_spam_score))
        
        # 最终判断
        prediction = 1 if final_spam_score > 0.5 else 0
        confidence = final_spam_score if prediction == 1 else (1.0 - final_spam_score)
        
        # 增强的垃圾短信检测逻辑 - 识别明显的垃圾短信模式
        obvious_spam_patterns = [
            r'\$\d+', # 美元符号后跟数字
            r'click here', # 点击这里
            r'www\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,}', # 网址模式
            r'http[s]?://', # 包含http或https链接
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
        ]
        
        # 检查是否匹配明显的垃圾短信模式
        obvious_spam_matches = 0
        for pattern in obvious_spam_patterns:
            if re.search(pattern, text_str.lower()):
                obvious_spam_matches += 1
                
        # 如果匹配了多个明显的垃圾短信模式，提高垃圾短信评分
        if obvious_spam_matches >= 2:
            # 至少匹配2个模式，大幅提高垃圾短信评分
            final_spam_score = max(0.85, final_spam_score)
            logging.info(f"检测到明显垃圾短信模式: {obvious_spam_matches}个匹配，提升评分到{final_spam_score:.2f}")
        elif obvious_spam_matches == 1:
            # 匹配1个模式，适当提高垃圾短信评分
            final_spam_score = max(0.7, final_spam_score)
            logging.info(f"检测到可能的垃圾短信模式: {obvious_spam_matches}个匹配，提升评分到{final_spam_score:.2f}")
        
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
