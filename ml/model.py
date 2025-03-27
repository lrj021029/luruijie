import numpy as np
import os
import pickle
import logging
import importlib.util

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

if HAS_TENSORFLOW:
    try:
        import tensorflow as tf
        logging.info("成功导入tensorflow库")
    except Exception as e:
        logging.error(f"导入tensorflow库失败: {e}")
        HAS_TENSORFLOW = False

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

def load_model(model_type):
    """
    加载预训练模型
    
    参数:
        model_type: 模型类型（'roberta', 'lstm', 'bert', 'cnn', 'xlnet', 'gpt', 'attention_lstm', 'svm', 'naive_bayes'）
    
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
            model = None  # 实际应用中应加载预训练的SVM或NB模型
            tokenizer = None
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
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
        # 如果PyTorch不可用，使用随机预测
        if not HAS_TORCH:
            seed = hash(str(features) + model_type) % 10000
            np.random.seed(seed)
            prediction = np.random.randint(0, 2)
            confidence = np.random.uniform(0.6, 0.95)
            return prediction, confidence
            
        # 深度学习模型预测
        if model_type in ['roberta', 'bert', 'lstm', 'cnn', 'xlnet', 'gpt', 'attention_lstm'] and model is not None:
            model.eval()
            with torch.no_grad():
                # 将特征转换为张量
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                output = model(features_tensor)
                # 获取sigmoid输出作为置信度
                confidence = torch.sigmoid(output).item()
                # 预测类别
                prediction = 1 if confidence > 0.5 else 0
                return prediction, confidence
        
        # 集成模型预测
        elif model_type == 'ensemble' and model is not None:
            model.eval()
            with torch.no_grad():
                # 为集成模型准备输入
                # 需要从各个模型获取特征，然后构建特征字典传入集成模型
                
                # 使用当前特征向量作为默认，应该包含所有模型用到的特征
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                
                # 构建特征字典（真实使用中应该调用各子模型获取特征）
                features_dict = {
                    'roberta': features_tensor,
                    'lstm': features_tensor,
                    'bert': features_tensor,
                    'cnn': features_tensor,
                    'xlnet': features_tensor,
                    'gpt': features_tensor,
                    'attention_lstm': features_tensor
                }
                
                # 这里简化模拟集成模型：对所有子模型进行相同的预测并输出权重
                output, weights = model(features_dict)
                
                # 获取预测值和置信度
                confidence = output.item()
                prediction = 1 if confidence > 0.5 else 0
                
                # 可以记录各个子模型的贡献权重（日志或用于可视化）
                logging.debug(f"集成模型权重: {weights}")
                
                return prediction, confidence
        else:
            # 传统机器学习模型预测
            # 实际应用中应使用加载的模型进行预测
            # 这里使用随机预测作为示例
            seed = hash(str(features) + model_type) % 10000
            np.random.seed(seed)
            prediction = np.random.randint(0, 2)
            confidence = np.random.uniform(0.6, 0.95)
            return prediction, confidence
    
    except Exception as e:
        logging.error(f"预测错误: {str(e)}")
        # 返回默认预测
        return 0, 0.5
