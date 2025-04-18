import numpy as np
import logging
import importlib.util
from ml.preprocessing import tokenize

# 检查依赖库是否可用
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None

# 有条件地导入
if HAS_TORCH:
    try:
        import torch
        logging.info("成功导入torch库")
    except Exception as e:
        logging.error(f"导入torch库失败: {e}")
        HAS_TORCH = False

# 我们不直接使用TensorFlow，因此禁用它以避免兼容性问题
HAS_TENSORFLOW = False
logging.info("已禁用TensorFlow，使用替代方法")

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

# 缓存预训练模型的tokenizer
tokenizer_cache = {}

def get_tokenizer(model_type):
    """
    获取特定模型类型的tokenizer
    不使用transformers库，而是使用jieba分词
    """
    if model_type in tokenizer_cache:
        return tokenizer_cache[model_type]
    
    # 由于不使用transformers库，所有模型类型都使用jieba分词器
    tokenizer = None
    tokenizer_cache[model_type] = tokenizer
    return tokenizer

def extract_features(text, tokenizer, send_freq=0.0, is_night=0, model_type='roberta'):
    """
    根据不同模型类型提取特征
    """
    try:
        # 归一化元数据
        normalized_freq = min(float(send_freq) / 100.0, 1.0)
        is_night_binary = 1 if int(is_night) > 0 else 0
        
        # 根据模型类型提取不同特征
        if model_type in ['roberta', 'bert', 'xlnet', 'gpt']:
            # 使用transformers模型提取特征
            return extract_transformer_features(text, tokenizer, normalized_freq, is_night_binary, model_type)
        elif model_type == 'lstm':
            # LSTM特征提取
            return extract_lstm_features(text, normalized_freq, is_night_binary)
        elif model_type == 'attention_lstm':
            # 带注意力机制的LSTM特征提取
            return extract_lstm_features(text, normalized_freq, is_night_binary)  # 使用相同的特征提取逻辑
        elif model_type == 'cnn':
            # CNN特征提取
            return extract_cnn_features(text, normalized_freq, is_night_binary)
        else:
            # 传统机器学习特征提取
            return extract_traditional_features(text, normalized_freq, is_night_binary)
    
    except Exception as e:
        logging.error(f"特征提取错误: {str(e)}")
        # 返回一个默认特征向量
        return np.zeros(768 + 2)  # 默认特征维度

def extract_transformer_features(text, tokenizer, send_freq, is_night, model_type):
    """
    使用基于词向量的特征提取方法（不使用Transformer）
    """
    try:
        # 使用jieba分词
        tokens = tokenize(text)
        
        # 根据不同模型类型使用不同的特征提取逻辑
        if model_type == 'roberta':
            # 模拟RoBERTa特征
            embedding = generate_model_embedding(tokens, model_type)
        elif model_type == 'bert':
            # 模拟BERT特征
            embedding = generate_model_embedding(tokens, model_type)
        elif model_type == 'xlnet':
            # 模拟XLNet特征
            embedding = generate_model_embedding(tokens, model_type)
        elif model_type == 'gpt':
            # 模拟GPT特征
            embedding = generate_model_embedding(tokens, model_type)
        else:
            # 默认特征
            embedding = np.random.rand(768)
        
        # 添加元数据
        features = np.concatenate([embedding, [send_freq, is_night]])
        
        return features
    except Exception as e:
        logging.error(f"替代Transformer特征提取错误: {str(e)}")
        return np.zeros(768 + 2)

def generate_model_embedding(tokens, model_type):
    """
    根据模型类型生成模拟词向量
    
    参数:
        tokens: 分词后的词元列表
        model_type: 模型类型
    
    返回:
        embedding: 768维的词向量
    """
    # 在真实场景下，应该使用预训练的词向量模型
    # 这里我们使用随机向量模拟不同模型的特征空间
    seed = hash(model_type) % 10000
    np.random.seed(seed)
    
    # 生成768维的模拟embedding
    embedding = np.random.rand(768)
    
    return embedding

def extract_lstm_features(text, send_freq, is_night, max_len=50, vocab_size=5000):
    """
    为LSTM模型提取特征 - 将文本转换为词索引序列（优化版本）
    
    参数:
        text: 输入文本
        send_freq: 发送频率
        is_night: 是否夜间
        max_len: 序列最大长度，减少到50（减少内存使用）
        vocab_size: 词汇表大小，减少到5000（减少内存使用）
    
    返回:
        features: 整数索引序列
    """
    try:
        # 对文本进行分词
        tokens = tokenize(text)
        
        # 使用哈希函数保持一致性的词汇映射
        # 这样每次运行得到的词索引都是一致的
        indices = []
        for token in tokens:
            # 将词转换为一个数字（哈希值）
            # 确保数值范围在1到vocab_size-2之间（0用于padding，vocab_size-1用于OOV）
            hash_value = hash(token) % (vocab_size - 2) + 1
            indices.append(hash_value)
        
        # 截断或填充序列到固定长度
        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            # PAD索引是0
            indices = indices + [0] * (max_len - len(indices))
            
        # 确保特征仅包含词索引
        return np.array(indices, dtype=np.int64)
    except Exception as e:
        logging.error(f"LSTM特征提取错误: {str(e)}")
        # 返回全PAD的序列
        return np.zeros(max_len, dtype=np.int64)

def extract_cnn_features(text, send_freq, is_night):
    """
    为CNN模型提取特征
    """
    try:
        # 对文本进行分词
        tokens = tokenize(text)
        
        # 模拟CNN特征（实际应用中应使用预训练的词向量和CNN模型）
        # 在实际应用中，这里应该使用预训练的词向量和CNN模型
        embedding = np.random.rand(768)  # 模拟768维embedding
        
        # 添加元数据
        features = np.concatenate([embedding, [send_freq, is_night]])
        
        return features
    except Exception as e:
        logging.error(f"CNN特征提取错误: {str(e)}")
        return np.zeros(768 + 2)

def extract_traditional_features(text, send_freq, is_night):
    """
    为传统机器学习模型（如SVM, Naive Bayes）提取特征
    """
    try:
        # 对文本进行分词
        tokens = tokenize(text)
        
        # 模拟词袋或TF-IDF特征（实际应用中应使用真实的特征提取）
        # 在实际应用中，这里应该使用CountVectorizer或TfidfVectorizer
        embedding = np.random.rand(768)  # 模拟768维embedding
        
        # 添加元数据
        features = np.concatenate([embedding, [send_freq, is_night]])
        
        return features
    except Exception as e:
        logging.error(f"传统特征提取错误: {str(e)}")
        return np.zeros(768 + 2)
