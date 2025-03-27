import numpy as np
import logging
import importlib.util
from ml.preprocessing import tokenize

# 检查依赖库是否可用
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None

# 有条件地导入
if HAS_TRANSFORMERS:
    try:
        from transformers import AutoTokenizer, AutoModel
        logging.info("成功导入transformers库")
    except Exception as e:
        logging.error(f"导入transformers库失败: {e}")
        HAS_TRANSFORMERS = False

if HAS_TORCH:
    try:
        import torch
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

# 缓存预训练模型的tokenizer
tokenizer_cache = {}

def get_tokenizer(model_type):
    """
    获取特定模型类型的tokenizer
    """
    if model_type in tokenizer_cache:
        return tokenizer_cache[model_type]
    
    # 检查transformers和torch是否可用
    if not HAS_TRANSFORMERS or not HAS_TORCH:
        logging.warning(f"获取tokenizer失败: transformers或torch库不可用")
        return None
    
    try:
        if model_type == 'roberta':
            tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        elif model_type == 'bert':
            tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        elif model_type == 'xlnet':
            tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-xlnet-base')
        elif model_type == 'gpt':
            tokenizer = AutoTokenizer.from_pretrained('uer/gpt2-chinese-cluecorpussmall')
        elif model_type in ['lstm', 'cnn', 'attention_lstm']:
            # 对于LSTM, CNN和Attention LSTM，我们使用jieba分词
            tokenizer = None
        else:
            # 默认使用jieba
            tokenizer = None
        
        tokenizer_cache[model_type] = tokenizer
        return tokenizer
    except Exception as e:
        logging.error(f"获取tokenizer错误: {str(e)}")
        return None

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
    使用Transformer模型提取文本特征
    """
    try:
        # 如果没有提供tokenizer，尝试获取，或使用随机特征
        if tokenizer is None:
            tokenizer = get_tokenizer(model_type)
            # 如果仍然为None，使用随机特征
            if tokenizer is None:
                logging.warning(f"无法获取{model_type}模型的tokenizer，使用随机特征向量")
                embedding = np.random.rand(768)  # 模拟768维embedding
                features = np.concatenate([embedding, [send_freq, is_night]])
                return features
        
        # 对文本进行编码
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # 模拟获取embedding（实际应用中应该使用预训练模型）
        # 在实际应用中，这里应该使用预训练模型获取真实的embedding
        # 例如：model = AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        #       with torch.no_grad():
        #           outputs = model(**inputs)
        #       embeddings = outputs.last_hidden_state[:,0,:].numpy()
        
        # 模拟embedding
        embedding = np.random.rand(768)  # 768维embedding
        
        # 添加元数据
        features = np.concatenate([embedding, [send_freq, is_night]])
        
        return features
    except Exception as e:
        logging.error(f"Transformer特征提取错误: {str(e)}")
        return np.zeros(768 + 2)

def extract_lstm_features(text, send_freq, is_night):
    """
    为LSTM模型提取特征
    """
    try:
        # 对文本进行分词
        tokens = tokenize(text)
        
        # 模拟LSTM特征（实际应用中应使用预训练的词向量和LSTM模型）
        # 在实际应用中，这里应该使用预训练的词向量和LSTM模型
        embedding = np.random.rand(768)  # 模拟768维embedding
        
        # 添加元数据
        features = np.concatenate([embedding, [send_freq, is_night]])
        
        return features
    except Exception as e:
        logging.error(f"LSTM特征提取错误: {str(e)}")
        return np.zeros(768 + 2)

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
