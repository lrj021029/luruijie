import re
import jieba
from collections import Counter
import logging

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

def clean_text(text):
    """
    清理文本：
    1. 转换为小写
    2. 移除标点符号
    3. 移除数字
    4. 移除多余空格
    """
    try:
        # 转小写
        text = text.lower()
        # 移除URL
        text = re.sub(r'http\S+', '', text)
        # 移除特殊符号和标点
        text = re.sub(r'[^\w\s]', '', text)
        # 移除数字
        text = re.sub(r'\d+', '', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logging.error(f"文本清理错误: {str(e)}")
        return text

def tokenize(text):
    """
    使用jieba分词器对中文文本进行分词
    """
    try:
        # 对文本进行分词
        tokens = jieba.lcut(text)
        # 过滤掉停用词和短词
        tokens = [token for token in tokens if len(token) > 1]
        return tokens
    except Exception as e:
        logging.error(f"分词错误: {str(e)}")
        return []

def get_word_frequencies(texts):
    """
    获取一组文本中的词频
    """
    try:
        all_tokens = []
        for text in texts:
            cleaned_text = clean_text(text)
            tokens = tokenize(cleaned_text)
            all_tokens.extend(tokens)
        
        # 计算词频
        word_freq = Counter(all_tokens)
        return dict(word_freq)
    except Exception as e:
        logging.error(f"词频统计错误: {str(e)}")
        return {}

def normalize_metadata(send_freq, is_night):
    """
    归一化元数据特征
    """
    try:
        # 确保send_freq是浮点数
        send_freq = float(send_freq)
        # 确保is_night是0或1
        is_night = 1 if int(is_night) > 0 else 0
        
        # 将send_freq归一化到[0,1]区间（假设最大频率是100）
        normalized_freq = min(send_freq / 100.0, 1.0)
        
        return normalized_freq, is_night
    except Exception as e:
        logging.error(f"元数据归一化错误: {str(e)}")
        return 0.0, 0
