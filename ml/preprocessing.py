import re
import jieba
from collections import Counter
import logging
import os

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
        
def get_stopwords():
    """
    获取停用词列表，支持中文和英文
    
    返回:
        stopwords: 停用词列表
    """
    try:
        # 常用中文停用词
        chinese_stopwords = {
            '的', '了', '和', '是', '就', '都', '而', '及', '与', '或', '一个', '没有', 
            '我们', '你们', '他们', '她们', '它们', '这个', '那个', '这些', '那些',
            '不', '在', '人', '上', '来', '到', '时', '大', '地', '为', '子', '中', '你',
            '说', '生', '国', '年', '着', '就', '那', '和', '要', '她', '出', '也', '得',
            '里', '后', '自己', '之', '去', '之后', '所', '然', '没', '很', '看', '起',
            '还', '有', '些', '最', '们', '怎么', '已', '把', '被', '好', '这', '会', '才',
            '能', '两', '多', '让', '做', '知道', '等', '如果', '但是', '因为', '所以'
        }
        
        # 常用英文停用词
        english_stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
            "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
            'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'having', 'do', 'does', 'did', 'doing', 'would', 'should', 'could', 'ought',
            'i\'m', 'you\'re', 'he\'s', 'she\'s', 'it\'s', 'we\'re', 'they\'re', 'i\'ve',
            'you\'ve', 'we\'ve', 'they\'ve', 'i\'d', 'you\'d', 'he\'d', 'she\'d', 'we\'d',
            'they\'d', 'i\'ll', 'you\'ll', 'he\'ll', 'she\'ll', 'we\'ll', 'they\'ll',
            'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'hasn\'t', 'haven\'t', 'hadn\'t',
            'doesn\'t', 'don\'t', 'didn\'t', 'won\'t', 'wouldn\'t', 'shan\'t', 'shouldn\'t',
            'can\'t', 'cannot', 'couldn\'t', 'mustn\'t', 'let\'s', 'that\'s', 'who\'s',
            'what\'s', 'here\'s', 'there\'s', 'when\'s', 'where\'s', 'why\'s', 'how\'s',
            'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
            'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
        }
        
        # 合并停用词
        all_stopwords = chinese_stopwords.union(english_stopwords)
        
        logging.info(f"已加载 {len(all_stopwords)} 个停用词")
        return list(all_stopwords)
    except Exception as e:
        logging.error(f"加载停用词错误: {str(e)}")
        return []
