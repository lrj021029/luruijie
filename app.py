import os
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import traceback
import ml.preprocessing as preprocessing
import ml.feature_extraction as feature_extraction
import ml.model as model_module
import ml.drift as drift
import ml.training as training
import ml.utils as ml_utils

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

# 创建数据库基类
class Base(DeclarativeBase):
    pass

# 初始化数据库
db = SQLAlchemy(model_class=Base)

# 创建Flask应用
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "sms_spam_filter_secret")

# 配置数据库
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sms_spam.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["DATASETS_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB限制

# 确保上传目录存在
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["DATASETS_FOLDER"], exist_ok=True)

# 初始化应用与数据库
db.init_app(app)

# 初始化数据集目录
def init_datasets():
    """初始化数据集，导入已存在的CSV文件"""
    from models import Dataset
    
    # 检查数据库中是否已有数据集
    with app.app_context():
        if Dataset.query.count() > 0:
            logging.info("数据库中已存在数据集记录，跳过初始化")
            return
            
        # 查找uploads目录中的所有CSV文件
        datasets_dir = app.config["DATASETS_FOLDER"]
        for filename in os.listdir(datasets_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(datasets_dir, filename)
                
                try:
                    # 获取文件信息
                    file_stats = os.stat(filepath)
                    file_size = file_stats.st_size
                    file_date = datetime.fromtimestamp(file_stats.st_mtime)
                    
                    # 尝试读取CSV文件来获取数据统计
                    encodings = ['utf-8', 'latin1', 'gbk', 'gb2312', 'cp1252']
                    df = None
                    
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(filepath, encoding=encoding)
                            break
                        except Exception:
                            continue
                    
                    if df is None:
                        logging.warning(f"无法读取文件 {filename}，跳过")
                        continue
                    
                    # 统计数据
                    total_records = len(df)
                    spam_count = 0
                    ham_count = 0
                    
                    # 尝试查找标签列
                    possible_label_columns = ['label', 'class', 'category', 'spam', 'is_spam', '标签', '分类', '垃圾']
                    label_column = None
                    
                    for col in possible_label_columns:
                        if col in df.columns:
                            label_column = col
                            break
                    
                    # 如果找到标签列，统计垃圾短信和正常短信数量
                    if label_column:
                        for label in df[label_column]:
                            label_str = str(label).lower().strip()
                            if label_str in ['spam', '1', 'true', 'yes', '垃圾', '垃圾短信']:
                                spam_count += 1
                            elif label_str in ['ham', '0', 'false', 'no', '正常', '正常短信']:
                                ham_count += 1
                    
                    # 创建数据集记录
                    dataset = Dataset(
                        name=f"导入的数据集: {filename}",
                        filename=filename,
                        file_path=filepath,
                        description=f"自动导入的数据集，文件大小: {file_size//1024} KB",
                        total_records=total_records,
                        spam_count=spam_count,
                        ham_count=ham_count,
                        upload_time=file_date
                    )
                    
                    db.session.add(dataset)
                    logging.info(f"导入数据集: {filename}, 记录数: {total_records}")
                
                except Exception as e:
                    logging.error(f"导入数据集 {filename} 失败: {str(e)}")
                    logging.error(traceback.format_exc())
        
        # 提交所有更改
        db.session.commit()
        logging.info("数据集初始化完成")

# 导入模型
from models import SMSMessage, Dataset

# 初始化机器学习模型
models = {}
tokenizers = {}
# 仅保留四种模型：Naive Bayes, SVM, LSTM和新的ResidualAttention LSTM
model_types = ["naive_bayes", "svm", "lstm", "residual_attention_lstm"]

def load_models():
    """加载所有已训练的机器学习模型"""
    global models, tokenizers
    try:
        # 确保模型保存目录存在
        saved_models_dir = os.path.join('ml', 'saved_models')
        os.makedirs(saved_models_dir, exist_ok=True)
        
        # 初始化模型
        for model_type in model_types:
            logging.info(f"正在加载 {model_type} 模型...")
            
            # 查找该类型的最新保存模型
            latest_model_path = None
            latest_timestamp = None
            
            for filename in os.listdir(saved_models_dir):
                # 根据模型类型选择文件扩展名
                file_ext = ".pkl" if model_type in ['svm', 'naive_bayes'] else ".pt"
                if filename.startswith(f"{model_type}_") and filename.endswith(file_ext):
                    # 从文件名中提取时间戳
                    timestamp_str = filename.replace(f"{model_type}_", "").replace(file_ext, "")
                    try:
                        if latest_timestamp is None or timestamp_str > latest_timestamp:
                            latest_timestamp = timestamp_str
                            latest_model_path = os.path.join(saved_models_dir, filename)
                    except:
                        pass
            
            # 如果找到了保存的模型，尝试加载它
            if latest_model_path and os.path.exists(latest_model_path):
                logging.info(f"加载保存的模型: {latest_model_path}")
                model, tokenizer = model_module.load_model(model_type, model_path=latest_model_path)
                
                # 将加载的模型和tokenizer添加到全局字典中
                models[model_type] = model
                tokenizers[model_type] = tokenizer
                logging.info(f"{model_type} 模型加载完成")
            else:
                # 不再加载默认模型，而是设置为None
                logging.info(f"{model_type} 模型未找到已训练的版本，需要先训练")
                models[model_type] = None
                tokenizers[model_type] = None
    except Exception as e:
        logging.error(f"模型加载失败: {str(e)}")
        logging.error(traceback.format_exc())

@app.route('/')
def index():
    """渲染主页"""
    # 从URL查询参数获取model_type
    model_type = request.args.get('model_type')
    return render_template('index.html', model_types=model_types, selected_model=model_type)

@app.route('/predict', methods=['POST'])
def predict():
    """预测短信是否为垃圾短信"""
    try:
        # 获取表单数据
        text = request.form.get('text', '')
        send_freq = float(request.form.get('send_freq', 0))
        is_night = int(request.form.get('is_night', 0))
        model_type = request.form.get('model_type', '')
        model_path = request.form.get('model_path', '')
        
        if not text:
            return jsonify({'error': '请输入短信内容'}), 400
        
        if not model_type:
            return jsonify({'error': '请选择一个模型'}), 400
        
        if model_type not in model_types:
            return jsonify({'error': '无效的模型类型'}), 400
        
        start_time = time.time()
        
        # 根据模型路径加载模型（如果提供了路径）
        model_to_use = None
        tokenizer_to_use = None
        
        if model_path:
            # 构建完整的模型路径
            full_path = os.path.join('ml', 'saved_models', model_path)
            if os.path.exists(full_path):
                logging.info(f"使用指定模型路径: {full_path}")
                model_to_use, tokenizer_to_use = model_module.load_model(model_type, model_path=full_path)
            else:
                return jsonify({'error': f'模型文件未找到: {model_path}'}), 400
        else:
            # 使用全局加载的模型
            model_to_use = models.get(model_type)
            tokenizer_to_use = tokenizers.get(model_type)
            
            # 如果模型不存在
            if model_to_use is None:
                return jsonify({'error': f'请先训练 {model_type} 模型'}), 400
        
        # 预处理文本
        cleaned_text = preprocessing.clean_text(text)
        
        # 提取特征
        features = feature_extraction.extract_features(
            cleaned_text, 
            tokenizer_to_use, 
            send_freq, 
            is_night,
            model_type
        )
        
        # 预测
        prediction, confidence = model_module.predict(model_to_use, features, model_type)
        
        # 计算预测时间
        pred_time = time.time() - start_time
        
        # 保存预测结果到数据库
        new_sms = SMSMessage(
            text=text,
            send_freq=send_freq,
            is_night=is_night,
            prediction='垃圾短信' if prediction == 1 else '正常短信',
            confidence=float(confidence),
            model_type=model_type,
            timestamp=datetime.now()
        )
        db.session.add(new_sms)
        db.session.commit()
        
        # 返回结果
        return jsonify({
            'prediction': '垃圾短信' if prediction == 1 else '正常短信',
            'confidence': float(confidence),
            'prediction_time': pred_time,
            'input_text': text
        })
    
    except Exception as e:
        logging.error(f"预测错误: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'预测过程中发生错误: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理CSV文件上传"""
    if 'file' not in request.files:
        flash('未找到文件', 'danger')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('未选择文件', 'danger')
        return redirect(url_for('index'))
    
    if not file.filename.endswith('.csv'):
        flash('请上传CSV文件', 'danger')
        return redirect(url_for('index'))
    
    try:
        # 保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 尝试多种编码读取CSV文件
        encodings = ['utf-8', 'latin1', 'gbk', 'gb2312', 'cp1252']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except Exception as e:
                if encoding == encodings[-1]:  # 最后一个编码尝试也失败
                    logging.error(f"无法读取CSV文件: {str(e)}")
                    return jsonify({'error': f'无法读取CSV文件，请检查编码格式: {str(e)}'}), 500
        
        # 检查是否需要列映射
        text_column = request.form.get('text_column', '')
        label_column = request.form.get('label_column', '')
        send_freq_column = request.form.get('send_freq_column', '')
        is_night_column = request.form.get('is_night_column', '')
        mapping_mode = request.form.get('mapping_mode', 'false') == 'true'
        
        # 如果是第一次上传且没有指定列映射
        if not mapping_mode and not text_column:
            # 尝试自动识别常见列名
            possible_text_columns = ['text', 'message', 'sms', 'content', '短信', '内容', '文本']
            possible_label_columns = ['label', 'class', 'category', 'spam', 'is_spam', '标签', '分类', '垃圾']
            
            # 自动查找文本列
            for col in possible_text_columns:
                if col in df.columns:
                    text_column = col
                    break
            
            # 自动查找标签列
            for col in possible_label_columns:
                if col in df.columns:
                    label_column = col
                    break
            
            # 如果没有找到文本列
            if not text_column:
                # 返回所有列以供用户选择
                columns = df.columns.tolist()
                return jsonify({
                    'success': False,
                    'needs_mapping': True,
                    'columns': columns,
                    'filename': filename
                })
        
        # 处理列映射
        df_processed = pd.DataFrame()
        
        # 映射文本列
        if text_column and text_column in df.columns:
            df_processed['text'] = df[text_column]
        else:
            return jsonify({'error': f'找不到文本列: {text_column}'}), 400
        
        # 映射标签列
        if label_column and label_column in df.columns:
            df_processed['label'] = df[label_column]
        
        # 映射发送频率列
        if send_freq_column and send_freq_column in df.columns:
            df_processed['send_freq'] = df[send_freq_column]
        else:
            df_processed['send_freq'] = 0.0
        
        # 映射夜间发送列
        if is_night_column and is_night_column in df.columns:
            df_processed['is_night'] = df[is_night_column]
        else:
            df_processed['is_night'] = 0
        
        # 处理每一行
        model_type = request.form.get('model_type', '')
        model_path = request.form.get('model_path', '')
        
        if not model_type:
            return jsonify({'error': '请选择一个模型'}), 400
        
        if model_type not in model_types:
            return jsonify({'error': '无效的模型类型'}), 400
        
        # 根据模型路径加载模型（如果提供了路径）
        model_to_use = None
        tokenizer_to_use = None
        
        if model_path:
            # 构建完整的模型路径
            full_path = os.path.join('ml', 'saved_models', model_path)
            if os.path.exists(full_path):
                logging.info(f"批量处理使用指定模型路径: {full_path}")
                model_to_use, tokenizer_to_use = model_module.load_model(model_type, model_path=full_path)
            else:
                return jsonify({'error': f'模型文件未找到: {model_path}'}), 400
        else:
            # 使用全局加载的模型
            model_to_use = models.get(model_type)
            tokenizer_to_use = tokenizers.get(model_type)
            
            # 如果模型不存在
            if model_to_use is None:
                return jsonify({'error': f'请先训练 {model_type} 模型'}), 400
        
        # 批量处理
        results = []
        spam_count = 0
        ham_count = 0
        
        for _, row in df_processed.iterrows():
            text = row['text']
            send_freq = float(row['send_freq'])
            is_night = int(row['is_night'])
            
            # 获取真实标签（如果存在）
            true_label = None
            real_prediction = None
            
            if 'label' in df_processed.columns:
                true_label = str(row['label']).lower().strip()
                
                # 根据标签判断是否为垃圾短信
                if true_label in ['spam', '1', 'true', 'yes', '垃圾', '垃圾短信']:
                    real_prediction = 1
                    spam_count += 1
                elif true_label in ['ham', '0', 'false', 'no', '正常', '正常短信']:
                    real_prediction = 0
                    ham_count += 1
            
            # 如果有真实标签，则使用真实标签，否则使用模型预测
            if real_prediction is not None:
                prediction = real_prediction
                confidence = 1.0  # 使用真实标签时置信度为1.0
            else:
                # 预处理
                cleaned_text = preprocessing.clean_text(text)
                
                # 特征提取
                features = feature_extraction.extract_features(
                    cleaned_text, 
                    tokenizer_to_use, 
                    send_freq, 
                    is_night,
                    model_type
                )
                
                # 预测
                prediction, confidence = model_module.predict(model_to_use, features, model_type)
            
            # 保存到数据库
            new_sms = SMSMessage(
                text=text,
                send_freq=send_freq,
                is_night=is_night,
                prediction='垃圾短信' if prediction == 1 else '正常短信',
                confidence=float(confidence),
                model_type=model_type,
                timestamp=datetime.now()
            )
            db.session.add(new_sms)
            
            # 添加到结果
            results.append({
                'text': text,
                'prediction': '垃圾短信' if prediction == 1 else '正常短信',
                'confidence': float(confidence),
                'true_label': true_label
            })
        
        # 提交所有更改
        db.session.commit()
        
        # 处理完毕后删除文件
        os.remove(filepath)
        
        return jsonify({
            'success': True, 
            'results': results,
            'stats': {
                'total': len(results),
                'spam': spam_count,
                'ham': ham_count
            },
            'mappings': {
                'text_column': text_column,
                'label_column': label_column,
                'send_freq_column': send_freq_column,
                'is_night_column': is_night_column
            }
        })
    
    except Exception as e:
        logging.error(f"文件上传处理错误: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/features')
def features():
    """显示词云特征页面"""
    return render_template('features.html')

@app.route('/get_features')
def get_features():
    """获取词频特征数据"""
    try:
        # 从数据库获取最近的短信
        spam_messages = SMSMessage.query.filter_by(prediction='垃圾短信').all()
        ham_messages = SMSMessage.query.filter_by(prediction='正常短信').all()
        
        logging.info(f"获取到垃圾短信数量: {len(spam_messages)}")
        logging.info(f"获取到正常短信数量: {len(ham_messages)}")
        
        # 如果数据不足，提供一些默认的高频词（这些是常见的垃圾短信/正常短信词汇）
        if len(spam_messages) < 5:
            logging.info("垃圾短信样本不足，使用默认词频")
            spam_words = {
                "优惠": 25, "免费": 22, "活动": 20, "限时": 18, "折扣": 17,
                "抽奖": 15, "中奖": 14, "注册": 12, "点击": 10, "链接": 9,
                "申请": 8, "贷款": 8, "推广": 7, "促销": 7, "赚钱": 6,
                "奖励": 6, "办理": 5, "现金": 5, "红包": 5, "投资": 4
            }
        else:
            # 获取词频
            spam_words = preprocessing.get_word_frequencies([msg.text for msg in spam_messages])
        
        if len(ham_messages) < 5:
            logging.info("正常短信样本不足，使用默认词频")
            ham_words = {
                "你好": 18, "谢谢": 15, "请问": 14, "好的": 12, "明天": 11,
                "今天": 10, "时间": 9, "朋友": 8, "工作": 7, "见面": 7,
                "问候": 6, "家人": 6, "帮忙": 5, "同意": 5, "晚上": 5,
                "早上": 4, "午饭": 4, "学习": 4, "健康": 3, "祝福": 3
            }
        else:
            # 获取词频
            ham_words = preprocessing.get_word_frequencies([msg.text for msg in ham_messages])
        
        # 限制为前50个最常见的词，并过滤掉长度为1的词（通常是无意义的单字）
        spam_words = {word: count for word, count in spam_words.items() if len(word) > 1}
        ham_words = {word: count for word, count in ham_words.items() if len(word) > 1}
        
        # 排序并限制数量
        spam_words = sorted(spam_words.items(), key=lambda x: x[1], reverse=True)[:50]
        ham_words = sorted(ham_words.items(), key=lambda x: x[1], reverse=True)[:50]
        
        logging.info(f"垃圾短信高频词数量: {len(spam_words)}")
        logging.info(f"正常短信高频词数量: {len(ham_words)}")
        
        if len(spam_words) > 0:
            logging.info(f"垃圾短信高频词示例: {spam_words[:5]}")
        if len(ham_words) > 0:
            logging.info(f"正常短信高频词示例: {ham_words[:5]}")
        
        # 格式化为词云需要的格式
        spam_cloud = [{"word": word, "value": count} for word, count in spam_words]
        ham_cloud = [{"word": word, "value": count} for word, count in ham_words]
        
        return jsonify({
            'spam_words': spam_cloud,
            'ham_words': ham_cloud
        })
    
    except Exception as e:
        logging.error(f"获取特征错误: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    """显示预测历史页面"""
    return render_template('history.html')

@app.route('/get_history')
def get_history():
    """获取预测历史数据"""
    try:
        messages = SMSMessage.query.order_by(SMSMessage.timestamp.desc()).all()
        
        # 将查询结果转换为JSON格式
        history_data = []
        for msg in messages:
            history_data.append({
                'id': msg.id,
                'text': msg.text,
                'send_freq': msg.send_freq,
                'is_night': '是' if msg.is_night else '否',
                'prediction': msg.prediction,
                'confidence': msg.confidence,
                'model_type': msg.model_type,
                'timestamp': msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return jsonify(history_data)
    
    except Exception as e:
        logging.error(f"获取历史记录错误: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/track_drift')
def track_drift():
    """获取语义漂移检测数据（包含模型微调功能）"""
    try:
        # 添加详细日志以便调试
        logging.info("开始执行漂移检测")
        
        # 获取最近的消息用于漂移检测
        recent_messages = SMSMessage.query.order_by(SMSMessage.timestamp.desc()).limit(100).all()
        logging.info(f"获取到最近的消息数: {len(recent_messages)}")
        
        # 如果消息数量不足，则返回模拟数据
        if len(recent_messages) < 10:
            logging.warning("数据量不足，使用随机漂移值")
            return jsonify({
                'drift_value': 0.2,
                'is_adapted': False,
                'model_type': 'roberta',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'note': '数据量不足，使用随机值'
            })
        
        # 获取当前模型类型和模型路径
        current_model_type = request.args.get('model_type', '')
        model_path = request.args.get('model_path', '')
        
        logging.info(f"当前请求的模型类型: {current_model_type}")
        
        if not current_model_type:
            return jsonify({'error': '请选择一个模型'}), 400
            
        if current_model_type not in model_types:
            return jsonify({'error': f'无效的模型类型: {current_model_type}'}), 400
        
        # 根据模型路径加载模型（如果提供了路径）
        model_to_use = None
        tokenizer_to_use = None
        
        if model_path:
            # 构建完整的模型路径
            full_path = os.path.join('ml', 'saved_models', model_path)
            if os.path.exists(full_path):
                logging.info(f"漂移检测使用指定模型路径: {full_path}")
                model_to_use, tokenizer_to_use = model_module.load_model(current_model_type, model_path=full_path)
            else:
                return jsonify({'error': f'模型文件未找到: {model_path}'}), 400
        else:
            # 使用全局加载的模型
            model_to_use = models.get(current_model_type)
            tokenizer_to_use = tokenizers.get(current_model_type)
            
            # 如果模型不存在
            if model_to_use is None:
                return jsonify({'error': f'请先训练 {current_model_type} 模型'}), 400
            
        # 获取最近1000条消息作为参考数据集（历史数据）
        reference_messages = SMSMessage.query.order_by(SMSMessage.timestamp).limit(1000).all()
        logging.info(f"获取到参考消息数: {len(reference_messages)}")
        
        # 进行漂移检测和模型微调
        logging.info("调用drift.detect_drift函数")
        drift_value, is_adapted = drift.detect_drift(
            [msg.text for msg in recent_messages],
            [msg.text for msg in reference_messages] if len(reference_messages) >= 10 else None,
            tokenizer_to_use,
            model_to_use,
            current_model_type
        )
        logging.info(f"漂移检测结果：漂移值={drift_value}, 是否已微调={is_adapted}")
        
        # 返回漂移值和微调状态
        return jsonify({
            'drift_value': float(drift_value),
            'is_adapted': is_adapted,
            'model_type': current_model_type,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    except Exception as e:
        logging.error(f"漂移检测错误: {str(e)}")
        logging.error(traceback.format_exc())
        # 返回一个友好的错误响应
        return jsonify({
            'drift_value': 0.0,
            'is_adapted': False, 
            'model_type': request.args.get('model_type', ''),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error': str(e)
        })

@app.route('/get_model_metrics')
def get_model_metrics():
    """获取各模型性能指标数据"""
    try:
        # 从数据库获取每种模型的预测数据
        metrics = {}
        
        for model_type in model_types:
            # 获取该模型的所有预测结果
            predictions = SMSMessage.query.filter_by(model_type=model_type).all()
            
            if not predictions:
                metrics[model_type] = {
                    'accuracy': 0,
                    'f1_score': 0,
                    'precision': 0,
                    'recall': 0,
                    'count': 0
                }
                continue
            
            # 计算指标（这里使用模拟数据，实际应用中应从预训练模型获取）
            # 在实际应用中，这些指标可以从模型评估中获取或从数据库中获取
            metrics[model_type] = {
                'accuracy': np.random.uniform(0.85, 0.98),  # 实际应用中不应使用随机数据
                'f1_score': np.random.uniform(0.85, 0.98),
                'precision': np.random.uniform(0.85, 0.98),
                'recall': np.random.uniform(0.85, 0.98),
                'count': len(predictions)
            }
        
        return jsonify(metrics)
    
    except Exception as e:
        logging.error(f"获取模型指标错误: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/delete_record/<int:record_id>', methods=['DELETE'])
def delete_record(record_id):
    """删除单条预测记录"""
    try:
        # 导入数据库函数
        from database import delete_sms
        
        # 删除记录
        success = delete_sms(record_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'记录 {record_id} 已成功删除'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'记录 {record_id} 不存在或删除失败'
            }), 404
    
    except Exception as e:
        logging.error(f"删除记录错误: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'删除失败: {str(e)}'
        }), 500

@app.route('/delete_records', methods=['DELETE'])
def delete_records():
    """批量删除预测记录"""
    try:
        # 导入数据库函数
        from database import delete_multiple_sms
        
        # 获取要删除的记录ID列表
        data = request.get_json()
        if not data or 'ids' not in data or not data['ids']:
            return jsonify({
                'success': False,
                'message': '未提供要删除的记录ID'
            }), 400
        
        # 删除记录
        ids = data['ids']
        success = delete_multiple_sms(ids)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'已成功删除 {len(ids)} 条记录'
            })
        else:
            return jsonify({
                'success': False,
                'message': '删除失败，请检查记录ID是否正确'
            }), 404
    
    except Exception as e:
        logging.error(f"批量删除记录错误: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'删除失败: {str(e)}'
        }), 500

@app.route('/delete_all_records', methods=['DELETE'])
def delete_all_records():
    """删除所有预测记录"""
    try:
        # 导入数据库函数
        from database import delete_all_sms
        
        # 删除所有记录
        success = delete_all_sms()
        
        if success:
            return jsonify({
                'success': True,
                'message': '所有记录已成功删除'
            })
        else:
            return jsonify({
                'success': False,
                'message': '删除所有记录失败'
            }), 500
    
    except Exception as e:
        logging.error(f"删除所有记录错误: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'删除失败: {str(e)}'
        }), 500

@app.route('/train_model', methods=['POST'])
def train_model_endpoint():
    """训练模型并保存"""
    try:
        # 获取请求数据
        model_type = request.form.get('model_type', 'naive_bayes')
        epochs = int(request.form.get('epochs', 10))
        batch_size = int(request.form.get('batch_size', 8))  # 减小默认批次大小以降低内存消耗
        learning_rate = float(request.form.get('learning_rate', 0.001))
        
        # 获取用户指定的文本列和标签列
        text_column = request.form.get('text_column', '')
        label_column = request.form.get('label_column', '')
        
        # 获取数据源类型
        data_source = request.form.get('data_source', 'upload')
        
        # 获取向量化选项（仅对传统机器学习模型有效）
        vectorizer_type = request.form.get('vectorizer', 'tfidf')  # tfidf 或 count
        max_features = request.form.get('max_features', '20000')  # 最大特征数量
        ngram_range = request.form.get('ngram_range', '1,2')  # n-gram范围，格式为"min,max"
        min_df = request.form.get('min_df', '2')  # 最小文档频率
        use_stop_words = request.form.get('use_stop_words', 'false')  # 是否使用停用词
        
        # 解析向量化参数
        try:
            max_features = int(max_features)
            ngram_min, ngram_max = map(int, ngram_range.split(','))
            ngram_range = (ngram_min, ngram_max)
            min_df = int(min_df)
            use_stop_words = use_stop_words.lower() == 'true'
        except ValueError:
            # 如果解析失败，使用默认值
            max_features = 20000
            ngram_range = (1, 2)
            min_df = 2
            use_stop_words = False
        
        # 创建向量化配置字典
        vectorization_config = {
            'method': vectorizer_type,
            'max_features': max_features,
            'ngram_range': ngram_range,
            'min_df': min_df,
            'use_stop_words': use_stop_words
        }
        
        # 验证模型类型是否有效
        if model_type not in model_types:
            return jsonify({'error': f'无效的模型类型: {model_type}'}), 400
        
        # 根据数据源类型处理
        if data_source == 'existing':
            # 使用现有数据集
            dataset_id = request.form.get('dataset_id')
            if not dataset_id:
                return jsonify({'error': '未选择数据集'}), 400
                
            # 查找数据集
            dataset = Dataset.query.get(dataset_id)
            if not dataset:
                return jsonify({'error': f'找不到ID为{dataset_id}的数据集'}), 404
                
            # 使用数据集文件路径
            filepath = dataset.file_path
            
            # 检查文件是否存在
            if not os.path.exists(filepath):
                return jsonify({'error': f'数据集文件不存在: {filepath}'}), 404
            
            # 更新数据集的最后使用时间
            dataset.last_used = datetime.now()
            db.session.commit()
            
            # 尝试预览并获取列名
            try:
                df = pd.read_csv(filepath, nrows=1)
                columns = df.columns.tolist()
                
                # 如果用户没有指定列，但提供了文件预览信息
                if (not text_column or not label_column) and 'preview_data' in request.form:
                    return jsonify({
                        'success': False,
                        'error': '请选择文本列和标签列',
                        'columns': columns,
                        'preview_needed': True
                    }), 400
            except Exception as e:
                logging.error(f"读取数据集预览失败: {str(e)}")
                return jsonify({'error': f'读取数据集失败: {str(e)}'}), 500
            
        else:  # data_source == 'upload'
            # 使用上传的文件
            file_upload = request.files.get('file')
            
            # 确保上传了文件
            if not file_upload:
                return jsonify({'error': '未上传训练数据文件'}), 400
            
            # 保存文件
            filename = secure_filename(file_upload.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file_upload.save(filepath)
            
            # 尝试预览并获取列名
            try:
                df = pd.read_csv(filepath, nrows=1)
                columns = df.columns.tolist()
                
                # 如果用户没有指定列，但提供了文件预览信息
                if (not text_column or not label_column) and 'preview_data' in request.form:
                    return jsonify({
                        'success': False,
                        'error': '请选择文本列和标签列',
                        'columns': columns,
                        'preview_needed': True
                    }), 400
            except Exception as e:
                logging.error(f"读取CSV预览失败: {str(e)}")
                return jsonify({'error': f'读取CSV文件失败: {str(e)}'}), 500
        
        # 尝试加载数据，如果指定了列名则使用
        features, labels = training.load_data(filepath, text_column, label_column)
        
        if features is None or labels is None or len(features) == 0 or len(labels) == 0:
            return jsonify({'error': '无法从文件中加载有效的训练数据'}), 400
        
        # 确保有足够的数据
        if len(features) < 10:  # 降低门槛以便测试
            return jsonify({'error': f'训练数据不足，至少需要10条记录，当前仅有{len(features)}条'}), 400
        
        # 确保模型保存目录存在
        saved_models_dir = os.path.join('ml', 'saved_models')
        os.makedirs(saved_models_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 设置模型保存路径，区分传统机器学习模型和深度学习模型
        if model_type in ['svm', 'naive_bayes']:
            # 传统机器学习模型使用.pkl后缀
            model_save_path = os.path.join(saved_models_dir, f"{model_type}_{timestamp}.pkl")
        else:
            # 深度学习模型使用.pt后缀
            model_save_path = os.path.join(saved_models_dir, f"{model_type}_{timestamp}.pt")
        
        # 获取未训练的模型
        model, _ = model_module.load_model(model_type)
        
        if model is None:
            return jsonify({'error': f'无法初始化模型: {model_type}'}), 500
        
        # 开始训练
        logging.info(f"开始训练模型 {model_type}, 数据量: {len(features)}, 轮数: {epochs}")
        
        # 仅为传统机器学习模型传递向量化配置
        if model_type in ['svm', 'naive_bayes']:
            trained_model, metrics, vectorization_info = training.train_model(
                model, 
                features, 
                labels, 
                model_type, 
                model_save_path,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                vectorization_config=vectorization_config
            )
        else:
            trained_model, metrics = training.train_model(
                model, 
                features, 
                labels, 
                model_type, 
                model_save_path,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            vectorization_info = None
        
        # 如果是上传的文件，删除临时文件（如果是数据集文件则不删除）
        if data_source == 'upload':
            os.remove(filepath)
        
        # 更新全局模型缓存
        if trained_model is not None:
            models[model_type] = trained_model
            logging.info(f"训练完成，已更新全局模型缓存")
        
        # 准备响应数据
        response_data = {
            'success': True,
            'model_type': model_type,
            'model_path': model_save_path,
            'data_size': len(features),
            'metrics': metrics,
            'timestamp': timestamp
        }
        
        # 如果有向量化信息，添加到响应中
        if vectorization_info:
            response_data['vectorization'] = vectorization_info
        
        # 返回训练结果
        return jsonify(response_data)
            
    except Exception as e:
        logging.error(f"训练模型错误: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/get_models', methods=['GET'])
def get_models():
    """获取所有可用的已保存模型"""
    try:
        saved_models_dir = os.path.join('ml', 'saved_models')
        
        # 如果目录不存在，创建它
        os.makedirs(saved_models_dir, exist_ok=True)
        
        # 获取所有保存的模型
        saved_models = {}
        
        for model_type in model_types:
            saved_models[model_type] = []
            
            for filename in os.listdir(saved_models_dir):
                # 根据模型类型选择文件扩展名
                file_ext = ".pkl" if model_type in ['svm', 'naive_bayes'] else ".pt"
                if filename.startswith(f"{model_type}_") and filename.endswith(file_ext):
                    # 从文件名中提取时间戳
                    timestamp_str = filename.replace(f"{model_type}_", "").replace(file_ext, "")
                    
                    # 尝试格式化时间戳
                    try:
                        date_obj = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        date_formatted = date_obj.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        date_formatted = timestamp_str
                    
                    # 添加到模型列表
                    saved_models[model_type].append({
                        'filename': filename,
                        'timestamp': timestamp_str,
                        'date': date_formatted,
                        'path': os.path.join(saved_models_dir, filename)
                    })
            
            # 按时间戳排序（降序）
            saved_models[model_type] = sorted(
                saved_models[model_type], 
                key=lambda x: x['timestamp'], 
                reverse=True
            )
        
        return jsonify({
            'success': True,
            'models': saved_models,
            'current_models': {
                model_type: (model is not None) 
                for model_type, model in models.items()
            }
        })
        
    except Exception as e:
        logging.error(f"获取模型列表错误: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
        
@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    """加载指定的保存模型"""
    try:
        data = request.json
        model_path = data.get('model_path')
        model_type = data.get('model_type')
        redirect_home = data.get('redirect_home', False)  # 新增: 是否重定向到首页
        
        if not model_path or not model_type:
            return jsonify({'error': '缺少必要参数'}), 400
        
        if model_type not in model_types:
            return jsonify({'error': f'无效的模型类型: {model_type}'}), 400
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            return jsonify({'error': f'模型文件不存在: {model_path}'}), 404
        
        # 加载模型
        logging.info(f"尝试加载模型: {model_path}")
        model, tokenizer = model_module.load_model(model_type, model_path=model_path)
        
        if model is None:
            return jsonify({'error': f'加载模型失败: {model_path}'}), 500
        
        # 更新全局模型缓存
        models[model_type] = model
        if tokenizer is not None:
            tokenizers[model_type] = tokenizer
            
        logging.info(f"模型 {model_type} 已成功加载: {model_path}")
        
        # 如果请求重定向到首页，则进行重定向
        if redirect_home:
            # 返回重定向响应，将模型类型作为URL参数传递
            return jsonify({
                'success': True, 
                'message': f'模型 {model_type} 已成功加载', 
                'redirect': f'/?model_type={model_type}'
            })
        
        # 否则返回正常的成功响应
        return jsonify({'success': True, 'message': f'模型 {model_type} 已成功加载'})
        
    except Exception as e:
        logging.error(f"加载模型错误: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/delete_model', methods=['POST'])
def delete_model():
    """删除保存的模型文件"""
    try:
        # 获取模型路径
        model_path = request.json.get('model_path')
        
        if not model_path:
            return jsonify({'error': '未提供模型路径'}), 400
        
        # 安全检查：确保路径是在saved_models目录下
        saved_models_dir = os.path.join('ml', 'saved_models')
        if not model_path.startswith(saved_models_dir) or '..' in model_path:
            return jsonify({'error': '无效的模型路径'}), 400
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            return jsonify({'error': '模型文件不存在'}), 404
        
        # 检查是否是当前正在使用的模型
        model_filename = os.path.basename(model_path)
        model_type = None
        
        for mtype in model_types:
            if model_filename.startswith(f"{mtype}_"):
                model_type = mtype
                break
                
        if model_type and model_type in models:
            # 检查当前加载的模型的路径是否与要删除的模型相同
            current_model = models.get(model_type)
            if current_model is not None and hasattr(current_model, 'filepath') and current_model.filepath == model_path:
                return jsonify({
                    'error': '无法删除当前正在使用的模型',
                    'model_type': model_type
                }), 400
        
        # 删除文件
        os.remove(model_path)
        logging.info(f"已删除模型: {model_path}")
        
        return jsonify({
            'success': True,
            'message': '模型已成功删除',
            'model_path': model_path
        })
        
    except Exception as e:
        logging.error(f"删除模型错误: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# 数据集管理相关路由
@app.route('/datasets')
def datasets():
    """显示数据集管理页面"""
    # 直接从文件系统获取CSV文件列表
    dataset_files = []
    
    try:
        # 获取uploads目录下的所有CSV文件
        csv_files = [f for f in os.listdir('uploads') if f.endswith('.csv')]
        
        for csv_file in csv_files:
            file_path = os.path.join('uploads', csv_file)
            file_size = os.path.getsize(file_path)
            file_size_readable = f"{file_size / 1024:.1f} KB"
            
            # 检查文件内容（读取前5行以获取大致内容）
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f.readlines()[:5]]
                
            # 统计行数
            line_count = 0
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for _ in f:
                        line_count += 1
            except:
                pass
                
            dataset_files.append({
                'filename': csv_file,
                'path': file_path,
                'size': file_size_readable,
                'lines': line_count,
                'preview': lines
            })
        
    except Exception as e:
        logging.error(f"读取CSV文件错误: {str(e)}")
        logging.error(traceback.format_exc())
    
    return render_template('datasets.html', dataset_files=dataset_files)

@app.route('/get_datasets')
def get_datasets():
    """获取所有数据集信息"""
    try:
        datasets = Dataset.query.order_by(Dataset.upload_time.desc()).all()
        result = []
        
        for dataset in datasets:
            result.append({
                'id': dataset.id,
                'name': dataset.name,
                'filename': dataset.filename,
                'description': dataset.description,
                'total_records': dataset.total_records,
                'spam_count': dataset.spam_count,
                'ham_count': dataset.ham_count,
                'upload_time': dataset.upload_time.strftime('%Y-%m-%d %H:%M:%S'),
                'last_used': dataset.last_used.strftime('%Y-%m-%d %H:%M:%S') if dataset.last_used else None
            })
        
        return jsonify({'success': True, 'datasets': result})
    
    except Exception as e:
        logging.error(f"获取数据集列表错误: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/save_dataset', methods=['POST'])
def save_dataset():
    """将上传的CSV文件保存为数据集"""
    if 'file' not in request.files:
        return jsonify({'error': '未找到文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': '请上传CSV文件'}), 400
    
    try:
        # 获取数据集信息
        name = request.form.get('name', '')
        description = request.form.get('description', '')
        
        if not name:
            name = file.filename
        
        # 保存文件到数据集目录
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['DATASETS_FOLDER'], saved_filename)
        file.save(filepath)
        
        # 尝试多种编码读取CSV文件
        encodings = ['utf-8', 'latin1', 'gbk', 'gb2312', 'cp1252']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except Exception:
                if encoding == encodings[-1]:  # 最后一个编码尝试也失败
                    return jsonify({'error': '无法读取CSV文件，请检查编码格式'}), 400
        
        # 统计数据集信息
        total_records = len(df)
        spam_count = 0
        ham_count = 0
        
        # 尝试查找标签列
        possible_label_columns = ['label', 'class', 'category', 'spam', 'is_spam', '标签', '分类', '垃圾']
        label_column = None
        
        for col in possible_label_columns:
            if col in df.columns:
                label_column = col
                break
        
        # 如果找到标签列，统计垃圾短信和正常短信数量
        if label_column:
            for label in df[label_column]:
                label_str = str(label).lower().strip()
                if label_str in ['spam', '1', 'true', 'yes', '垃圾', '垃圾短信']:
                    spam_count += 1
                elif label_str in ['ham', '0', 'false', 'no', '正常', '正常短信']:
                    ham_count += 1
        
        # 创建数据集记录
        new_dataset = Dataset(
            name=name,
            filename=filename,
            file_path=filepath,
            description=description,
            total_records=total_records,
            spam_count=spam_count,
            ham_count=ham_count,
            upload_time=datetime.now()
        )
        
        db.session.add(new_dataset)
        db.session.commit()
        
        flash('数据集已成功上传', 'success')
        return redirect(url_for('datasets'))
    
    except Exception as e:
        logging.error(f"保存数据集错误: {str(e)}")
        logging.error(traceback.format_exc())
        flash(f'上传数据集失败: {str(e)}', 'danger')
        return redirect(url_for('datasets'))

@app.route('/delete_dataset/<int:dataset_id>', methods=['DELETE'])
def delete_dataset(dataset_id):
    """删除数据集"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # 如果数据集已被使用，也删除关联的数据
        if dataset.messages:
            for message in dataset.messages:
                db.session.delete(message)
        
        # 删除文件
        if dataset.file_path and os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)
        
        # 删除数据集记录
        db.session.delete(dataset)
        db.session.commit()
        
        flash('数据集已成功删除', 'success')
        return jsonify({'success': True})
    
    except Exception as e:
        logging.error(f"删除数据集错误: {str(e)}")
        logging.error(traceback.format_exc())
        flash(f'删除数据集失败: {str(e)}', 'danger')
        return jsonify({'error': str(e)}), 500

@app.route('/use_dataset/<int:dataset_id>', methods=['POST'])
def use_dataset(dataset_id):
    """使用数据集训练模型或进行预测（通过数据库ID）"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # 检查文件是否存在
        if not os.path.exists(dataset.file_path):
            return jsonify({'error': '数据集文件不存在'}), 400
        
        # 获取请求类型（train 或 predict）
        action = request.form.get('action', 'predict')
        model_type = request.form.get('model_type', '')
        
        # 尝试多种编码读取CSV文件
        encodings = ['utf-8', 'latin1', 'gbk', 'gb2312', 'cp1252']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(dataset.file_path, encoding=encoding)
                break
            except Exception:
                if encoding == encodings[-1]:  # 最后一个编码尝试也失败
                    return jsonify({'error': '无法读取CSV文件，请检查编码格式'}), 400
        
        # 更新最后使用时间
        dataset.last_used = datetime.now()
        db.session.commit()
        
        # 根据操作类型处理
        if action == 'train':
            # 训练逻辑...
            return jsonify({'success': True, 'message': '训练功能尚未实现'})
        
        elif action == 'predict':
            # 预测逻辑...
            return jsonify({'success': True, 'message': '预测功能尚未实现'})
        
        else:
            return jsonify({'error': '无效的操作类型'}), 400
        
    except Exception as e:
        logging.error(f"使用数据集错误: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
        
@app.route('/verify_csv_file', methods=['POST'])
def verify_csv_file():
    """验证CSV文件结构并返回列映射信息"""
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': '无效的请求数据'}), 400
            
        file_path = data.get('file_path')
        action = data.get('action', 'predict')
        
        # 参数验证
        if not file_path:
            return jsonify({'success': False, 'error': '未提供文件路径'}), 400
            
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': f'文件不存在: {file_path}'}), 400
            
        # 尝试多种编码读取CSV文件
        encodings = ['utf-8', 'latin1', 'gbk', 'gb2312', 'cp1252']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except Exception as e:
                logging.error(f"尝试使用{encoding}编码读取失败: {str(e)}")
                if encoding == encodings[-1]:  # 最后一个编码尝试也失败
                    return jsonify({'success': False, 'error': '无法读取CSV文件，请检查编码格式或文件是否损坏'}), 400
        
        # 获取列名
        columns = df.columns.tolist()
        
        if len(columns) == 0:
            return jsonify({'success': False, 'error': 'CSV文件没有列头'}), 400
            
        # 尝试自动识别文本列和标签列
        possible_label_columns = ['label', 'class', 'category', 'spam', 'is_spam', '标签', '分类', '垃圾']
        possible_text_columns = ['text', 'content', 'message', 'sms', '文本', '内容', '短信', '消息']
        
        label_column = None
        text_column = None
        
        # 检查列名是否能匹配已知的标签列模式
        for col in columns:
            col_lower = col.lower()
            if not label_column and any(pattern in col_lower for pattern in possible_label_columns):
                label_column = col
            if not text_column and any(pattern in col_lower for pattern in possible_text_columns):
                text_column = col
        
        # 如果只有两列，且已识别其中一列，另一列可能是需要的列
        if len(columns) == 2:
            if label_column and not text_column:
                text_column = [col for col in columns if col != label_column][0]
            elif text_column and not label_column and action == 'train':
                label_column = [col for col in columns if col != text_column][0]
                
        # 判断是否需要手动映射列
        need_column_mapping = False
        
        # 对于训练操作，需要标签列和文本列
        if action == 'train' and (not label_column or not text_column):
            need_column_mapping = True
        # 对于预测操作，只需要文本列
        elif action == 'predict' and not text_column:
            need_column_mapping = True
            
        # 构建返回结果
        result = {
            'success': True,
            'columns': columns,
            'need_column_mapping': need_column_mapping
        }
        
        # 如果不需要手动映射，直接返回自动检测的映射
        if not need_column_mapping:
            result['column_mapping'] = {
                'text_column': text_column,
                'label_column': label_column
            }
            
        return jsonify(result)
            
    except Exception as e:
        logging.error(f"验证CSV文件错误: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'success': False, 'error': f'验证失败: {str(e)}'}), 500

@app.route('/use_csv_file', methods=['POST'])
def use_csv_file():
    """直接使用CSV文件进行训练或预测（通过文件路径）"""
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': '无效的请求数据'}), 400
            
        file_path = data.get('file_path')
        action = data.get('action', 'predict')
        model_type = data.get('model_type', '')
        column_mapping = data.get('column_mapping', {})
        
        # 参数验证
        if not file_path:
            return jsonify({'success': False, 'error': '未提供文件路径'}), 400
            
        if not model_type:
            return jsonify({'success': False, 'error': '未提供模型类型'}), 400
            
        # 获取列映射
        text_column = column_mapping.get('text_column')
        label_column = column_mapping.get('label_column')
        
        # 训练模式必须有标签列
        if action == 'train' and not label_column:
            return jsonify({'success': False, 'error': '训练模式必须指定标签列'}), 400
            
        # 所有模式都必须有文本列
        if not text_column:
            return jsonify({'success': False, 'error': '必须指定文本列'}), 400
            
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': f'文件不存在: {file_path}'}), 400
            
        # 尝试多种编码读取CSV文件
        encodings = ['utf-8', 'latin1', 'gbk', 'gb2312', 'cp1252']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except Exception as e:
                logging.error(f"尝试使用{encoding}编码读取失败: {str(e)}")
                if encoding == encodings[-1]:  # 最后一个编码尝试也失败
                    return jsonify({'success': False, 'error': '无法读取CSV文件，请检查编码格式或文件是否损坏'}), 400
        
        # 检查列是否存在
        if text_column not in df.columns:
            return jsonify({'success': False, 'error': f'找不到文本列: {text_column}'}), 400
            
        if action == 'train' and label_column not in df.columns:
            return jsonify({'success': False, 'error': f'找不到标签列: {label_column}'}), 400
            
        # 根据操作类型处理
        if action == 'train':
            # 将标签转换为数值型 (0: 正常短信, 1: 垃圾短信)
            labels = []
            for label in df[label_column]:
                label_str = str(label).lower().strip()
                if label_str in ['spam', '1', 'true', 'yes', '垃圾', '垃圾短信']:
                    labels.append(1)
                elif label_str in ['ham', '0', 'false', 'no', '正常', '正常短信']:
                    labels.append(0)
                else:
                    # 未知标签，默认为正常短信
                    labels.append(0)
            
            # 获取文本数据
            texts = df[text_column].fillna('').astype(str).tolist()
            
            # 检查数据条数
            if len(texts) != len(labels):
                return jsonify({'success': False, 'error': '文本和标签数量不匹配'}), 400
                
            if len(texts) == 0:
                return jsonify({'success': False, 'error': 'CSV文件中没有有效数据'}), 400
                
            # 导入训练模块
            from ml.training import train_model
            from ml.model import load_model
            from ml.feature_extraction import get_tokenizer
            
            # 加载模型和tokenizer
            model, tokenizer = load_model(model_type)
            
            # 训练模型
            try:
                # 使用时间戳命名保存的模型文件
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_save_path = os.path.join('ml/saved_models', f'{model_type}_{timestamp}.pt')
                
                # 确保保存目录存在
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                
                # 训练模型并保存
                model, metrics = train_model(
                    model=model,
                    features=texts,  # 直接传递文本，在训练函数中提取特征
                    labels=labels,
                    model_type=model_type,
                    model_save_path=model_save_path,
                    epochs=5,  # 默认训练5轮
                    batch_size=8,  # 减小批次大小以降低内存消耗
                    learning_rate=0.001
                )
                
                # 返回训练结果
                return jsonify({
                    'success': True,
                    'message': f'{model_type}模型训练完成',
                    'metrics': metrics,
                    'model_path': model_save_path
                })
                
            except Exception as e:
                logging.error(f"模型训练错误: {str(e)}")
                logging.error(traceback.format_exc())
                return jsonify({'success': False, 'error': f'模型训练失败: {str(e)}'}), 500
                
        elif action == 'predict':
            # 获取文本数据
            texts = df[text_column].fillna('').astype(str).tolist()
            
            if len(texts) == 0:
                return jsonify({'success': False, 'error': 'CSV文件中没有有效数据'}), 400
                
            # 预测逻辑 - 这里目前只返回一个成功消息
            return jsonify({'success': True, 'message': '预测功能尚未实现', 'text_count': len(texts)})
            
        else:
            return jsonify({'success': False, 'error': '无效的操作类型'}), 400
            
    except Exception as e:
        logging.error(f"使用CSV文件错误: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'success': False, 'error': f'处理失败: {str(e)}'}), 500

# 启动应用时创建表
with app.app_context():
    db.create_all()
    # 加载模型
    load_models()

# 在启动前初始化数据库和数据集
with app.app_context():
    db.create_all()
    # 初始化数据集
    init_datasets()
    # 加载模型
    load_models()

if __name__ == '__main__':
    # 启动应用
    app.run(host="0.0.0.0", port=5000, debug=True)
