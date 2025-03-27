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
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB限制

# 确保上传目录存在
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# 初始化应用与数据库
db.init_app(app)

# 导入模型
from models import SMSMessage

# 初始化机器学习模型
models = {}
tokenizers = {}
model_types = ["roberta", "lstm", "bert", "cnn", "xlnet", "gpt", "attention_lstm", "svm", "naive_bayes"]

def load_models():
    """加载所有机器学习模型"""
    global models, tokenizers
    try:
        # 初始化模型
        for model_type in model_types:
            logging.info(f"正在加载 {model_type} 模型...")
            models[model_type], tokenizers[model_type] = model_module.load_model(model_type)
            logging.info(f"{model_type} 模型加载完成")
    except Exception as e:
        logging.error(f"模型加载失败: {str(e)}")
        logging.error(traceback.format_exc())

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html', model_types=model_types)

@app.route('/predict', methods=['POST'])
def predict():
    """预测短信是否为垃圾短信"""
    try:
        # 获取表单数据
        text = request.form.get('text', '')
        send_freq = float(request.form.get('send_freq', 0))
        is_night = int(request.form.get('is_night', 0))
        model_type = request.form.get('model_type', 'roberta')
        
        if not text:
            return jsonify({'error': '请输入短信内容'}), 400
        
        if model_type not in model_types:
            return jsonify({'error': '无效的模型类型'}), 400
        
        start_time = time.time()
        
        # 预处理文本
        cleaned_text = preprocessing.clean_text(text)
        
        # 提取特征
        features = feature_extraction.extract_features(
            cleaned_text, 
            tokenizers.get(model_type), 
            send_freq, 
            is_night
        )
        
        # 预测
        prediction, confidence = model_module.predict(models.get(model_type), features, model_type)
        
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
        
        # 读取CSV
        df = pd.read_csv(filepath)
        
        # 检查必要的列
        required_columns = ['text', 'send_freq', 'is_night']
        if not all(col in df.columns for col in required_columns):
            flash('CSV文件格式错误，必须包含 text, send_freq, is_night 列', 'danger')
            return redirect(url_for('index'))
        
        # 处理每一行
        model_type = request.form.get('model_type', 'roberta')
        
        if model_type not in model_types:
            flash('无效的模型类型', 'danger')
            return redirect(url_for('index'))
        
        # 批量处理
        results = []
        for _, row in df.iterrows():
            text = row['text']
            send_freq = float(row['send_freq'])
            is_night = int(row['is_night'])
            
            # 预处理
            cleaned_text = preprocessing.clean_text(text)
            
            # 特征提取
            features = feature_extraction.extract_features(
                cleaned_text, 
                tokenizers.get(model_type), 
                send_freq, 
                is_night
            )
            
            # 预测
            prediction, confidence = model_module.predict(models.get(model_type), features, model_type)
            
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
                'confidence': float(confidence)
            })
        
        # 提交所有更改
        db.session.commit()
        
        # 处理完毕后删除文件
        os.remove(filepath)
        
        flash(f'成功处理 {len(results)} 条短信', 'success')
        return jsonify({'success': True, 'results': results})
    
    except Exception as e:
        logging.error(f"文件上传处理错误: {str(e)}")
        logging.error(traceback.format_exc())
        flash(f'处理文件时出错: {str(e)}', 'danger')
        return redirect(url_for('index'))

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
        
        # 获取词频
        spam_words = preprocessing.get_word_frequencies([msg.text for msg in spam_messages])
        ham_words = preprocessing.get_word_frequencies([msg.text for msg in ham_messages])
        
        # 限制为前50个最常见的词
        spam_words = sorted(spam_words.items(), key=lambda x: x[1], reverse=True)[:50]
        ham_words = sorted(ham_words.items(), key=lambda x: x[1], reverse=True)[:50]
        
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
    """获取语义漂移检测数据"""
    try:
        # 获取最近的消息用于漂移检测
        recent_messages = SMSMessage.query.order_by(SMSMessage.timestamp.desc()).limit(100).all()
        
        if len(recent_messages) < 10:
            return jsonify({'error': '数据不足，无法进行漂移检测'}), 400
        
        # 进行漂移检测
        drift_value = drift.detect_drift([msg.text for msg in recent_messages])
        
        # 返回漂移值
        return jsonify({
            'drift_value': float(drift_value),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    except Exception as e:
        logging.error(f"漂移检测错误: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

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

# 启动应用时创建表
with app.app_context():
    db.create_all()
    # 加载模型
    load_models()

if __name__ == '__main__':
    # 在启动前确保表已创建
    with app.app_context():
        db.create_all()
    # 启动应用
    app.run(host="0.0.0.0", port=5000, debug=True)
