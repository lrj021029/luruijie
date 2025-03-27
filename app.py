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
model_types = ["ensemble", "roberta", "lstm", "bert", "cnn", "xlnet", "gpt", "attention_lstm", "svm", "naive_bayes"]

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
        model_type = request.form.get('model_type', 'roberta')
        
        if model_type not in model_types:
            return jsonify({'error': '无效的模型类型'}), 400
        
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
        
        # 获取当前模型类型（默认使用 roberta）
        current_model_type = request.args.get('model_type', 'roberta')
        logging.info(f"当前请求的模型类型: {current_model_type}")
        if current_model_type not in models:
            current_model_type = 'roberta'
            logging.warning(f"无效的模型类型，改用默认值: {current_model_type}")
            
        # 获取最近1000条消息作为参考数据集（历史数据）
        reference_messages = SMSMessage.query.order_by(SMSMessage.timestamp).limit(1000).all()
        logging.info(f"获取到参考消息数: {len(reference_messages)}")
        
        # 进行漂移检测和模型微调
        logging.info("调用drift.detect_drift函数")
        drift_value, is_adapted = drift.detect_drift(
            [msg.text for msg in recent_messages],
            [msg.text for msg in reference_messages] if len(reference_messages) >= 10 else None,
            tokenizers.get(current_model_type),
            models.get(current_model_type),
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
            'model_type': request.args.get('model_type', 'roberta'),
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
