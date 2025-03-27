from app import db
from models import SMSMessage
from datetime import datetime

def add_sms(text, send_freq, is_night, prediction, confidence, model_type):
    """添加一条短信记录到数据库"""
    sms = SMSMessage(
        text=text,
        send_freq=send_freq,
        is_night=is_night,
        prediction=prediction,
        confidence=confidence,
        model_type=model_type,
        timestamp=datetime.now()
    )
    db.session.add(sms)
    db.session.commit()
    return sms

def add_sms_batch(sms_list):
    """批量添加短信记录到数据库"""
    for sms_data in sms_list:
        sms = SMSMessage(
            text=sms_data['text'],
            send_freq=sms_data['send_freq'],
            is_night=sms_data['is_night'],
            prediction=sms_data['prediction'],
            confidence=sms_data['confidence'],
            model_type=sms_data['model_type'],
            timestamp=datetime.now()
        )
        db.session.add(sms)
    db.session.commit()

def get_all_sms():
    """获取所有短信记录"""
    return SMSMessage.query.all()

def get_recent_sms(limit=100):
    """获取最近的n条短信记录"""
    return SMSMessage.query.order_by(SMSMessage.timestamp.desc()).limit(limit).all()

def get_spam_sms():
    """获取所有垃圾短信记录"""
    return SMSMessage.query.filter_by(prediction='垃圾短信').all()

def get_ham_sms():
    """获取所有正常短信记录"""
    return SMSMessage.query.filter_by(prediction='正常短信').all()

def get_sms_by_model(model_type):
    """获取特定模型的短信记录"""
    return SMSMessage.query.filter_by(model_type=model_type).all()
