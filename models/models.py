from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String(150), unique=True, nullable=False)
    password      = db.Column(db.String(200), nullable=False)
    name          = db.Column(db.String(100), nullable=False)
    watchlist     = db.Column(db.Text, default='')   # comma-separated symbols
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    predictions   = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    stock_symbol    = db.Column(db.String(10), nullable=False)
    date            = db.Column(db.DateTime, default=datetime.utcnow)
    current_price   = db.Column(db.Float)
    predicted_price = db.Column(db.Float)
    actual_price    = db.Column(db.Float, nullable=True)
    sentiment_score = db.Column(db.Float, default=0.0)
    sentiment_label = db.Column(db.String(20), default='NEUTRAL')
    signal          = db.Column(db.String(10))   # BUY / SELL / HOLD
    news_headlines  = db.Column(db.Text, default='')

class Portfolio(db.Model):
    __tablename__ = 'portfolio'
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    stock_symbol = db.Column(db.String(10), nullable=False)
    buy_price    = db.Column(db.Float, nullable=False)
    quantity     = db.Column(db.Float, nullable=False)
    buy_date     = db.Column(db.DateTime, default=datetime.utcnow)
    notes        = db.Column(db.Text, default='')
