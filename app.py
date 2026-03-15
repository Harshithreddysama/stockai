import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from flask_cors import CORS

from models.models import db, User, Prediction, Portfolio
from utils.predict import get_prediction, fetch_stock_data
from utils.sentiment import get_news_sentiment, combine_signals
from utils.notifications import mail, start_scheduler

# ── App factory ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY']               = os.getenv('SECRET_KEY', 'dev-secret-key-change-me')
app.config['SQLALCHEMY_DATABASE_URI']  = os.getenv('DATABASE_URL', 'sqlite:///stocks.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Mail
app.config['MAIL_SERVER']   = 'smtp.gmail.com'
app.config['MAIL_PORT']     = 587
app.config['MAIL_USE_TLS']  = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

# Extensions
db.init_app(app)
mail.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ── Auth routes ───────────────────────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        data  = request.get_json() or request.form
        email = data.get('email', '').strip().lower()
        name  = data.get('name', '').strip()
        pwd   = data.get('password', '')
        if User.query.filter_by(email=email).first():
            if request.is_json:
                return jsonify({'error': 'Email already registered'}), 400
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        hashed = bcrypt.generate_password_hash(pwd).decode('utf-8')
        user = User(email=email, name=name, password=hashed)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        if request.is_json:
            return jsonify({'message': 'Registered successfully', 'redirect': '/dashboard'})
        return redirect(url_for('dashboard'))
    return render_template('auth.html', mode='register')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        data  = request.get_json() or request.form
        email = data.get('email', '').strip().lower()
        pwd   = data.get('password', '')
        user  = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, pwd):
            login_user(user, remember=True)
            if request.is_json:
                return jsonify({'message': 'Logged in', 'redirect': '/dashboard'})
            return redirect(url_for('dashboard'))
        if request.is_json:
            return jsonify({'error': 'Invalid email or password'}), 401
        flash('Invalid credentials', 'danger')
    return render_template('auth.html', mode='login')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/dashboard')
@login_required
def dashboard():
    watchlist = [s.strip().upper() for s in current_user.watchlist.split(',') if s.strip()]
    recent    = Prediction.query.filter_by(user_id=current_user.id)\
                                .order_by(Prediction.date.desc()).limit(10).all()
    portfolio = Portfolio.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html',
                           user=current_user,
                           watchlist=watchlist,
                           recent_predictions=recent,
                           portfolio=portfolio)


# ── API endpoints ──────────────────────────────────────────────────────────────
@app.route('/api/analyze', methods=['POST'])
@login_required
def analyze():
    """Run full ML + sentiment analysis on a stock symbol."""
    data   = request.get_json()
    symbol = data.get('symbol', '').strip().upper()
    if not symbol:
        return jsonify({'error': 'Symbol required'}), 400
    try:
        pred   = get_prediction(symbol)
        senti  = get_news_sentiment(symbol, pred.get('name', ''))
        signal = combine_signals(pred['signal'], senti['label'],
                                 senti['score'], pred['change_pct'])

        # Persist to DB
        record = Prediction(
            user_id         = current_user.id,
            stock_symbol    = symbol,
            current_price   = pred['current_price'],
            predicted_price = pred['predicted_price'],
            sentiment_score = senti['score'],
            sentiment_label = senti['label'],
            signal          = signal,
            news_headlines  = json.dumps(senti['headlines'])
        )
        db.session.add(record)
        db.session.commit()

        return jsonify({
            **pred,
            'sentiment':    senti,
            'final_signal': signal,
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Analysis error for {symbol}: {e}")
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500


@app.route('/api/watchlist', methods=['POST'])
@login_required
def update_watchlist():
    data    = request.get_json()
    symbols = data.get('symbols', [])
    current_user.watchlist = ','.join(s.upper() for s in symbols)
    db.session.commit()
    return jsonify({'message': 'Watchlist updated', 'watchlist': symbols})


@app.route('/api/portfolio', methods=['GET'])
@login_required
def get_portfolio():
    items = Portfolio.query.filter_by(user_id=current_user.id).all()
    result = []
    for item in items:
        try:
            current = get_prediction(item.stock_symbol)['current_price']
        except Exception:
            current = item.buy_price
        pnl = (current - item.buy_price) * item.quantity
        result.append({
            'id':           item.id,
            'symbol':       item.stock_symbol,
            'buy_price':    item.buy_price,
            'current_price': current,
            'quantity':     item.quantity,
            'pnl':          round(pnl, 2),
            'pnl_pct':      round((current - item.buy_price) / item.buy_price * 100, 2),
            'buy_date':     item.buy_date.strftime('%Y-%m-%d'),
        })
    return jsonify(result)


@app.route('/api/portfolio', methods=['POST'])
@login_required
def add_portfolio():
    data = request.get_json()
    item = Portfolio(
        user_id      = current_user.id,
        stock_symbol = data['symbol'].upper(),
        buy_price    = float(data['buy_price']),
        quantity     = float(data['quantity']),
        notes        = data.get('notes', '')
    )
    db.session.add(item)
    db.session.commit()
    return jsonify({'message': f"{data['symbol']} added to portfolio"})


@app.route('/api/portfolio/<int:item_id>', methods=['DELETE'])
@login_required
def delete_portfolio(item_id):
    item = Portfolio.query.filter_by(id=item_id, user_id=current_user.id).first_or_404()
    db.session.delete(item)
    db.session.commit()
    return jsonify({'message': 'Removed from portfolio'})


@app.route('/api/history')
@login_required
def prediction_history():
    records = Prediction.query.filter_by(user_id=current_user.id)\
                              .order_by(Prediction.date.desc()).limit(20).all()
    return jsonify([{
        'symbol':          r.stock_symbol,
        'date':            r.date.strftime('%Y-%m-%d %H:%M'),
        'current_price':   r.current_price,
        'predicted_price': r.predicted_price,
        'signal':          r.signal,
        'sentiment':       r.sentiment_label,
    } for r in records])


# ── Bootstrap ──────────────────────────────────────────────────────────────────
with app.app_context():
    db.create_all()

if os.getenv('FLASK_ENV') != 'development':
    start_scheduler(app)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
