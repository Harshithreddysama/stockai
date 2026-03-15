import os
from flask_mail import Mail, Message
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import logging

mail = Mail()
logger = logging.getLogger(__name__)


def send_signal_alert(app, user_email: str, user_name: str,
                      symbol: str, signal: str, current_price: float,
                      predicted_price: float, sentiment_label: str,
                      headlines: list):
    """Send a BUY/SELL/HOLD alert email to the user."""
    with app.app_context():
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(signal, "⚪")
        change = round((predicted_price - current_price) / current_price * 100, 2)
        sign   = "+" if change > 0 else ""
        headlines_text = "\n".join(f"  • {h}" for h in headlines[:3])

        msg = Message(
            subject=f"{emoji} StockAI Alert: {signal} signal for {symbol}",
            sender=os.getenv("MAIL_USERNAME"),
            recipients=[user_email]
        )
        msg.html = f"""
<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;max-width:600px;margin:auto;padding:20px;background:#f9f9f9">
  <div style="background:#0f172a;padding:24px;border-radius:12px 12px 0 0">
    <h1 style="color:#e2e8f0;margin:0;font-size:22px">📈 StockAI Signal Alert</h1>
  </div>
  <div style="background:#fff;padding:24px;border-radius:0 0 12px 12px;border:1px solid #e2e8f0">
    <p style="color:#64748b">Hello {user_name},</p>
    <p style="color:#1e293b">Our AI has detected a <strong>{signal}</strong> signal for <strong>{symbol}</strong>.</p>

    <div style="background:{"#dcfce7" if signal=="BUY" else "#fef2f2" if signal=="SELL" else "#fefce8"};
                border-radius:8px;padding:16px;margin:16px 0;
                border-left:4px solid {"#16a34a" if signal=="BUY" else "#dc2626" if signal=="SELL" else "#ca8a04"}">
      <p style="margin:0;font-size:28px;font-weight:bold;
                color:{"#16a34a" if signal=="BUY" else "#dc2626" if signal=="SELL" else "#ca8a04"}">
        {emoji} {signal}
      </p>
      <p style="margin:4px 0 0;color:#475569">
        Current: <strong>${current_price}</strong> &nbsp;→&nbsp;
        Predicted: <strong>${predicted_price}</strong>
        ({sign}{change}%)
      </p>
    </div>

    <h3 style="color:#1e293b;font-size:15px">📰 News Sentiment: {sentiment_label}</h3>
    <div style="background:#f8fafc;border-radius:6px;padding:12px">
      <p style="color:#64748b;font-size:13px;margin:0">Recent headlines analyzed:</p>
      {"".join(f'<p style="color:#475569;font-size:13px;margin:4px 0">• {h}</p>' for h in headlines[:3])}
    </div>

    <div style="margin-top:20px;padding:12px;background:#fef9c3;border-radius:6px">
      <p style="color:#854d0e;font-size:12px;margin:0">
        ⚠️ <strong>Disclaimer:</strong> This is an AI-generated signal for informational purposes only.
        It is NOT financial advice. Always do your own research before investing.
        Past predictions do not guarantee future results.
      </p>
    </div>

    <p style="margin-top:20px;text-align:center">
      <a href="{os.getenv('APP_URL','http://localhost:5000')}"
         style="background:#3b82f6;color:#fff;padding:10px 24px;border-radius:6px;
                text-decoration:none;font-weight:bold">
        View Full Analysis
      </a>
    </p>
    <p style="color:#94a3b8;font-size:12px;text-align:center;margin-top:16px">
      StockAI · {datetime.now().strftime('%B %d, %Y')} · Unsubscribe by updating your watchlist
    </p>
  </div>
</body>
</html>
"""
        try:
            mail.send(msg)
            logger.info(f"Alert sent to {user_email} for {symbol} ({signal})")
        except Exception as e:
            logger.error(f"Failed to send email to {user_email}: {e}")


def check_all_users_and_notify(app):
    """
    Scheduled job: run predictions for every user's watchlist
    and send alerts for BUY/SELL signals.
    """
    from utils.predict import get_prediction
    from utils.sentiment import get_news_sentiment, combine_signals
    from models.models import User, db

    with app.app_context():
        users = User.query.all()
        for user in users:
            if not user.watchlist:
                continue
            symbols = [s.strip().upper() for s in user.watchlist.split(',') if s.strip()]
            for symbol in symbols:
                try:
                    pred  = get_prediction(symbol)
                    senti = get_news_sentiment(symbol, pred.get("name", ""))
                    final = combine_signals(
                        pred["signal"], senti["label"],
                        senti["score"], pred["change_pct"]
                    )
                    if final in ("BUY", "SELL"):   # only alert on actionable signals
                        send_signal_alert(
                            app=app,
                            user_email=user.email,
                            user_name=user.name,
                            symbol=symbol,
                            signal=final,
                            current_price=pred["current_price"],
                            predicted_price=pred["predicted_price"],
                            sentiment_label=senti["label"],
                            headlines=senti["headlines"],
                        )
                except Exception as e:
                    logger.error(f"Notification error for {symbol}: {e}")


def start_scheduler(app):
    """Start the background scheduler (runs daily at 9 AM)."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=lambda: check_all_users_and_notify(app),
        trigger='cron',
        hour=9, minute=0,
        id='daily_alerts',
        replace_existing=True
    )
    scheduler.start()
    logger.info("Scheduler started — daily alerts at 09:00")
    return scheduler
