# config.py

from typing import Dict, List, Tuple # Removed OrderedDict as it's directly used from collections
from collections import OrderedDict 
import os

# --- Import KPI Definitions ---
try:
    from kpi_definitions import KPI_CONFIG
except ImportError:
    print("Warning (config.py): Could not import KPI_CONFIG from kpi_definitions.py.")
    KPI_CONFIG = {}

# --- General Settings ---
APP_TITLE: str = "Trading Mastery Hub"
RISK_FREE_RATE: float = 0.02
FORECAST_HORIZON: int = 30

# --- Database Configuration ---
DATABASE_TYPE: str = os.getenv("DATABASE_TYPE", "sqlite")
SQLITE_DB_FILE: str = os.getenv("SQLITE_DB_FILE", "trading_dashboard.db")
DATABASE_URL: str = ""
if DATABASE_TYPE == "sqlite": DATABASE_URL = f"sqlite:///{SQLITE_DB_FILE}"
elif DATABASE_TYPE == "postgresql":
    DB_USER = os.getenv("DB_USER", "user"); DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_HOST = os.getenv("DB_HOST", "localhost"); DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "trading_dashboard_pg")
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else: DATABASE_URL = f"sqlite:///{SQLITE_DB_FILE}" # Fallback

# --- Benchmark Configuration ---
DEFAULT_BENCHMARK_TICKER: str = "SPY"
AVAILABLE_BENCHMARKS: Dict[str, str] = {
    "S&P 500 (SPY)": "SPY", "Nasdaq 100 (QQQ)": "QQQ", "Dow Jones (DIA)": "DIA",
    "Russell 2000 (IWM)": "IWM", "Gold (GLD)": "GLD", "None": ""
}

# --- Conceptual Column Definitions (as before) ---
CONCEPTUAL_COLUMNS: Dict[str, str] = {
    "date": "Trade Date/Time (Primary)", "pnl": "Profit or Loss (PnL)", "symbol": "Trading Symbol/Ticker", 
    # ... (all other conceptual columns as previously defined) ...
    "strategy": "Strategy Name/Identifier", "trade_id": "Unique Trade ID", "entry_price": "Entry Price", 
    "exit_price": "Exit Price", "risk_pct": "Risk Percentage (%)", "notes": "Trade Notes/Comments", 
    "signal_confidence": "Signal Confidence Score", "duration_minutes": "Trade Duration (minutes)", 
    "entry_time_str": "Entry Time (Raw String, if separate)", "trade_month_str": "Trade Month (Raw String)", 
    "trade_day_str": "Trade Day of Week (Raw String)", "trade_plan_str": "Trade Plan/Setup", "bias_str": "Market Bias", 
    "tag_str": "Custom Tags", "time_frame_str": "Trading Time Frame", "direction_str": "Trade Direction (Long/Short)", 
    "trade_size_num": "Trade Size/Quantity", "r_r_csv_num": "Risk:Reward Ratio (from CSV)", "session_str": "Trading Session", 
    "market_conditions_str": "Market Conditions Description", "event_type_str": "Economic Event Type", 
    "events_details_str": "Economic Event Details", "market_sentiment_str": "Market Sentiment", 
    "psychological_factors_str": "Psychological Factors", "compliance_check_str": "Compliance Check Status", 
    "account_str": "Trading Account ID/Name", "initial_balance_num": "Account Initial Balance", 
    "current_balance_num": "Account Current Balance", "drawdown_value_csv": "Drawdown Value (from CSV)", 
    "trade_outcome_csv_str": "Trade Outcome (Win/Loss/BE from CSV)", "exit_type_csv_str": "Exit Type/Reason", 
    "loss_indicator_num": "Loss Indicator (Numeric)", "win_indicator_num": "Win Indicator (Numeric)", 
    "stop_distance_num": "Stop Loss Distance", "candle_count_num": "Candle Count for Trade Duration", 
    "cumulative_equity_csv": "Cumulative Equity (from CSV)", "absolute_daily_pnl_csv": "Absolute Daily PnL (from CSV)", 
    "error_exit_type_related_str": "Error Type (Exit Related)", "profit_value_csv": "Gross Profit Value (from CSV)", 
    "loss_value_csv": "Gross Loss Value (from CSV)", "duration_hrs_csv": "Trade Duration (hours, from CSV)", 
    "peak_value_csv": "Peak Excursion/Value (from CSV)", "take_profit_price": "Take Profit Price Level", 
    "stop_loss_price": "Stop Loss Price Level", "trade_result_screenshot": "Trade Result Screenshot URL/Path", 
    "event_release_time": "Event Release Time", "htf_key_level_focus": "HTF Key Level Focus", 
    "market_reaction_on_news": "Market Reaction on News Release", "planned_entry_price": "Planned Entry Price", 
    "planned_rrr": "Planned Risk:Reward Ratio", "planned_sl_price": "Planned Stop Loss Price", 
    "planned_tp_price": "Planned Take Profit Price", "pre_market_emotions": "Pre-Market Emotions", 
    "pre_market_notes": "Pre-Market Notes", "trade_plan_screenshot": "Trade Plan Screenshot URL/Path", 
    "actionable_steps_notes": "Actionable Steps (Post-Trade)", "ror_pct": "Return on Risk (%)", 
    "trade_performance_summary_text": "Trade Performance Summary (Text Block)", "multiplier_value": "Multiplier Value"
}
CONCEPTUAL_COLUMN_TYPES: Dict[str, str] = {key: "text" for key in CONCEPTUAL_COLUMNS} # Simplified for brevity, use full map
for k in ["pnl", "entry_price", "exit_price", "risk_pct", "signal_confidence", "duration_minutes", "trade_size_num", "r_r_csv_num", "initial_balance_num", "current_balance_num", "drawdown_value_csv", "loss_indicator_num", "win_indicator_num", "stop_distance_num", "candle_count_num", "cumulative_equity_csv", "absolute_daily_pnl_csv", "profit_value_csv", "loss_value_csv", "duration_hrs_csv", "peak_value_csv", "take_profit_price", "stop_loss_price", "planned_entry_price", "planned_rrr", "planned_sl_price", "planned_tp_price", "ror_pct", "multiplier_value"]: CONCEPTUAL_COLUMN_TYPES[k] = "numeric"
CONCEPTUAL_COLUMN_TYPES["date"] = "datetime"
CONCEPTUAL_COLUMN_SYNONYMS: Dict[str, List[str]] = {key: [] for key in CONCEPTUAL_COLUMNS} # Simplified, use full map
CRITICAL_CONCEPTUAL_COLUMNS: List[str] = ["date", "pnl"]
CONCEPTUAL_COLUMN_CATEGORIES: OrderedDict[str, List[str]] = OrderedDict([("Core", ["date", "pnl", "symbol"])]) # Simplified

# --- UI and Plotting Colors (as before) ---
COLORS: Dict[str, str] = {
    "royal_blue": "#1E88E5", "green": "#4CAF50", "red": "#F44336", "gray": "#808080", "orange": "#FFA500",
    "purple": "#8A2BE2", "dark_background": "#0E1117", "light_background": "#FFFFFF",
    "text_dark": "#FAFAFA", "text_light": "#212529", "text_muted_color": "#A0A2B3",
    "card_background_dark": "#1C1E25", "card_border_dark": "#3E4049",
    "card_background_light": "#F0F2F6", "card_border_light": "#DEE2E6"
}
# --- KPI Groupings & Order (as before) ---
KPI_GROUPS_OVERVIEW: Dict[str, List[str]] = {"Overall": ["total_pnl", "total_trades"]} # Simplified
DEFAULT_KPI_DISPLAY_ORDER: List[str] = ["total_pnl", "total_trades"] # Simplified
KPI_GROUPS_RISK_DURATION: Dict[str, List[str]] = {"Drawdown": ["max_drawdown_abs", "max_drawdown_pct"]} # Simplified

# --- Plotting Themes (as before) ---
PLOTLY_THEME_DARK: str = "plotly_dark"; PLOTLY_THEME_LIGHT: str = "plotly_white" # etc.

# --- Analysis Settings (as before) ---
BOOTSTRAP_ITERATIONS: int = 1000; CONFIDENCE_LEVEL: float = 0.95
DISTRIBUTIONS_TO_FIT: List[str] = ['norm', 't']; MARKOV_MAX_LAG: int = 1

# --- Logging Configuration (as before) ---
LOG_FILE: str = "logs/trading_dashboard_app.log"
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"

# --- EXPECTED_COLUMNS (as before) ---
EXPECTED_COLUMNS: Dict[str, str] = {key: key for key in CONCEPTUAL_COLUMNS.keys()}

# --- NEW: Email Configuration (Placeholders) ---
EMAIL_SENDER_ADDRESS: str = os.getenv("EMAIL_SENDER_ADDRESS", "noreply@yourtradinghub.com")
PASSWORD_RESET_SUBJECT: str = f"Password Reset Request - {APP_TITLE}"
PASSWORD_RESET_LINK_EXPIRY_HOURS: int = 1 # Link expires in 1 hour

# --- Load environment variables from .env ---
from dotenv import load_dotenv
load_dotenv()

# Re-evaluate DATABASE_URL (as before)
if DATABASE_TYPE == "postgresql":
    DB_USER = os.getenv("DB_USER", "user"); DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_HOST = os.getenv("DB_HOST", "localhost"); DB_PORT = os.getenv("DB_PORT", "5432"); DB_NAME = os.getenv("DB_NAME", "trading_dashboard_pg")
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
elif DATABASE_TYPE == "sqlite":
    SQLITE_DB_FILE = os.getenv("SQLITE_DB_FILE", "trading_dashboard.db"); DATABASE_URL = f"sqlite:///{SQLITE_DB_FILE}"
if not DATABASE_URL:
    print("CRITICAL WARNING (config.py): DATABASE_URL not set."); DATABASE_URL = "sqlite:///trading_dashboard_fallback.db"

# --- NEW: Application Base URL (for constructing reset links) ---
# This should be the URL where your Streamlit app is accessible.
# For local development: "http://localhost:8501"
# For Streamlit Cloud: "https://your-app-name.streamlit.app"
# IMPORTANT: Set this via an environment variable in production/deployment.
APP_BASE_URL: str = os.getenv("APP_BASE_URL", "http://localhost:8501")
