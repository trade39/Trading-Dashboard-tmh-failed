# config.py

from typing import Dict, List, Tuple # Removed OrderedDict as it's directly used from collections
from collections import OrderedDict
import os

# --- Import KPI Definitions ---
try:
    from kpi_definitions import KPI_CONFIG # Assuming kpi_definitions.py is in the same directory or PYTHONPATH
except ImportError:
    print("Warning (config.py): Could not import KPI_CONFIG from kpi_definitions.py. Using empty KPI_CONFIG.")
    KPI_CONFIG = {}

# --- General Settings ---
APP_TITLE: str = "Trading Mastery Hub"
RISK_FREE_RATE: float = 0.02
FORECAST_HORIZON: int = 30

# --- Asset Paths ---
# Ensure these paths are relative to the root of your Streamlit application (where app.py is)
# Example: if you have an 'assets' folder in your project root:
ASSETS_DIR = "assets" # Define a base assets directory
LOGO_PATH_FOR_BROWSER_TAB: str = os.path.join(ASSETS_DIR, "Trading_Mastery_Hub_600x600.png")
LOGO_PATH_SIDEBAR: str = os.path.join(ASSETS_DIR, "Trading_Mastery_Hub_600x600.png")


# --- Database Configuration ---
DATABASE_TYPE: str = os.getenv("DATABASE_TYPE", "sqlite")
SQLITE_DB_FILE: str = os.getenv("SQLITE_DB_FILE", "trading_dashboard.db") # Default filename
DATABASE_URL: str = ""

# Construct DATABASE_URL based on DATABASE_TYPE
# For SQLite, ensure the path is correct relative to where the app runs.
# If SQLITE_DB_FILE is just a name, it will be in the current working directory.
# For robustness, especially with services in subdirectories, an absolute path or path relative to project root is better.
# However, for Streamlit Cloud, a simple filename might be placed in the root.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # Gets directory of config.py

if DATABASE_TYPE == "sqlite":
    # If SQLITE_DB_FILE is just a name, make it relative to the project root
    if not os.path.isabs(SQLITE_DB_FILE) and '/' not in SQLITE_DB_FILE and '\\' not in SQLITE_DB_FILE:
        db_path = os.path.join(PROJECT_ROOT, SQLITE_DB_FILE)
    else: # Assumes SQLITE_DB_FILE is already a full path or correctly relative
        db_path = SQLITE_DB_FILE
    DATABASE_URL = f"sqlite:///{db_path}"
elif DATABASE_TYPE == "postgresql":
    DB_USER = os.getenv("DB_USER", "user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "trading_dashboard_pg")
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    print(f"Warning (config.py): Unknown DATABASE_TYPE '{DATABASE_TYPE}'. Defaulting to SQLite file 'trading_dashboard_fallback.db' in project root.")
    db_path_fallback = os.path.join(PROJECT_ROOT, "trading_dashboard_fallback.db")
    DATABASE_URL = f"sqlite:///{db_path_fallback}"


# --- Benchmark Configuration ---
DEFAULT_BENCHMARK_TICKER: str = "SPY" # Default benchmark
AVAILABLE_BENCHMARKS: Dict[str, str] = {
    "S&P 500 (SPY)": "SPY",
    "Nasdaq 100 (QQQ)": "QQQ",
    "Dow Jones (DIA)": "DIA",
    "Russell 2000 (IWM)": "IWM",
    "Gold (GLD)": "GLD",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "None": "" # Option for no benchmark
}

# --- Conceptual Column Definitions ---
# These are the standardized internal names the application will use.
# The user maps their CSV columns to these conceptual names.
CONCEPTUAL_COLUMNS: Dict[str, str] = {
    "date": "Trade Date/Time (Primary)",
    "pnl": "Profit or Loss (PnL)",
    "symbol": "Trading Symbol/Ticker",
    "strategy": "Strategy Name/Identifier",
    "trade_id": "Unique Trade ID",
    "entry_price": "Entry Price",
    "exit_price": "Exit Price",
    "risk_pct": "Risk Percentage (%)", # e.g., 1 for 1%
    "notes": "Trade Notes/Comments",
    "signal_confidence": "Signal Confidence Score", # e.g., 1-10 or 0-1
    "duration_minutes": "Trade Duration (minutes)",
    "entry_time_str": "Entry Time (Raw String, if separate from date)",
    "trade_month_str": "Trade Month (Raw String, e.g., 'January', 'Jan')",
    "trade_day_of_week_str": "Trade Day of Week (Raw String, e.g., 'Monday', 'Mon')",
    "trade_plan_str": "Trade Plan/Setup Description",
    "bias_str": "Market Bias (e.g., Bullish, Bearish, Neutral)",
    "tag_str": "Custom Tags (comma-separated)",
    "time_frame_str": "Trading Time Frame (e.g., M5, H1, Daily)",
    "direction_str": "Trade Direction (e.g., Long, Short, Buy, Sell)",
    "trade_size_num": "Trade Size/Quantity/Volume/Contracts",
    "r_r_csv_num": "Risk:Reward Ratio (from CSV, if pre-calculated)",
    "session_str": "Trading Session (e.g., London, New York, Asian)",
    "market_conditions_str": "Market Conditions Description (e.g., Trending, Ranging)",
    "event_type_str": "Economic Event Type (e.g., NFP, CPI, FOMC)",
    "events_details_str": "Economic Event Details/Impact",
    "market_sentiment_str": "Market Sentiment (e.g., Fear, Greed, Neutral)",
    "psychological_factors_str": "Psychological Factors (e.g., FOMO, Overconfidence, Discipline)",
    "compliance_check_str": "Compliance Check Status (e.g., Passed, Failed)",
    "account_str": "Trading Account ID/Name",
    "initial_balance_num": "Account Initial Balance (for this period/trade set)",
    "current_balance_num": "Account Current Balance (after trades)",
    "drawdown_value_csv": "Drawdown Value (from CSV, if pre-calculated)",
    "trade_outcome_csv_str": "Trade Outcome (Win/Loss/BE from CSV, if pre-defined)",
    "exit_type_csv_str": "Exit Type/Reason (e.g., Target Hit, Stop Loss, Manual Close)",
    "loss_indicator_num": "Loss Indicator (Numeric, e.g., 1 for loss, 0 for not)",
    "win_indicator_num": "Win Indicator (Numeric, e.g., 1 for win, 0 for not)",
    "stop_distance_num": "Stop Loss Distance (Points/Pips/Price)",
    "candle_count_num": "Candle Count for Trade Duration",
    "cumulative_equity_csv": "Cumulative Equity (from CSV, if pre-calculated)",
    "absolute_daily_pnl_csv": "Absolute Daily PnL (from CSV, if pre-calculated)",
    "error_exit_type_related_str": "Error Type (Exit Related, e.g., Slippage, Platform Error)",
    "profit_value_csv": "Gross Profit Value (from CSV, if separate from PnL)",
    "loss_value_csv": "Gross Loss Value (from CSV, if separate from PnL)",
    "duration_hrs_csv": "Trade Duration (hours, from CSV)",
    "peak_value_csv": "Peak Excursion/Value during trade (from CSV)",
    "take_profit_price": "Take Profit Price Level",
    "stop_loss_price": "Stop Loss Price Level",
    "trade_result_screenshot": "Trade Result Screenshot URL/Path",
    "event_release_time": "Event Release Time",
    "htf_key_level_focus": "HTF Key Level Focus",
    "market_reaction_on_news": "Market Reaction on News Release",
    "planned_entry_price": "Planned Entry Price",
    "planned_rrr": "Planned Risk:Reward Ratio",
    "planned_sl_price": "Planned Stop Loss Price",
    "planned_tp_price": "Planned Take Profit Price",
    "pre_market_emotions": "Pre-Market Emotions",
    "pre_market_notes": "Pre-Market Notes",
    "trade_plan_screenshot": "Trade Plan Screenshot URL/Path",
    "actionable_steps_notes": "Actionable Steps (Post-Trade Review)",
    "ror_pct": "Return on Risk (%)", # e.g., PnL / (Risked Amount) * 100
    "trade_performance_summary_text": "Trade Performance Summary (Text Block)",
    "multiplier_value": "Multiplier Value (e.g., points to currency)"
}

# Define expected data types for each conceptual column for validation and conversion
CONCEPTUAL_COLUMN_TYPES: Dict[str, str] = {
    "date": "datetime",
    "pnl": "numeric",
    "symbol": "text",
    "strategy": "text",
    "trade_id": "text", # Usually alphanumeric
    "entry_price": "numeric",
    "exit_price": "numeric",
    "risk_pct": "numeric", # Stored as percentage value, e.g., 1.0 for 1%
    "notes": "text",
    "signal_confidence": "numeric", # Could be float or int
    "duration_minutes": "numeric",
    "entry_time_str": "text", # To be combined with date
    "trade_month_str": "text",
    "trade_day_of_week_str": "text",
    "trade_plan_str": "text",
    "bias_str": "text",
    "tag_str": "text",
    "time_frame_str": "text",
    "direction_str": "text",
    "trade_size_num": "numeric",
    "r_r_csv_num": "numeric",
    "session_str": "text",
    "market_conditions_str": "text",
    "event_type_str": "text",
    "events_details_str": "text",
    "market_sentiment_str": "text",
    "psychological_factors_str": "text",
    "compliance_check_str": "text", # Could be boolean-like text
    "account_str": "text",
    "initial_balance_num": "numeric",
    "current_balance_num": "numeric",
    "drawdown_value_csv": "numeric",
    "trade_outcome_csv_str": "text",
    "exit_type_csv_str": "text",
    "loss_indicator_num": "numeric", # Boolean-like (0 or 1)
    "win_indicator_num": "numeric",  # Boolean-like (0 or 1)
    "stop_distance_num": "numeric",
    "candle_count_num": "numeric",
    "cumulative_equity_csv": "numeric",
    "absolute_daily_pnl_csv": "numeric",
    "error_exit_type_related_str": "text",
    "profit_value_csv": "numeric",
    "loss_value_csv": "numeric",
    "duration_hrs_csv": "numeric",
    "peak_value_csv": "numeric",
    "take_profit_price": "numeric",
    "stop_loss_price": "numeric",
    "trade_result_screenshot": "text", # URL or path
    "event_release_time": "text", # Could be datetime if consistent
    "htf_key_level_focus": "text",
    "market_reaction_on_news": "text",
    "planned_entry_price": "numeric",
    "planned_rrr": "numeric",
    "planned_sl_price": "numeric",
    "planned_tp_price": "numeric",
    "pre_market_emotions": "text",
    "pre_market_notes": "text",
    "trade_plan_screenshot": "text", # URL or path
    "actionable_steps_notes": "text",
    "ror_pct": "numeric", # Stored as percentage value, e.g., 150 for 150%
    "trade_performance_summary_text": "text",
    "multiplier_value": "numeric"
}

# Synonyms for column headers to aid automatic mapping
CONCEPTUAL_COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "date": ["trade date", "timestamp", "execution time", "time", "datetime", "date_time"],
    "pnl": ["profit", "loss", "net pnl", "profit_loss", "realized_pnl", "p&l", "net profit"],
    "symbol": ["ticker", "instrument", "market", "asset", "security"],
    "strategy": ["strategy_name", "model_name", "algo_id", "trade_model", "setup"],
    "trade_id": ["order_id", "execution_id", "transaction_id", "deal_id"],
    "entry_price": ["open_price", "buy_price", "sell_price", "avg_entry_price", "entry"],
    "exit_price": ["close_price", "avg_exit_price", "exit"],
    "risk_pct": ["risk_percentage", "risk %", "risk_on_trade_%"],
    "notes": ["comments", "description", "trade_rationale", "lesson_learned", "journal_entry"],
    "duration_minutes": ["trade_duration_min", "holding_period_min", "duration_mins", "duration (min)"],
    "trade_size_num": ["quantity", "volume", "contracts", "shares", "lots", "position_size", "size"],
    "r_r_csv_num": ["r:r", "rr_ratio", "risk_reward", "reward_risk", "r/r"],
    "direction_str": ["direction", "side", "long_short", "trade_direction", "position_type", "type"],
    "account_str": ["account_id", "account_name", "portfolio_id", "account"],
    "entry_time_str": ["entry time", "time_entry"],
    "trade_day_of_week_str": ["day_of_week", "weekday", "trade_dow"],
    "trade_month_str": ["month", "trade_month"],
    "market_conditions_str": ["market_condition", "market_type", "market_state"],
    "session_str": ["trading_session", "market_session"],
    "exit_type_csv_str": ["exit_reason", "close_reason", "exit_type"],
    # Add more synonyms as identified from various broker reports
}

# Columns that are absolutely essential for the dashboard's core functionality
CRITICAL_CONCEPTUAL_COLUMNS: List[str] = ["date", "pnl"] # Symbol could also be critical

# Define categories for the column mapper UI to group conceptual columns logically
CONCEPTUAL_COLUMN_CATEGORIES: OrderedDict[str, List[str]] = OrderedDict([
    ("Critical Trade Info", ["date", "pnl", "symbol", "trade_id"]),
    ("Trade Execution Details", ["entry_price", "exit_price", "trade_size_num", "direction_str", "entry_time_str", "multiplier_value"]),
    ("Performance & Risk (from CSV)", ["r_r_csv_num", "risk_pct", "ror_pct", "drawdown_value_csv", "cumulative_equity_csv", "profit_value_csv", "loss_value_csv"]),
    ("Strategy & Context", ["strategy", "trade_plan_str", "time_frame_str", "session_str", "bias_str", "market_conditions_str", "market_sentiment_str"]),
    ("Trade Management", ["take_profit_price", "stop_loss_price", "stop_distance_num", "exit_type_csv_str"]),
    ("Descriptive & Qualitative", ["notes", "tag_str", "psychological_factors_str", "actionable_steps_notes", "trade_performance_summary_text"]),
    ("Indicators & Signals (Optional)", ["signal_confidence", "loss_indicator_num", "win_indicator_num"]),
    ("Temporal & Duration (Optional)", ["duration_minutes", "duration_hrs_csv", "candle_count_num", "trade_day_of_week_str", "trade_month_str"]),
    ("Account & Balance (Optional)", ["account_str", "initial_balance_num", "current_balance_num"]),
    ("Events & Compliance (Optional)", ["event_type_str", "events_details_str", "event_release_time", "market_reaction_on_news", "compliance_check_str"]),
    ("Planning & Screenshots (Optional)", ["planned_entry_price", "planned_sl_price", "planned_tp_price", "planned_rrr", "pre_market_emotions", "pre_market_notes", "trade_plan_screenshot", "trade_result_screenshot"]),
    ("Other Numeric/Categorical (Optional)", ["absolute_daily_pnl_csv", "error_exit_type_related_str", "htf_key_level_focus", "trade_outcome_csv_str", "peak_value_csv"])
])


# --- UI and Plotting Colors ---
COLORS: Dict[str, str] = {
    "royal_blue": "#1E88E5", "green": "#4CAF50", "red": "#F44336", "gray": "#808080", "orange": "#FFA500",
    "purple": "#8A2BE2", "dark_background": "#0E1117", "light_background": "#FFFFFF",
    "text_dark": "#FAFAFA", "text_light": "#212529", "text_muted_color": "#A0A2B3",
    "card_background_dark": "#1C1E25", "card_border_dark": "#3E4049",
    "card_background_light": "#F0F2F6", "card_border_light": "#DEE2E6"
}
PLOTLY_THEME_DARK: str = "plotly_dark"
PLOTLY_THEME_LIGHT: str = "plotly_white"
PLOT_BG_COLOR_DARK = '#0E1117'
PLOT_PAPER_BG_COLOR_DARK = '#0E1117'
PLOT_FONT_COLOR_DARK = '#FAFAFA'
PLOT_BG_COLOR_LIGHT = '#FFFFFF'
PLOT_PAPER_BG_COLOR_LIGHT = '#FFFFFF'
PLOT_FONT_COLOR_LIGHT = '#212529'
PLOT_LINE_COLOR = COLORS.get('royal_blue')
PLOT_MARKER_PROFIT_COLOR = COLORS.get('green')
PLOT_MARKER_LOSS_COLOR = COLORS.get('red')
PLOT_BENCHMARK_LINE_COLOR = COLORS.get('orange')


# --- KPI Groupings & Order ---
KPI_GROUPS_OVERVIEW: OrderedDict[str, List[str]] = OrderedDict([
    ("Overall Performance", [
        "total_pnl", "total_trades", "avg_trade_pnl", "avg_daily_pnl"
    ]),
    ("Win/Loss Analysis", [
        "win_rate", "loss_rate", "profit_factor", "win_loss_ratio",
        "avg_win", "avg_loss", "max_win_streak", "max_loss_streak"
    ]),
    ("Risk-Adjusted Returns", [
        "sharpe_ratio", "sortino_ratio", "calmar_ratio"
    ]),
    ("Risk Metrics (Drawdown & VaR)", [
        "max_drawdown_abs", "max_drawdown_pct",
        "var_95_loss", "cvar_95_loss", "var_99_loss", "cvar_99_loss"
    ]),
    ("Distribution & Volatility", [
        "pnl_skewness", "pnl_kurtosis", "annual_volatility", "expected_annual_return"
    ]),
    ("Benchmark Comparison", [
        "benchmark_total_return", "alpha", "beta",
        "benchmark_correlation", "tracking_error", "information_ratio"
    ])
])

DEFAULT_KPI_DISPLAY_ORDER: List[str] = [
    "total_pnl", "total_trades", "win_rate", "profit_factor",
    "sharpe_ratio", "max_drawdown_pct", "avg_trade_pnl", "alpha"
]

KPI_GROUPS_RISK_DURATION: OrderedDict[str, List[str]] = OrderedDict([
    ("Key Drawdown Metrics", [
        "max_drawdown_abs", "max_drawdown_pct", "calmar_ratio"
    ]),
    ("Value at Risk (VaR & CVaR)", [
        "var_95_loss", "cvar_95_loss", "var_99_loss", "cvar_99_loss"
    ]),
    ("Market Risk & Relative Performance", [
        "alpha", "beta", "benchmark_correlation", "tracking_error", "information_ratio"
    ]),
    ("Return Volatility & Distribution", [
        "annual_volatility", "expected_annual_return", "pnl_skewness", "pnl_kurtosis"
    ])
])


# --- Analysis Settings ---
BOOTSTRAP_ITERATIONS: int = 1000
CONFIDENCE_LEVEL: float = 0.95
DISTRIBUTIONS_TO_FIT: List[str] = ['norm', 't', 'laplace', 'cauchy', 'skewnorm']
MARKOV_MAX_LAG: int = 1

# --- Logging Configuration ---
LOG_DIR = os.path.join(PROJECT_ROOT, "logs") # Place logs in a 'logs' subdir of project root
LOG_FILE: str = os.path.join(LOG_DIR, "trading_dashboard_app.log")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"

# --- EXPECTED_COLUMNS ---
EXPECTED_COLUMNS: Dict[str, str] = {key: key for key in CONCEPTUAL_COLUMNS.keys()}

# --- Email Configuration ---
EMAIL_SENDER_ADDRESS: str = os.getenv("EMAIL_SENDER_ADDRESS", "noreply@yourtradinghub.com")
PASSWORD_RESET_SUBJECT: str = f"Password Reset Request - {APP_TITLE}"
PASSWORD_RESET_LINK_EXPIRY_HOURS: int = int(os.getenv("PASSWORD_RESET_LINK_EXPIRY_HOURS", "1"))

# --- Application Base URL (for constructing reset links) ---
APP_BASE_URL: str = os.getenv("APP_BASE_URL", "http://localhost:8501")


# --- Load environment variables from .env ---
# This should be done early, but after defining defaults that might be overridden.
from dotenv import load_dotenv
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(dotenv_path=dotenv_path)

# Re-evaluate any variables that depend on .env values after loading
# Example: If DATABASE_URL or API keys are set in .env, they would be available now.
# The DATABASE_URL logic above already uses os.getenv, so it will pick up .env values if set.
# Ensure LOG_LEVEL, EMAIL_SENDER_ADDRESS etc. are also re-read if they can be set by .env
LOG_LEVEL = os.getenv("LOG_LEVEL", LOG_LEVEL).upper() # Use existing LOG_LEVEL as default if not in .env
EMAIL_SENDER_ADDRESS = os.getenv("EMAIL_SENDER_ADDRESS", EMAIL_SENDER_ADDRESS)
APP_BASE_URL = os.getenv("APP_BASE_URL", APP_BASE_URL)
PASSWORD_RESET_LINK_EXPIRY_HOURS = int(os.getenv("PASSWORD_RESET_LINK_EXPIRY_HOURS", str(PASSWORD_RESET_LINK_EXPIRY_HOURS)))

# Final check for DATABASE_URL to ensure it's set
if not DATABASE_URL:
    print(f"CRITICAL WARNING (config.py): DATABASE_URL is not set even after .env load. Defaulting to SQLite file 'trading_dashboard_critical_fallback.db' in project root: {PROJECT_ROOT}")
    db_path_critical_fallback = os.path.join(PROJECT_ROOT, "trading_dashboard_critical_fallback.db")
    DATABASE_URL = f"sqlite:///{db_path_critical_fallback}"

