import os
from dotenv import load_dotenv
import requests
import pandas as pd
import re
from datetime import datetime, timedelta
import pytz
import logging
import streamlit as st
import schedule
import time
import http.client
import json

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
WAGERGPT_API_URL = "http://api.wagergpt.co/daily-picks"
API_KEY = os.getenv('WAGERGPT_API_KEY')
SPORT = 'baseball_mlb'
REGIONS = 'us'  # Multiple regions can be specified if comma delimited
MARKETS = 'h2h,spreads'  # Multiple markets can be specified if comma delimited
ODDS_FORMAT = 'decimal'
DATE_FORMAT = 'iso'
RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY')
RAPIDAPI_HOST = "odds.p.rapidapi.com"
HISTORICAL_DATA_FILE = "odds_data.csv"

KELLY_FRACTIONS = {
    "Full Kelly": 1.0,
    "Half Kelly": 0.5,
    "Quarter Kelly": 0.25
}

# Initialize session state variables
if 'df_bets' not in st.session_state:
    st.session_state.df_bets = None

if 'df_odds' not in st.session_state:
    st.session_state.df_odds = None

if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 1000

if 'kelly_type' not in st.session_state:
    st.session_state.kelly_type = "Half Kelly"

if 'win_rate' not in st.session_state:
    st.session_state.win_rate = 0.5

if 'team_edges' not in st.session_state:
    st.session_state.team_edges = {}

if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# Streamlit App
st.set_page_config(page_title="WagerGPT Betting Recommendations", layout="wide")

# Sidebar
with st.sidebar:
    st.title("Settings")
    st.session_state.api_key = st.text_input("WagerGPT API Key", type="password", value=st.session_state.api_key)
    st.session_state.bankroll = st.number_input("Bankroll ($)", min_value=100, value=st.session_state.bankroll, step=100)
    st.session_state.kelly_type = st.selectbox("Kelly Criterion", list(KELLY_FRACTIONS.keys()), index=list(KELLY_FRACTIONS.keys()).index(st.session_state.kelly_type))
    st.session_state.win_rate = st.slider("Default Win Rate", min_value=0.0, max_value=1.0, value=st.session_state.win_rate, step=0.01)

# Main content
st.title("WagerGPT Betting Recommendations")

# Navigation
tabs = st.tabs(["Predictions", "Latest Odds", "Analysis", "History"])

@st.cache_data
def fetch_daily_picks(access_token):
    params = {'access_token': access_token}
    try:
        response = requests.get(WAGERGPT_API_URL, params=params)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"WagerGPT API request failed: {e}")
        return None
    return response.json()

def get_bet_recommendations():
    if not st.session_state.api_key:
        st.error("Please enter your WagerGPT API key in the sidebar.")
        return None

    daily_picks = fetch_daily_picks(st.session_state.api_key)
    
    if daily_picks is None:
        st.error("Failed to fetch data from WagerGPT API. Please check your API key and try again.")
        return None

    picks_string = daily_picks.get('picks', '')
    if not picks_string:
        st.warning("No picks found in the API response.")
        return None
    
    pattern = re.compile(r"(\d+\.\s)([\w\s]+)\s([-+]?\d+(?:\.\d+)?)\s@\s(\d+\.\d+)")
    picks = pattern.findall(picks_string)
    
    bets = []
    for pick in picks:
        bet_number, team, point_spread, odds = pick
        bets.append({
            "Sport": "MLB",
            "Team": team.strip(),
            "Spread": float(point_spread),
            "Bet Recommendation": f"{team.strip()} {point_spread}",
            "Odds": float(odds),
            "WinRate": st.session_state.win_rate,
            "Team Edge (%)": st.session_state.team_edges.get(team.strip(), 0.0)
        })

    df_bets = pd.DataFrame(bets)

    if df_bets.empty:
        st.warning("No bets available after parsing.")
        return None

    st.session_state.df_bets = df_bets

def update_team_edges():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Team Edge Adjustments")
        for i, row in st.session_state.df_bets.iloc[:len(st.session_state.df_bets)//2].iterrows():
            team = row['Team']
            unique_key = f"{team}_{i}"
            edge = st.slider(
                f"{team} Edge", min_value=-50, max_value=50, value=int(row['Team Edge (%)'] * 100), step=1, key=unique_key)
            st.session_state.team_edges[team] = edge / 100
            st.session_state.df_bets.at[i, 'Team Edge (%)'] = edge / 100
    
    with col2:
        for i, row in st.session_state.df_bets.iloc[len(st.session_state.df_bets)//2:].iterrows():
            team = row['Team']
            unique_key = f"{team}_{i}"
            edge = st.slider(
                f"{team} Edge", min_value=-50, max_value=50, value=int(row['Team Edge (%)'] * 100), step=1, key=unique_key)
            st.session_state.team_edges[team] = edge / 100
            st.session_state.df_bets.at[i, 'Team Edge (%)'] = edge / 100
    
    st.session_state.df_bets['Adjusted Win Rate'] = st.session_state.df_bets.apply(lambda x: round(x['WinRate'] + x['Team Edge (%)'], 4), axis=1)
    
    def calculate_ev(row):
        return round((row['Adjusted Win Rate'] * (row['Odds'] - 1) - (1 - row['Adjusted Win Rate'])), 4)

    st.session_state.df_bets['EV'] = st.session_state.df_bets.apply(calculate_ev, axis=1)

    kelly_fraction = KELLY_FRACTIONS[st.session_state.kelly_type]

    st.session_state.df_bets['Recommended Bet Size'] = st.session_state.df_bets.apply(
        lambda x: round(kelly_fraction * st.session_state.bankroll * x['EV'] / x['Odds'], 4), axis=1)

    max_daily_risk = 0.5 * st.session_state.bankroll
    if st.session_state.df_bets['Recommended Bet Size'].sum() > max_daily_risk:
        st.session_state.df_bets['Recommended Bet Size'] = round(
            (st.session_state.df_bets['Recommended Bet Size'] / st.session_state.df_bets['Recommended Bet Size'].sum()) * max_daily_risk, 4)

def display_predictions():
    if st.session_state.df_bets is not None:
        st.dataframe(st.session_state.df_bets.style.format({
            "Spread": "{:.2f}", 
            "Odds": "{:.2f}", 
            "WinRate": "{:.2%}", 
            "Team Edge (%)": "{:.2%}", 
            "Adjusted Win Rate": "{:.2%}", 
            "EV": "{:.4f}",  # Displayed as decimal
            "Recommended Bet Size": "${:.2f}"
        }).background_gradient(subset=['EV'], cmap='RdYlGn'))
    else:
        st.warning("No betting recommendations available. Please fetch the data first.")

def display_analysis():
    if st.session_state.df_bets is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Expected Value Distribution")
            ev_chart = st.session_state.df_bets[['Team', 'EV']].set_index('Team')
            st.bar_chart(ev_chart)
        
        with col2:
            st.subheader("Bet Size Distribution")
            bet_size_chart = st.session_state.df_bets[['Team', 'Recommended Bet Size']].set_index('Team')
            st.bar_chart(bet_size_chart)
        
        st.subheader("Summary Statistics")
        summary_stats = st.session_state.df_bets[['EV', 'Recommended Bet Size']].describe()
        st.dataframe(summary_stats.style.format("{:.4f}"))

        # Load and display historical data
        historical_data = load_historical_data()
        if historical_data is not None:
            display_historical_charts(historical_data)
    else:
        st.warning("No data available for analysis. Please fetch predictions first.")

def display_history():
    st.info("Betting history feature coming soon!")

def fetch_odds():
    conn = http.client.HTTPSConnection(RAPIDAPI_HOST)
    headers = {
        'x-rapidapi-key': RAPIDAPI_KEY,
        'x-rapidapi-host': RAPIDAPI_HOST
    }
    
    conn.request("GET", f"/v4/sports/{SPORT}/odds?regions={REGIONS}&oddsFormat={ODDS_FORMAT}&markets={MARKETS}&dateFormat={DATE_FORMAT}", headers=headers)
    
    res = conn.getresponse()
    data = res.read()
    
    odds_json = json.loads(data.decode("utf-8"))
    
    # Create a DataFrame from the JSON data
    events = []
    for event in odds_json:
        for bookmaker in event['bookmakers']:
            for market in bookmaker['markets']:
                for outcome in market['outcomes']:
                    events.append([
                        event['commence_time'],
                        event['sport_key'],
                        event['home_team'],
                        event['away_team'],
                        bookmaker['title'],
                        market['key'],
                        outcome['name'],
                        outcome['price']
                    ])
    
    df = pd.DataFrame(events, columns=[
        'Commence Time', 'Sport Key', 'Home Team', 'Away Team',
        'Bookmaker', 'Market', 'Outcome', 'Odds'
    ])
    
    df['Commence Time'] = pd.to_datetime(df['Commence Time'])
    df['Timestamp'] = datetime.now()
    
    # Highlight the most favorable odds (lowest odds)
    idx = df.groupby(['Home Team', 'Away Team', 'Market'])['Odds'].transform(min) == df['Odds']
    favorable_odds_df = df[idx]
    
    return favorable_odds_df

def display_odds():
    st.subheader("Latest Odds")
    if st.button("Fetch Latest Odds"):
        with st.spinner("Fetching latest odds..."):
            st.session_state.df_odds = fetch_odds()
    
    if st.session_state.df_odds is not None:
        st.dataframe(st.session_state.df_odds.style.format({
            "Odds": "{:.2f}"
        }).background_gradient(subset=['Odds'], cmap='RdYlGn', low=0.5, high=0))
    else:
        st.warning("No odds data available. Please fetch the latest odds.")

def load_historical_data():
    if os.path.exists(HISTORICAL_DATA_FILE):
        return pd.read_csv(HISTORICAL_DATA_FILE)
    return None

def display_historical_charts(historical_data):
    last_24_hours = datetime.now() - timedelta(hours=24)
    historical_data['Timestamp'] = pd.to_datetime(historical_data['Timestamp'])
    recent_data = historical_data[historical_data['Timestamp'] >= last_24_hours]

    st.subheader("24-hour Change in Expected Value")
    ev_chart = recent_data.pivot(index='Timestamp', columns='Team', values='EV').fillna(0)
    st.line_chart(ev_chart)

    st.subheader("24-hour Change in Recommended Bet Size")
    bet_size_chart = recent_data.pivot(index='Timestamp', columns='Team', values='Recommended Bet Size').fillna(0)
    st.line_chart(bet_size_chart)

# Predictions tab
with tabs[0]:
    if st.button("Fetch Predictions"):
        get_bet_recommendations()
    
    if st.session_state.df_bets is not None:
        update_team_edges()
        display_predictions()

# Latest Odds tab
with tabs[1]:
    display_odds()

# Analysis tab
with tabs[2]:
    display_analysis()

# History tab
with tabs[3]:
    display_history()

st.sidebar.info("Developed by Your Company Name")
st.sidebar.text("Version 2.1")

# Scheduling functionality
def scheduled_odds_fetch():
    odds_df = fetch_odds()
    if odds_df is not None:
        odds_df.to_csv(HISTORICAL_DATA_FILE, mode='a', header=not os.path.exists(HISTORICAL_DATA_FILE), index=False)
    st.experimental_rerun()

def schedule_jobs():
    pst = pytz.timezone('America/Los_Angeles')
    times = ["06:00", "09:00", "12:00", "15:00"]
    for time_str in times:
        schedule_time = datetime.strptime(time_str, "%H:%M").replace(tzinfo=pst)
        schedule.every().day.at(schedule_time.strftime("%H:%M")).do(scheduled_odds_fetch)

if __name__ == "__main__":
    schedule_jobs()

    while True:
        schedule.run_pending()
        time.sleep(1)
