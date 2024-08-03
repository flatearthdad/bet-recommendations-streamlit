import os
from dotenv import load_dotenv
import requests
import pandas as pd
import re
from datetime import datetime
import logging
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load environment variables from .env file
load_dotenv()

# Import configuration settings
from config import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Streamlit App
st.set_page_config(page_title="WagerGPT Betting Recommendations", layout="wide")

# Custom CSS for dark theme and improved visual appeal
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stDataFrame {
        background-color: #2D2D2D;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,.1);
    }
    .stSlider>div>div>div>div {
        background-color: #4CAF50;
    }
    .stTextInput>div>div>input {
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
    .stSelectbox>div>div>select {
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
    .stTab {
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
    .stTab[data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTab[data-baseweb="tab"] {
        height: 50px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTab[aria-selected="true"] {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Settings")
    wagergpt_api_key = st.text_input("WagerGPT API Key", type="password")
    odds_api_key = st.text_input("The Odds API Key", type="password")
    st.session_state.bankroll = st.number_input("Bankroll ($)", min_value=100, value=st.session_state.bankroll, step=100)
    st.session_state.kelly_type = st.selectbox("Kelly Criterion", list(KELLY_FRACTIONS.keys()), index=list(KELLY_FRACTIONS.keys()).index(st.session_state.kelly_type))
    st.session_state.win_rate = st.slider("Default Win Rate", min_value=0.0, max_value=1.0, value=st.session_state.win_rate, step=0.01)

# Check API keys
if not wagergpt_api_key or not odds_api_key:
    st.error("Please enter your API keys in the sidebar.")
else:
    # Debugging: Print API keys (remove or comment out in production)
    print(f"WagerGPT API Key: {wagergpt_api_key}")
    print(f"The Odds API Key: {odds_api_key}")

    # Store API keys in session state
    st.session_state.wagergpt_api_key = wagergpt_api_key
    st.session_state.odds_api_key = odds_api_key

# Main content
st.title("WagerGPT Betting Recommendations")

# Navigation
tabs = st.tabs(["Predictions", "Latest Odds", "Analysis"])

# Setup a session for requests with retry logic
session = requests.Session()
retry = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)
session.mount("http://", adapter)

@st.cache_data
def fetch_daily_picks(access_token):
    params = {'access_token': access_token}
    try:
        response = session.get(WAGERGPT_API_URL, params=params)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"WagerGPT API request failed: {e}")
        return None
    return response.json()

def get_bet_recommendations():
    if not st.session_state.wagergpt_api_key:
        st.error("Please enter your WagerGPT API key in the sidebar.")
        return None

    daily_picks = fetch_daily_picks(st.session_state.wagergpt_api_key)
    
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
            "Odds": float(odds),
            "WinRate": st.session_state.win_rate,
            "Team Edge (%)": st.session_state.team_edges.get(team.strip(), 0.0)
        })

    df_bets = pd.DataFrame(bets)

    if df_bets.empty:
        st.warning("No bets available after parsing.")
        return None

    st.session_state.df_bets = df_bets

    # Ensure 'EV' column is created before using it
    calculate_ev_column()

def calculate_ev_column():
    if st.session_state.df_bets is not None:
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

        # Update the dynamic table
        st.session_state.df_dynamic_bets = st.session_state.df_bets[['Team', 'Spread', 'Odds', 'WinRate', 'Team Edge (%)', 'Adjusted Win Rate', 'EV', 'Recommended Bet Size']]

def update_team_edges():
    st.subheader("Team Edge Adjustments")
    col1, col2 = st.columns(2)
    with col1:
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
    
    calculate_ev_column()

def display_predictions():
    if st.session_state.df_bets is not None:
        st.dataframe(st.session_state.df_bets.style.format({
            "Spread": "{:.2f}", 
            "Odds": "{:.2f}", 
            "WinRate": "{:.2%}", 
            "Team Edge (%)": "{:.2%}", 
            "Adjusted Win Rate": "{:.2%}", 
            "EV": "{:.4f}",
            "Recommended Bet Size": "${:.2f}"
        }).background_gradient(subset=['EV'], cmap='RdYlGn'))

        # Improved bet size distribution graphic as a bar chart
        fig = px.bar(st.session_state.df_bets, x='Team', y='Recommended Bet Size', color='Team', title='Bet Size Distribution by Team')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#FFFFFF',
            xaxis_title='Team',
            yaxis_title='Recommended Bet Size ($)'
        )
        st.plotly_chart(fig)
    else:
        st.warning("No betting recommendations available. Please fetch the data first.")

def fetch_odds():
    if not st.session_state.odds_api_key:
        st.error("Please enter your The Odds API key in the sidebar.")
        return None

    params = {
        'apiKey': st.session_state.odds_api_key,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT
    }
    
    try:
        response = session.get(ODDS_API_URL, params=params)
        response.raise_for_status()
        odds_json = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching odds data: {e}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON: {e}")
        return None
    
    events = []
    for event in odds_json:
        home_team = event['home_team']
        away_team = event['away_team']
        commence_time = event['commence_time']
        
        for bookmaker in event['bookmakers']:
            for market in bookmaker['markets']:
                for outcome in market['outcomes']:
                    events.append([
                        event['commence_time'],
                        event['home_team'],
                        event['away_team'],
                        bookmaker['title'],
                        market['key'],
                        outcome['name'],
                        outcome.get('price', 'N/A'),
                        outcome.get('point', 'N/A')
                    ])
    
    df = pd.DataFrame(events, columns=[
        'Commence Time', 'Home Team', 'Away Team',
        'Bookmaker', 'Market', 'Team', 'Odds', 'Point'
    ])
    
    df['Commence Time'] = pd.to_datetime(df['Commence Time'])
    
    return df

def find_best_odds():
    if st.session_state.df_odds is not None and st.session_state.df_bets is not None:
        best_odds = []
        for _, bet in st.session_state.df_bets.iterrows():
            team = bet['Team']
            odds_data = st.session_state.df_odds[st.session_state.df_odds['Team'] == team]
            best_odd = odds_data.loc[odds_data['Odds'].idxmax()]
            best_odds.append({
                'Team': team,
                'Best Bookmaker': best_odd['Bookmaker'],
                'Best Odds': best_odd['Odds']
            })
        return pd.DataFrame(best_odds)
    else:
        st.warning("No odds or bets data available to find best odds.")
        return None

def display_analysis():
    if st.session_state.df_bets is not None:
        # Display dynamic table
        st.subheader("Dynamic Bet Sizing Table")
        if 'df_dynamic_bets' in st.session_state:
            st.dataframe(st.session_state.df_dynamic_bets.style.format({
                "Spread": "{:.2f}", 
                "Odds": "{:.2f}", 
                "WinRate": "{:.2%}", 
                "Team Edge (%)": "{:.2%}", 
                "Adjusted Win Rate": "{:.2%}", 
                "EV": "{:.4f}",
                "Recommended Bet Size": "${:.2f}"
            }).background_gradient(subset=['EV'], cmap='RdYlGn'))

        update_team_edges()

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Expected Value Distribution")
            fig = px.bar(st.session_state.df_bets, x='Team', y='EV', title='Expected Value by Team')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#FFFFFF'
            )
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Bet Size Distribution")
            fig = px.bar(st.session_state.df_bets, x='Team', y='Recommended Bet Size', title='Recommended Bet Size by Team')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#FFFFFF'
            )
            st.plotly_chart(fig)
        
        st.subheader("Summary Statistics")
        summary_stats = st.session_state.df_bets[['EV', 'Recommended Bet Size']].describe()
        st.dataframe(summary_stats.style.format("{:.4f}"))

        # Add a heat map for correlations
        correlation_matrix = st.session_state.df_bets[['Spread', 'Odds', 'WinRate', 'Team Edge (%)', 'EV', 'Recommended Bet Size']].corr()
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu'
        ))
        fig.update_layout(
            title='Correlation Heatmap',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#FFFFFF'
        )
        st.plotly_chart(fig)
    else:
        st.warning("No data available for analysis. Please fetch predictions first.")

def display_odds():
    st.subheader("Latest Odds")
    if st.button("Fetch Latest Odds"):
        with st.spinner("Fetching latest odds..."):
            st.session_state.df_odds = fetch_odds()
    if st.session_state.df_odds is not None and not st.session_state.df_odds.empty:
        st.dataframe(
            st.session_state.df_odds.style.format({
                "Odds": lambda x: f"{x:+d}" if isinstance(x, (int, float)) else x,
                "Point": lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x
            }).background_gradient(subset=['Odds'], cmap='RdYlGn', low=0.5, high=0)
        )

        # Add a bar chart for odds comparison
        fig = px.bar(
            st.session_state.df_odds,
            x='Team',
            y='Odds',
            color='Bookmaker',
            title='Odds Comparison by Bookmaker',
            barmode='group'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#FFFFFF',
            xaxis_title='Team',
            yaxis_title='Odds'
        )
        st.plotly_chart(fig)

        st.subheader("Best Odds by Bookmaker")
        best_odds_df = find_best_odds()
        if best_odds_df is not None:
            st.dataframe(best_odds_df)
    else:
        st.warning("No odds data available. Please fetch the latest odds.")

# Fetch and display data
with tabs[0]:
    st.subheader("Betting Predictions")
    if st.button("Fetch Betting Predictions"):
        with st.spinner("Fetching predictions..."):
            get_bet_recommendations()
    display_predictions()

with tabs[1]:
    display_odds()

with tabs[2]:
    display_analysis()
