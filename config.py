# config.py
WAGERGPT_API_URL = "http://api.wagergpt.co/daily-picks"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
SPORT = 'baseball_mlb'
REGIONS = 'us'
MARKETS = 'h2h,spreads'
ODDS_FORMAT = 'american'

KELLY_FRACTIONS = {
    "Full Kelly": 1.0,
    "Half Kelly": 0.5,
    "Quarter Kelly": 0.25
}
