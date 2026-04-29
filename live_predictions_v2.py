#!/usr/bin/env python3
"""
LIVE ПРОГНОЗЫ v2 — Признаки из истории + модель + рекомендации
"""
import requests, pickle, pandas as pd, numpy as np
from datetime import datetime

print("=" * 60)
print(f"🔴 LIVE ПРОГНОЗ — {datetime.now().strftime('%d.%m.%Y %H:%M')}")
print("=" * 60)

# ============================================================
# 1. ЗАГРУЖАЕМ МОДЕЛЬ И ИСТОРИЧЕСКИЕ ДАННЫЕ
# ============================================================
print("\n📥 Загрузка модели и данных...")
with open('model_meta.pkl', 'rb') as f:
    data = pickle.load(f)
    models, FEATURES = data['models'], data['features']

# Исторические данные для расчёта признаков
hist = pd.read_csv('data/all_leagues.csv', low_memory=False)
hist['Date'] = pd.to_datetime(hist['Date'], dayfirst=True)
hist = hist.dropna(subset=['FTHG', 'FTAG'])
hist = hist.sort_values('Date')

def get_team_stats(team, league, current_date):
    """Считает признаки для команды на основе истории ДО current_date"""
    team_matches = hist[(hist['HomeTeam'] == team) | (hist['AwayTeam'] == team)]
    team_matches = team_matches[team_matches['Date'] < pd.to_datetime(current_date).tz_localize(None)]
    
    if len(team_matches) == 0:
        return {'goals_for_5': 1.5, 'goals_against_5': 1.2, 
                'goals_for_3': 1.5, 'goals_against_3': 1.2,
                'over_25_pct': 0.5, 'over_home_pct': 0.5, 'over_away_pct': 0.5,
                'position': 10, 'rest_days': 7}
    
    last_matches = team_matches.tail(20)
    
    # Голы
    goals_for = []
    goals_against = []
    for _, m in last_matches.iterrows():
        if m['HomeTeam'] == team:
            goals_for.append(m['FTHG'])
            goals_against.append(m['FTAG'])
        else:
            goals_for.append(m['FTAG'])
            goals_against.append(m['FTHG'])
    
    goals_for = np.array(goals_for, dtype=float)
    goals_against = np.array(goals_against, dtype=float)
    
    # Процент Over 2.5
    total_goals = (last_matches['FTHG'] + last_matches['FTAG']).values
    over_count = (total_goals > 2.5).sum()
    
    # Домашние/выездные Over%
    home_matches = last_matches[last_matches['HomeTeam'] == team]
    away_matches = last_matches[last_matches['AwayTeam'] == team]
    home_over = ((home_matches['FTHG'] + home_matches['FTAG']) > 2.5).mean() if len(home_matches) > 0 else 0.5
    away_over = ((away_matches['FTHG'] + away_matches['FTAG']) > 2.5).mean() if len(away_matches) > 0 else 0.5
    
    # Позиция (упрощённо)
    league_teams = hist[(hist['league'] == league) & (hist['Date'] < pd.to_datetime(current_date).tz_localize(None))]
    if len(league_teams) > 0:
        # Считаем очки
        pts = {}
        for _, m in league_teams.iterrows():
            for t in [m['HomeTeam'], m['AwayTeam']]:
                if t not in pts: pts[t] = 0
            if m['FTHG'] > m['FTAG']: pts[m['HomeTeam']] += 3
            elif m['FTHG'] == m['FTAG']:
                pts[m['HomeTeam']] += 1; pts[m['AwayTeam']] += 1
            else: pts[m['AwayTeam']] += 3
        sorted_pts = sorted(pts.values(), reverse=True)
        team_pts = pts.get(team, 0)
        position = list(sorted_pts).index(team_pts) + 1 if team_pts in sorted_pts else 10
    else:
        position = 10
    
    # Отдых
    if len(team_matches) > 0:
        last_date = team_matches['Date'].max()
        rest_days = (pd.to_datetime(current_date).tz_localize(None) - last_date).days
    else:
        rest_days = 7
    
    return {
        'goals_for_5': np.mean(goals_for[-5:]) if len(goals_for) >= 5 else np.mean(goals_for),
        'goals_against_5': np.mean(goals_against[-5:]) if len(goals_against) >= 5 else np.mean(goals_against),
        'goals_for_3': np.mean(goals_for[-3:]) if len(goals_for) >= 3 else np.mean(goals_for),
        'goals_against_3': np.mean(goals_against[-3:]) if len(goals_against) >= 3 else np.mean(goals_against),
        'over_25_pct': over_count / len(last_matches) if len(last_matches) > 0 else 0.5,
        'over_home_pct': home_over, 'over_away_pct': away_over,
        'position': min(position, 20), 'rest_days': min(rest_days, 14)
    }

# ============================================================
# 2. ПОЛУЧАЕМ LIVE-МАТЧИ
# ============================================================
print("📡 Загрузка live-матчей...")
API_KEY = 'fd01eee7eae66cd820dcb35166d0e868'
sports = {
    'soccer_epl': 'England_Premier',
    'soccer_spain_la_liga': 'Spain_LaLiga',
    'soccer_germany_bundesliga': 'Germany_Bundesliga',
    'soccer_italy_serie_a': 'Italy_SerieA',
    'soccer_france_ligue_one': 'France_Ligue1',
    'soccer_netherlands_eredivisie': 'Netherlands_Eredivisie',
    'soccer_portugal_primeira_liga': 'Portugal_Primeira',
}

matches = []
for sport, league_name in sports.items():
    resp = requests.get(f"https://api.the-odds-api.com/v4/sports/{sport}/odds/",
                        params={'apiKey': API_KEY, 'regions': 'eu', 'markets': 'h2h,totals', 'oddsFormat': 'decimal'})
    if resp.status_code == 200:
        for m in resp.json():
            bm = m['bookmakers'][0] if m.get('bookmakers') else None
            if not bm: continue
            
            h2h = next((mk for mk in bm['markets'] if mk['key']=='h2h'), None)
            totals = next((mk for mk in bm['markets'] if mk['key']=='totals'), None)
            if not h2h: continue
            
            matches.append({
                'home': m['home_team'], 'away': m['away_team'],
                'league': league_name, 'time': m['commence_time'],
                'odds_over': next((o['price'] for o in totals['outcomes'] if o['name']=='Over'), None) if totals else None,
                'odds_under': next((o['price'] for o in totals['outcomes'] if o['name']=='Under'), None) if totals else None,
            })

print(f"   Загружено: {len(matches)} матчей")

# ============================================================
# 3. ПРОГНОЗ ДЛЯ КАЖДОГО МАТЧА
# ============================================================
print("\n🧠 Расчёт прогнозов...")
predictions = []

for m in matches:
    if not m['odds_over']: continue
    
    # Признаки
    hs = get_team_stats(m['home'], m['league'], m['time'])
    as_ = get_team_stats(m['away'], m['league'], m['time'])
    
    league_avg = 2.5  # упрощённо
    
    features = {
        'home_position': hs['position'], 'away_position': as_['position'],
        'home_goals_for_5': hs['goals_for_5'], 'home_goals_against_5': hs['goals_against_5'],
        'home_goals_for_3': hs['goals_for_3'], 'home_goals_against_3': hs['goals_against_3'],
        'away_goals_for_5': as_['goals_for_5'], 'away_goals_against_5': as_['goals_against_5'],
        'away_goals_for_3': as_['goals_for_3'], 'away_goals_against_3': as_['goals_against_3'],
        'home_over_25_pct': hs['over_25_pct'], 'away_over_25_pct': as_['over_25_pct'],
        'home_over_home_pct': hs['over_home_pct'], 'away_over_away_pct': as_['over_away_pct'],
        'league_avg_goals': league_avg,
        'h2h_over_pct': 0.5,  # H2H пока упрощённо
        'home_rest_days': hs['rest_days'], 'away_rest_days': as_['rest_days'],
    }
    
    X = pd.DataFrame([features])[FEATURES]
    
    # Прогнозы
    probs = {}
    for name, model in models.items():
        probs[name] = model.predict_proba(X)[0, 1]
    
    # Лучший исход по EV
    best_ev = -999
    best_bet = None
    
    checks = [
        ('OU 2.5', 'ou25', m['odds_over'], m['odds_under']),
        ('Home 1.5', 'h15', 1.70, 1.70),  # приблизительные кэфы
        ('Away 1.5', 'a15', 1.80, 1.80),
        ('Home 0.5', 'h05', 1.25, 3.50),
        ('Away 0.5', 'a05', 1.30, 3.00),
    ]
    
    for line_name, model_key, odds_o, odds_u in checks:
        p = probs[model_key]
        if p > 0.70 and odds_o:
            ev = p * odds_o - 1
            if ev > best_ev and ev > 0.05:
                best_ev = ev
                best_bet = (line_name, 'Over', odds_o, p, ev)
        if (1-p) > 0.65 and odds_u:
            ev = (1-p) * odds_u - 1
            if ev > best_ev and ev > 0.05:
                best_ev = ev
                best_bet = (line_name, 'Under', odds_u, 1-p, ev)
    
    if best_bet:
        predictions.append({
            'home': m['home'], 'away': m['away'],
            'league': m['league'], 'time': m['time'][:16],
            'line': best_bet[0], 'direction': best_bet[1],
            'odds': best_bet[2], 'prob': best_bet[3], 'ev': best_bet[4],
            'stake': 200
        })

# ============================================================
# 4. ВЫВОД
# ============================================================
print(f"\n{'='*80}")
print(f"🎯 РЕКОМЕНДАЦИИ ({len(predictions)} ставок)")
print(f"{'='*80}")
print(f"{'#':<3} {'Матч':<35} {'Линия':<12} {'Напр':<6} {'Кэф':<6} {'Вер':<6} {'EV':<7} {'Ставка'}")
print(f"{'-'*80}")

total_stake = 0
for i, p in enumerate(sorted(predictions, key=lambda x: x['ev'], reverse=True)[:20]):
    total_stake += p['stake']
    print(f"{i+1:<3} {p['home']+' vs '+p['away']:<35} {p['line']:<12} {p['direction']:<6} "
          f"{p['odds']:<6.2f} {p['prob']:<5.0%} {p['ev']:<+6.1%} {p['stake']}₽")

print(f"\n💰 Всего ставок: {len(predictions)} на сумму {total_stake:,}₽")
print(f"📁 Сохранено в live_predictions.csv")

pd.DataFrame(predictions).to_csv('live_predictions.csv', index=False)
print("\n✅ ГОТОВО!")
