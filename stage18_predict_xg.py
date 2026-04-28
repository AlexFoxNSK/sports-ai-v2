#!/usr/bin/env python3
"""
ЭТАП 18: ОБУЧАЕМ МОДЕЛЬ ПРЕДСКАЗЫВАТЬ xG
Используем xG для Team Totals на ВСЕХ матчах.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("🚀 ЭТАП 18: ПРЕДСКАЗАНИЕ xG ДЛЯ TEAM TOTALS")
print("=" * 55)

# ============================================================
# 1. MERGE И ПРИЗНАКИ (все матчи)
# ============================================================
print("\n📥 ЗАГРУЗКА И ПРИЗНАКИ")
df = pd.read_csv('data/all_leagues.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.dropna(subset=['FTHG', 'FTAG'])
df = df.sort_values('Date').reset_index(drop=True)

# Merge с xG (только для АПЛ)
xgd = pd.read_csv('data/sstats_epl.csv')
xgd['date'] = pd.to_datetime(xgd['date'])
df = df.merge(
    xgd[['date', 'home', 'homeXg', 'awayXg']],
    left_on=['Date', 'HomeTeam'], right_on=['date', 'home'], how='left'
)
df = df.drop(columns=['date', 'home'], errors='ignore')

# Флаг: есть ли реальный xG
df['has_real_xg'] = df['homeXg'].notna().astype(int)
real_xg_count = df['has_real_xg'].sum()
print(f"   С реальным xG: {real_xg_count} матчей")

# --- Позиции ---
st = {}; hp, ap = [], []
for _, row in df.iterrows():
    for team in [row['HomeTeam'], row['AwayTeam']]:
        if team not in st: st[team] = {'pts': 0}
    s = sorted(st.items(), key=lambda x: x[1]['pts'], reverse=True)
    p = {t: i+1 for i, (t, _) in enumerate(s)}
    hp.append(p[row['HomeTeam']]); ap.append(p[row['AwayTeam']])
    hg, ag = row['FTHG'], row['FTAG']
    if hg > ag: st[row['HomeTeam']]['pts'] += 3
    elif hg == ag: st[row['HomeTeam']]['pts'] += 1; st[row['AwayTeam']]['pts'] += 1
    else: st[row['AwayTeam']]['pts'] += 3
df['home_position'] = hp; df['away_position'] = ap

# --- Голы ---
tg, ta = {}, {}
hg5, ha5, hg3, ha3 = [], [], [], []
ag5, aa5, ag3, aa3 = [], [], [], []
for _, row in df.iterrows():
    for team, g5, a5, g3, a3, gf, ga in [
        (row['HomeTeam'], hg5, ha5, hg3, ha3, row['FTHG'], row['FTAG']),
        (row['AwayTeam'], ag5, aa5, ag3, aa3, row['FTAG'], row['FTHG'])
    ]:
        gh = tg.get(team, []); ah = ta.get(team, [])
        g5.append(np.mean(gh[-5:]) if gh else 0); a5.append(np.mean(ah[-5:]) if ah else 0)
        g3.append(np.mean(gh[-3:]) if gh else 0); a3.append(np.mean(ah[-3:]) if ah else 0)
        gh.append(gf); ah.append(ga); tg[team] = gh; ta[team] = ah
df['home_goals_for_5'] = hg5; df['home_goals_against_5'] = ha5
df['home_goals_for_3'] = hg3; df['home_goals_against_3'] = ha3
df['away_goals_for_5'] = ag5; df['away_goals_against_5'] = aa5
df['away_goals_for_3'] = ag3; df['away_goals_against_3'] = aa3

# --- Over% ---
to = {}; ho, ao = [], []
to_h, to_a = {}, {}; ho_h, ao_a = [], []
for _, row in df.iterrows():
    ov = 1 if row['FTHG'] + row['FTAG'] > 2.5 else 0
    for team, lst in [(row['HomeTeam'], ho), (row['AwayTeam'], ao)]:
        h = to.get(team, []); lst.append(np.mean(h[-20:]) if h else 0.5); h.append(ov); to[team] = h
    hh = to_h.get(row['HomeTeam'], []); ho_h.append(np.mean(hh[-10:]) if hh else 0.5); hh.append(ov); to_h[row['HomeTeam']] = hh
    ah = to_a.get(row['AwayTeam'], []); ao_a.append(np.mean(ah[-10:]) if ah else 0.5); ah.append(ov); to_a[row['AwayTeam']] = ah
df['home_over_25_pct'] = ho; df['away_over_25_pct'] = ao
df['home_over_home_pct'] = ho_h; df['away_over_away_pct'] = ao_a

# --- Лига ---
lg = {}; la = []
for _, row in df.iterrows():
    league = row['league']; t = row['FTHG'] + row['FTAG']
    if league not in lg: lg[league] = []
    la.append(np.mean(lg[league]) if lg[league] else 2.5); lg[league].append(t)
df['league_avg_goals'] = la

# --- H2H ---
h2h = {}; hl = []
for _, row in df.iterrows():
    key = tuple(sorted([row['HomeTeam'], row['AwayTeam']]))
    h = h2h.get(key, []); hl.append(np.mean(h[-5:]) if h else 0.5)
    h.append(1 if row['FTHG'] + row['FTAG'] > 2.5 else 0); h2h[key] = h
df['h2h_over_pct'] = hl

# --- Отдых ---
last = {}; hr, ar = [], []
for _, row in df.iterrows():
    d = row['Date']
    for team, lst in [(row['HomeTeam'], hr), (row['AwayTeam'], ar)]:
        prev = last.get(team, d - pd.Timedelta(days=7))
        lst.append(min((d - prev).days, 14)); last[team] = d
df['home_rest_days'] = hr; df['away_rest_days'] = ar

FEATURES = [
    'home_over_25_pct', 'away_position', 'home_position', 'league_avg_goals',
    'home_goals_against_3', 'away_goals_against_5', 'away_over_25_pct',
    'home_goals_for_5', 'h2h_over_pct', 'away_goals_for_3',
    'home_rest_days', 'away_goals_for_5', 'home_goals_against_5',
    'home_goals_for_3', 'away_rest_days', 'away_goals_against_3',
    'home_over_home_pct', 'away_over_away_pct'
]

# ============================================================
# 2. ОБУЧАЕМ МОДЕЛЬ ПРЕДСКАЗЫВАТЬ homeXg
# ============================================================
print("\n🧠 ОБУЧАЕМ XGBoost РЕГРЕССОР ДЛЯ homeXg")

# Только матчи с реальным xG
train_xg = df[df['has_real_xg'] == 1].copy()
split_xg = int(len(train_xg) * 0.8)
train_xg_train = train_xg.iloc[:split_xg]
train_xg_test = train_xg.iloc[split_xg:]

# Модель для homeXg
model_home_xg = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
model_home_xg.fit(train_xg_train[FEATURES], train_xg_train['homeXg'])

# Модель для awayXg
model_away_xg = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
model_away_xg.fit(train_xg_train[FEATURES], train_xg_train['awayXg'])

# Оценка
pred_home = model_home_xg.predict(train_xg_test[FEATURES])
pred_away = model_away_xg.predict(train_xg_test[FEATURES])
mae_home = mean_absolute_error(train_xg_test['homeXg'], pred_home)
mae_away = mean_absolute_error(train_xg_test['awayXg'], pred_away)
print(f"   MAE homeXg: {mae_home:.3f}")
print(f"   MAE awayXg: {mae_away:.3f}")

# ============================================================
# 3. ПРЕДСКАЗЫВАЕМ xG ДЛЯ ВСЕХ МАТЧЕЙ
# ============================================================
print("\n🔮 ПРЕДСКАЗЫВАЕМ xG ДЛЯ ВСЕХ 14 282 МАТЧЕЙ")
df['pred_homeXg'] = model_home_xg.predict(df[FEATURES])
df['pred_awayXg'] = model_away_xg.predict(df[FEATURES])

# Там где есть реальный xG — используем его
df['final_homeXg'] = df['homeXg'].fillna(df['pred_homeXg'])
df['final_awayXg'] = df['awayXg'].fillna(df['pred_awayXg'])

# ============================================================
# 4. ТЕСТИРУЕМ TEAM TOTALS НА ВСЕХ МАТЧАХ
# ============================================================
print("\n💵 ТЕСТИРУЕМ TEAM TOTALS")

split = int(len(df) * 0.8)
test = df.iloc[split:]

def get_team_odds(xg_val, is_home=True):
    """Приблизительный кэф на основе xG (из нашей таблицы)"""
    if xg_val > 2.5: return 1.12
    elif xg_val > 2.0: return 1.54 if is_home else 1.27
    elif xg_val > 1.5: return 1.77 if is_home else 1.91
    elif xg_val > 1.0: return 2.00 if is_home else 2.51
    elif xg_val > 0.5: return 3.85 if is_home else 5.67
    else: return 6.50

bank = 10000
bets = wins = 0

for _, row in test.iterrows():
    hxg = row['final_homeXg']
    axg = row['final_awayXg']
    home_goals = row['FTHG']
    away_goals = row['FTAG']
    stake = bank * 0.02

    # Home Team Over 1.5
    if hxg > 2.0:
        odds = get_team_odds(hxg, True)
        bets += 1
        if home_goals > 1.5:
            bank += stake * (odds - 1)
            wins += 1
        else:
            bank -= stake

    # Away Team Over 1.5
    if axg > 2.0:
        odds = get_team_odds(axg, False)
        bets += 1
        if away_goals > 1.5:
            bank += stake * (odds - 1)
            wins += 1
        else:
            bank -= stake

roi = (bank - 10000) / 100
hit = wins / bets * 100 if bets > 0 else 0
print(f"   Ставок: {bets} | Банк: {bank:.0f}₽ | ROI: {roi:+.2f}% | Hit: {hit:.1f}%")

print(f"\n📊 СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ:")
print(f"   OU 2.5:       88 ставок, ROI +13.45%, Hit 65.9%")
print(f"   Team Totals:  {bets} ставок, ROI {roi:+.2f}%, Hit {hit:.1f}%")
print(f"   ВМЕСТЕ:       {88+bets} ставок")

print("\n✅ ЭТАП 18 ЗАВЕРШЁН")
