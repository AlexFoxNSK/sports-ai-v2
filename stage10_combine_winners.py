#!/usr/bin/env python3
"""
ЭТАП 10: КОМБИНАЦИЯ ЛУЧШИХ ПРИЗНАКОВ
- Over% дома/в гостях + Rest advantage
- Два порога уверенности (70% и 65%)
- Несколько размеров стейка
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("🚀 ЭТАП 10: КОМБИНАЦИЯ ПОБЕДИТЕЛЕЙ + ТЮНИНГ")
print("=" * 55)

# ============================================================
# ЗАГРУЗКА И ПРИЗНАКИ
# ============================================================
df = pd.read_csv('data/all_leagues.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.dropna(subset=['FTHG', 'FTAG'])
df = df.sort_values('Date').reset_index(drop=True)
df['odds_over'] = pd.to_numeric(df.get('B365>2.5', None), errors='coerce')
df['odds_under'] = pd.to_numeric(df.get('B365<2.5', None), errors='coerce')
df = df.dropna(subset=['odds_over', 'odds_under'])

# --- Турнирная таблица ---
standings = {}
home_pos, away_pos = [], []
for _, row in df.iterrows():
    for team in [row['HomeTeam'], row['AwayTeam']]:
        if team not in standings: standings[team] = {'pts': 0}
    s = sorted(standings.items(), key=lambda x: x[1]['pts'], reverse=True)
    p = {t: i+1 for i, (t, _) in enumerate(s)}
    home_pos.append(p[row['HomeTeam']]); away_pos.append(p[row['AwayTeam']])
    hg, ag = row['FTHG'], row['FTAG']
    if hg > ag: standings[row['HomeTeam']]['pts'] += 3
    elif hg == ag: standings[row['HomeTeam']]['pts'] += 1; standings[row['AwayTeam']]['pts'] += 1
    else: standings[row['AwayTeam']]['pts'] += 3
df['home_position'] = home_pos; df['away_position'] = away_pos

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

# --- Over% общий + дома/в гостях ---
to = {}; ho, ao = [], []
to_h, to_a = {}, {}
ho_home, ao_away = [], []
for _, row in df.iterrows():
    ov = 1 if row['FTHG'] + row['FTAG'] > 2.5 else 0
    for team, lst in [(row['HomeTeam'], ho), (row['AwayTeam'], ao)]:
        h = to.get(team, []); lst.append(np.mean(h[-20:]) if h else 0.5); h.append(ov); to[team] = h
    hh = to_h.get(row['HomeTeam'], []); ho_home.append(np.mean(hh[-10:]) if hh else 0.5); hh.append(ov); to_h[row['HomeTeam']] = hh
    ah = to_a.get(row['AwayTeam'], []); ao_away.append(np.mean(ah[-10:]) if ah else 0.5); ah.append(ov); to_a[row['AwayTeam']] = ah
df['home_over_25_pct'] = ho; df['away_over_25_pct'] = ao
df['home_over_home_pct'] = ho_home; df['away_over_away_pct'] = ao_away

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
df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']

df['target'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)
df['total_goals'] = df['FTHG'] + df['FTAG']

# ============================================================
# ТЕСТИРУЕМ КОМБИНАЦИИ
# ============================================================
split = int(len(df) * 0.8)
train = df.iloc[:split]; test = df.iloc[split:]

# Лучшие признаки (16 базовых + 2 Over% + 1 отдых)
FEATURES_BEST = [
    'home_over_25_pct', 'away_position', 'home_position', 'league_avg_goals',
    'home_goals_against_3', 'away_goals_against_5', 'away_over_25_pct',
    'home_goals_for_5', 'h2h_over_pct', 'away_goals_for_3',
    'home_rest_days', 'away_goals_for_5', 'home_goals_against_5',
    'home_goals_for_3', 'away_rest_days', 'away_goals_against_3',
    'home_over_home_pct', 'away_over_away_pct',  # 🆕 Over% дома/в гостях
    'rest_advantage',                             # 🆕 Разница в отдыхе
]

print(f"\n🔧 Финалистов: {len(FEATURES_BEST)} признаков")

# Тестируем разные пороги и стейки
configs = [
    ('Порог 70%, Стейк 2%', 0.70, 0.02),
    ('Порог 65%, Стейк 2%', 0.65, 0.02),
    ('Порог 70%, Стейк 1.5%', 0.70, 0.015),
    ('Порог 65%, Стейк 1.5%', 0.65, 0.015),
    ('Порог 60%, Стейк 2%', 0.60, 0.02),
]

print(f"\n{'='*75}")
print(f"{'Конфигурация':<30} {'Ставок':<8} {'W/L':<10} {'Банк':<10} {'ROI':<8} {'Hit Rate':<9}")
print(f"{'-'*75}")

best_roi = -999
best_config = None

for name, threshold, stake_pct in configs:
    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
    model.fit(train[FEATURES_BEST], train['target'])
    probs = model.predict_proba(test[FEATURES_BEST])[:, 1]

    bank = 10000
    bets = wins = 0
    for i, (_, row) in enumerate(test.iterrows()):
        p = probs[i]; total = row['total_goals']; stake = bank * stake_pct
        if p > threshold and row['odds_over'] > 0:
            bets += 1; bank += stake * (row['odds_over'] - 1) if total > 2.5 else -stake
            wins += 1 if total > 2.5 else 0
        elif p < (1 - threshold) and row['odds_under'] > 0:
            bets += 1; bank += stake * (row['odds_under'] - 1) if total < 2.5 else -stake
            wins += 1 if total < 2.5 else 0

    roi = (bank - 10000) / 100
    hit = wins / bets * 100 if bets > 0 else 0
    print(f"{name:<30} {bets:<8} {wins}/{bets-wins:<9} {bank:<10.0f} {roi:<+7.2f}% {hit:<8.1f}%")

    if roi > best_roi:
        best_roi = roi
        best_config = (name, threshold, stake_pct, bets, wins, bank, hit)

print(f"\n🏆 ЛУЧШАЯ: {best_config[0]} → ROI {best_roi:+.2f}%, Банк {best_config[5]:.0f}₽, Ставок {best_config[3]}, Hit {best_config[6]:.1f}%")

# Важность признаков
model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
model.fit(train[FEATURES_BEST], train['target'])
print(f"\n📊 ВАЖНОСТЬ:")
imp = model.feature_importances_
for i, j in enumerate(np.argsort(imp)[::-1][:10]):
    tag = ' 🆕' if FEATURES_BEST[j] in ['home_over_home_pct', 'away_over_away_pct', 'rest_advantage'] else ''
    print(f"   {i+1}. {FEATURES_BEST[j]:<25}: {imp[j]:.4f}{tag}")

print("\n✅ ЭТАП 10 ЗАВЕРШЁН")
