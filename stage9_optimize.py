#!/usr/bin/env python3
"""
ЭТАП 9: МЕТОДИЧНЫЙ ПОДБОР ПРИЗНАКОВ
Тестируем добавление по 2 признака за запуск.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("🚀 ЭТАП 9: ПОШАГОВОЕ УЛУЧШЕНИЕ ПРИЗНАКОВ")
print("=" * 55)

# ============================================================
# 1. ЗАГРУЗКА
# ============================================================
print("\n📥 ЗАГРУЗКА")
df = pd.read_csv('data/all_leagues.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.dropna(subset=['FTHG', 'FTAG'])
df = df.sort_values('Date').reset_index(drop=True)
df['odds_over'] = pd.to_numeric(df.get('B365>2.5', None), errors='coerce')
df['odds_under'] = pd.to_numeric(df.get('B365<2.5', None), errors='coerce')
df = df.dropna(subset=['odds_over', 'odds_under'])
print(f"   {len(df):,} матчей")

# ============================================================
# 2. ВСЕ ПРИЗНАКИ (создаём один раз)
# ============================================================
print("🔧 Создаём все признаки...")

# --- Позиции + мотивация ---
standings = {}
home_pos, away_pos = [], []
home_pts_to_top4, away_pts_to_top4 = [], []
home_pts_to_releg, away_pts_to_releg = [], []

for _, row in df.iterrows():
    league = row['league']
    for team in [row['HomeTeam'], row['AwayTeam']]:
        if team not in standings:
            standings[team] = {'pts': 0, 'league': league}
    league_teams = {t: s for t, s in standings.items() if s['league'] == league}
    s = sorted(league_teams.items(), key=lambda x: x[1]['pts'], reverse=True)
    p = {t: i+1 for i, (t, _) in enumerate(s)}
    pts = sorted([st['pts'] for st in league_teams.values()], reverse=True)
    hp = p.get(row['HomeTeam'], 10); ap = p.get(row['AwayTeam'], 10)
    home_pos.append(hp); away_pos.append(ap)
    top4 = pts[3] if len(pts) > 3 else 0; releg = pts[16] if len(pts) > 16 else 0
    home_pts_to_top4.append(max(0, top4 - standings[row['HomeTeam']]['pts']))
    away_pts_to_top4.append(max(0, top4 - standings[row['AwayTeam']]['pts']))
    home_pts_to_releg.append(max(0, standings[row['HomeTeam']]['pts'] - releg))
    away_pts_to_releg.append(max(0, standings[row['AwayTeam']]['pts'] - releg))
    hg, ag = row['FTHG'], row['FTAG']
    if hg > ag: standings[row['HomeTeam']]['pts'] += 3
    elif hg == ag: standings[row['HomeTeam']]['pts'] += 1; standings[row['AwayTeam']]['pts'] += 1
    else: standings[row['AwayTeam']]['pts'] += 3

df['home_position'] = home_pos; df['away_position'] = away_pos
df['home_pts_to_top4'] = home_pts_to_top4; df['away_pts_to_top4'] = away_pts_to_top4
df['home_pts_to_releg'] = home_pts_to_releg; df['away_pts_to_releg'] = away_pts_to_releg

# --- Взвешенная форма ---
tg_w, ta_w = {}, {}
hg_w, ha_w, ag_w, aa_w = [], [], [], []
for _, row in df.iterrows():
    for team, gf_l, ga_l, gf, ga in [
        (row['HomeTeam'], hg_w, ha_w, row['FTHG'], row['FTAG']),
        (row['AwayTeam'], ag_w, aa_w, row['FTAG'], row['FTHG'])
    ]:
        gh = tg_w.get(team, []); ah = ta_w.get(team, [])
        if len(gh) >= 5:
            w = [0.5, 0.7, 1.0, 1.5, 2.0]
            gf_l.append(np.average(gh[-5:], weights=w))
            ga_l.append(np.average(ah[-5:], weights=w))
        else:
            gf_l.append(np.mean(gh) if gh else 0)
            ga_l.append(np.mean(ah) if ah else 0)
        gh.append(gf); ah.append(ga)
        tg_w[team] = gh; ta_w[team] = ah
df['home_goals_for_w'] = hg_w; df['home_goals_against_w'] = ha_w
df['away_goals_for_w'] = ag_w; df['away_goals_against_w'] = aa_w

# --- Обычные голы ---
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
to_home, to_away = {}, {}
ho_home, ao_away = [], []
for _, row in df.iterrows():
    ov = 1 if row['FTHG'] + row['FTAG'] > 2.5 else 0
    for team, lst in [(row['HomeTeam'], ho), (row['AwayTeam'], ao)]:
        h = to.get(team, []); lst.append(np.mean(h[-20:]) if h else 0.5); h.append(ov); to[team] = h
    # Дома/в гостях
    hh = to_home.get(row['HomeTeam'], []); ho_home.append(np.mean(hh[-10:]) if hh else 0.5); hh.append(ov); to_home[row['HomeTeam']] = hh
    ah = to_away.get(row['AwayTeam'], []); ao_away.append(np.mean(ah[-10:]) if ah else 0.5); ah.append(ov); to_away[row['AwayTeam']] = ah
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
# 3. ТЕСТИРУЕМ КОМБИНАЦИИ ПРИЗНАКОВ
# ============================================================
split = int(len(df) * 0.8)
train = df.iloc[:split]; test = df.iloc[split:]

# База из Этапа 7 (16 признаков)
BASE = [
    'home_over_25_pct', 'away_position', 'home_position', 'league_avg_goals',
    'home_goals_against_3', 'away_goals_against_5', 'away_over_25_pct',
    'home_goals_for_5', 'h2h_over_pct', 'away_goals_for_3',
    'home_rest_days', 'away_goals_for_5', 'home_goals_against_5',
    'home_goals_for_3', 'away_rest_days', 'away_goals_against_3'
]

# Комбинации для теста
combos = [
    ('База (16)', BASE),
    ('+ Мотивация', BASE + ['home_pts_to_top4', 'away_pts_to_top4', 'home_pts_to_releg', 'away_pts_to_releg']),
    ('+ Over% дома/в гостях', BASE + ['home_over_home_pct', 'away_over_away_pct']),
    ('+ Взвешенная форма', BASE + ['home_goals_for_w', 'home_goals_against_w', 'away_goals_for_w', 'away_goals_against_w']),
    ('+ Rest advantage', BASE + ['rest_advantage']),
]

results = []

for name, features in combos:
    print(f"\n🧪 {name} ({len(features)} признаков)...", end=' ')
    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
    model.fit(train[features], train['target'])
    probs = model.predict_proba(test[features])[:, 1]
    ll = log_loss(test['target'], probs)

    # Считаем ставки
    bank = 10000
    bets = wins = losses = 0
    for i, (_, row) in enumerate(test.iterrows()):
        p = probs[i]; total = row['total_goals']; stake = bank * 0.02
        if p > 0.70 and row['odds_over'] > 0:
            bets += 1; bank += stake * (row['odds_over'] - 1) if total > 2.5 else -stake
            wins += 1 if total > 2.5 else 0; losses += 0 if total > 2.5 else 1
        elif p < 0.30 and row['odds_under'] > 0:
            bets += 1; bank += stake * (row['odds_under'] - 1) if total < 2.5 else -stake
            wins += 1 if total < 2.5 else 0; losses += 0 if total < 2.5 else 1
    roi = (bank - 10000) / 100

    results.append({'name': name, 'features': len(features), 'logloss': ll, 'bets': bets, 'wins': wins, 'losses': losses, 'roi': roi, 'bank': bank})
    print(f"LogLoss: {ll:.4f} | Ставок: {bets} | ROI: {roi:+.2f}% | Банк: {bank:.0f}")

# Итоговая таблица
print(f"\n{'='*70}")
print(f"{'Комбинация':<30} {'Призн':<6} {'LogLoss':<9} {'Ставок':<8} {'ROI':<9} {'Банк':<8}")
print(f"{'-'*70}")
best = min(results, key=lambda r: r['logloss'])
for r in sorted(results, key=lambda r: r['logloss']):
    marker = ' ✅' if r == best else ''
    print(f"{r['name']:<30} {r['features']:<6} {r['logloss']:<9.4f} {r['bets']:<8} {r['roi']:<+8.2f}% {r['bank']:<8.0f}{marker}")

print(f"\n✅ ЭТАП 9 ЗАВЕРШЁН")
