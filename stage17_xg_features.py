#!/usr/bin/env python3
"""
ЭТАП 17: ДОБАВЛЯЕМ xG + GLICKO ПРИЗНАКИ
Мержим sstats_epl.csv с all_leagues.csv, добавляем xG в модель.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("🚀 ЭТАП 17: xG + GLICKO ПРИЗНАКИ")
print("=" * 55)

# ============================================================
# 1. MERGE ДАННЫХ
# ============================================================
print("\n📥 MERGE ДАННЫХ")

# Основные данные
df = pd.read_csv('data/all_leagues.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.dropna(subset=['FTHG', 'FTAG'])
df = df.sort_values('Date').reset_index(drop=True)

# SStats данные
xgd = pd.read_csv('data/sstats_epl.csv')
xgd['date'] = pd.to_datetime(xgd['date'])

# Merge по дате и домашней команде
df = df.merge(
    xgd[['date', 'home', 'homeXg', 'awayXg', 'homeRating', 'awayRating', 'homeWinProb', 'awayWinProb']],
    left_on=['Date', 'HomeTeam'],
    right_on=['date', 'home'],
    how='left'
)

# Удаляем дубликаты колонок от merge
df = df.drop(columns=['date', 'home'], errors='ignore')

# Заполняем пропуски медианой (для xgboost это ок)
for col in ['homeXg', 'awayXg', 'homeRating', 'awayRating', 'homeWinProb', 'awayWinProb']:
    if col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

print(f"   Всего: {len(df):,} матчей")
print(f"   С xG: {df['homeXg'].notna().sum():,}")

# ============================================================
# 2. ВСЕ ПРИЗНАКИ (старые + xG)
# ============================================================
print("\n🔧 ПРИЗНАКИ")

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

# --- 🆕 xG + Glicko ---
df['xg_diff'] = df['homeXg'] - df['awayXg']
df['rating_diff'] = df['homeRating'] - df['awayRating']

df['target'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)
df['total_goals'] = df['FTHG'] + df['FTAG']

# ============================================================
# 3. ДВА НАБОРА ПРИЗНАКОВ (с xG и без)
# ============================================================
FEATURES_NO_XG = [
    'home_over_25_pct', 'away_position', 'home_position', 'league_avg_goals',
    'home_goals_against_3', 'away_goals_against_5', 'away_over_25_pct',
    'home_goals_for_5', 'h2h_over_pct', 'away_goals_for_3',
    'home_rest_days', 'away_goals_for_5', 'home_goals_against_5',
    'home_goals_for_3', 'away_rest_days', 'away_goals_against_3',
    'home_over_home_pct', 'away_over_away_pct'
]

FEATURES_WITH_XG = FEATURES_NO_XG + ['homeXg', 'awayXg', 'xg_diff', 'homeRating', 'awayRating', 'rating_diff', 'homeWinProb']

# ============================================================
# 4. СРАВНЕНИЕ МОДЕЛЕЙ
# ============================================================
split = int(len(df) * 0.8)
train = df.iloc[:split]; test = df.iloc[split:]

print(f"\n{'='*60}")
print(f"{'Модель':<25} {'Признаков':<10} {'LogLoss':<10} {'Точность':<10}")
print(f"{'-'*60}")

for name, features in [('БЕЗ xG', FEATURES_NO_XG), ('С xG + Glicko', FEATURES_WITH_XG)]:
    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
    model.fit(train[features], train['target'])
    probs = model.predict_proba(test[features])[:, 1]
    ll = log_loss(test['target'], probs)
    acc = accuracy_score(test['target'], model.predict(test[features]))
    print(f"{name:<25} {len(features):<10} {ll:<10.4f} {acc:<10.2%}")

# Важность для модели с xG
model_xg = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
model_xg.fit(train[FEATURES_WITH_XG], train['target'])

print(f"\n📊 ВАЖНОСТЬ (топ-10 с xG):")
imp = model_xg.feature_importances_
for i, j in enumerate(np.argsort(imp)[::-1][:10]):
    tag = ' 🆕' if FEATURES_WITH_XG[j] in ['homeXg','awayXg','xg_diff','homeRating','awayRating','rating_diff','homeWinProb'] else ''
    print(f"   {i+1}. {FEATURES_WITH_XG[j]:<25}: {imp[j]:.4f}{tag}")

print("\n✅ ЭТАП 17 ЗАВЕРШЁН")
