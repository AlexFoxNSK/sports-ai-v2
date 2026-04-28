#!/usr/bin/env python3
"""
ЭТАП 4: ОТБОР ЛУЧШИХ ПРИЗНАКОВ
Оставляем только 15 самых важных признаков из Этапов 1-3.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("🚀 ЭТАП 4: ОТБОР ЛУЧШИХ 15 ПРИЗНАКОВ")
print("=" * 55)

# ============================================================
# 1. ЗАГРУЗКА
# ============================================================
print("\n📥 ЗАГРУЗКА")
df = pd.read_csv('data/all_leagues.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.dropna(subset=['FTHG', 'FTAG'])
df = df.sort_values('Date').reset_index(drop=True)
print(f"   {len(df):,} матчей")

# ============================================================
# 2. ПРИЗНАКИ (только лучшие 15)
# ============================================================
print("\n🔧 ТОЛЬКО ЛУЧШИЕ 15 ПРИЗНАКОВ")

# --- Позиции ---
standings = {}
home_pos, away_pos = [], []
for _, row in df.iterrows():
    for team in [row['HomeTeam'], row['AwayTeam']]:
        if team not in standings:
            standings[team] = {'pts': 0}
    sorted_teams = sorted(standings.items(), key=lambda x: x[1]['pts'], reverse=True)
    positions = {t: i+1 for i, (t, _) in enumerate(sorted_teams)}
    home_pos.append(positions[row['HomeTeam']])
    away_pos.append(positions[row['AwayTeam']])
    hg, ag = row['FTHG'], row['FTAG']
    if hg > ag: standings[row['HomeTeam']]['pts'] += 3
    elif hg == ag: standings[row['HomeTeam']]['pts'] += 1; standings[row['AwayTeam']]['pts'] += 1
    else: standings[row['AwayTeam']]['pts'] += 3
df['home_position'] = home_pos
df['away_position'] = away_pos

# --- Голы 5 и 3 ---
team_gf, team_ga = {}, {}
h_gf5, h_ga5, h_gf3, h_ga3 = [], [], [], []
a_gf5, a_ga5, a_gf3, a_ga3 = [], [], [], []
for _, row in df.iterrows():
    for team, gf5, ga5, gf3, ga3, gf, ga in [
        (row['HomeTeam'], h_gf5, h_ga5, h_gf3, h_ga3, row['FTHG'], row['FTAG']),
        (row['AwayTeam'], a_gf5, a_ga5, a_gf3, a_ga3, row['FTAG'], row['FTHG'])
    ]:
        gfh = team_gf.get(team, []); gah = team_ga.get(team, [])
        gf5.append(np.mean(gfh[-5:]) if gfh else 0)
        ga5.append(np.mean(gah[-5:]) if gah else 0)
        gf3.append(np.mean(gfh[-3:]) if gfh else 0)
        ga3.append(np.mean(gah[-3:]) if gah else 0)
        gfh.append(gf); gah.append(ga)
        team_gf[team] = gfh; team_ga[team] = gah

df['home_goals_for_5'] = h_gf5
df['home_goals_against_5'] = h_ga5
df['home_goals_for_3'] = h_gf3
df['home_goals_against_3'] = h_ga3
df['away_goals_for_5'] = a_gf5
df['away_goals_against_5'] = a_ga5
df['away_goals_for_3'] = a_gf3
df['away_goals_against_3'] = a_ga3

# --- Over 2.5 % ---
team_over = {}
h_over, a_over = [], []
for _, row in df.iterrows():
    is_over = 1 if row['FTHG'] + row['FTAG'] > 2.5 else 0
    for team, lst in [(row['HomeTeam'], h_over), (row['AwayTeam'], a_over)]:
        hist = team_over.get(team, [])
        lst.append(np.mean(hist[-20:]) if hist else 0.5)
        hist.append(is_over); team_over[team] = hist
df['home_over_25_pct'] = h_over
df['away_over_25_pct'] = a_over

# --- Лига ---
lg = {}
lg_avg = []
for _, row in df.iterrows():
    league = row['league']; total = row['FTHG'] + row['FTAG']
    if league not in lg: lg[league] = []
    lg_avg.append(np.mean(lg[league]) if lg[league] else 2.5)
    lg[league].append(total)
df['league_avg_goals'] = lg_avg

# --- H2H ---
h2h = {}
h2h_list = []
for _, row in df.iterrows():
    key = tuple(sorted([row['HomeTeam'], row['AwayTeam']]))
    hist = h2h.get(key, [])
    h2h_list.append(np.mean(hist[-5:]) if hist else 0.5)
    hist.append(1 if row['FTHG'] + row['FTAG'] > 2.5 else 0)
    h2h[key] = hist
df['h2h_over_pct'] = h2h_list

# --- Кэфы ---
df['odds_over'] = pd.to_numeric(df.get('B365>2.5', 1.85), errors='coerce')
df['odds_under'] = pd.to_numeric(df.get('B365<2.5', 1.85), errors='coerce')

# --- Отдых ---
last = {}
h_rest, a_rest = [], []
for _, row in df.iterrows():
    d = row['Date']
    for team, lst in [(row['HomeTeam'], h_rest), (row['AwayTeam'], a_rest)]:
        prev = last.get(team, d - pd.Timedelta(days=7))
        lst.append(min((d - prev).days, 14))
        last[team] = d
df['home_rest_days'] = h_rest
df['away_rest_days'] = a_rest

# --- Целевая ---
df['target'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)

# ============================================================
# 3. ТОЛЬКО ЛУЧШИЕ 15 ПРИЗНАКОВ
# ============================================================
BEST_FEATURES = [
    'odds_under',              # 24% важности — самый сильный
    'odds_over',               # 7%
    'home_goals_against_3',    # форма защиты
    'away_goals_for_5',        # форма атаки гостей
    'h2h_over_pct',            # история встреч
    'home_goals_for_3',        # форма атаки
    'away_position',           # позиция гостей
    'away_goals_against_3',    # форма защиты гостей
    'league_avg_goals',        # стиль лиги
    'home_position',           # позиция хозяев
    'away_over_25_pct',        # стиль гостей
    'home_over_25_pct',        # стиль хозяев
    'home_goals_for_5',        # долгосрочная форма атаки
    'home_rest_days',          # отдых хозяев
    'away_rest_days',          # отдых гостей
]

print(f"   Используем {len(BEST_FEATURES)} признаков")

# ============================================================
# 4. ОБУЧЕНИЕ
# ============================================================
print("\n🧠 ОБУЧЕНИЕ")
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]; test = df.iloc[split_idx:]
X_train = train[BEST_FEATURES]; y_train = train['target']
X_test = test[BEST_FEATURES]; y_test = test['target']

print(f"   Train: {len(train):,} | Test: {len(test):,}")

model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)
ll = log_loss(y_test, probs)
acc = accuracy_score(y_test, preds)

print(f"\n📊 РЕЗУЛЬТАТЫ:")
print(f"   LogLoss: {ll:.4f}")
print(f"   Этап 1: 0.6840 | Этап 2: 0.6849 | Этап 3: 0.7123 | Этап 4: {ll:.4f}")
if ll < 0.6840:
    print(f"   🎉 ЛУЧШИЙ РЕЗУЛЬТАТ! Улучшение: {0.6840-ll:+.4f}")
else:
    print(f"   Улучшение относительно Этапа 3: {0.7123-ll:+.4f}")
print(f"   Точность: {acc:.2%}")

print(f"\n📊 ВАЖНОСТЬ ПРИЗНАКОВ:")
imp = model.feature_importances_
idx = np.argsort(imp)[::-1]
for i, j in enumerate(idx):
    name = BEST_FEATURES[j]
    bar = '█' * int(imp[j] * 50)
    print(f"   {i+1:>2}. {name:<25}: {imp[j]:.4f} {bar}")

print("\n✅ ЭТАП 4 ЗАВЕРШЁН")
