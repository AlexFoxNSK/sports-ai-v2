#!/usr/bin/env python3
"""
ЭТАП 2: ДОБАВЛЯЕМ ПРИЗНАКИ
Новые признаки:
- Форма за 3 матча (а не только 5)
- История личных встреч (H2H)
- Средний тотал команд
- Коэффициенты букмекера
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("🚀 ЭТАП 2: БОЛЬШЕ ПРИЗНАКОВ ДЛЯ ТОТАЛОВ")
print("=" * 55)

# ============================================================
# 1. ЗАГРУЗКА
# ============================================================
print("\n📥 ЗАГРУЗКА ДАННЫХ")
df = pd.read_csv('data/all_leagues.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.dropna(subset=['FTHG', 'FTAG'])
df = df.sort_values('Date').reset_index(drop=True)
print(f"   {len(df):,} матчей, {df['league'].nunique()} лиг")

# ============================================================
# 2. ПРИЗНАКИ
# ============================================================
print("🔧 СОЗДАНИЕ ПРИЗНАКОВ")

# --- Турнирная таблица (позиции) ---
standings = {}
home_positions, away_positions = [], []

for _, row in df.iterrows():
    for team in [row['HomeTeam'], row['AwayTeam']]:
        if team not in standings:
            standings[team] = {'points': 0, 'goals_for': 0, 'goals_against': 0}

    sorted_teams = sorted(standings.items(), key=lambda x: x[1]['points'], reverse=True)
    positions = {team: i+1 for i, (team, _) in enumerate(sorted_teams)}
    home_positions.append(positions[row['HomeTeam']])
    away_positions.append(positions[row['AwayTeam']])

    hg, ag = row['FTHG'], row['FTAG']
    if hg > ag:
        standings[row['HomeTeam']]['points'] += 3
    elif hg == ag:
        standings[row['HomeTeam']]['points'] += 1
        standings[row['AwayTeam']]['points'] += 1
    else:
        standings[row['AwayTeam']]['points'] += 3
    standings[row['HomeTeam']]['goals_for'] += hg
    standings[row['HomeTeam']]['goals_against'] += ag
    standings[row['AwayTeam']]['goals_for'] += ag
    standings[row['AwayTeam']]['goals_against'] += hg

df['home_position'] = home_positions
df['away_position'] = away_positions

# --- Голы за 5 и за 3 матча ---
team_gf, team_ga = {}, {}
home_gf5, home_ga5, home_gf3, home_ga3 = [], [], [], []
away_gf5, away_ga5, away_gf3, away_ga3 = [], [], [], []

for _, row in df.iterrows():
    for team, gf5_list, ga5_list, gf3_list, ga3_list, gf, ga in [
        (row['HomeTeam'], home_gf5, home_ga5, home_gf3, home_ga3, row['FTHG'], row['FTAG']),
        (row['AwayTeam'], away_gf5, away_ga5, away_gf3, away_ga3, row['FTAG'], row['FTHG'])
    ]:
        gf_hist = team_gf.get(team, [])
        ga_hist = team_ga.get(team, [])

        gf5_list.append(np.mean(gf_hist[-5:]) if gf_hist else 0)
        ga5_list.append(np.mean(ga_hist[-5:]) if ga_hist else 0)
        gf3_list.append(np.mean(gf_hist[-3:]) if gf_hist else 0)
        ga3_list.append(np.mean(ga_hist[-3:]) if ga_hist else 0)

        gf_hist.append(gf)
        ga_hist.append(ga)
        team_gf[team] = gf_hist
        team_ga[team] = ga_hist

df['home_goals_for_5'] = home_gf5
df['home_goals_against_5'] = home_ga5
df['home_goals_for_3'] = home_gf3
df['home_goals_against_3'] = home_ga3
df['away_goals_for_5'] = away_gf5
df['away_goals_against_5'] = away_ga5
df['away_goals_for_3'] = away_gf3
df['away_goals_against_3'] = away_ga3

# --- Процент Over 2.5 ---
team_over = {}
home_over_pct, away_over_pct = [], []

for _, row in df.iterrows():
    is_over = 1 if row['FTHG'] + row['FTAG'] > 2.5 else 0
    for team, pct_list in [(row['HomeTeam'], home_over_pct), (row['AwayTeam'], away_over_pct)]:
        hist = team_over.get(team, [])
        pct = np.mean(hist[-20:]) if hist else 0.5
        pct_list.append(pct)
        hist.append(is_over)
        team_over[team] = hist

df['home_over_25_pct'] = home_over_pct
df['away_over_25_pct'] = away_over_pct

# --- Средний тотал команд ---
team_total = {}
home_avg_total, away_avg_total = [], []

for _, row in df.iterrows():
    total = row['FTHG'] + row['FTAG']
    for team, alist in [(row['HomeTeam'], home_avg_total), (row['AwayTeam'], away_avg_total)]:
        hist = team_total.get(team, [])
        alist.append(np.mean(hist[-10:]) if hist else 2.5)
        hist.append(total)
        team_total[team] = hist

df['home_avg_total'] = home_avg_total
df['away_avg_total'] = away_avg_total

# --- Разница в тотале (чья форма "верховее") ---
df['total_form_diff'] = df['home_avg_total'] - df['away_avg_total']

# --- Средний тотал лиги ---
league_goals = {}
league_avg_list = []
for _, row in df.iterrows():
    league = row['league']
    total = row['FTHG'] + row['FTAG']
    if league not in league_goals:
        league_goals[league] = []
    league_avg_list.append(np.mean(league_goals[league]) if league_goals[league] else 2.5)
    league_goals[league].append(total)
df['league_avg_goals'] = league_avg_list

# --- H2H: история личных встреч ---
h2h_history = {}
h2h_over_pct_list = []

for _, row in df.iterrows():
    home = row['HomeTeam']
    away = row['AwayTeam']
    key = tuple(sorted([home, away]))

    hist = h2h_history.get(key, [])
    pct = np.mean(hist[-5:]) if hist else 0.5
    h2h_over_pct_list.append(pct)

    is_over = 1 if row['FTHG'] + row['FTAG'] > 2.5 else 0
    hist.append(is_over)
    h2h_history[key] = hist

df['h2h_over_pct'] = h2h_over_pct_list

# --- Коэффициенты букмекера ---
if 'B365>2.5' in df.columns and 'B365<2.5' in df.columns:
    df['odds_over'] = pd.to_numeric(df['B365>2.5'], errors='coerce')
    df['odds_under'] = pd.to_numeric(df['B365<2.5'], errors='coerce')
else:
    df['odds_over'] = 1.85
    df['odds_under'] = 1.85

# --- Целевая ---
df['target'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)
print(f"   Over: {df['target'].sum()} ({(df['target'].mean()*100):.1f}%)")

# ============================================================
# 3. ОБУЧЕНИЕ
# ============================================================
print("\n🧠 ОБУЧЕНИЕ")

feature_columns = [
    # Старые признаки (9 штук)
    'home_position', 'away_position',
    'home_goals_for_5', 'home_goals_against_5',
    'away_goals_for_5', 'away_goals_against_5',
    'home_over_25_pct', 'away_over_25_pct',
    'league_avg_goals',

    # Новые признаки (10 штук)
    'home_goals_for_3', 'home_goals_against_3',    # Форма за 3 матча
    'away_goals_for_3', 'away_goals_against_3',
    'h2h_over_pct',                                 # История встреч
    'home_avg_total', 'away_avg_total',             # Средний тотал команд
    'total_form_diff',                              # Разница в тотале
    'odds_over', 'odds_under',                      # Кэфы букмекера
]

# Проверяем что все колонки есть
available_features = [c for c in feature_columns if c in df.columns]
print(f"   Признаков: {len(available_features)}")

# Сплит 80/20
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

X_train = train[available_features]
y_train = train['target']
X_test = test[available_features]
y_test = test['target']

print(f"   Train: {len(train):,} | Test: {len(test):,}")
print(f"   Test период: {test['Date'].min().date()} → {test['Date'].max().date()}")

# Обучение
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    random_state=42, verbosity=0, n_jobs=-1
)
model.fit(X_train, y_train)

# Оценка
probs = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)
ll = log_loss(y_test, probs)
acc = accuracy_score(y_test, preds)
baseline = 1 - y_test.mean()

print(f"\n📊 РЕЗУЛЬТАТЫ:")
print(f"   LogLoss: {ll:.4f} (этап 1: 0.6840, улучшение: {0.6840-ll:+.4f})")
print(f"   Точность: {acc:.2%} (базовая: {baseline:.2%})")

# Важность
print(f"\n📊 ВАЖНОСТЬ ПРИЗНАКОВ:")
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]

for i, idx in enumerate(indices):
    name = available_features[idx]
    imp = importance[idx]
    bar = '█' * int(imp * 50)
    marker = ' 🆕' if '3' in name or 'h2h' in name or 'avg_total' in name or 'odds' in name or 'form_diff' in name else ''
    print(f"   {i+1:>2}. {name:<25}: {imp:.4f} {bar}{marker}")

# Сравнение старых vs новых
old_imp = sum(importance[available_features.index(f)] for f in feature_columns[:9] if f in available_features)
new_imp = sum(importance[available_features.index(f)] for f in feature_columns[9:] if f in available_features)
print(f"\n   Важность старых признаков: {old_imp:.2%}")
print(f"   Важность новых признаков: {new_imp:.2%}")

print("\n✅ ЭТАП 2 ЗАВЕРШЁН")
