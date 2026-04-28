#!/usr/bin/env python3
"""
ЭТАП 3: МОТИВАЦИЯ + ОТДЫХ + ДОП. ПРИЗНАКИ
Новые признаки:
- Мотивация (очки до зоны ЛЧ и зоны вылета)
- Дни отдыха перед матчем
- Плотность календаря (3+ матчей за 7 дней)
- Удары в створ (HST/AST) как proxy xG
- Угловые (HC/AC) — косвенный признак атак
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("🚀 ЭТАП 3: МОТИВАЦИЯ + ОТДЫХ + УДАРЫ")
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
# 2. ВСЕ ПРИЗНАКИ (старые + новые)
# ============================================================
print("🔧 СОЗДАНИЕ ПРИЗНАКОВ")

# --- Турнирная таблица (позиции + мотивация) ---
standings = {}
home_positions, away_positions = [], []
home_pts_to_top4, away_pts_to_top4 = [], []
home_pts_to_releg, away_pts_to_releg = [], []
max_points_per_league = {}  # максимум очков в лиге для нормировки

for _, row in df.iterrows():
    league = row['league']
    for team in [row['HomeTeam'], row['AwayTeam']]:
        if team not in standings:
            standings[team] = {'points': 0, 'goals_for': 0, 'goals_against': 0, 'matches': 0, 'league': league}

    sorted_teams = sorted(standings.items(), key=lambda x: x[1]['points'], reverse=True)
    positions = {team: i+1 for i, (team, _) in enumerate(sorted_teams)}

    # Позиции
    hp = positions[row['HomeTeam']]
    ap = positions[row['AwayTeam']]
    home_positions.append(hp)
    away_positions.append(ap)

    # Мотивация: очки до 4 места (зона ЛЧ) и до 17 (зона вылета)
    # Сортируем по очкам и находим 4-е и 17-е места
    all_points = sorted([s['points'] for s in standings.values() if s['league'] == league], reverse=True)
    top4_cutoff = all_points[3] if len(all_points) > 3 else 0   # очки 4-го места
    releg_cutoff = all_points[16] if len(all_points) > 16 else 0  # очки 17-го места

    home_pts_to_top4.append(max(0, top4_cutoff - standings[row['HomeTeam']]['points']))
    away_pts_to_top4.append(max(0, top4_cutoff - standings[row['AwayTeam']]['points']))
    home_pts_to_releg.append(max(0, standings[row['HomeTeam']]['points'] - releg_cutoff))
    away_pts_to_releg.append(max(0, standings[row['AwayTeam']]['points'] - releg_cutoff))

    # Обновляем таблицу
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
    standings[row['HomeTeam']]['matches'] += 1
    standings[row['AwayTeam']]['matches'] += 1

df['home_position'] = home_positions
df['away_position'] = away_positions
df['home_pts_to_top4'] = home_pts_to_top4
df['away_pts_to_top4'] = away_pts_to_top4
df['home_pts_to_releg'] = home_pts_to_releg
df['away_pts_to_releg'] = away_pts_to_releg

# --- Голы за 5 и 3 матча ---
team_gf, team_ga = {}, {}
home_gf5, home_ga5, home_gf3, home_ga3 = [], [], [], []
away_gf5, away_ga5, away_gf3, away_ga3 = [], [], [], []

for _, row in df.iterrows():
    for team, gf5, ga5, gf3, ga3, gf, ga in [
        (row['HomeTeam'], home_gf5, home_ga5, home_gf3, home_ga3, row['FTHG'], row['FTAG']),
        (row['AwayTeam'], away_gf5, away_ga5, away_gf3, away_ga3, row['FTAG'], row['FTHG'])
    ]:
        gf_hist = team_gf.get(team, [])
        ga_hist = team_ga.get(team, [])
        gf5.append(np.mean(gf_hist[-5:]) if gf_hist else 0)
        ga5.append(np.mean(ga_hist[-5:]) if ga_hist else 0)
        gf3.append(np.mean(gf_hist[-3:]) if gf_hist else 0)
        ga3.append(np.mean(ga_hist[-3:]) if ga_hist else 0)
        gf_hist.append(gf); ga_hist.append(ga)
        team_gf[team] = gf_hist; team_ga[team] = ga_hist

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
    for team, pct in [(row['HomeTeam'], home_over_pct), (row['AwayTeam'], away_over_pct)]:
        hist = team_over.get(team, [])
        pct.append(np.mean(hist[-20:]) if hist else 0.5)
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
df['total_form_diff'] = df['home_avg_total'] - df['away_avg_total']

# --- Средний тотал лиги ---
league_goals = {}
league_avg_list = []
for _, row in df.iterrows():
    league = row['league']
    total = row['FTHG'] + row['FTAG']
    if league not in league_goals: league_goals[league] = []
    league_avg_list.append(np.mean(league_goals[league]) if league_goals[league] else 2.5)
    league_goals[league].append(total)
df['league_avg_goals'] = league_avg_list

# --- H2H ---
h2h_history = {}
h2h_over_pct_list = []
for _, row in df.iterrows():
    key = tuple(sorted([row['HomeTeam'], row['AwayTeam']]))
    hist = h2h_history.get(key, [])
    h2h_over_pct_list.append(np.mean(hist[-5:]) if hist else 0.5)
    hist.append(1 if row['FTHG'] + row['FTAG'] > 2.5 else 0)
    h2h_history[key] = hist
df['h2h_over_pct'] = h2h_over_pct_list

# --- Коэффициенты ---
if 'B365>2.5' in df.columns:
    df['odds_over'] = pd.to_numeric(df['B365>2.5'], errors='coerce')
    df['odds_under'] = pd.to_numeric(df['B365<2.5'], errors='coerce')
else:
    df['odds_over'] = 1.85
    df['odds_under'] = 1.85

# --- 🆕 ДНИ ОТДЫХА ---
last_match = {}
home_rest, away_rest = [], []
for _, row in df.iterrows():
    date = row['Date']
    for team, rest_list in [(row['HomeTeam'], home_rest), (row['AwayTeam'], away_rest)]:
        last = last_match.get(team, date - pd.Timedelta(days=7))
        days = min((date - last).days, 14)  # кэп на 14 дней
        rest_list.append(days)
        last_match[team] = date
df['home_rest_days'] = home_rest
df['away_rest_days'] = away_rest
df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']

# --- 🆕 ПЛОТНОСТЬ КАЛЕНДАРЯ ---
team_dates = {}
home_busy, away_busy = [], []
for _, row in df.iterrows():
    date = row['Date']
    for team, busy_list in [(row['HomeTeam'], home_busy), (row['AwayTeam'], away_busy)]:
        dates = team_dates.get(team, [])
        # Сколько матчей за последние 7 дней
        count = sum(1 for d in dates if (date - d).days <= 7)
        busy_list.append(count)
        dates.append(date)
        team_dates[team] = dates
df['home_busy_7d'] = home_busy
df['away_busy_7d'] = away_busy

# --- 🆕 УДАРЫ В СТВОР (proxy xG) ---
if 'HST' in df.columns and 'AST' in df.columns:
    df['HST'] = pd.to_numeric(df['HST'], errors='coerce').fillna(4)
    df['AST'] = pd.to_numeric(df['AST'], errors='coerce').fillna(3)
else:
    df['HST'] = 4
    df['AST'] = 3

team_hst, team_ast = {}, {}
home_hst_5, away_ast_5 = [], []
for _, row in df.iterrows():
    for team, hst_list, ast_list, hst_val, ast_val in [
        (row['HomeTeam'], home_hst_5, away_ast_5, row['HST'], row['AST']),
    ]:
        hst_hist = team_hst.get(team, [])
        ast_hist = team_ast.get(team, [])
        home_hst_5.append(np.mean(hst_hist[-5:]) if hst_hist else 4)
        # Для гостей берём их удары на выезде (используем AST)
        hst_hist.append(hst_val)
        team_hst[team] = hst_hist

    for team, ast_list, hst_list, ast_val, hst_val in [
        (row['AwayTeam'], away_ast_5, home_hst_5, row['AST'], row['HST']),
    ]:
        ast_hist = team_ast.get(team, [])
        away_ast_5.append(np.mean(ast_hist[-5:]) if ast_hist else 3)
        ast_hist.append(ast_val)
        team_ast[team] = ast_hist

# Упростим: используем среднее для всех
df['home_shots_on_target_5'] = home_hst_5
df['away_shots_on_target_5'] = away_ast_5[:len(home_hst_5)]  # обрежем до нужной длины

# Убедимся что длины совпадают
min_len = min(len(df), len(home_hst_5), len(away_ast_5))
df = df.iloc[:min_len]
df['home_shots_on_target_5'] = home_hst_5[:min_len]
df['away_shots_on_target_5'] = away_ast_5[:min_len]

# --- 🆕 УГЛОВЫЕ ---
if 'HC' in df.columns and 'AC' in df.columns:
    df['HC'] = pd.to_numeric(df['HC'], errors='coerce').fillna(5)
    df['AC'] = pd.to_numeric(df['AC'], errors='coerce').fillna(4)
else:
    df['HC'] = 5
    df['AC'] = 4

team_hc, team_ac = {}, {}
home_corners_5, away_corners_5 = [], []
for _, row in df.iterrows():
    for team, c_list, c_val in [(row['HomeTeam'], home_corners_5, row['HC'])]:
        hist = team_hc.get(team, [])
        c_list.append(np.mean(hist[-5:]) if hist else 5)
        hist.append(c_val); team_hc[team] = hist
    for team, c_list, c_val in [(row['AwayTeam'], away_corners_5, row['AC'])]:
        hist = team_ac.get(team, [])
        c_list.append(np.mean(hist[-5:]) if hist else 4)
        hist.append(c_val); team_ac[team] = hist

min_len = min(len(df), len(home_corners_5), len(away_corners_5))
df = df.iloc[:min_len]
df['home_corners_5'] = home_corners_5[:min_len]
df['away_corners_5'] = away_corners_5[:min_len]

# --- Целевая ---
df['target'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)
print(f"   Over: {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")

# ============================================================
# 3. ОБУЧЕНИЕ
# ============================================================
print("\n🧠 ОБУЧЕНИЕ")

feature_columns = [
    # Старые
    'home_position', 'away_position',
    'home_goals_for_5', 'home_goals_against_5',
    'away_goals_for_5', 'away_goals_against_5',
    'home_over_25_pct', 'away_over_25_pct',
    'league_avg_goals',
    'home_goals_for_3', 'home_goals_against_3',
    'away_goals_for_3', 'away_goals_against_3',
    'h2h_over_pct',
    'home_avg_total', 'away_avg_total', 'total_form_diff',
    'odds_over', 'odds_under',
    # 🆕 Новые
    'home_pts_to_top4', 'away_pts_to_top4',          # Мотивация
    'home_pts_to_releg', 'away_pts_to_releg',
    'home_rest_days', 'away_rest_days',              # Отдых
    'rest_advantage',
    'home_busy_7d', 'away_busy_7d',                  # Плотность календаря
    'home_shots_on_target_5', 'away_shots_on_target_5',  # Удары в створ
    'home_corners_5', 'away_corners_5',              # Угловые
]

available = [c for c in feature_columns if c in df.columns]
print(f"   Признаков: {len(available)}")

split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]
X_train = train[available]; y_train = train['target']
X_test = test[available]; y_test = test['target']

print(f"   Train: {len(train):,} | Test: {len(test):,}")

model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)
ll = log_loss(y_test, probs)
acc = accuracy_score(y_test, preds)
baseline = 1 - y_test.mean()

print(f"\n📊 РЕЗУЛЬТАТЫ:")
print(f"   LogLoss: {ll:.4f} (этап 1: 0.6840, этап 2: 0.6849)")
print(f"   Точность: {acc:.2%} (базовая: {baseline:.2%})")

print(f"\n📊 ВАЖНОСТЬ ПРИЗНАКОВ:")
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]
for i, idx in enumerate(indices[:20]):
    name = available[idx]
    imp = importance[idx]
    bar = '█' * int(imp * 50)
    tag = ' 🆕' if any(k in name for k in ['pts_to','rest','busy','shots','corners']) else ''
    print(f"   {i+1:>2}. {name:<28}: {imp:.4f} {bar}{tag}")

print("\n✅ ЭТАП 3 ЗАВЕРШЁН")
