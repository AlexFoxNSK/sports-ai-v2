#!/usr/bin/env python3
"""
ЭТАП 16: МОДЕЛЬ ИНДИВИДУАЛЬНЫХ ТОТАЛОВ (HOME TEAM OVER 1.5)
Обучаем XGBoost предсказывать: забьёт ли домашняя команда 2+ гола?
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("🚀 ЭТАП 16: HOME TEAM OVER 1.5")
print("=" * 55)

# ============================================================
# 1. ЗАГРУЗКА И ПРИЗНАКИ
# ============================================================
print("\n📥 ЗАГРУЗКА")
df = pd.read_csv('data/all_leagues.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.dropna(subset=['FTHG', 'FTAG'])
df = df.sort_values('Date').reset_index(drop=True)
print(f"   {len(df):,} матчей")

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

# --- Голы (только для домашней команды) ---
tg = {}
home_goals_for_5 = []
home_goals_for_3 = []
home_goals_home_5 = []  # голы дома за 5 матчей

for _, row in df.iterrows():
    team = row['HomeTeam']
    gh = tg.get(team, [])
    # Все голы
    home_goals_for_5.append(np.mean(gh[-5:]) if gh else 0)
    home_goals_for_3.append(np.mean(gh[-3:]) if gh else 0)
    gh.append(row['FTHG'])
    tg[team] = gh

df['home_goals_for_5'] = home_goals_for_5
df['home_goals_for_3'] = home_goals_for_3

# --- Противник: сколько пропускает гостевая команда ---
ta = {}
away_goals_against_5 = []
away_goals_against_3 = []

for _, row in df.iterrows():
    team = row['AwayTeam']
    ah = ta.get(team, [])
    away_goals_against_5.append(np.mean(ah[-5:]) if ah else 0)
    away_goals_against_3.append(np.mean(ah[-3:]) if ah else 0)
    ah.append(row['FTHG'])  # сколько пропустили от хозяев
    ta[team] = ah

df['away_goals_against_5'] = away_goals_against_5
df['away_goals_against_3'] = away_goals_against_3

# --- Over 1.5 % для домашней команды ---
team_over_15 = {}
home_over_15_pct = []

for _, row in df.iterrows():
    team = row['HomeTeam']
    hist = team_over_15.get(team, [])
    home_over_15_pct.append(np.mean(hist[-20:]) if hist else 0.5)
    hist.append(1 if row['FTHG'] > 1.5 else 0)
    team_over_15[team] = hist

df['home_over_15_pct'] = home_over_15_pct

# --- Стиль защиты соперника: % матчей где соперник пропускает <2 ---
team_under_15 = {}
away_under_15_pct = []

for _, row in df.iterrows():
    team = row['AwayTeam']
    hist = team_under_15.get(team, [])
    away_under_15_pct.append(np.mean(hist[-20:]) if hist else 0.5)
    hist.append(1 if row['FTHG'] < 2 else 0)  # пропустили <2 от хозяев
    team_under_15[team] = hist

df['away_under_15_pct'] = away_under_15_pct

# --- Отдых ---
last = {}; hr, ar = [], []
for _, row in df.iterrows():
    d = row['Date']
    for team, lst in [(row['HomeTeam'], hr), (row['AwayTeam'], ar)]:
        prev = last.get(team, d - pd.Timedelta(days=7))
        lst.append(min((d - prev).days, 14)); last[team] = d
df['home_rest_days'] = hr; df['away_rest_days'] = ar

# --- Целевая: Home Team Over 1.5 ---
df['target_home_15'] = (df['FTHG'] > 1.5).astype(int)
over_count = df['target_home_15'].sum()
print(f"   Home Over 1.5: {over_count} ({over_count/len(df)*100:.1f}%)")

# ============================================================
# 2. ПРИЗНАКИ ДЛЯ МОДЕЛИ
# ============================================================
FEATURES_TEAM = [
    'home_position', 'away_position',
    'home_goals_for_5', 'home_goals_for_3',
    'away_goals_against_5', 'away_goals_against_3',
    'home_over_15_pct', 'away_under_15_pct',
    'home_rest_days', 'away_rest_days',
]

# ============================================================
# 3. ОБУЧЕНИЕ
# ============================================================
print("\n🧠 ОБУЧЕНИЕ")
split = int(len(df) * 0.8)
train = df.iloc[:split]; test = df.iloc[split:]

model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
model.fit(train[FEATURES_TEAM], train['target_home_15'])

probs = model.predict_proba(test[FEATURES_TEAM])[:, 1]
preds = model.predict(test[FEATURES_TEAM])
ll = log_loss(test['target_home_15'], probs)
acc = accuracy_score(test['target_home_15'], preds)
baseline = 1 - test['target_home_15'].mean()

print(f"   LogLoss: {ll:.4f}")
print(f"   Точность: {acc:.2%} (базовая: {baseline:.2%})")

# ============================================================
# 4. ВАЖНОСТЬ ПРИЗНАКОВ
# ============================================================
print(f"\n📊 ВАЖНОСТЬ ПРИЗНАКОВ:")
imp = model.feature_importances_
for i, j in enumerate(np.argsort(imp)[::-1]):
    print(f"   {i+1}. {FEATURES_TEAM[j]:<25}: {imp[j]:.4f} {'█'*int(imp[j]*50)}")

# ============================================================
# 5. БЫСТРЫЙ БЭКТЕСТ (оценочный)
# ============================================================
print(f"\n💵 ОЦЕНОЧНЫЙ БЭКТЕСТ")

# Берём кэфы из общего тотала (приблизительно)
# Team Over 1.5 обычно ~1.40-1.70
# Используем odds_over из CSV как прокси (грубо)
if 'B365>2.5' in df.columns:
    test_odds = pd.to_numeric(test['B365>2.5'], errors='coerce').fillna(1.7)
else:
    test_odds = 1.7

bank = 10000
bets = wins = 0
for i, (_, row) in enumerate(test.iterrows()):
    p = probs[i]
    total_home = row['FTHG']
    stake = bank * 0.02
    
    # Используем приблизительный кэф для Team Over 1.5
    # Он коррелирует с общим тоталом
    implied_odds = 1 + (test_odds.iloc[i] - 1) * 0.7  # грубая оценка
    
    if p > 0.75:
        bets += 1
        if total_home > 1.5:
            bank += stake * (implied_odds - 1)
            wins += 1
        else:
            bank -= stake

roi = (bank - 10000) / 100
hit = wins / bets * 100 if bets > 0 else 0
print(f"   Ставок: {bets} | Банк: {bank:.0f}₽ | ROI: {roi:+.2f}% | Hit: {hit:.1f}%")

# Сравнение с основной моделью
print(f"\n📊 СРАВНЕНИЕ С ОСНОВНОЙ МОДЕЛЬЮ:")
print(f"   OU 2.5:  88 ставок, ROI +13.45%, Hit 65.9%")
print(f"   Team 1.5: {bets} ставок, ROI {roi:+.2f}%, Hit {hit:.1f}%")
print(f"   ВМЕСТЕ:  {88+bets} ставок")

print("\n✅ ЭТАП 16 ЗАВЕРШЁН")
