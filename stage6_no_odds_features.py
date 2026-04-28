#!/usr/bin/env python3
"""
ЭТАП 6: ЧЕСТНАЯ МОДЕЛЬ БЕЗ КОЭФФИЦИЕНТОВ В ПРИЗНАКАХ
Модель учится ТОЛЬКО на статистике. Кэфы используем для поиска value.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("🚀 ЭТАП 6: ЧЕСТНАЯ МОДЕЛЬ (БЕЗ КЭФОВ В ПРИЗНАКАХ)")
print("=" * 55)

# ============================================================
# 1. ЗАГРУЗКА И ВСЕ ПРИЗНАКИ
# ============================================================
print("\n📥 ЗАГРУЗКА")
df = pd.read_csv('data/all_leagues.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.dropna(subset=['FTHG', 'FTAG'])
df = df.sort_values('Date').reset_index(drop=True)
print(f"   {len(df):,} матчей")

# --- Позиции ---
print("🔧 Признаки...")
standings = {}
home_pos, away_pos = [], []
for _, row in df.iterrows():
    for team in [row['HomeTeam'], row['AwayTeam']]:
        if team not in standings: standings[team] = {'pts': 0}
    s = sorted(standings.items(), key=lambda x: x[1]['pts'], reverse=True)
    pos = {t: i+1 for i, (t, _) in enumerate(s)}
    home_pos.append(pos[row['HomeTeam']]); away_pos.append(pos[row['AwayTeam']])
    hg, ag = row['FTHG'], row['FTAG']
    if hg > ag: standings[row['HomeTeam']]['pts'] += 3
    elif hg == ag: standings[row['HomeTeam']]['pts'] += 1; standings[row['AwayTeam']]['pts'] += 1
    else: standings[row['AwayTeam']]['pts'] += 3
df['home_position'] = home_pos; df['away_position'] = away_pos

# --- Голы 5 и 3 ---
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

# --- Over 2.5 % ---
to = {}; ho, ao = [], []
for _, row in df.iterrows():
    ov = 1 if row['FTHG'] + row['FTAG'] > 2.5 else 0
    for team, lst in [(row['HomeTeam'], ho), (row['AwayTeam'], ao)]:
        h = to.get(team, []); lst.append(np.mean(h[-20:]) if h else 0.5)
        h.append(ov); to[team] = h
df['home_over_25_pct'] = ho; df['away_over_25_pct'] = ao

# --- Лига ---
lg = {}; la = []
for _, row in df.iterrows():
    league = row['league']; t = row['FTHG'] + row['FTAG']
    if league not in lg: lg[league] = []
    la.append(np.mean(lg[league]) if lg[league] else 2.5)
    lg[league].append(t)
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

# --- Целевая ---
df['target'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)

# ============================================================
# 2. ПРИЗНАКИ БЕЗ КЭФОВ
# ============================================================
# ❌ Убрали odds_under и odds_over
FEATURES_NO_ODDS = [
    'h2h_over_pct',
    'home_goals_against_3', 'away_goals_for_5',
    'away_position', 'away_rest_days',
    'home_position', 'home_rest_days',
    'league_avg_goals', 'away_goals_against_3',
    'home_goals_for_5', 'away_over_25_pct',
    'home_goals_for_3', 'home_over_25_pct',
    'home_goals_against_5', 'away_goals_against_5',
    'away_goals_for_3'
]

print(f"   Признаков: {len(FEATURES_NO_ODDS)} (без кэфов)")

# ============================================================
# 3. ОБУЧЕНИЕ
# ============================================================
print("\n🧠 ОБУЧЕНИЕ")
split = int(len(df) * 0.8)
train = df.iloc[:split]; test = df.iloc[split:]

model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
model.fit(train[FEATURES_NO_ODDS], train['target'])

probs = model.predict_proba(test[FEATURES_NO_ODDS])[:, 1]
preds = model.predict(test[FEATURES_NO_ODDS])
ll = log_loss(test['target'], probs)
acc = accuracy_score(test['target'], preds)
print(f"   LogLoss: {ll:.4f} | Точность: {acc:.2%}")

# ============================================================
# 4. СТРАТЕГИЯ СТАВОК
# ============================================================
print("\n💵 СТРАТЕГИЯ СТАВОК")

BANK_START = 10000
STAKE_PCT = 0.02
ODDS_TB25 = 1.90; ODDS_TM25 = 1.90
ODDS_TB15 = 1.30; ODDS_TM35 = 1.40

bank = BANK_START
bets = []
results = []

for idx, row in test.iterrows():
    prob_over = probs[len(results)]
    total_goals = row['FTHG'] + row['FTAG']
    stake = bank * STAKE_PCT
    bet_type = None

    if prob_over > 0.70:
        bet_type = 'TB2.5'
        bank += stake * (ODDS_TB25 - 1) if total_goals > 2.5 else -stake
        results.append('W' if total_goals > 2.5 else 'L')

    elif prob_over < 0.30:
        bet_type = 'TM2.5'
        bank += stake * (ODDS_TM25 - 1) if total_goals < 2.5 else -stake
        results.append('W' if total_goals < 2.5 else 'L')

    elif prob_over > 0.55:
        bet_type = 'TB1.5'
        bank += stake * (ODDS_TB15 - 1) if total_goals > 1.5 else -stake
        results.append('W' if total_goals > 1.5 else 'L')

    elif prob_over < 0.45:
        bet_type = 'TM3.5'
        bank += stake * (ODDS_TM35 - 1) if total_goals < 3.5 else -stake
        results.append('W' if total_goals < 3.5 else 'L')

    else:
        results.append('SKIP')

    bets.append(bet_type)

total_bets = len([b for b in bets if b is not None])
wins = results.count('W'); losses = results.count('L'); skips = results.count('SKIP')
roi = (bank - BANK_START) / BANK_START * 100
hit_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

print(f"\n📊 РЕЗУЛЬТАТЫ:")
print(f"   Банк: {BANK_START:,} → {bank:,.0f}₽ | PnL: {bank - BANK_START:+,.0f}₽")
print(f"   ROI: {roi:+.2f}% | Проходимость: {hit_rate:.1f}%")
print(f"   Ставок: {total_bets} | Выиграно: {wins} | Проиграно: {losses} | Пропущено: {skips}")

print(f"\n📊 ПО ТИПАМ:")
for bt in ['TB2.5','TM2.5','TB1.5','TM3.5']:
    c = bets.count(bt)
    if c > 0: print(f"   {bt}: {c}")

print(f"\n📊 ВАЖНОСТЬ:")
imp = model.feature_importances_
for i, j in enumerate(np.argsort(imp)[::-1][:12]):
    print(f"   {i+1}. {FEATURES_NO_ODDS[j]:<25}: {imp[j]:.4f}")

print("\n✅ ЭТАП 6 ЗАВЕРШЁН")
