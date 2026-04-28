#!/usr/bin/env python3
"""
ЭТАП 7: ЧЕСТНЫЙ БЭКТЕСТ С РЕАЛЬНЫМИ КЭФАМИ
- Только ТБ 2.5 и ТМ 2.5 (кэфы из CSV)
- Фикс. стейк 2% от банка
- Фильтр уверенности: >70% для ставки
- Пропускаем если нет кэфа
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("🚀 ЭТАП 7: ЧЕСТНЫЙ БЭКТЕСТ С РЕАЛЬНЫМИ КЭФАМИ")
print("=" * 55)

# ============================================================
# 1. ЗАГРУЗКА И ПРИЗНАКИ
# ============================================================
print("\n📥 ЗАГРУЗКА")
df = pd.read_csv('data/all_leagues.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.dropna(subset=['FTHG', 'FTAG'])
df = df.sort_values('Date').reset_index(drop=True)

# Кэфы как числа
df['odds_over'] = pd.to_numeric(df.get('B365>2.5', None), errors='coerce')
df['odds_under'] = pd.to_numeric(df.get('B365<2.5', None), errors='coerce')

# Удаляем строки без кэфов (не можем посчитать PnL)
df = df.dropna(subset=['odds_over', 'odds_under'])
print(f"   {len(df):,} матчей с кэфами")

# --- Позиции ---
print("🔧 Признаки...")
st = {}
hp, ap = [], []
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

# --- Over % ---
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
df['total_goals'] = df['FTHG'] + df['FTAG']

# ============================================================
# 2. ОБУЧЕНИЕ
# ============================================================
print("\n🧠 ОБУЧЕНИЕ")

FEATURES = [
    'home_over_25_pct', 'away_position', 'home_position', 'league_avg_goals',
    'home_goals_against_3', 'away_goals_against_5', 'away_over_25_pct',
    'home_goals_for_5', 'h2h_over_pct', 'away_goals_for_3',
    'home_rest_days', 'away_goals_for_5', 'home_goals_against_5',
    'home_goals_for_3', 'away_rest_days', 'away_goals_against_3'
]

split = int(len(df) * 0.8)
train = df.iloc[:split]; test = df.iloc[split:]

model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
model.fit(train[FEATURES], train['target'])

probs = model.predict_proba(test[FEATURES])[:, 1]
ll = log_loss(test['target'], probs)
acc = accuracy_score(test['target'], model.predict(test[FEATURES]))
print(f"   LogLoss: {ll:.4f} | Точность: {acc:.2%}")

# ============================================================
# 3. БЭКТЕСТ С РЕАЛЬНЫМИ КЭФАМИ
# ============================================================
print("\n💵 БЭКТЕСТ С РЕАЛЬНЫМИ КЭФАМИ")

BANK_START = 10000
STAKE_PCT = 0.02

bank = BANK_START
bank_history = [bank]
bets_over = 0
bets_under = 0
wins = 0
losses = 0
skips = 0

for idx, row in test.iterrows():
    prob_over = probs[len(bank_history) - 1]
    total = row['total_goals']
    stake = bank * STAKE_PCT

    bet_made = False

    # ТБ 2.5 — только при уверенности > 70%
    if prob_over > 0.70 and row['odds_over'] > 0:
        bets_over += 1
        bet_made = True
        if total > 2.5:
            bank += stake * (row['odds_over'] - 1)
            wins += 1
        else:
            bank -= stake
            losses += 1

    # ТМ 2.5 — только при уверенности < 30% (Under > 70%)
    elif prob_over < 0.30 and row['odds_under'] > 0:
        bets_under += 1
        bet_made = True
        if total < 2.5:
            bank += stake * (row['odds_under'] - 1)
            wins += 1
        else:
            bank -= stake
            losses += 1

    if not bet_made:
        skips += 1

    bank_history.append(bank)

total_bets = bets_over + bets_under
roi = (bank - BANK_START) / BANK_START * 100
hit_rate = wins / total_bets * 100 if total_bets > 0 else 0

print(f"\n📊 РЕЗУЛЬТАТЫ:")
print(f"   Банк: {BANK_START:,} → {bank:,.0f}₽")
print(f"   PnL:  {bank - BANK_START:+,.0f}₽")
print(f"   ROI:  {roi:+.2f}%")
print(f"   Всего ставок: {total_bets} (ТБ: {bets_over}, ТМ: {bets_under})")
print(f"   Выиграно: {wins} | Проиграно: {losses} | Пропущено: {skips}")
print(f"   Проходимость: {hit_rate:.1f}%")
print(f"   Средний кэф ТБ: {test.loc[probs>0.70,'odds_over'].mean():.2f}" if bets_over > 0 else "")
print(f"   Средний кэф ТМ: {test.loc[probs<0.30,'odds_under'].mean():.2f}" if bets_under > 0 else "")

# Кривая баланса (сокращённо)
print(f"\n📈 КРИВАЯ БАЛАНСА:")
for i in range(0, len(bank_history), max(1, len(bank_history)//10)):
    print(f"   Тур {i}: {bank_history[i]:,.0f}₽")

print("\n✅ ЭТАП 7 ЗАВЕРШЁН")
