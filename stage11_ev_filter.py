#!/usr/bin/env python3
"""
ЭТАП 11: + EV ФИЛЬТР (СТАВКА ТОЛЬКО ПРИ ПЕРЕВЕСЕ > 5%)
- База 16 признаков + Over% дома/в гостях
- Порог уверенности 65%
- Дополнительно: p × odds > 1.05 (EV > 5%)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("🚀 ЭТАП 11: + EV ФИЛЬТР (СТАВКА ТОЛЬКО ПРИ ПЕРЕВЕСЕ > 5%)")
print("=" * 55)

# ============================================================
# ЗАГРУЗКА И ПРИЗНАКИ (18 лучших)
# ============================================================
df = pd.read_csv('data/all_leagues.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.dropna(subset=['FTHG', 'FTAG'])
df = df.sort_values('Date').reset_index(drop=True)
df['odds_over'] = pd.to_numeric(df.get('B365>2.5', None), errors='coerce')
df['odds_under'] = pd.to_numeric(df.get('B365<2.5', None), errors='coerce')
df = df.dropna(subset=['odds_over', 'odds_under'])

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

df['target'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)
df['total_goals'] = df['FTHG'] + df['FTAG']

# ============================================================
# 18 ЛУЧШИХ ПРИЗНАКОВ
# ============================================================
FEATURES = [
    'home_over_25_pct', 'away_position', 'home_position', 'league_avg_goals',
    'home_goals_against_3', 'away_goals_against_5', 'away_over_25_pct',
    'home_goals_for_5', 'h2h_over_pct', 'away_goals_for_3',
    'home_rest_days', 'away_goals_for_5', 'home_goals_against_5',
    'home_goals_for_3', 'away_rest_days', 'away_goals_against_3',
    'home_over_home_pct', 'away_over_away_pct'
]

# ============================================================
# ТЕСТИРУЕМ: С EV-ФИЛЬТРОМ и БЕЗ
# ============================================================
split = int(len(df) * 0.8)
train = df.iloc[:split]; test = df.iloc[split:]

model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
model.fit(train[FEATURES], train['target'])
probs = model.predict_proba(test[FEATURES])[:, 1]
ll = log_loss(test['target'], probs)
print(f"\n🧠 LogLoss: {ll:.4f}")

# Конфигурации для сравнения
configs = [
    ('БЕЗ EV фильтра (порог 65%)', 0.65, 0.02, False),
    ('С EV фильтром (порог 65%, EV>5%)', 0.65, 0.02, True),
    ('БЕЗ EV фильтра (порог 70%)', 0.70, 0.02, False),
    ('С EV фильтром (порог 70%, EV>5%)', 0.70, 0.02, True),
]

print(f"\n{'='*85}")
print(f"{'Конфигурация':<42} {'Ставок':<7} {'Банк':<10} {'ROI':<8} {'Hit':<8} {'Ср. EV':<8}")
print(f"{'-'*85}")

for name, threshold, stake_pct, use_ev in configs:
    bank = 10000
    bets = wins = 0
    total_ev = 0

    for i, (_, row) in enumerate(test.iterrows()):
        p = probs[i]; total = row['total_goals']; stake = bank * stake_pct

        # Проверяем Over
        if p > threshold and row['odds_over'] > 0:
            ev = p * row['odds_over'] - 1  # математическое ожидание
            if not use_ev or ev > 0.05:  # EV фильтр
                bets += 1; total_ev += ev
                bank += stake * (row['odds_over'] - 1) if total > 2.5 else -stake
                wins += 1 if total > 2.5 else 0

        # Проверяем Under
        elif p < (1 - threshold) and row['odds_under'] > 0:
            ev = (1 - p) * row['odds_under'] - 1
            if not use_ev or ev > 0.05:
                bets += 1; total_ev += ev
                bank += stake * (row['odds_under'] - 1) if total < 2.5 else -stake
                wins += 1 if total < 2.5 else 0

    roi = (bank - 10000) / 100
    hit = wins / bets * 100 if bets > 0 else 0
    avg_ev = total_ev / bets * 100 if bets > 0 else 0
    print(f"{name:<42} {bets:<7} {bank:<10.0f} {roi:<+7.2f}% {hit:<7.1f}% {avg_ev:<+7.2f}%")

print("\n✅ ЭТАП 11 ЗАВЕРШЁН")
