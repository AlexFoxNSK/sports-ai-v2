#!/usr/bin/env python3
"""
ЭТАП 20: СРАВНЕНИЕ СТРАТЕГИЙ СТАВОК
- Одиночные
- Экспрессы (2-3-4 события)
- Системы (2/3, 3/4)
- Добавлены ТБ/ТМ 0.5 и Team Totals 0.5
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from itertools import combinations
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🚀 ЭТАП 20: СРАВНЕНИЕ СТРАТЕГИЙ СТАВОК")
print("=" * 60)

# ============================================================
# ЗАГРУЗКА И МОДЕЛИ (КОРОТКО)
# ============================================================
df = pd.read_csv('data/all_leagues.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.dropna(subset=['FTHG', 'FTAG'])
df = df.sort_values('Date').reset_index(drop=True)
df['odds_over'] = pd.to_numeric(df.get('B365>2.5', None), errors='coerce')
df['odds_under'] = pd.to_numeric(df.get('B365<2.5', None), errors='coerce')
df = df.dropna(subset=['odds_over', 'odds_under'])

# Merge xG
xgd = pd.read_csv('data/sstats_epl.csv')
xgd['date'] = pd.to_datetime(xgd['date'])
df = df.merge(xgd[['date','home','homeXg','awayXg']], left_on=['Date','HomeTeam'], right_on=['date','home'], how='left')
df = df.drop(columns=['date','home'], errors='ignore')
df['has_real_xg'] = df['homeXg'].notna().astype(int)

# --- Все признаки (сокращённо) ---
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

lg = {}; la = []
for _, row in df.iterrows():
    league = row['league']; t = row['FTHG'] + row['FTAG']
    if league not in lg: lg[league] = []
    la.append(np.mean(lg[league]) if lg[league] else 2.5); lg[league].append(t)
df['league_avg_goals'] = la

h2h = {}; hl = []
for _, row in df.iterrows():
    key = tuple(sorted([row['HomeTeam'], row['AwayTeam']]))
    h = h2h.get(key, []); hl.append(np.mean(h[-5:]) if h else 0.5)
    h.append(1 if row['FTHG'] + row['FTAG'] > 2.5 else 0); h2h[key] = h
df['h2h_over_pct'] = hl

last = {}; hr, ar = [], []
for _, row in df.iterrows():
    d = row['Date']
    for team, lst in [(row['HomeTeam'], hr), (row['AwayTeam'], ar)]:
        prev = last.get(team, d - pd.Timedelta(days=7))
        lst.append(min((d - prev).days, 14)); last[team] = d
df['home_rest_days'] = hr; df['away_rest_days'] = ar

df['target'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)
df['total_goals'] = df['FTHG'] + df['FTAG']

FEATURES = [
    'home_over_25_pct', 'away_position', 'home_position', 'league_avg_goals',
    'home_goals_against_3', 'away_goals_against_5', 'away_over_25_pct',
    'home_goals_for_5', 'h2h_over_pct', 'away_goals_for_3',
    'home_rest_days', 'away_goals_for_5', 'home_goals_against_5',
    'home_goals_for_3', 'away_rest_days', 'away_goals_against_3',
    'home_over_home_pct', 'away_over_away_pct'
]

# ============================================================
# ОБУЧЕНИЕ
# ============================================================
split = int(len(df) * 0.8)
train = df.iloc[:split]; test = df.iloc[split:]

model_ou = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
model_ou.fit(train[FEATURES], train['target'])
probs_ou = model_ou.predict_proba(test[FEATURES])[:, 1]

# xG регрессоры
train_xg = train[train['has_real_xg'] == 1]
model_hxg = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
model_hxg.fit(train_xg[FEATURES], train_xg['homeXg'])
model_axg = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
model_axg.fit(train_xg[FEATURES], train_xg['awayXg'])
test_hxg = model_hxg.predict(test[FEATURES])
test_axg = model_axg.predict(test[FEATURES])

def calibrate(p):
    if p > 0.70 and p <= 0.75: return p - 0.13
    elif p >= 0.35 and p < 0.40: return p + 0.14
    elif p > 0.65 and p <= 0.70: return p - 0.02
    elif p >= 0.60 and p <= 0.65: return p + 0.06
    return p

# ============================================================
# ГЕНЕРАЦИЯ СИГНАЛОВ ДЛЯ КАЖДОГО МАТЧА
# ============================================================
signals = []  # список всех сигналов: (дата, описание, кэф, вероятность, исход)

for i, (_, row) in enumerate(test.iterrows()):
    date = row['Date']
    total = row['total_goals']
    hg = row['FTHG']
    ag = row['FTAG']
    hxg = test_hxg[i]
    axg = test_axg[i]
    p_ou = calibrate(probs_ou[i])

    # OU 2.5
    if p_ou > 0.75 and row['odds_over'] > 0:
        ev = p_ou * row['odds_over'] - 1
        if ev > 0.05:
            outcome = 1 if total > 2.5 else 0
            signals.append({'date': date, 'desc': f"{row['HomeTeam']} vs {row['AwayTeam']} TB2.5",
                           'odds': row['odds_over'], 'prob': p_ou, 'outcome': outcome})
    if p_ou < 0.35 and row['odds_under'] > 0:
        ev = (1-p_ou) * row['odds_under'] - 1
        if ev > 0.05:
            outcome = 1 if total < 2.5 else 0
            signals.append({'date': date, 'desc': f"{row['HomeTeam']} vs {row['AwayTeam']} TM2.5",
                           'odds': row['odds_under'], 'prob': 1-p_ou, 'outcome': outcome})

    # Home Team 1.5
    if hxg > 2.0:
        odds = 1.54 if hxg > 2.5 else 1.77
        outcome = 1 if hg > 1.5 else 0
        signals.append({'date': date, 'desc': f"{row['HomeTeam']} TB1.5(H)", 'odds': odds, 'prob': 0.75, 'outcome': outcome})
    elif hxg < 1.0:
        odds = 1.80
        outcome = 1 if hg < 1.5 else 0
        signals.append({'date': date, 'desc': f"{row['HomeTeam']} TM1.5(H)", 'odds': odds, 'prob': 0.78, 'outcome': outcome})

    # Home Team 0.5
    if hxg > 1.0:
        odds = 1.25
        outcome = 1 if hg > 0.5 else 0
        signals.append({'date': date, 'desc': f"{row['HomeTeam']} TB0.5(H)", 'odds': odds, 'prob': 0.82, 'outcome': outcome})
    elif hxg < 0.5:
        odds = 3.50
        outcome = 1 if hg < 0.5 else 0
        signals.append({'date': date, 'desc': f"{row['HomeTeam']} TM0.5(H)", 'odds': odds, 'prob': 0.90, 'outcome': outcome})

    # Away Team 1.5
    if axg > 2.0:
        odds = 1.27 if axg > 2.5 else 1.91
        outcome = 1 if ag > 1.5 else 0
        signals.append({'date': date, 'desc': f"{row['AwayTeam']} TB1.5(A)", 'odds': odds, 'prob': 0.79, 'outcome': outcome})
    elif axg < 1.0:
        odds = 1.90
        outcome = 1 if ag < 1.5 else 0
        signals.append({'date': date, 'desc': f"{row['AwayTeam']} TM1.5(A)", 'odds': odds, 'prob': 0.74, 'outcome': outcome})

    # Away Team 0.5
    if axg > 1.0:
        odds = 1.30
        outcome = 1 if ag > 0.5 else 0
        signals.append({'date': date, 'desc': f"{row['AwayTeam']} TB0.5(A)", 'odds': odds, 'prob': 0.78, 'outcome': outcome})
    elif axg < 0.5:
        odds = 3.00
        outcome = 1 if ag < 0.5 else 0
        signals.append({'date': date, 'desc': f"{row['AwayTeam']} TM0.5(A)", 'odds': odds, 'prob': 0.88, 'outcome': outcome})

print(f"\n📊 Всего сигналов: {len(signals)}")

# ============================================================
# ТЕСТИРУЕМ СТРАТЕГИИ
# ============================================================
BANK = 10000
STAKE_PCT = 0.02
results = {}

# --- СТРАТЕГИЯ 1: Одиночные ---
bank = BANK
bets = wins = 0
for s in signals:
    stake = bank * STAKE_PCT
    if s['outcome'] == 1:
        bank += stake * (s['odds'] - 1)
        wins += 1
    else:
        bank -= stake
    bets += 1
roi = (bank - BANK) / BANK * 100
results['Одиночные'] = {'bets': bets, 'bank': bank, 'roi': roi, 'hit': wins/bets*100}

# --- СТРАТЕГИЯ 2: Экспрессы по 2 (из одного матча) ---
bank = BANK
bets = wins = 0
by_date = {}
for s in signals:
    d = str(s['date'])[:10]
    if d not in by_date: by_date[d] = []
    by_date[d].append(s)

for date, day_signals in by_date.items():
    if len(day_signals) >= 2:
        for combo in combinations(day_signals, 2):
            stake = bank * 0.01  # 1% на экспресс
            total_odds = combo[0]['odds'] * combo[1]['odds']
            if combo[0]['outcome'] == 1 and combo[1]['outcome'] == 1:
                bank += stake * (total_odds - 1)
                wins += 1
            else:
                bank -= stake
            bets += 1
if bets > 0:
    roi = (bank - BANK) / BANK * 100
    results['Экспрессы x2'] = {'bets': bets, 'bank': bank, 'roi': roi, 'hit': wins/bets*100}

# --- СТРАТЕГИЯ 3: Система 2/3 ---
bank = BANK
bets = wins = 0
for date, day_signals in by_date.items():
    if len(day_signals) >= 3:
        for combo in combinations(day_signals, 3):
            stake_per = bank * 0.005  # 0.5% на каждый экспресс внутри системы
            total_stake = stake_per * 3  # 3 экспресса по 2
            win_count = sum(s['outcome'] for s in combo)
            if win_count >= 2:
                # Считаем выигрыш по экспрессам
                won_combos = list(combinations(combo, 2))
                payout = sum(stake_per * (c[0]['odds'] * c[1]['odds']) for c in won_combos if c[0]['outcome']==1 and c[1]['outcome']==1)
                bank += payout - total_stake
                if payout > total_stake: wins += 1
            else:
                bank -= total_stake
            bets += 1
if bets > 0:
    roi = (bank - BANK) / BANK * 100
    results['Система 2/3'] = {'bets': bets, 'bank': bank, 'roi': roi, 'hit': wins/bets*100}

# ============================================================
# ВЫВОД
# ============================================================
print(f"\n{'='*70}")
print(f"{'Стратегия':<20} {'Ставок':<8} {'Банк':<12} {'ROI':<10} {'Hit Rate':<10}")
print(f"{'-'*70}")
for name, r in results.items():
    print(f"{name:<20} {r['bets']:<8} {r['bank']:<12,.0f} {r['roi']:<+9.2f}% {r['hit']:<9.1f}%")

print(f"\n✅ ЭТАП 20 ЗАВЕРШЁН")
