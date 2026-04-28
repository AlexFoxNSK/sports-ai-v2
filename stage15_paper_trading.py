#!/usr/bin/env python3
"""
ЭТАП 15: PAPER TRADING
Симуляция реальных ставок с лучшей стратегией:
- ТБ 2.5 при p > 0.75 (калиброванная)
- ТМ 2.5 при p < 0.35 (калиброванная)
- EV > 5%, стейк 2%
- Логирование всех ставок в CSV
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("🚀 ЭТАП 15: PAPER TRADING")
print("=" * 55)

# ============================================================
# ЗАГРУЗКА И ПРИЗНАКИ
# ============================================================
df = pd.read_csv('data/all_leagues.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.dropna(subset=['FTHG', 'FTAG'])
df = df.sort_values('Date').reset_index(drop=True)
df['odds_over'] = pd.to_numeric(df.get('B365>2.5', None), errors='coerce')
df['odds_under'] = pd.to_numeric(df.get('B365<2.5', None), errors='coerce')
df = df.dropna(subset=['odds_over', 'odds_under'])

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
# МОДЕЛЬ
# ============================================================
split = int(len(df) * 0.8)
train = df.iloc[:split]; test = df.iloc[split:]

model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
model.fit(train[FEATURES], train['target'])
probs = model.predict_proba(test[FEATURES])[:, 1]

# ============================================================
# ФУНКЦИЯ КАЛИБРОВКИ
# ============================================================
def calibrate_prob(p):
    if p > 0.70 and p <= 0.75: return p - 0.13
    elif p >= 0.35 and p < 0.40: return p + 0.14
    elif p > 0.65 and p <= 0.70: return p - 0.02
    elif p >= 0.60 and p <= 0.65: return p + 0.06
    return p

# ============================================================
# PAPER TRADING
# ============================================================
BANK_START = 10000
STAKE_PCT = 0.02
TB_THRESH = 0.75
TM_THRESH = 0.35

bank = BANK_START
bets_log = []

for i, (_, row) in enumerate(test.iterrows()):
    p_raw = probs[i]
    p_cal = calibrate_prob(p_raw)
    total = row['total_goals']
    stake = bank * STAKE_PCT

    bet = None
    outcome = None
    pnl = 0
    ev = 0

    # ТБ 2.5
    if p_cal > TB_THRESH and row['odds_over'] > 0:
        ev = p_cal * row['odds_over'] - 1
        if ev > 0.05:
            bet = 'TB2.5'
            outcome = 'W' if total > 2.5 else 'L'
            pnl = stake * (row['odds_over'] - 1) if total > 2.5 else -stake
            bank += pnl

    # ТМ 2.5
    elif p_cal < TM_THRESH and row['odds_under'] > 0:
        ev = (1 - p_cal) * row['odds_under'] - 1
        if ev > 0.05:
            bet = 'TM2.5'
            outcome = 'W' if total < 2.5 else 'L'
            pnl = stake * (row['odds_under'] - 1) if total < 2.5 else -stake
            bank += pnl

    if bet:
        bets_log.append({
            'date': row['Date'].strftime('%Y-%m-%d'),
            'league': row['league'],
            'match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
            'bet': bet,
            'odds': row['odds_over'] if bet == 'TB2.5' else row['odds_under'],
            'prob_raw': round(p_raw, 4),
            'prob_cal': round(p_cal, 4),
            'ev': round(ev * 100, 2),
            'stake': round(stake, 2),
            'outcome': outcome,
            'goals': int(total),
            'pnl': round(pnl, 2),
            'bank': round(bank, 2),
        })

bets_df = pd.DataFrame(bets_log)

# ============================================================
# ОТЧЁТ
# ============================================================
roi = (bank - BANK_START) / BANK_START * 100
total_bets = len(bets_df)
wins = (bets_df['outcome'] == 'W').sum() if total_bets > 0 else 0
hit = wins / total_bets * 100 if total_bets > 0 else 0

print(f"\n{'='*55}")
print(f"📊 ИТОГИ PAPER TRADING")
print(f"{'='*55}")
print(f"   Банк:        {BANK_START:,} → {bank:,.0f}₽")
print(f"   PnL:         {bank - BANK_START:+,.0f}₽")
print(f"   ROI:         {roi:+.2f}%")
print(f"   Ставок:      {total_bets}")
print(f"   Выиграно:    {wins}")
print(f"   Проиграно:   {total_bets - wins}")
print(f"   Проходимость: {hit:.1f}%")
print(f"   Средний EV:  {bets_df['ev'].mean():+.1f}%" if total_bets > 0 else "")

# По месяцам
if total_bets > 0:
    bets_df['month'] = pd.to_datetime(bets_df['date']).dt.month
    print(f"\n📅 ПО МЕСЯЦАМ:")
    for m in sorted(bets_df['month'].unique()):
        sub = bets_df[bets_df['month'] == m]
        w = (sub['outcome'] == 'W').sum()
        print(f"   Месяц {m:>2}: {len(sub):>3} ст | Win: {w}/{len(sub)} | PnL: {sub['pnl'].sum():+,.0f}₽")

# По лигам
    print(f"\n🏆 ПО ЛИГАМ:")
    for league in bets_df['league'].unique():
        sub = bets_df[bets_df['league'] == league]
        w = (sub['outcome'] == 'W').sum()
        print(f"   {league:<25}: {len(sub):>3} ст | Win: {w}/{len(sub)} | PnL: {sub['pnl'].sum():+,.0f}₽")

# Сохраняем лог
log_file = f"paper_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
bets_df.to_csv(log_file, index=False)
print(f"\n📁 Лог сохранён: {log_file}")

print("\n✅ ЭТАП 15 ЗАВЕРШЁН")
