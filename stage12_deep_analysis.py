#!/usr/bin/env python3
"""
ЭТАП 12: ДЕТАЛЬНЫЙ АНАЛИЗ СТАВОК
Лучшая конфигурация: 18 признаков + порог 65% + EV > 5%
Анализ по лигам, месяцам, кэфам, просадкам.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("🚀 ЭТАП 12: ДЕТАЛЬНЫЙ АНАЛИЗ СТАВОК")
print("=" * 65)

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

# [ВСЕ ПРИЗНАКИ — СОКРАЩЁННО]
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

# ============================================================
# МОДЕЛЬ + БЭКТЕСТ
# ============================================================
FEATURES = [
    'home_over_25_pct', 'away_position', 'home_position', 'league_avg_goals',
    'home_goals_against_3', 'away_goals_against_5', 'away_over_25_pct',
    'home_goals_for_5', 'h2h_over_pct', 'away_goals_for_3',
    'home_rest_days', 'away_goals_for_5', 'home_goals_against_5',
    'home_goals_for_3', 'away_rest_days', 'away_goals_against_3',
    'home_over_home_pct', 'away_over_away_pct'
]

split = int(len(df) * 0.8)
train = df.iloc[:split]; test = df.iloc[split:]

model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
model.fit(train[FEATURES], train['target'])
probs = model.predict_proba(test[FEATURES])[:, 1]

# Бэктест с лучшей конфигурацией (порог 65%, EV > 5%)
bank = 10000
bets_data = []  # список словарей для анализа

for i, (_, row) in enumerate(test.iterrows()):
    p = probs[i]; total = row['total_goals']; stake = bank * 0.02

    bet_type = None
    if p > 0.65 and row['odds_over'] > 0:
        ev = p * row['odds_over'] - 1
        if ev > 0.05:
            bet_type = 'Over'
            bank += stake * (row['odds_over'] - 1) if total > 2.5 else -stake
            bets_data.append({
                'date': row['Date'], 'league': row['league'],
                'home': row['HomeTeam'], 'away': row['AwayTeam'],
                'bet': 'TB2.5', 'odds': row['odds_over'],
                'prob': p, 'ev': ev, 'result': 'W' if total > 2.5 else 'L',
                'stake': stake, 'pnl': stake * (row['odds_over'] - 1) if total > 2.5 else -stake,
                'total_goals': total, 'bank': bank
            })

    elif p < 0.35 and row['odds_under'] > 0:
        ev = (1 - p) * row['odds_under'] - 1
        if ev > 0.05:
            bet_type = 'Under'
            bank += stake * (row['odds_under'] - 1) if total < 2.5 else -stake
            bets_data.append({
                'date': row['Date'], 'league': row['league'],
                'home': row['HomeTeam'], 'away': row['AwayTeam'],
                'bet': 'TM2.5', 'odds': row['odds_under'],
                'prob': 1 - p, 'ev': ev, 'result': 'W' if total < 2.5 else 'L',
                'stake': stake, 'pnl': stake * (row['odds_under'] - 1) if total < 2.5 else -stake,
                'total_goals': total, 'bank': bank
            })

bets_df = pd.DataFrame(bets_data)
if len(bets_df) == 0:
    print("Нет ставок для анализа!")
    exit()

roi = (bank - 10000) / 100
hit = (bets_df['result'] == 'W').mean() * 100

print(f"\n📊 ОБЩИЕ РЕЗУЛЬТАТЫ")
print(f"   Банк: 10 000 → {bank:,.0f}₽ | ROI: {roi:+.2f}%")
print(f"   Ставок: {len(bets_df)} | Проходимость: {hit:.1f}%")
print(f"   ТБ 2.5: {len(bets_df[bets_df['bet']=='TB2.5'])} | ТМ 2.5: {len(bets_df[bets_df['bet']=='TM2.5'])}")

# По лигам
print(f"\n📊 ПО ЛИГАМ:")
for league in bets_df['league'].unique():
    sub = bets_df[bets_df['league'] == league]
    w = (sub['result'] == 'W').sum()
    pnl = sub['pnl'].sum()
    stake_sum = sub['stake'].sum()
    print(f"   {league:<25}: {len(sub):>3} ст | Hit: {w/len(sub)*100:5.1f}% | PnL: {pnl:>+8.0f}₽ | ROI: {pnl/stake_sum*100:>+6.2f}%")

# По месяцам
print(f"\n📊 ПО МЕСЯЦАМ:")
bets_df['month'] = bets_df['date'].dt.month
for m in sorted(bets_df['month'].unique()):
    sub = bets_df[bets_df['month'] == m]
    w = (sub['result'] == 'W').sum()
    pnl = sub['pnl'].sum()
    print(f"   Месяц {m:>2}: {len(sub):>3} ст | Hit: {w/len(sub)*100:5.1f}% | PnL: {pnl:>+8.0f}₽")

# По кэфам
print(f"\n📊 ПО КЭФАМ:")
for lo, hi in [(1.3, 1.5), (1.5, 1.7), (1.7, 2.0), (2.0, 2.5), (2.5, 4.0)]:
    sub = bets_df[(bets_df['odds'] >= lo) & (bets_df['odds'] < hi)]
    if len(sub) > 0:
        w = (sub['result'] == 'W').sum()
        pnl = sub['pnl'].sum()
        print(f"   {lo:.1f}-{hi:.1f}: {len(sub):>3} ст | Hit: {w/len(sub)*100:5.1f}% | PnL: {pnl:>+8.0f}₽")

# Просадки
print(f"\n📊 ПРОСАДКИ:")
bets_df['cum_pnl'] = bets_df['pnl'].cumsum()
bets_df['peak'] = bets_df['cum_pnl'].cummax()
bets_df['drawdown'] = bets_df['cum_pnl'] - bets_df['peak']
max_dd = bets_df['drawdown'].min()
max_dd_idx = bets_df['drawdown'].idxmin()
print(f"   Макс. просадка: {max_dd:,.0f}₽ (после ставки #{max_dd_idx})")
print(f"   Текущий PnL: {bets_df['cum_pnl'].iloc[-1]:,.0f}₽")
sharpe = bets_df['pnl'].mean() / bets_df['pnl'].std() * np.sqrt(252) if bets_df['pnl'].std() > 0 else 0
print(f"   Sharpe Ratio: {sharpe:.2f}")

print("\n✅ ЭТАП 12 ЗАВЕРШЁН")
