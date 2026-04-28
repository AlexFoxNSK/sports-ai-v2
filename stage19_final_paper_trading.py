#!/usr/bin/env python3
"""
🏆 ФИНАЛЬНЫЙ PAPER TRADING — ВСЕ ТРИ МОДЕЛИ
1. Over/Under 2.5 (XGBoost классификатор)
2. Home Team Over 1.5 (на основе xG)
3. Away Team Over 1.5 (на основе xG)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🏆 ФИНАЛЬНЫЙ PAPER TRADING — ТРИ МОДЕЛИ")
print("=" * 60)

# ============================================================
# 1. ЗАГРУЗКА И ВСЕ ПРИЗНАКИ
# ============================================================
print("\n📥 ЗАГРУЗКА")
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

FEATURES = [
    'home_over_25_pct', 'away_position', 'home_position', 'league_avg_goals',
    'home_goals_against_3', 'away_goals_against_5', 'away_over_25_pct',
    'home_goals_for_5', 'h2h_over_pct', 'away_goals_for_3',
    'home_rest_days', 'away_goals_for_5', 'home_goals_against_5',
    'home_goals_for_3', 'away_rest_days', 'away_goals_against_3',
    'home_over_home_pct', 'away_over_away_pct'
]

# ============================================================
# 2. ОБУЧАЕМ ВСЕ МОДЕЛИ
# ============================================================
print("\n🧠 ОБУЧЕНИЕ ВСЕХ МОДЕЛЕЙ")
split = int(len(df) * 0.8)
train = df.iloc[:split]; test = df.iloc[split:]

# Модель 1: OU 2.5
model_ou = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
model_ou.fit(train[FEATURES], train['target'])
probs_ou = model_ou.predict_proba(test[FEATURES])[:, 1]
print(f"   OU 2.5: LogLoss={log_loss(test['target'], probs_ou):.4f}")

# Модель 2-3: xG регрессоры (обучаем на матчах с реальным xG)
train_xg = train[train['has_real_xg'] == 1]
model_home_xg = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
model_home_xg.fit(train_xg[FEATURES], train_xg['homeXg'])
model_away_xg = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
model_away_xg.fit(train_xg[FEATURES], train_xg['awayXg'])

# Предсказываем xG для теста
test_hxg = model_home_xg.predict(test[FEATURES])
test_axg = model_away_xg.predict(test[FEATURES])

# ============================================================
# 3. ФУНКЦИИ
# ============================================================
def calibrate_prob(p):
    if p > 0.70 and p <= 0.75: return p - 0.13
    elif p >= 0.35 and p < 0.40: return p + 0.14
    elif p > 0.65 and p <= 0.70: return p - 0.02
    elif p >= 0.60 and p <= 0.65: return p + 0.06
    return p

def get_team_odds(xg_val, is_home=True):
    if xg_val > 2.5: return 1.12
    elif xg_val > 2.0: return 1.54 if is_home else 1.27
    elif xg_val > 1.5: return 1.77 if is_home else 1.91
    elif xg_val > 1.0: return 2.00 if is_home else 2.51
    elif xg_val > 0.5: return 3.85 if is_home else 5.67
    else: return 6.50

# ============================================================
# 4. PAPER TRADING
# ============================================================
print("\n💵 PAPER TRADING")
BANK_START = 10000
STAKE_PCT = 0.02
TB_THRESH = 0.75
TM_THRESH = 0.35

bank = BANK_START
bets_log = []
stats = {'ou': 0, 'home_team': 0, 'away_team': 0, 'wins': 0, 'losses': 0}

for i, (_, row) in enumerate(test.iterrows()):
    stake = bank * STAKE_PCT
    total = row['total_goals']

    # === МОДЕЛЬ 1: OU 2.5 ===
    p_ou_raw = probs_ou[i]
    p_ou_cal = calibrate_prob(p_ou_raw)

    if p_ou_cal > TB_THRESH and row['odds_over'] > 0:
        ev = p_ou_cal * row['odds_over'] - 1
        if ev > 0.05:
            bet = 'TB2.5'
            outcome = 'W' if total > 2.5 else 'L'
            pnl = stake * (row['odds_over'] - 1) if total > 2.5 else -stake
            bank += pnl
            stats['ou'] += 1
            stats['wins' if outcome == 'W' else 'losses'] += 1
            bets_log.append({'date': row['Date'].strftime('%Y-%m-%d'), 'match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                             'model': 'OU 2.5', 'bet': bet, 'odds': row['odds_over'], 'prob': round(p_ou_cal, 3),
                             'ev': round(ev*100, 1), 'outcome': outcome, 'goals': int(total), 'pnl': round(pnl, 2), 'bank': round(bank, 2)})
            continue

    if p_ou_cal < TM_THRESH and row['odds_under'] > 0:
        ev = (1-p_ou_cal) * row['odds_under'] - 1
        if ev > 0.05:
            bet = 'TM2.5'
            outcome = 'W' if total < 2.5 else 'L'
            pnl = stake * (row['odds_under'] - 1) if total < 2.5 else -stake
            bank += pnl
            stats['ou'] += 1
            stats['wins' if outcome == 'W' else 'losses'] += 1
            bets_log.append({'date': row['Date'].strftime('%Y-%m-%d'), 'match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                             'model': 'OU 2.5', 'bet': bet, 'odds': row['odds_under'], 'prob': round(1-p_ou_cal, 3),
                             'ev': round(ev*100, 1), 'outcome': outcome, 'goals': int(total), 'pnl': round(pnl, 2), 'bank': round(bank, 2)})
            continue

    # === МОДЕЛЬ 2: Home Team Over 1.5 ===
    hxg = test_hxg[i]
    if hxg > 2.0:
        odds = get_team_odds(hxg, True)
        ev = 0.75 * odds - 1
        if ev > 0.05:
            outcome = 'W' if row['FTHG'] > 1.5 else 'L'
            pnl = stake * (odds - 1) if row['FTHG'] > 1.5 else -stake
            bank += pnl
            stats['home_team'] += 1
            stats['wins' if outcome == 'W' else 'losses'] += 1
            bets_log.append({'date': row['Date'].strftime('%Y-%m-%d'), 'match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                             'model': 'Team H1.5', 'bet': 'TB1.5(H)', 'odds': odds, 'prob': 0.75,
                             'ev': round(ev*100, 1), 'outcome': outcome, 'goals': f"{int(row['FTHG'])}", 'pnl': round(pnl, 2), 'bank': round(bank, 2)})
    elif hxg < 1.0:
        odds = 1.80  # ТМ 1.5 кэф ~1.80 для xG<1.0
        ev = 0.78 * odds - 1  # ~78% что будет Under
        if ev > 0.05:
            outcome = 'W' if row['FTHG'] < 1.5 else 'L'
            pnl = stake * (odds - 1) if row['FTHG'] < 1.5 else -stake
            bank += pnl
            stats['home_team'] += 1
            stats['wins' if outcome == 'W' else 'losses'] += 1
            bets_log.append({'date': row['Date'].strftime('%Y-%m-%d'), 'match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                             'model': 'Team H1.5', 'bet': 'TM1.5(H)', 'odds': odds, 'prob': 0.92,
                             'ev': round(ev*100, 1), 'outcome': outcome, 'goals': f"{int(row['FTHG'])}", 'pnl': round(pnl, 2), 'bank': round(bank, 2)})

    # === МОДЕЛЬ 3: Away Team Over 1.5 ===
    axg = test_axg[i]
    if axg > 2.0:
        odds = get_team_odds(axg, False)
        ev = 0.79 * odds - 1
        if ev > 0.05:
            outcome = 'W' if row['FTAG'] > 1.5 else 'L'
            pnl = stake * (odds - 1) if row['FTAG'] > 1.5 else -stake
            bank += pnl
            stats['away_team'] += 1
            stats['wins' if outcome == 'W' else 'losses'] += 1
            bets_log.append({'date': row['Date'].strftime('%Y-%m-%d'), 'match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                             'model': 'Team A1.5', 'bet': 'TB1.5(A)', 'odds': odds, 'prob': 0.79,
                             'ev': round(ev*100, 1), 'outcome': outcome, 'goals': f"{int(row['FTAG'])}", 'pnl': round(pnl, 2), 'bank': round(bank, 2)})
    elif axg < 1.0:
        odds = 1.90  # ТМ 1.5 кэф ~1.90 для xG<1.0
        ev = 0.74 * odds - 1  # ~74% что будет Under
        if ev > 0.05:
            outcome = 'W' if row['FTAG'] < 1.5 else 'L'
            pnl = stake * (odds - 1) if row['FTAG'] < 1.5 else -stake
            bank += pnl
            stats['away_team'] += 1
            stats['wins' if outcome == 'W' else 'losses'] += 1
            bets_log.append({'date': row['Date'].strftime('%Y-%m-%d'), 'match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                             'model': 'Team A1.5', 'bet': 'TM1.5(A)', 'odds': odds, 'prob': 0.88,
                             'ev': round(ev*100, 1), 'outcome': outcome, 'goals': f"{int(row['FTAG'])}", 'pnl': round(pnl, 2), 'bank': round(bank, 2)})

# ============================================================
# 5. ОТЧЁТ
# ============================================================
bets_df = pd.DataFrame(bets_log)
total_bets = len(bets_df)
roi = (bank - BANK_START) / BANK_START * 100
hit = stats['wins'] / total_bets * 100 if total_bets > 0 else 0

print(f"\n{'='*60}")
print(f"🏆 ИТОГИ PAPER TRADING")
print(f"{'='*60}")
print(f"   Банк:        {BANK_START:,} → {bank:,.0f}₽")
print(f"   PnL:         {bank - BANK_START:+,.0f}₽")
print(f"   ROI:         {roi:+.2f}%")
print(f"   Ставок:      {total_bets}")
print(f"   Выиграно:    {stats['wins']}")
print(f"   Проходимость: {hit:.1f}%")
print()
print(f"   По моделям:")
print(f"   OU 2.5:       {stats['ou']} ставок")
print(f"   Team H 1.5:   {stats['home_team']} ставок")
print(f"   Team A 1.5:   {stats['away_team']} ставок")

# Сохраняем
bets_df.to_csv('final_paper_trading.csv', index=False)
print(f"\n📁 Лог: final_paper_trading.csv ({total_bets} ставок)")
print("\n✅ ГОТОВО!")
