#!/usr/bin/env python3
import pandas as pd
from flask import Flask, render_template_string, request
from datetime import datetime
import math
app = Flask(__name__)

HTML = '''<!DOCTYPE html>
<html lang="ru"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>⚽ Sports AI</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;500;700&display=swap');
:root{--bg:#06060f;--card:rgba(10,10,35,0.9);--neon:#00d4ff;--purple:#a855f7;--pink:#ec4899;--text:#e2e8f0;--green:#22c55e;--red:#ef4444;--gold:#fbbf24}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Rajdhani',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;
  background:radial-gradient(ellipse at 50% 0%,#0a0a30 0%,#020210 60%),url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><line x1="0" y1="50" x2="100" y2="50" stroke="%23111" stroke-width="0.3"/><line x1="50" y1="0" x2="50" y2="100" stroke="%23111" stroke-width="0.3"/></svg>');
  padding:10px}
#particles{position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0}
.header{position:relative;z-index:1;display:flex;justify-content:space-between;align-items:center;padding:14px 24px;
  background:var(--card);backdrop-filter:blur(30px);-webkit-backdrop-filter:blur(30px);
  border:1px solid rgba(168,85,247,0.2);border-radius:16px;margin-bottom:12px;
  box-shadow:0 0 40px rgba(168,85,247,0.1),inset 0 1px 0 rgba(255,255,255,0.05)}
.header h1{font-family:'Orbitron',sans-serif;font-size:1.3em;font-weight:900;letter-spacing:3px;
  background:linear-gradient(135deg,var(--neon),var(--purple),var(--pink));-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  text-shadow:0 0 30px rgba(0,212,255,0.5)}
.header span{color:#64748b;font-family:'Orbitron',sans-serif;font-size:.7em;letter-spacing:2px}
.stats{position:relative;z-index:1;display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.stat{background:var(--card);backdrop-filter:blur(30px);-webkit-backdrop-filter:blur(30px);
  border:1px solid rgba(255,255,255,0.06);padding:10px 16px;border-radius:12px;text-align:center;flex:1;min-width:80px;
  transition:all .3s;position:relative;overflow:hidden}
.stat::before{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;
  background:conic-gradient(from 0deg,transparent,var(--neon),transparent,var(--purple),transparent);
  opacity:0;transition:opacity .3s}
.stat:hover::before{opacity:.05;animation:rotate 4s linear infinite}
.stat:hover{transform:translateY(-3px);box-shadow:0 12px 30px rgba(168,85,247,0.2);border-color:rgba(168,85,247,0.4)}
.stat .val{font-family:'Orbitron',sans-serif;font-size:1.3em;font-weight:700;position:relative;z-index:1}
.stat .lbl{font-size:.6em;color:#64748b;text-transform:uppercase;letter-spacing:2px;margin-top:4px;position:relative;z-index:1}
@keyframes rotate{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}
.g{color:var(--green)}.r{color:var(--red)}.n{color:var(--neon)}.p{color:var(--purple)}
.panels{position:relative;z-index:1}
.match-card{background:var(--card);backdrop-filter:blur(30px);-webkit-backdrop-filter:blur(30px);
  border:1px solid rgba(255,255,255,0.04);border-radius:12px;padding:12px 16px;margin-bottom:6px;
  display:grid;grid-template-columns:2fr 1.5fr 0.6fr 0.6fr 0.6fr 0.5fr 0.8fr 1.5fr;gap:8px;align-items:center;
  transition:all .25s;position:relative;overflow:hidden}
.match-card::after{content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;
  background:linear-gradient(90deg,transparent,rgba(168,85,247,0.05),transparent);transition:left .5s}
.match-card:hover::after{left:100%}
.match-card:hover{border-color:rgba(168,85,247,0.3);transform:translateX(4px);box-shadow:0 8px 25px rgba(0,0,0,0.4)}
.match-info{font-weight:500;font-size:.8em}
.match-teams{font-size:.85em;font-weight:700}
.badge{padding:3px 10px;border-radius:6px;font-size:.65em;font-weight:700;letter-spacing:1px;text-align:center}
.bo{background:linear-gradient(135deg,rgba(0,212,255,0.2),rgba(168,85,247,0.2));color:var(--neon);border:1px solid rgba(0,212,255,0.3)}
.bu{background:linear-gradient(135deg,rgba(236,72,153,0.2),rgba(239,68,68,0.2));color:var(--pink);border:1px solid rgba(236,72,153,0.3)}
.won{color:var(--green);font-weight:700}.lost{color:var(--red)}.pending{color:var(--neon);animation:glow 2s infinite}
@keyframes glow{0%,100%{opacity:1}50%{opacity:0.5}}
.reason{font-size:.62em;color:#64748b;line-height:1.4;max-width:200px}
.pagination{position:relative;z-index:1;display:flex;justify-content:center;gap:6px;margin-top:12px}
.pagination a{color:var(--text);text-decoration:none;padding:5px 12px;background:var(--card);
  border:1px solid rgba(255,255,255,0.06);border-radius:8px;font-size:.7em;transition:all .2s;font-family:'Orbitron',sans-serif}
.pagination a:hover{background:rgba(168,85,247,0.15);border-color:var(--purple);color:var(--purple)}
.pagination a.active{background:linear-gradient(135deg,var(--neon),var(--purple));color:#000;font-weight:700;border:none;box-shadow:0 0 20px rgba(0,212,255,0.4)}
.hologram{position:fixed;bottom:20px;right:20px;width:120px;height:120px;pointer-events:none;z-index:0;
  background:radial-gradient(circle,rgba(0,212,255,0.1),transparent);border-radius:50%;
  animation:float 3s ease-in-out infinite}
.hologram::before{content:'⚽';font-size:3em;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  animation:kick 1s ease-in-out infinite;opacity:.3}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-20px)}}
@keyframes kick{0%,100%{transform:translate(-50%,-50%) rotate(0deg)}50%{transform:translate(-50%,-50%) rotate(180deg)}}
@media(max-width:768px){.match-card{grid-template-columns:1fr 1fr}.stat{min-width:60px}}
</style></head><body>
<canvas id="particles"></canvas>
<div class="hologram"></div>
<div class="header"><h1>⚽ SPORTS AI v2.0</h1><span>{{ update_time }}</span></div>
<div class="stats">
<div class="stat"><div class="val n">{{ total }}</div><div class="lbl">Ставок</div></div>
<div class="stat"><div class="val {{ 'g' if bank_val>=10000 else 'r' }}">{{ bank }}</div><div class="lbl">Банк</div></div>
<div class="stat"><div class="val {{ 'g' if roi_val>0 else 'r' }}">{{ roi }}</div><div class="lbl">ROI</div></div>
<div class="stat"><div class="val p">{{ hit }}</div><div class="lbl">Проходимость</div></div>
<div class="stat"><div class="val {{ 'g' if pnl_val>0 else 'r' }}">{{ pnl }}</div><div class="lbl">Прибыль</div></div>
</div>
<div class="panels">
{% for b in bets %}
<div class="match-card">
<div class="match-date" style="font-size:.65em;color:var(--neon);grid-column:1/-1;margin-bottom:-4px">{{ b.match_date }}</div>
<div class="match-teams">{{ b.match_ru }}</div>
<div><span class="badge {{ 'bo' if 'ТБ' in b.bet_ru or 'ИТБ' in b.bet_ru else 'bu' }}">{{ b.bet_ru }}</span></div>
<div style="font-weight:700;color:var(--gold)">{{ b.odds }}</div>
<div>{{ (b.prob*100)|int }}%</div>
<div class="{{ 'g' if b.ev>0 else 'r' }}" style="font-weight:600">{{ '%+.0f'|format(b.ev*100) }}%</div>
<div style="font-weight:600">{{ b.stake }}₽</div>
<div><span class="{{ 'won' if b.status=='Won' else 'lost' if b.status=='Lost' else 'pending' }}">{{ b.status }}</span></div>
<div class="reason">{{ b.reason }}</div>
</div>
{% endfor %}
</div>
<div class="pagination">
{% for p in range(1, pages+1) %}<a href="?page={{ p }}" class="{{ 'active' if p==page }}">{{ p }}</a>{% endfor %}
</div>
<script>
setTimeout(function(){location.reload()},60000);
// Particles
(function(){var c=document.getElementById('particles'),ctx=c.getContext('2d'),w=c.width=window.innerWidth,h=c.height=window.innerHeight,pts=[];for(var i=0;i<80;i++){pts.push({x:Math.random()*w,y:Math.random()*h,vx:(Math.random()-.5)*.5,vy:(Math.random()-.5)*.5,s:1+Math.random()*2,o:Math.random()*.5})}
function draw(){ctx.clearRect(0,0,w,h);pts.forEach(function(p,i){p.x+=p.vx;p.y+=p.vy;if(p.x<0)p.x=w;if(p.x>w)p.x=0;if(p.y<0)p.y=h;if(p.y>h)p.y=0;ctx.beginPath();ctx.arc(p.x,p.y,p.s,0,Math.PI*2);ctx.fillStyle='rgba(168,85,247,'+p.o+')';ctx.fill();
for(var j=i+1;j<pts.length;j++){var d=Math.hypot(p.x-pts[j].x,p.y-pts[j].y);if(d<80){ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(pts[j].x,pts[j].y);ctx.strokeStyle='rgba(0,212,255,'+(.1-d/800)+')';ctx.lineWidth=.3;ctx.stroke()}}});requestAnimationFrame(draw)}draw()})();
</script></body></html>'''

@app.route('/')
def index():
    try:
        df = pd.read_csv('live_predictions.csv')
        if 'bet_ru' not in df.columns:
            def tr(r):
                l,d,h,a=str(r['line']),str(r['direction']),str(r['home']),str(r['away'])
                if 'OU 2.5' in l:return f"{'ТБ' if d=='Over' else 'ТМ'} Тотал 2.5"
                if 'Home 1.5' in l:return f"{'ИТБ' if d=='Over' else 'ИТМ'}1.5({h})"
                if 'Away 1.5' in l:return f"{'ИТБ' if d=='Over' else 'ИТМ'}1.5({a})"
                if 'Home 0.5' in l:return f"{'ИТБ' if d=='Over' else 'ИТМ'}0.5({h})"
                if 'Away 0.5' in l:return f"{'ИТБ' if d=='Over' else 'ИТМ'}0.5({a})"
                return f"{'ТБ' if d=='Over' else 'ТМ'} {l}"
            df['bet_ru']=df.apply(tr,axis=1)
            df['match_ru']=df.apply(lambda r:f"{r['home']} — {r['away']}",axis=1)
            if 'match_date' not in df.columns:
                df['match_date']=pd.to_datetime(df['match_time'] if 'match_time' in df.columns else '2026-05-01').dt.strftime('%d.%m %H:%M')
            df.to_csv('live_predictions.csv',index=False)
        bank=10000;won=lost=0
        for _,r in df.iterrows():
            s=str(r.get('status','Pending'))
            if s=='Won':bank+=float(r['stake'])*(float(r['odds'])-1);won+=1
            elif s=='Lost':bank-=float(r['stake']);lost+=1
        page=request.args.get('page',1,type=int);per_page=10
        total=len(df);pages=max(1,(total+per_page-1)//per_page)
        bets=df.iloc[(page-1)*per_page:page*per_page].fillna('').to_dict('records')
        for b in bets:
            for k in b:
                if hasattr(b[k],'item'):b[k]=b[k].item()
        settled=len(df[df['status'].isin(['Won','Lost'])])
        hit_rate=f"{won/settled*100:.0f}%" if settled>0 else '—'
        roi_val=(bank-10000)/10000*100;pnl_val=bank-10000
    except Exception as e:
        bets=[];total=0;pages=1;page=1;bank=10000;bank_val=10000
        bank_str='10,000₽';roi='+0.0%';roi_val=0;hit_rate='—';pnl='0₽';pnl_val=0
    bank_str=f"{bank:,.0f}₽";roi=f"{roi_val:+.1f}%";pnl=f"{pnl_val:+}₽";bank_val=bank
    return render_template_string(HTML,bets=bets,total=total,bank=bank_str,bank_val=bank_val,roi=roi,roi_val=roi_val,hit=hit_rate,pnl=pnl,pnl_val=pnl_val,page=page,pages=pages,update_time=datetime.now().strftime('%d.%m %H:%M'))

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000,debug=False)
