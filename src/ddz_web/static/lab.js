'use strict';
// Operator-local research projections, never browser-side checkpoint evaluation.
const lab = id => document.getElementById(id);
const labApi = async path => {
  const r = await fetch(path, {headers: {'Accept': 'application/json'}});
  const value = await r.json();
  if (!r.ok) throw new Error(value.detail || r.statusText);
  return value;
};
const addText = (parent, tag, value, cls) => {
  const el = document.createElement(tag);
  el.textContent = String(value ?? '—');
  if (cls) el.className = cls;
  parent.append(el);
  return el;
};
const empty = (el, msg) => {el.replaceChildren(); addText(el, 'p', msg, 'empty');};
const fmt = (x, digits=3) => Number.isFinite(Number(x)) && x !== null && x !== '' ? Number(x).toFixed(digits) : '—';
function metricsCard(parent, label, val) {
  const box=addText(parent,'div','', 'kpi');
  addText(box,'strong',val);addText(box,'small',label);
}
function updateSelect(el, values, placeholder, chosen) {
  el.replaceChildren();
  for (const value of values) el.add(new Option(value,value));
  if (chosen && values.includes(chosen)) el.value=chosen;
  if (!values.length) el.add(new Option(placeholder,''));
}
function drawLine(id, records, xKey, yKey) {
  const svg=lab(id);svg.replaceChildren();
  const values=records.map(r=>[Number(r[xKey]),Number(r[yKey])]).filter(([x,y])=>Number.isFinite(x)&&Number.isFinite(y));
  if (!values.length) {const t=document.createElementNS('http://www.w3.org/2000/svg','text');t.setAttribute('x','20');t.setAttribute('y','103');t.setAttribute('fill','#90aca7');t.textContent='尚无有效训练数据';svg.append(t);return;}
  const minX=Math.min(...values.map(v=>v[0])), maxX=Math.max(...values.map(v=>v[0]));
  const minY=Math.min(0,...values.map(v=>v[1])), maxY=Math.max(...values.map(v=>v[1]));
  const point=([x,y])=>[48+588*(x-minX)/(maxX-minX||1),167-133*(y-minY)/(maxY-minY||1)];
  const path=document.createElementNS('http://www.w3.org/2000/svg','polyline');
  path.setAttribute('points',values.map(v=>point(v).join(',')).join(' '));
  path.setAttribute('fill','none');path.setAttribute('stroke','#b8ebc9');path.setAttribute('stroke-width','3');path.setAttribute('stroke-linejoin','round');svg.append(path);
  for(const [x,y,label] of [[13,30,fmt(maxY)],[13,170,fmt(minY)],[48,199,String(minX)],[600,199,String(maxX)]]) {
    const t=document.createElementNS('http://www.w3.org/2000/svg','text');t.setAttribute('x',x);t.setAttribute('y',y);t.setAttribute('font-size','15');t.setAttribute('fill','#90aca7');t.textContent=label;svg.append(t);
  }
}
let overview=null;
async function refreshOverview() {
  overview=await labApi('/api/lab/overview');
  updateSelect(lab('run-select'),overview.runs.map(r=>r.run_id),'没有训练任务',lab('run-select').value);
  updateSelect(lab('compare-run-select'),overview.runs.map(r=>r.run_id),'没有比较对象',lab('compare-run-select').value);
  updateSelect(lab('report-select'),overview.reports.map(r=>r.report_id),'没有 Tournament 报告',lab('report-select').value);
  updateSelect(lab('replay-select'),overview.replays.map(r=>r.replay_id),'没有已结束的对局',lab('replay-select').value);
  updateSelect(lab('matrix-select'),(overview.matrices||[]).map(r=>r.report_id),'没有组合评测',lab('matrix-select').value);
  const body=lab('registry-list');body.replaceChildren();
  const all=Object.entries(overview.models.models);
  if(!all.length)empty(body,'暂无已注册 Checkpoint；使用 ddz-league register 进行注册。');
  for(const [id,info] of all) {const row=addText(body,'div','','data-row');
    addText(row,'strong',id+(id===overview.models.active?' · ACTIVE':''));
    addText(row,'small','SHA256: '+String(info.sha256 || '').slice(0,20)+'…');
    addText(row,'small','训练局数：'+String(info.episodes_seen ?? '—'));
  }
}
async function loadRun() {
  const id=lab('run-select').value;const box=lab('run-summary');box.replaceChildren();
  if(!id){empty(box,'没有可读取的训练数据');drawLine('score-chart',[],'episodes_seen','balanced_score');drawLine('throughput-chart',[],'episodes_seen','episodes_per_s');return;}
  const data=await labApi('/api/lab/runs/'+encodeURIComponent(id)+'/metrics');
  const evals=data.evaluations.records, throughput=data.throughput.records;
  const last=evals.at(-1)||{},recent=throughput.at(-1)||{};
  metricsCard(box,'Episodes',last.episodes_seen??recent.episodes_seen??'—');
  metricsCard(box,'Balanced Score',fmt(last.balanced_score));
  metricsCard(box,'Best Score',fmt(last.best_score));
  metricsCard(box,'Episodes/s',fmt(recent.episodes_per_s,2));
  metricsCard(box,'Samples/s',fmt(recent.samples_per_s,1));
  metricsCard(box,'Replay Samples',recent.replay_size??last.replay_size??'—');
  drawLine('score-chart',evals,'episodes_seen','balanced_score');
  drawLine('throughput-chart',throughput,'episodes_seen','episodes_per_s');
  lab('run-hint').textContent=(data.throughput.truncated||data.evaluations.truncated?'仅显示文件尾部部分数据。':'')+' 指标仅代表训练记录，不能证明策略已收敛。';
}
async function loadReport() {
  const id=lab('report-select').value,summary=lab('report-summary'),items=lab('report-results');
  summary.replaceChildren();items.replaceChildren();if(!id){empty(items,'没有已生成的 Tournament 报告');return;}
  const r=await labApi('/api/lab/reports/'+encodeURIComponent(id));
  metricsCard(summary,'Incumbent',r.incumbent_id);metricsCard(summary,'Recommended',r.recommended_id||'未建议晋升');
  metricsCard(summary,'Deal Blocks',r.seed_starts?.length??'—');metricsCard(summary,'Min Deals',r.min_deals);
  for(const key of r.ranked_ids||Object.keys(r.candidates)){
    const v=r.candidates[key],row=addText(items,'div','','data-row');addText(row,'strong',key);
    addText(row,'small',`Score: ${fmt(v.score?.mean)} · 95% CI [${fmt(v.score?.ci?.[0])}, ${fmt(v.score?.ci?.[1])}] · 样本簇 ${v.score?.deal_clusters??'—'}`);
    if(v.vs_incumbent)addText(row,'small',`vs champion: Δ ${fmt(v.vs_incumbent.delta)} · CI [${fmt(v.vs_incumbent.ci[0])}, ${fmt(v.vs_incumbent.ci[1])}] · ${v.eligible?'满足统计门槛':'证据不足'}`);
  }
}
let replayFrames=[];
function viewReplayStep() {
  const idx=Number(lab('replay-seek').value),v=replayFrames[idx];if(!v)return;
  lab('replay-step').textContent=`${idx} / ${replayFrames.length-1}`;
  const parent=lab('replay-details');parent.replaceChildren();const row=addText(parent,'div','','data-row');
  addText(row,'strong',`阶段 ${v.phase} · 当前行动 ${v.acting_seat===null?'结束':'座位 '+v.acting_seat}`);
  addText(row,'small',`剩余牌数：${v.remaining.join(' / ')} · 地主座位：${v.landlord??'未定'}`);
  addText(row,'small',`本轮出牌：${v.last_play_counts.reduce((a,b)=>a+b,0)} 张 · 胜者：${v.winner||'—'}`);
  for(const evt of v.events) addText(parent,'div',`#${evt.sequence} ${evt.kind} · ${evt.actor===null?'系统':'座位 '+evt.actor}`,'data-row');
}
async function loadReplay() {
  const id=lab('replay-select').value,info=lab('replay-information');info.replaceChildren();
  replayFrames=[];lab('replay-seek').disabled=true;if(!id){empty(lab('replay-details'),'没有可用的已完成回放');return;}
  const r=await labApi('/api/lab/replays/'+encodeURIComponent(id));replayFrames=r.frames;
  metricsCard(info,'Integrity',r.verified?'SHA Verified':'—');metricsCard(info,'Winner',r.winner);metricsCard(info,'Turns',r.frames.length-1);
  lab('replay-seek').max=String(r.frames.length-1);lab('replay-seek').value='0';lab('replay-seek').disabled=false;viewReplayStep();
}
async function loadLogs(){const r=await labApi('/api/lab/logs');const el=lab('log-list');el.replaceChildren();if(!r.lines.length){empty(el,'没有可显示的 Loguru JSONL 日志');return;}
  for(const entry of [...r.lines].reverse()) {const li=addText(el,'li','');addText(li,'span',entry.time||'','muted');addText(li,'span',entry.level||'','log-level');addText(li,'span',entry.message||'');}
}
async function loadJobs() {
  const result=await labApi('/api/lab/jobs');const area=lab('jobs-list');area.replaceChildren();
  if(!result.jobs.length){empty(area,'暂无调度任务，请通过 ddz-jobs submit 添加');return;}
  for(const job of result.jobs){const row=addText(area,'div','','data-row');
    addText(row,'strong',job.experiment_id+' · '+job.status);
    addText(row,'small','Job '+job.job_id.slice(0,12)+' · '+(job.created_at||''));
    if(job.parent_experiment)addText(row,'small','Parent: '+job.parent_experiment);
    if(job.exit_code!==null)addText(row,'small','Exit code: '+job.exit_code);
  }
}
async function loadMatrix() {
  const area=lab('matrix-list');area.replaceChildren();const id=lab('matrix-select').value;
  if(!id){empty(area,'暂无 Cross-Play Matrix 报告');return;}
  const data=await labApi('/api/lab/matrices/'+encodeURIComponent(id));
  for(const row of data.rows){const el=addText(area,'div','','data-row');
    addText(el,'strong',Object.entries(row.models).map(([role,model])=>role+': '+model).join(' · '));
    addText(el,'small','Landlord win rate: '+fmt(row.landlord_win_rate)+' · '+row.completed+' seeded games');
  }
  addText(area,'p','组合相互对抗的胜率不能直接用于 Checkpoint 自动晋升。','muted');
}
async function safely(fn) {try{await fn();}catch(e){const pane=document.querySelector('.lab-view:not([hidden])');const msg=pane.querySelector('.lab-error')||addText(pane,'p','','lab-error');msg.textContent='数据读取失败：'+e.message;}}
const loaders={arena:async()=>{},runs:async()=>{await refreshOverview();await loadRun();},models:refreshOverview,reports:async()=>{await refreshOverview();await loadReport();},replays:async()=>{await refreshOverview();await loadReplay();},logs:loadLogs,jobs:loadJobs,matrix:async()=>{await refreshOverview();await loadMatrix();}};
for (const btn of document.querySelectorAll('[data-tab]'))btn.onclick=()=>{
  const tab=btn.dataset.tab;for(const view of document.querySelectorAll('.lab-view'))view.hidden=view.id!=='tab-'+tab;
  for(const el of document.querySelectorAll('[data-tab]'))el.classList.toggle('selected',el===btn);
  safely(loaders[tab]);
};
lab('run-select').onchange=()=>safely(loadRun);lab('report-select').onchange=()=>safely(loadReport);
lab('replay-select').onchange=()=>safely(loadReplay);lab('replay-seek').oninput=viewReplayStep;
lab('run-refresh').onclick=()=>safely(loaders.runs);lab('models-refresh').onclick=()=>safely(loaders.models);
lab('reports-refresh').onclick=()=>safely(loaders.reports);lab('replays-refresh').onclick=()=>safely(loaders.replays);
lab('logs-refresh').onclick=()=>safely(loadLogs);
lab('jobs-refresh').onclick=()=>safely(loadJobs);
lab('matrix-select').onchange=()=>safely(loadMatrix);
lab('matrix-refresh').onclick=()=>safely(loaders.matrix);

async function compareRuns() {
  const left=lab('run-select').value,right=lab('compare-run-select').value;
  const area=lab('compare-run-result');area.replaceChildren();
  if (!left||!right||left===right) {empty(area,'请选择两个不同的实验');return;}
  const data=await labApi('/api/lab/compare?left='+encodeURIComponent(left)+'&right='+encodeURIComponent(right));
  const a=data.results[left],b=data.results[right];
  const row=addText(area,'div','','data-row');
  addText(row,'strong',left+' vs '+right);
  addText(row,'small','Balanced Score: '+fmt(a.latest.balanced_score)+' → '+fmt(b.latest.balanced_score));
  addText(row,'small','Δ (right - left): '+fmt(data.balanced_score_delta));
  addText(row,'small','Episodes: '+String(a.latest.episodes_seen??'—')+' → '+String(b.latest.episodes_seen??'—'));
  addText(row,'small','Status: '+String(a.run_status??'—')+' / '+String(b.run_status??'—'));
}
lab('compare-run-btn').onclick=()=>safely(compareRuns);
