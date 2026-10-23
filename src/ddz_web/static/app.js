'use strict';
const $ = id => document.getElementById(id);
let gameId = null, socket = null, current = null;
let hand = null, selected = Array(15).fill(0);
const names = ['3','4','5','6','7','8','9','10','J','Q','K','A','2','小王','大王'];
async function api(url, init) { const r = await fetch(url, init); const d = await r.json(); if (!r.ok) throw new Error(d.detail || r.statusText); return d; }
function notice(message) { $('notice').textContent = message || ''; }
function models(select, data) { select.replaceChildren(); const ids = Object.keys(data.models); if (data.baseline_enabled) ids.unshift('@reasonable','@random'); if(select.id==='m0') ids.unshift('@human'); for (const id of ids) { const label=id==='@human'?'我来出牌':id==='@reasonable'?'合理随机 Baseline（非 RL）':id==='@random'?'旧版均匀随机（非 RL）':id; select.add(new Option(label,id)); } if (data.active && ids.includes(data.active)) select.value = data.active; }
async function init() { try { const data = await api('/api/models'); for(const n of ['m0','m1','m2']) models($(n),data); if (!Object.keys(data.models).length && !data.baseline_enabled) notice('请先注册并激活训练好的模型，或以 --allow-baseline 启动演示。'); $('connection').textContent='服务已连接'; const saved=localStorage.getItem('ddz-last-game');if(saved){try{const state=await api(`/api/games/${saved}`);update(state);connect(saved);}catch{localStorage.removeItem('ddz-last-game');}} } catch(e) { $('connection').textContent='连接失败'; notice(e.message); } }
function update(data) {
 const changed=gameId!==data.game_id; current=data; gameId=data.game_id;
 if(changed){hand=null;selected=Array(15).fill(0);}
 $('counter').textContent=`Game ${gameId} · ${data.step_index} 步`;
 $('phase').textContent=data.winner ? `获胜：${data.winner==='LANDLORD'?'地主':'农民阵营'}` : ({CALL:'叫地主',ROB:'抢地主',PLAY:'出牌阶段',FINISHED:'牌局结束'}[data.phase]||data.phase);
 $('turn').textContent=data.error|| (data.acting_seat===null?'本局已结束':`行动中：${data.model_ids[data.acting_seat]==='@human'?'你':'Agent '+data.acting_seat} · ${data.status==='awaiting_human'?'请出牌':data.running?'自动':'暂停'}${data.random_bidding && (data.phase==='CALL'||data.phase==='ROB')?' · AI 随机叫牌':''}`);
 for(let i=0;i<3;i++){const seat=$('seat'+i); seat.classList.toggle('active',data.acting_seat===i);seat.classList.toggle('landlord',data.landlord===i);seat.querySelector('small').textContent=`剩余 ${data.remaining[i]} 张 · ${data.model_ids[i]}`;}
 const cards=$('cards');cards.replaceChildren(); data.last_play_counts.forEach((count,idx)=>{ for(let x=0;x<count;x++){const c=document.createElement('div');c.className='card'+(idx>12?' red':'');c.textContent=names[idx];cards.append(c);} });
 const events=$('events');events.replaceChildren(); for(const event of [...data.events].reverse()) {const item=document.createElement('li');item.textContent=`#${event.sequence} ${event.kind} · ${event.actor===null?'系统':'座位 '+event.actor}`;events.append(item);} 
 for(const id of ['play','pause','step','speed']) $(id).disabled=data.status==='finished'||data.status==='truncated'||data.status==='error'||(id==='step'&&data.running);
 $('speed').value=data.delay;$('speed-value').textContent=`${data.delay.toFixed(2)} 秒`;
 const replay=$('replay');const finished=data.status==='finished'||data.status==='truncated';replay.classList.toggle('disabled',!finished);replay.setAttribute('aria-disabled',String(!finished));if(finished)replay.href=`/api/games/${gameId}/replay`;
 $('human-panel').hidden=data.model_ids[0]!=='@human';
 if(data.model_ids[0]==='@human' && (!hand || hand.step_index!==data.step_index)) refreshHand();
}
async function refreshHand(){
 const id=gameId;if(!id)return;
 try{const data=await api(`/api/games/${id}/hand`);if(gameId!==id)return;hand=data;selected=Array(15).fill(0);renderHand();}
 catch(e){notice(e.message);}
}
function renderHand(){
 if(!hand)return;
 const tray=$('human-cards'),actions=$('human-actions');tray.replaceChildren();actions.replaceChildren();
 hand.own_counts.forEach((count,rank)=>{for(let copy=0;copy<count;copy++){
  const card=document.createElement('button');card.type='button';card.className='card human-card'+(rank>12?' red':'')+(copy<selected[rank]?' selected':'');card.textContent=names[rank];card.disabled=!hand.your_turn;
  card.onclick=()=>{selected[rank]=selected[rank]===count?0:selected[rank]+1;renderHand();};tray.append(card);
 }});
 $('human-hint').textContent=hand.your_turn?'轮到你了。选牌后出牌，或选择叫牌/不出。':'等待 AI 行动；你的手牌只在本机对局接口显示。';
 if(!hand.your_turn)return;
 for(const [index,action] of hand.legal_actions.entries()){
  if(action.kind==='PLAY' && !action.counts.every((n,i)=>n===selected[i]))continue;
  if(action.kind!=='PLAY' && !['CALL','DECLINE_CALL','ROB','DECLINE_ROB','PASS'].includes(action.kind))continue;
  const button=document.createElement('button');button.type='button';button.textContent=({CALL:'叫地主',DECLINE_CALL:'不叫',ROB:'抢地主',DECLINE_ROB:'不抢',PASS:'不出',PLAY:'出牌'}[action.kind]);
  button.onclick=()=>submitAction(index);actions.append(button);
 }
 if(selected.some(Boolean) && ![...actions.children].some(b=>b.textContent==='出牌'))$('human-hint').textContent='所选牌型当前不合法，请调整选择。';
}
async function submitAction(index){try{const state=await api(`/api/games/${gameId}/action`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index,expected_step:hand.step_index})});update(state);}catch(e){notice(e.message);refreshHand();}}
function connect(id) { if(socket)socket.close();const scheme=location.protocol==='https:'?'wss:':'ws:';socket=new WebSocket(`${scheme}//${location.host}/api/games/${id}/events`);socket.onmessage=e=>{const packet=JSON.parse(e.data);if(packet.type==='snapshot')update(packet);};socket.onclose=()=>{if(gameId===id)$('connection').textContent='实时连接断开 · 可刷新状态';};socket.onopen=()=>{$('connection').textContent='实时连接正常';};}
async function create(){try{const selected=['m0','m1','m2'].map(id=>$(id).value);if(selected.some(x=>!x)) throw new Error('请先选择三个座位');const s=$('seed').value.trim();const data=await api('/api/games',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model_ids:selected,seed:s===''?null:Number(s),delay:Number($('speed').value),random_bidding:$('random-bidding').checked})});localStorage.setItem('ddz-last-game',data.game_id);notice(selected[0]==='@human'?'牌局已创建；点击播放，轮到你时选择操作。':'牌局已创建；点击播放或逐步执行。');update(data);connect(data.game_id);}catch(e){notice(e.message);}}
async function control(action,delay){if(!gameId)return;try{const state=await api(`/api/games/${gameId}/control`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,delay})});update(state);}catch(e){notice(e.message);}}
$('create').onclick=create;for(const act of ['play','pause','step'])$(act).onclick=()=>control(act);$('speed').onchange=()=>control('speed',Number($('speed').value));init();
