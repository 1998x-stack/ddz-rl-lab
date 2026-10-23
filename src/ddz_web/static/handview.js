'use strict';
/* Opt-in debug only; normal snapshots never reveal hidden cards. */
(() => {
  const panel = document.getElementById('hands-analyzer');
  const rows = document.getElementById('hands-rows');
  const hint = document.getElementById('trick-explanation');
  if (!panel || !rows || !hint) return;
  const faces = ['3','4','5','6','7','8','9','10','J','Q','K','A','2','小王','大王'];
  const suits = ['♠','♥','♣','♦'];
  let enabled = false, serial = 0;
  const originalUpdate = update;
  update = function (data) {
    originalUpdate(data);
    if (data.phase === 'PLAY') {
      const hasLead = data.last_play_counts && data.last_play_counts.some(n => n > 0);
      const last = data.events && data.events.length ? data.events[data.events.length - 1].kind : '';
      hint.textContent = hasLead
        ? '跟牌阶段：普通牌必须同牌型、同张数且更大，合法炸弹除外；当前需压座位 ' + data.last_play_seat
        : last === 'TRICK_RESET'
          ? '两家连续 PASS 后，新一轮自由领牌；对Q这时不是用来压对2。'
          : '新一轮：当前玩家可自由选择合法牌型，无需压上一轮的牌。';
    } else {
      hint.textContent = data.phase === 'FINISHED' ? '本局结束' : '叫地主阶段，不进行牌型比较';
    }
    if (enabled) void refresh(data.game_id, data.sequence);
  };
  async function refresh(id, sequence) {
    if (!id) return;
    const ticket = ++serial;
    try {
      const response = await fetch('/api/games/' + encodeURIComponent(id) + '/debug/hands', {cache:'no-store'});
      if (!response.ok) throw Error('HTTP ' + response.status);
      const state = await response.json();
      if (ticket !== serial || state.game_id !== id || state.sequence < sequence) return;
      rows.replaceChildren();
      for (let seat=0;seat<3;seat++) {
        const row = document.createElement('div'); row.className = 'debug-hand-row';
        const heading = document.createElement('div'); heading.className = 'debug-hand-heading';
        heading.textContent = 'Agent ' + seat + ' · ' + (state.landlord===seat?'地主':state.landlord===null?'待定':'农民')
          + ' · ' + state.hands[seat].length + ' 张' + (state.acting_seat===seat?' · 行动中':'');
        const tray = document.createElement('div'); tray.className='debug-hand-tray';
        tray.setAttribute('aria-label', 'Agent ' + seat + ' 的已排序手牌');
        for (const cid of state.hands[seat]) {
          const rank = cid>=52 ? cid-39 : cid%13;
          const suitIndex = cid<52 ? Math.floor(cid/13) : -1;
          const card = document.createElement('span');
          card.className = 'debug-playing-card' + (suitIndex===1||suitIndex===3||cid===53?' red':'');
          const number = document.createElement('b'); number.textContent=faces[rank];
          const symbol = document.createElement('small'); symbol.textContent=suitIndex<0?'★':suits[suitIndex];
          card.append(number,symbol); tray.append(card);
        }
        row.append(heading,tray); rows.append(row);
      }
    } catch (error) { if(ticket===serial)rows.textContent='调试牌面不可用：' + error.message; }
  }
  fetch('/api/debug/config', {cache:'no-store'})
    .then(r=>r.ok?r.json():{enabled:false})
    .then(cfg=>{enabled=cfg.enabled===true;panel.hidden=!enabled;
      if(enabled && typeof current!=='undefined' && current)void refresh(current.game_id,current.sequence);})
    .catch(()=>{panel.hidden=true;});
})();
