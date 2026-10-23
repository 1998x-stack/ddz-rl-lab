'use strict';
// The operator token is submitted once; only an HttpOnly session cookie remains.
(async function setupOperatorLogin(){
 const box=document.getElementById('operator-login');
 try {
  const response=await fetch('/api/auth/status',{credentials:'same-origin'});
  const status=await response.json();
  if (!status.required || status.authenticated) return;
  box.hidden=false;
  box.querySelector('form').addEventListener('submit',async(event)=>{
    event.preventDefault();
    const input=box.querySelector('input');
    const token=input.value;
    input.value='';
    const result=await fetch('/api/auth/login',{
      method:'POST',headers:{'X-Operator-Token':token},credentials:'same-origin'
    });
    box.querySelector('[role="status"]').textContent=result.ok?'登录成功':'凭据无效';
    if(result.ok) location.reload();
  });
 } catch (_) {box.hidden=true;}
})();
