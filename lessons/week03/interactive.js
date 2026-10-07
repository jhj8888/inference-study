'use strict';
const W03 = (() => {
  const el=id=>document.getElementById(id);
  const copy=x=>JSON.parse(JSON.stringify(x));
  function simulate(budget=4,max_running=2,block_size=2,capacity=8,cancel_b=false){
    for(const v of [budget,max_running,block_size,capacity])if(!Number.isInteger(v)||v<1)throw Error('positive integers required');
    const requests=[['A',4,2,0],['B',2,3,0],['C',6,1,1]].map(([name,prompt,target,arrival])=>({name,prompt,target,arrival,computed:0,emitted:0,state:'NOT_ARRIVED',blocks:0,first_output:null,finish:null}));
    const byName=Object.fromEntries(requests.map(r=>[r.name,r]));
    let running=[],waiting=[],trace=[],status='COMPLETE';
    for(let step=0;step<1000;step++){
      const events=[];
      for(const r of requests)if(r.arrival===step){r.state='WAITING';waiting.push(r.name);events.push(r.name+':arrive');}
      if(cancel_b&&step===2){const r=byName.B;if(['WAITING','RUNNING'].includes(r.state)){running=running.filter(n=>n!=='B');waiting=waiting.filter(n=>n!=='B');r.state='ABORTED';r.blocks=0;r.finish=step;events.push('B:abort');}}
      const before=copy(requests),plans={};let remaining=budget;
      const reserve=(r,n)=>{const needed=Math.ceil((r.computed+n)/block_size),others=requests.filter(q=>q!==r).reduce((s,q)=>s+q.blocks,0);if(needed+others>capacity){events.push(r.name+':allocation_denied');return false;}r.blocks=needed;plans[r.name]=n;return true;};
      for(const name of [...running]){if(!remaining)break;const r=byName[name],n=Math.min(r.prompt+r.emitted-r.computed,remaining);if(n&&reserve(r,n))remaining-=n;}
      while(waiting.length&&remaining&&running.length<max_running){const r=byName[waiting[0]],n=Math.min(r.prompt+r.emitted-r.computed,remaining);if(!reserve(r,n))break;remaining-=n;waiting.shift();running.push(r.name);r.state='RUNNING';}
      const peak_blocks=requests.reduce((s,r)=>s+r.blocks,0),outputs=[];
      for(const [name,n] of Object.entries(plans)){const r=byName[name],known=r.prompt+r.emitted;r.computed+=n;if(r.computed===known){r.emitted++;outputs.push(name);if(r.first_output===null)r.first_output=step+1;if(r.emitted===r.target){r.state='FINISHED_LENGTH';r.finish=step+1;r.blocks=0;running=running.filter(n=>n!==r.name);events.push(r.name+':finish');}}}
      trace.push({step,plans,outputs,before,after:copy(requests),running_after:[...running],waiting_after:[...waiting],peak_blocks,used_after:requests.reduce((s,r)=>s+r.blocks,0),events});
      if(requests.every(r=>['FINISHED_LENGTH','ABORTED'].includes(r.state)))break;
      if(!Object.keys(plans).length&&(running.length||waiting.length)){status='BLOCKED_NO_PREEMPTION';break;}
      if(step===999)throw Error('did not terminate');
    }
    return {kind:'teaching_simulation_not_vllm_measurement',status,config:{budget,max_running,block_size,capacity,cancel_b},trace,requests:copy(requests)};
  }
  const journey=[
    ['HTTP / Serving','解析协议并渲染输入','ChatCompletionRequest → engine inputs。这里解释消息、模板与参数，尚未执行模型。','S01','router / OpenAIServingChat'],
    ['AsyncLLM','登记请求与结果接收点','generate 消费期间调用 add_request，生成 EngineCoreRequest，并先登记本地输出处理。','S04','AsyncLLM.generate / _add_request'],
    ['CoreClient / IPC','提交 ADD 消息','AsyncMPClient 编码请求，经 ZMQ 发给 Core；发送成功并不代表已经计算。','S07','AsyncMPClient.add_request_async'],
    ['EngineCore','接收与分发','输入线程预处理为 Request，经内部队列让主循环分发，再进入 Scheduler。','S09','process_input_sockets / _handle_client_request'],
    ['Scheduler','形成这一步的计划','给请求分配 token 预算和 KV 槽位，形成 SchedulerOutput；逻辑 token 数不必等于 padding 后形状。','S14','Scheduler.schedule'],
    ['Executor / Worker','组织执行','单卡 UniProcExecutor 用 run_method 调 driver_worker；这个边界不必跨进程。','S18','UniProcExecutor.collective_rpc'],
    ['Runner / sampling','执行模型并产生 token','准备输入与 attention 元数据；本路线的 execute_model 与 sample_tokens 可以分开。','S20','GPUModelRunner.execute_model'],
    ['Core update','更新序列与结束状态','追加输出 token、检查停止条件、组织 core 输出；完成请求进入资源清理。','S22','Scheduler.update_from_output'],
    ['Output / SSE','形成用户可见文本','detokenize 和 stop-string 检查后进入 collector，再由 generate / Serving 送出。','S27','OutputProcessor.process_outputs'],
    ['Finish / free','结束并解除资源占用','正常结束、取消的入口不同；共享引用和在途工作可能推迟块复用。','S32','Scheduler._free_request']
  ];
  function showJourney(i){
    const j=journey[i];el('journey-detail').innerHTML=`<div class="widget-kicker">阶段 ${i+1} / ${journey.length}</div><h4>${j[1]}</h4><p>${j[2]}</p><code>${j[4]}</code> <a href="#${j[3].toLowerCase()}">查看 ${j[3]} 源码锚点</a>`;
    journey.forEach((_,k)=>{const b=el('journey-'+k);b.setAttribute('aria-pressed',String(k===i));});
  }
  let model,selectedStep=0;
  function render(){
    const s=model.trace[selectedStep],total=Object.values(s.plans).reduce((a,b)=>a+b,0);
    el('sim-step-label').textContent=`step ${s.step} / ${model.trace.length-1}`;
    el('sim-prev').disabled=selectedStep===0;el('sim-next').disabled=selectedStep===model.trace.length-1;
    el('sim-summary').innerHTML=`<div class="stat"><strong>${total} / ${model.config.budget}</strong><span>本步安排的输入位置 / 预算</span></div><div class="stat"><strong>${s.outputs.length}</strong><span>本步产生的输出 token</span></div><div class="stat"><strong>${s.peak_blocks} → ${s.used_after}</strong><span>计划峰值块 → 步末占用块</span></div>`;
    el('sim-chart').innerHTML='<table class="sim-matrix"><caption>整段计划：数字是输入位置数，● 表示该步为该请求产生一个输出</caption><thead><tr><th>请求</th>'+model.trace.map(t=>`<th${t.step===selectedStep?' class="current"':''}>${t.step}</th>`).join('')+'</tr></thead><tbody>'+['A','B','C'].map(name=>'<tr><th>'+name+'</th>'+model.trace.map(t=>`<td class="${t.step===selectedStep?'current ':''}${t.plans[name]?'work':''}">${t.plans[name]||'—'}${t.outputs.includes(name)?' ●':''}</td>`).join('')+'</tr>').join('')+'</tbody></table>';
    el('sim-state').innerHTML='<table><caption>所选步骤结束后的请求状态</caption><thead><tr><th>请求</th><th>computed</th><th>已输出</th><th>占用块</th><th>状态</th></tr></thead><tbody>'+s.after.map(r=>`<tr><th>${r.name}</th><td>${r.computed}</td><td>${r.emitted}/${r.target}</td><td>${r.blocks}</td><td>${r.state}</td></tr>`).join('')+'</tbody></table>';
    const blocked=model.status!=='COMPLETE'&&selectedStep===model.trace.length-1;
    el('sim-explain').textContent=`步末 running: ${s.running_after.join(', ')||'空'}；waiting: ${s.waiting_after.join(', ')||'空'}。事件：${s.events.join('；')||'无到达或结束事件'}。`+(blocked?' BLOCKED：现有容量不能推进，教学模型未实现抢占。':'同步教学模型：computed 在本步执行完后更新；真实源码还要区分 in-flight。');
  }
  function resetSimulation(){selectedStep=0;model=simulate(Number(el('sim-budget').value),Number(el('sim-running').value),2,Number(el('sim-capacity').value),el('sim-cancel').checked);render();}
  let owners={A:true,B:true};
  function ownership(){const n=Object.values(owners).filter(Boolean).length;el('own-a').disabled=!owners.A;el('own-b').disabled=!owners.B;el('own-result').textContent=`ref_cnt = ${n} · `+(n?'仍被请求占用，不可覆盖':'引用归零，可进入复用队列');el('own-note').textContent=n?'A/B 谁还持有引用，就必须保留该块。':el('own-cache').checked?'前缀缓存身份仍保留：可复用并不等于内容已经清零；被重新分配时还需处理缓存身份。':'不保留前缀身份：仍只是回到块池供复用，不表示 GPU 内存已交还操作系统。';}
  el('journey-buttons').innerHTML=journey.map((j,i)=>`<button id="journey-${i}" type="button" aria-pressed="false">${i+1}. ${j[0]}</button>`).join('');
  journey.forEach((_,i)=>el('journey-'+i).addEventListener('click',()=>showJourney(i)));showJourney(0);
  for(const id of ['sim-budget','sim-running','sim-capacity','sim-cancel'])el(id).addEventListener('change',resetSimulation);
  el('sim-prev').addEventListener('click',()=>{if(selectedStep>0){selectedStep--;render();}});
  el('sim-next').addEventListener('click',()=>{if(selectedStep<model.trace.length-1){selectedStep++;render();}});
  el('sim-reset').addEventListener('click',()=>{el('sim-budget').value='4';el('sim-running').value='2';el('sim-capacity').value='8';el('sim-cancel').checked=false;resetSimulation();});
  el('own-a').addEventListener('click',()=>{owners.A=false;ownership();});el('own-b').addEventListener('click',()=>{owners.B=false;ownership();});el('own-reset').addEventListener('click',()=>{owners={A:true,B:true};ownership();});el('own-cache').addEventListener('change',ownership);
  el('print-button').addEventListener('click',()=>window.print());resetSimulation();ownership();
  return {simulate,showJourney,getState:()=>({model,selectedStep})};
})();
