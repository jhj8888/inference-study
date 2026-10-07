// Pure arithmetic and DOM rendering; all values are theoretical or constructed.
const W02 = (() => {
  function gemm(m, peak=200, bw=1500) {
    const flops=2*m*4096*4096, bytes=2*(m*4096+4096*4096+m*4096);
    const computeMs=flops/(peak*1e12)*1000, memoryMs=bytes/(bw*1e9)*1000;
    return {intensity:flops/bytes,computeMs,memoryMs,lowerMs:Math.max(computeMs,memoryMs),ridge:peak*1000/bw};
  }
  function queue(rate) {
    const rho=rate/200;
    if(rate>=200)return {rho,stable:false,meanMs:null,waitMs:null,resident:null};
    const meanMs=1000/(200-rate);
    return {rho,stable:true,meanMs,waitMs:meanMs-5,resident:rate*meanMs/1000};
  }
  function metrics(mode,gap=200) {
    const sent=mode==='stall'?20:0;
    const times=mode==='bundle'?[80,110,170]:mode==='single'?[100]:[120,140,160,160+gap];
    const counts=mode==='bundle'?[1,2,3]:times.map(()=>1);
    const gaps=times.slice(1).map((t,i)=>t-times[i]);
    const tokens=counts.reduce((a,b)=>a+b,0), first=times[0],last=times[times.length-1];
    return {times,counts,gaps,tokens,ttft:first-sent,e2e:last-sent,completion:last+20-sent,
      tpot:tokens>1?(last-first)/(tokens-1):null,meanItl:gaps.length?gaps.reduce((a,b)=>a+b,0)/gaps.length:null,
      maxItl:gaps.length?Math.max(...gaps):null};
  }
  function transfer(batch,context,bw,chunkMiB) {
    const bytes=2*32*batch*context*8*128*2;
    const chunks=Math.ceil(bytes/(chunkMiB*2**20));
    const payloadMs=bytes/(bw*1e9)*1000,setupMs=chunks*.04;
    return {gib:bytes/2**30,chunks,payloadMs,setupMs,totalMs:payloadMs+setupMs};
  }
  return {gemm,queue,metrics,transfer};
})();
globalThis.W02=W02;

const el=id=>document.getElementById(id), value=id=>Number(el(id).value);
const stat=(v,label)=>`<div class="stat"><strong>${v}</strong><span>${label}</span></div>`;
function bars(items,unit='ms') {
  const max=Math.max(...items.map(x=>x[1]),1e-12),height=items.length*58+8;
  const format=n=>n.toFixed(n>0&&n<.01?5:3);
  return `<svg viewBox="0 0 650 ${height}" role="img" aria-label="${items.map(x=>`${x[0]} ${format(x[1])} ${unit}`).join('；')}">`+
    items.map(([label,n,color],i)=>`<text x="0" y="${i*58+15}" font-size="14" fill="#263f54">${label} · ${format(n)} ${unit}</text><rect x="0" y="${i*58+23}" width="${Math.max(0,n/max*630)}" height="18" rx="3" fill="${color}"/>`).join('')+'</svg>';
}
function renderRoof() {
  const r=W02.gemm(value('roof-m'),value('roof-peak'),value('roof-bw'));
  el('roof-stats').innerHTML=stat(r.intensity.toFixed(2),'FLOP/byte 算术强度')+stat(r.ridge.toFixed(2),'FLOP/byte 屋脊点')+stat(r.lowerMs.toFixed(5)+' ms','单层理论时间下界');
  el('roof-chart').innerHTML=bars([['计算项 F/P',r.computeMs,'#0072B2'],['搬运项 Q/BW',r.memoryMs,'#D55E00']]);
  el('roof-explain').textContent=`此模型由${r.computeMs>=r.memoryMs?'计算':'搬运'}项主导，取两项最大值。改变硬件后再判断；这不是实测延迟，也不包含启动、排队与通信。`;
}
function renderQueue() {
  const rate=value('queue-rate'),r=W02.queue(rate);
  el('queue-rate-label').textContent=rate+' 请求/s';
  el('queue-stats').innerHTML=stat((r.rho*100).toFixed(0)+'%','ρ=λ/μ')+stat(r.stable?r.meanMs.toFixed(2)+' ms':'无有限稳态','平均系统时间 W')+stat(r.stable?r.resident.toFixed(2):'不适用','平均系统内请求数 λW');
  el('queue-chart').innerHTML=r.stable?bars([['服务均值',5,'#0072B2'],['等待均值',r.waitMs,'#D55E00']]):'<p class="status-bad">λ ≥ μ：停止使用有限稳态公式，队列没有有限的稳态平均长度。</p>';
  el('queue-explain').textContent=r.stable?'平均系统时间 = 排队 + 服务。该解析结果要求 M/M/1 的假设，不能直接当作 vLLM 的 TTFT。':'有限的一轮模拟仍然可能结束，但不代表系统在持续该到达率下存在有限的稳态均值。';
}
function renderMetrics() {
  const mode=el('metric-mode').value,gap=value('metric-gap'),r=W02.metrics(mode,gap);
  el('metric-gap').disabled=mode!=='stall';el('metric-gap-label').textContent=gap+' ms';
  el('metric-stats').innerHTML=stat(r.ttft+' ms','TTFT')+stat(r.tpot===null?'N/A':r.tpot.toFixed(2)+' ms','每 token 的 TPOT')+stat(r.meanItl===null?'N/A':r.meanItl.toFixed(2)+' ms','观测 ITL 的平均')+stat(r.maxItl===null?'N/A':r.maxItl+' ms','最大观测 ITL');
  el('metric-chart').innerHTML=r.gaps.length?bars(r.gaps.map((g,i)=>['第 '+(i+1)+' 个流间隔',g,g>80?'#D55E00':'#0072B2'])):'<p>只有首 token，没有后续间隔。不要把 N/A 当作零延迟样本。</p>';
  const prefix=`${r.times.length} 次流输出，共 ${r.tokens} token；最后输出延迟 ${r.e2e} ms，HTTP 完成延迟 ${r.completion} ms。`;
  el('metric-explain').textContent=prefix+(mode==='bundle'?'合并流块内部没有逐 token 到达时间。不要补零间隔；此处 mean ITL=45 ms，而 TPOT=18 ms/token。':mode==='single'?'TPOT 与 ITL 不适用。':`TPOT≤100 ms/token：${r.tpot<=100?'通过':'未通过'}；max ITL≤80 ms：${r.maxItl<=80?'通过':'未通过'}。平均速度与停顿要分别判断。`);
}
function renderTransfer() {
  const r=W02.transfer(value('transfer-batch'),value('transfer-context'),value('transfer-bw'),value('transfer-chunk'));
  el('transfer-stats').innerHTML=stat(r.gib.toFixed(3)+' GiB','逻辑 KV 载荷')+stat(r.payloadMs.toFixed(2)+' ms','理想载荷时间')+stat(r.totalMs.toFixed(2)+' ms','加入假设串行开销');
  el('transfer-chart').innerHTML=bars([['载荷时间',r.payloadMs,'#0072B2'],['逐块开销',r.setupMs,'#D55E00']]);
  el('transfer-explain').textContent=`${r.chunks} 块 × 40 μs = ${r.setupMs.toFixed(2)} ms 开销。GB/s 是十进制 byte/s；GiB、MiB 为二进制容量。此处假定完全串行，不含查找、转换或竞争。`;
}
for(const [ids,fn] of [[['roof-m','roof-peak','roof-bw'],renderRoof],[['queue-rate'],renderQueue],[['metric-mode','metric-gap'],renderMetrics],[['transfer-batch','transfer-context','transfer-bw','transfer-chunk'],renderTransfer]]) {
  for(const id of ids){el(id).addEventListener('input',fn);el(id).addEventListener('change',fn);}fn();
}
el('print-button').addEventListener('click',()=>window.print());
