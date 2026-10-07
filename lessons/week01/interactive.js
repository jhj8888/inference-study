'use strict';
const W01 = {
  attention(causal = true) {
    const Q = [[1,0],[0,1],[1,1]], K = Q, V = [[1,0],[0,2],[3,1]];
    const scores = Q.map(q => K.map(k => (q[0]*k[0]+q[1]*k[1])/Math.sqrt(2)));
    const weights = scores.map((row,i) => {
      const visible = row.map((x,j) => causal && j>i ? -Infinity : x);
      const largest = Math.max(...visible);
      const exp = visible.map(x => Math.exp(x-largest)), total=exp.reduce((a,b)=>a+b,0);
      return exp.map(x=>x/total);
    });
    const output=weights.map(row=>[0,1].map(d=>row.reduce((sum,w,j)=>sum+w*V[j][d],0)));
    return {scores,weights,output};
  },
  kvBytes(layers,requests,tokens,heads,dimension,bytes) {
    return 2*layers*requests*tokens*heads*dimension*bytes;
  },
  decode(step) {
    const cache = step === 0 ? 0 : 4+step-1;
    return {cache, emitted: step, oldCache: step<=1 ? 0 : cache-1,
      input: step===0?'尚未开始':step===1?'p0 p1 p2 p3':'g'+(step-2),
      next: step===0?null:'g'+(step-1)};
  }
};
globalThis.W01=W01;
if (typeof document !== 'undefined') {
  const el=id=>document.getElementById(id);
  function drawAttention() {
    const index=Number(el('query-row').value), causal=el('causal').checked, data=W01.attention(causal);
    el('query-label').textContent=String(index);
    let table='<table class="matrix"><caption>Attention 权重 A</caption><thead><tr><th>Q / K</th><th>K0</th><th>K1</th><th>K2</th></tr></thead><tbody>';
    data.weights.forEach((row,i)=>{
      table+='<tr class="'+(i===index?'active-row':'')+'"><th>Q'+i+'</th>';
      row.forEach((w,j)=>{table+='<td class="'+(causal&&j>i?'masked':'')+'" style="background-color:'+(causal&&j>i?'#e8eeee':'rgba(0,114,178,'+(0.05+w*.55)+')')+'">'+(causal&&j>i?'遮住':w.toFixed(4))+'</td>';});
      table+='</tr>';
    });
    el('attention-matrix').innerHTML=table+'</tbody></table>';
    let svg='<svg viewBox="0 0 340 150" role="img" aria-label="当前 query 对各 key 的权重"><line x1="43" y1="20" x2="43" y2="128" stroke="#a9c0bb"/>';
    data.weights[index].forEach((w,j)=>{let y=20+j*37;svg+='<text x="5" y="'+(y+18)+'" font-size="13">K'+j+'</text><rect x="43" y="'+y+'" width="'+(w*220)+'" height="25" rx="4" fill="#0072b2"/><text x="'+(50+w*220)+'" y="'+(y+18)+'" font-size="12">'+w.toFixed(4)+'</text>';});
    el('attention-bars').innerHTML=svg+'</svg>';
    el('attention-result').textContent='O'+index+' = ['+data.output[index].map(x=>x.toFixed(4)).join(', ')+']';
    el('attention-explain').textContent=causal?'当前 Q'+index+' 只能看 K0…K'+index+'；先屏蔽，再对允许位置归一化，最后对 V 加权。':'因果掩码已关闭：较早的 Query 也能读取未来 Key。观察 Q0、Q1 的输出如何改变。';
  }
  let step=0;
  function drawDecode() {
    const state=W01.decode(step);
    el('decode-stage').textContent=step===0?'准备开始':step===1?'第 1 次调用 · prefill':'第 '+step+' 次调用 · decode '+(step-1);
    let tokens='';
    for(let i=0;i<8;i++){
      const label=i<4?'p'+i:'g'+(i-4);
      const kind=i<state.oldCache?'old':i<state.cache?'new':i>=4&&i<4+state.emitted?'emitted':'future';
      tokens+='<span class="token '+kind+'">'+label+'</span>';
    }
    el('decode-tokens').innerHTML=tokens;
    el('decode-action').textContent=step===0?'尚未执行 forward。点击“下一步”，一次处理完整 prompt。':'送入 '+state.input+'，由最后位置的 logits 选出 '+state.next+'。'+(step===4?'目标已达到，g3 不再送回模型。':'');
    el('decode-cache').textContent='每层 '+state.cache+' 个位置；已生成 '+state.emitted+' 个 token。';
    el('decode-prev').disabled=step===0;el('decode-next').disabled=step===4;
  }
  function drawCapacity() {
    const heads=Number(el('kv-mode').value),requests=Number(el('kv-requests').value),bytes=Number(el('kv-bytes').value),tokens=Number(el('kv-context').value);
    const budget=Number(el('kv-budget').value);
    if(!Number.isFinite(budget)||budget<=0){el('kv-explain').textContent='请输入大于 0 的净 KV 预算。';el('kv-stats').innerHTML='';el('kv-chart').innerHTML='';return;}
    const total=W01.kvBytes(32,requests,tokens,heads,128,bytes)/2**30;
    const single=total/requests,maxRequests=Math.floor(budget/single);
    el('kv-context-label').textContent=tokens+' token';
    el('kv-stats').innerHTML='<div class="stat"><strong>'+total.toFixed(3)+' GiB</strong><span>全部请求的理想 KV 载荷</span></div><div class="stat"><strong>'+single.toFixed(3)+' GiB</strong><span>单请求 KV 载荷</span></div><div class="stat"><strong>'+maxRequests+' 个</strong><span>按净预算估计的最大同长度请求数</span></div>';
    let max=Math.max(budget,total)*1.18;
    let svg='<svg viewBox="0 0 650 125" role="img" aria-label="KV 载荷与净预算比较">';
    [['理论载荷',total,total>budget?'#d55e00':'#0072b2'],['净 KV 预算',budget,'#009e73']].forEach(([label,value,color],i)=>{
      let y=15+i*47;svg+='<text x="0" y="'+(y+21)+'" font-size="14">'+label+'</text><rect x="102" y="'+y+'" width="'+(value/max*430)+'" height="30" rx="5" fill="'+color+'"/><text x="'+(112+value/max*430)+'" y="'+(y+21)+'" font-size="13">'+value.toFixed(2)+' GiB</text>';
    });
    el('kv-chart').innerHTML=svg+'</svg>';
    el('kv-explain').textContent='2 × 32 × '+requests+' × '+tokens+' × '+heads+' × 128 × '+bytes+' / 2³⁰ = '+total.toFixed(3)+' GiB。'+(total>budget?'已超过设定的净 KV 预算。':'在理想 payload 预算内。')+'未计碎片、元数据与其他运行时开销。'+(bytes===1?'1 byte 只是量化载荷假设，不保证实际支持或质量。':'');
  }
  el('query-row').addEventListener('input',drawAttention);el('causal').addEventListener('change',drawAttention);
  el('decode-prev').addEventListener('click',()=>{step=Math.max(0,step-1);drawDecode();});
  el('decode-next').addEventListener('click',()=>{step=Math.min(4,step+1);drawDecode();});
  el('decode-reset').addEventListener('click',()=>{step=0;drawDecode();});
  ['kv-mode','kv-requests','kv-bytes','kv-context','kv-budget'].forEach(id=>el(id).addEventListener('input',drawCapacity));
  el('print-button').addEventListener('click',()=>window.print());
  drawAttention();drawDecode();drawCapacity();
}
