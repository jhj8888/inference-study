// Numerical cross-checks against Python outputs plus meaningful UI transitions.
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path'),assert=require('node:assert/strict');
const defaults={'roof-m':'1','roof-peak':'200','roof-bw':'1500','queue-rate':'180','metric-mode':'stall','metric-gap':'200','transfer-batch':'1','transfer-context':'8192','transfer-bw':'25','transfer-chunk':'4'};
const elements=new Map();
const get=id=>{if(!elements.has(id))elements.set(id,{value:defaults[id]||'',innerHTML:'',textContent:'',disabled:false,events:{},addEventListener(n,fn){this.events[n]=fn;}});return elements.get(id);};
const ctx={document:{getElementById:get},window:{print(){}}};vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname,'interactive.js'),'utf8'),ctx);
const api=ctx.W02,near=(a,b)=>assert.ok(Math.abs(a-b)<1e-9,`${a} != ${b}`);
const csv=name=>{const lines=fs.readFileSync(path.join(__dirname,'../../experiments/w02-performance/data',name),'utf8').trim().split(/\r?\n/);const keys=lines.shift().split(',');return lines.map(l=>Object.fromEntries(l.split(',').map((v,i)=>[keys[i],Number(v)])));};
for(const r of csv('gemm.csv')){const a=api.gemm(r.m);near(a.intensity,r.intensity);near(a.lowerMs,r.lower_ms);}
for(const r of csv('transfer.csv')){const a=api.transfer(r.requests,r.context,r.bandwidth_gbs,r.chunk_mib);near(a.gib,r.payload_gib);near(a.totalMs,r.serialized_ms);assert.equal(a.chunks,r.chunks);}
const py=JSON.parse(fs.readFileSync(path.join(__dirname,'../../experiments/w02-performance/data/metrics.json'),'utf8')).per_request[0];
const a=api.metrics('stall');near(a.ttft,py.ttft_ms);near(a.tpot,py.tpot_ms);near(a.completion,py.completion_ms);
near(api.metrics('bundle').tpot,18);near(api.metrics('bundle').meanItl,45);assert.equal(api.metrics('single').tpot,null);
near(api.queue(180).meanMs,50);assert.equal(api.queue(200).stable,false);assert.equal(api.queue(220).meanMs,null);
get('queue-rate').value='200';get('queue-rate').events.input();assert.ok(get('queue-chart').innerHTML.includes('没有有限'));
get('queue-rate').value='180';get('queue-rate').events.input();assert.ok(get('queue-stats').innerHTML.includes('50.00'));
get('roof-m').value='2048';get('roof-m').events.change();assert.ok(get('roof-explain').textContent.includes('计算项'));
get('metric-mode').value='bundle';get('metric-mode').events.change();assert.equal(get('metric-gap').disabled,true);assert.ok(get('metric-stats').innerHTML.includes('18.00'));
get('metric-mode').value='single';get('metric-mode').events.change();assert.ok(get('metric-chart').innerHTML.includes('没有后续间隔'));
get('metric-mode').value='stall';get('metric-gap').value='20';get('metric-mode').events.change();assert.equal(get('metric-gap').disabled,false);assert.ok(!get('metric-explain').textContent.includes('未通过'));
get('transfer-chunk').value='0.25';get('transfer-chunk').events.change();assert.ok(get('transfer-explain').textContent.includes('4096 块'));
console.log('PASS: JS/Python agreement for 6 GEMMs, 90 transfers and request A; queue instability, bundled/single output, SLO and DOM transitions.');
