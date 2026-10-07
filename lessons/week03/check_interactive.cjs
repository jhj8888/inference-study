// Test simulation behavior against independent hand fixtures and Python reference traces.
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path'),assert=require('node:assert/strict');
const {spawnSync}=require('node:child_process');
const defaults={'sim-budget':'4','sim-running':'2','sim-capacity':'8'};
const elements=new Map();
const get=id=>{if(!elements.has(id))elements.set(id,{value:defaults[id]||'',checked:id==='own-cache',innerHTML:'',textContent:'',disabled:false,attrs:{},events:{},setAttribute(k,v){this.attrs[k]=v},addEventListener(n,f){this.events[n]=f}});return elements.get(id);};
const ctx={document:{getElementById:get},window:{print(){}}};vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname,'interactive.js'),'utf8'),ctx);
const api=vm.runInContext('W03',ctx),plain=x=>JSON.parse(JSON.stringify(x));
for(const name of ['baseline','budget2','serial','wide','cancel','blocked']){
  const py=JSON.parse(fs.readFileSync(path.join(__dirname,'../../experiments/w03-lifecycle/data',name+'.json'),'utf8'));
  const c=py.config;
  assert.deepEqual(plain(api.simulate(c.budget,c.max_running,c.block_size,c.capacity,c.cancel_b)),py,name);
}
const pyCode="import sys,json;sys.path.insert(0,'experiments/w03-lifecycle');from simulate import simulate;print(json.dumps([simulate(b,s,2,c,x) for b in range(1,9) for s in range(1,4) for c in [2,4,8,16] for x in [False,True]]))";
const process=spawnSync('py',['-3.11','-c',pyCode],{cwd:path.join(__dirname,'../..'),encoding:'utf8',maxBuffer:8*1024*1024});
assert.equal(process.status,0,process.stderr);
const sweeps=JSON.parse(process.stdout);
for(const py of sweeps){const c=py.config;assert.deepEqual(plain(api.simulate(c.budget,c.max_running,c.block_size,c.capacity,c.cancel_b)),py);}
assert.equal(api.getState().selectedStep,0);assert.equal(get('sim-prev').disabled,true);
get('sim-next').events.click();assert.equal(api.getState().selectedStep,1);assert.ok(get('sim-summary').innerHTML.includes('3 / 4'));
get('sim-capacity').value='2';get('sim-capacity').events.change();get('sim-next').events.click();assert.ok(get('sim-explain').textContent.includes('BLOCKED'));assert.equal(get('sim-next').disabled,true);
get('sim-reset').events.click();assert.equal(get('sim-capacity').value,'8');assert.equal(api.getState().selectedStep,0);
get('sim-cancel').checked=true;get('sim-cancel').events.change();get('sim-next').events.click();get('sim-next').events.click();assert.ok(get('sim-state').innerHTML.includes('ABORTED'));
get('own-a').events.click();assert.ok(get('own-result').textContent.includes('ref_cnt = 1'));assert.equal(get('own-a').disabled,true);
get('own-b').events.click();assert.ok(get('own-note').textContent.includes('身份仍保留'));get('own-cache').checked=false;get('own-cache').events.change();assert.ok(get('own-note').textContent.includes('不表示 GPU 内存已交还'));
get('own-reset').events.click();assert.ok(get('own-result').textContent.includes('ref_cnt = 2'));
get('journey-6').events.click();assert.ok(get('journey-detail').innerHTML.includes('GPUModelRunner'));assert.equal(get('journey-6').attrs['aria-pressed'],'true');
console.log(`PASS: 6 stored scenarios and ${sweeps.length} Python/JS traces; step navigation, reset, cancellation, blocked status, ownership and source journey transitions.`);
