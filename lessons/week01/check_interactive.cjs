// Check numerical agreement and deterministic UI transitions without external packages.
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const assert = require('node:assert/strict');
const code = fs.readFileSync(path.join(__dirname, 'interactive.js'), 'utf8');
const elements = new Map();
const defaults = {'query-row':'2','kv-mode':'8','kv-requests':'8','kv-bytes':'2','kv-budget':'16','kv-context':'8192'};
const get = id => {
  if (!elements.has(id)) elements.set(id, {value:defaults[id]||'',checked:id==='causal',innerHTML:'',textContent:'',disabled:false,events:{},addEventListener(name,fn){this.events[name]=fn;}});
  return elements.get(id);
};
const context = {document:{getElementById:get},window:{print(){}}};
vm.createContext(context);vm.runInContext(code,context);
const api=context.W01;
const example=JSON.parse(fs.readFileSync(path.join(__dirname,'../../experiments/w01-attention/data/attention_example.json'),'utf8'));
const actual=api.attention(true);
for(const key of ['scores','weights','output']) for(let i=0;i<3;i++) for(let j=0;j<actual[key][i].length;j++) assert.ok(Math.abs(actual[key][i][j]-example[key][i][j])<1e-12);
assert.equal(api.kvBytes(32,8,8192,8,128,2)/2**30,8);
for(const heads of [32,8,1]) for(const bytes of [1,2,4]) assert.equal(api.kvBytes(32,4,16384,heads,128,bytes),2*32*4*16384*heads*128*bytes);
for(let step=0;step<=4;step++){assert.equal(api.decode(step).cache,step===0?0:step+3);assert.equal(api.decode(step).emitted,step);}
assert.ok(!get('decode-tokens').innerHTML.includes('token emitted'));
for(let i=0;i<4;i++) get('decode-next').events.click();
assert.equal(get('decode-next').disabled,true);
assert.ok(get('decode-cache').textContent.includes('7 个位置'));
assert.ok(get('decode-tokens').innerHTML.includes('emitted">g3'));
get('decode-reset').events.click();assert.equal(get('decode-prev').disabled,true);
get('query-row').value='0';get('causal').checked=false;get('causal').events.change();
assert.ok(get('attention-explain').textContent.includes('已关闭'));
assert.ok(api.attention(false).weights[0][2]>0);
get('kv-mode').value='32';get('kv-mode').events.input();
assert.ok(get('kv-stats').innerHTML.includes('32.000 GiB'));
assert.ok(get('kv-explain').textContent.includes('已超过'));
get('kv-budget').value='0';get('kv-budget').events.input();
assert.ok(get('kv-explain').textContent.includes('大于 0'));
console.log('PASS: JS/Python numerical agreement, KV calculations, mask toggle, decode transitions, budget handling.');
