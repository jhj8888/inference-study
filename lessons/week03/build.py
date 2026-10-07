"""Build week 3 teaching figures and a self-contained offline lesson."""
import base64
import csv
import html
import json
from pathlib import Path
import re
import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATA=ROOT/'experiments/w03-lifecycle/data'
FIG=HERE/'figures'
BLUE,ORANGE,GREEN,INK='#0072B2','#D55E00','#009E73','#203b52'


def load(name):
    return json.loads((DATA/(name+'.json')).read_text(encoding='utf-8'))


def setup():
    FIG.mkdir(exist_ok=True)
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    else: plt.rcParams['font.family']=['Noto Sans CJK SC','DejaVu Sans']
    plt.rcParams.update({'font.size':11,'axes.unicode_minus':False,'axes.titleweight':'bold',
        'axes.spines.top':False,'axes.spines.right':False,'text.color':INK,
        'axes.labelcolor':INK,'legend.frameon':False,'pdf.fonttype':42})


def save(fig,name,note):
    fig.text(.12,.018,note,fontsize=9,color='#536d7c')
    fig.savefig(FIG/f'{name}.png',dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def budget_figure():
    trace=load('baseline')['trace']
    values=np.array([[s['plans'].get(name,0) for s in trace] for name in 'ABC'])
    fig,ax=plt.subplots(figsize=(10.4,4.7))
    ax.imshow(values,cmap='Blues',vmin=0,vmax=5,aspect='auto')
    for i,name in enumerate('ABC'):
        for j,s in enumerate(trace):
            n=values[i,j]
            ax.text(j,i,f'{n} 个输入位置'+ ('\n产生 1 个输出' if name in s['outputs'] else '\n无输出'),
                    ha='center',va='center',fontsize=11,color='white' if n>=3 else INK)
    ax.set(xticks=range(4),xticklabels=[f'step {i}' for i in range(4)],yticks=range(3),
           yticklabels=['A (4→2)','B (2→3)','C (6→1)'],title='同一迭代里，prefill 和 decode 可以共同消耗预算')
    ax.tick_params(length=0,pad=10)
    fig.subplots_adjust(left=.15,bottom=.17,top=.84)
    save(fig,'budget','教学模拟｜预算 4、活动上限 2、容量 8 块。C 在 step 1 到达；0 不表示请求已结束。')


def kv_figure():
    trace=load('baseline')['trace'];x=np.arange(len(trace))
    fig,ax=plt.subplots(figsize=(10.4,4.8))
    for shift,key,color,label in [(-.18,'peak_blocks',BLUE,'计划完成 / 执行释放前'),(.18,'used_after',ORANGE,'步末完成释放后')]:
        values=[s[key] for s in trace]
        ax.bar(x+shift,values,.34,color=color,label=label)
        for a,b in zip(x+shift,values):ax.text(a,b+.13,str(b),ha='center')
    ax.axhline(8,color=GREEN,ls='--',label='总容量 8 块')
    ax.set(xticks=x,xticklabels=[f'step {i}' for i in x],ylim=(0,9.5),ylabel='已占用的抽象 KV 块数',title='区分资源峰值与释放后的占用')
    ax.legend(loc='upper left',ncols=2,fontsize=10);ax.grid(axis='y',alpha=.14)
    fig.subplots_adjust(bottom=.18,top=.85)
    save(fig,'kv','教学模拟｜每块 2 个输入位置，无共享和前缀复用；这些数值不是显存 GiB。')


def counter_figure():
    fig,ax=plt.subplots(figsize=(10.4,4.9))
    x=np.arange(3)
    ax.plot(x,[4,5,6],marker='o',lw=2.5,color=BLUE,label='已知序列 K = prompt + 输出')
    ax.plot(x,[0,4,5],marker='s',lw=2.5,color=ORANGE,label='已计算输入位置 C')
    for i,(known,computed) in enumerate(zip([4,5,6],[0,4,5])):
        ax.text(i,known+.18,str(known),ha='center',color=BLUE)
        ax.text(i,computed-.32,str(computed),ha='center',color=ORANGE)
    ax.set(xticks=x,xticklabels=['开始：P=4，O=0','prefill 后：O=1','下一步后：O=2，结束'],ylim=(-.6,7.2),ylabel='位置 / token 计数',title='A 的最后一个输出已被采样，但不需要再次 forward')
    ax.legend(loc='upper left',fontsize=10);ax.grid(alpha=.13)
    fig.subplots_adjust(bottom=.2,top=.85)
    save(fig,'counters','同步教学模型｜图中是步末逻辑计数；真实调度器的 optimistic computed 还需结合 in-flight 解读。')


def scenario_figure():
    with (DATA/'summary.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    labels=['基准','预算 2','活动上限 1','预算 8 / 上限 3','取消 B','容量 2 / 未完成']
    colors=[BLUE,BLUE,BLUE,BLUE,ORANGE,'#8797a1']
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    y=np.arange(len(rows))
    for ax,key,title in zip(axs,['steps','peak_blocks'],['观测迭代步数（无毫秒含义）','峰值 KV 块数（每块 2 个位置）']):
        vals=[int(r[key]) for r in rows]
        bars=ax.barh(y,vals,color=colors,height=.6)
        bars[-1].set_hatch('//')
        for a,b in zip(y,vals):ax.text(b+.12,a,str(b),va='center')
        ax.set(yticks=y,yticklabels=labels if ax is axs[0] else [],title=title,xlim=(0,10))
        ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    fig.subplots_adjust(left=.18,bottom=.18,top=.84,wspace=.15)
    save(fig,'scenarios','教学模拟｜取消 B 改变交付量；容量 2 在第 2 个观测步骤停住，并未完成，不能称为加速。')


def svg_text(x,y,lines,size=17,color=INK,anchor='middle'):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}" font-size="{size}">'+''.join(f'<tspan x="{x}" dy="{0 if i==0 else 24}">{html.escape(line)}</tspan>' for i,line in enumerate(lines))+'</text>'


def svg_box(x,y,w,h,lines,color='#e8f0f6'):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{color}" stroke="#b6c9d6"/>'+svg_text(x+w/2,y+30,lines)


def svg_start(title,w,h):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}"><rect width="100%" height="100%" fill="#fff"/><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#456b84"/></marker></defs><g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif">'+svg_text(32,38,[title],25,anchor='start')


def arrow(path):
    return f'<path d="{path}" fill="none" stroke="#456b84" stroke-width="2.3" marker-end="url(#arrow)"/>'


def diagrams():
    s=svg_start('在线单卡主线：分清逻辑职责与进程边界',1040,780)
    for x,w,title in [(30,410,'API 进程'),(585,425,'EngineCore 进程（所选 UniProc 路径）')]:
        s+=f'<rect x="{x}" y="75" width="{w}" height="595" rx="16" fill="#f8fafc" stroke="#809aac" stroke-dasharray="7 5"/>'+svg_text(x+w/2,108,[title],17)
    for y,lines in [(142,['HTTP / Serving','协议、渲染、参数']),(265,['AsyncLLM / OutputProcessor','登记、生成器、文本输出']),(388,['AsyncMPClient','请求与结果消息'])]:
        s+=svg_box(55,y,360,82,lines)
    for y,lines in [(142,['EngineCore','接收、分发、迭代']),(247,['Scheduler / KV Manager','预算、准入、块元数据']),(352,['UniProcExecutor → Worker','同进程方法分派']),(457,['GPUModelRunner','准备输入、forward、采样']),(562,['Core / Scheduler update','状态、清理、输出队列 / socket'])]:
        s+=svg_box(610,y,375,76,lines,'#e7f1ed')
    s+=arrow('M235 224 L235 265')+arrow('M235 347 L235 388')
    for y in [218,323,428,533]:s+=arrow(f'M797 {y} L797 {y+29}')
    s+=arrow('M415 414 L500 414 L500 180 L610 180')+svg_text(508,158,['ADD / ZMQ'],14)
    s+=arrow('M610 600 L545 600 L545 715 L235 715 L235 470')
    s+=svg_text(568,752,['返回：core 输出 → collector → Serving / SSE；Runner 负责设备执行，不是额外 Python 进程'],15)
    (FIG/'architecture.svg').write_text(s+'</g></svg>',encoding='utf-8')
    s=svg_start('主线状态：等待、活动、结束；抢占可以返回等待',1040,455)
    s+=svg_box(35,115,240,88,['WAITING','等准入 / 资源'])
    s+=svg_box(385,115,240,88,['RUNNING','活动，但不保证每步入选'])
    s+=svg_box(740,115,270,88,['FINISHED_*','完成 / 长度上限 / 取消'],'#e7f1ed')
    s+=svg_box(385,302,240,75,['PREEMPTED','等待恢复与可能重算'],'#fff0df')
    s+=arrow('M155 115 L155 90 L875 90 L875 115')+svg_text(520,77,['在 waiting 中也可被取消'],15)
    s+=arrow('M275 156 L385 156')+svg_text(330,134,['获准调度'],15)
    s+=arrow('M625 156 L740 156')+svg_text(682,134,['停止或取消'],15)
    s+=arrow('M505 203 L505 302')+svg_text(565,259,['资源不足等'],15)
    s+=arrow('M385 338 L155 338 L155 203')+svg_text(228,317,['重新放入 waiting'],15)
    s+=svg_text(520,421,['简化状态图；真实代码还包含 grammar、远程 KV、流式输入等等待状态。'],15)
    (FIG/'states.svg').write_text(s+'</g></svg>',encoding='utf-8')


def result_table():
    with (DATA/'summary.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    lines=['| 场景 | 结果 | 观测步数 | 输入位置数 | 已产生输出 | 峰值块 |','| --- | --- | ---: | ---: | ---: | ---: |']
    for r in rows:lines.append('| '+' | '.join(r[k] for k in ['scenario','status','steps','input_positions','output_tokens','peak_blocks'])+' |')
    return '\n'.join(lines)


def sources_table():
    ev=json.loads((HERE/'source-evidence.json').read_text(encoding='utf-8'))
    lines=['| 锚点 | 文件与完整符号 | 定义 / 证据行 |','| --- | --- | --- |']
    for a in ev['anchors']:
        lines.append(f'| <a id="{a["id"].lower()}"></a>{a["id"]} | [{a["file"]}](../../../vllm-main/{a["file"]})<br>`{a["symbol"]}` | {a["start"]} / {a["evidence_line"]} |')
    return '\n'.join(lines)


def document():
    source=HERE/'README.md';text=source.read_text(encoding='utf-8')
    for key,value in [('results',result_table()),('sources',sources_table())]:
        text=re.sub(rf'<!-- {key}:start -->.*?<!-- {key}:end -->',f'<!-- {key}:start -->\n{value}\n<!-- {key}:end -->',text,flags=re.S)
    source.write_text(text,encoding='utf-8',newline='\n')
    md=markdown.Markdown(extensions=['tables','fenced_code','toc','md_in_html'],extension_configs={'toc':{'toc_depth':'2-2'}})
    body=md.convert(text.replace('<details>','<details markdown="1">'))
    body=body.replace('<table>','<div class="table-scroll"><table>').replace('</table>','</table></div>')
    widgets=(HERE/'widgets.html').read_text(encoding='utf-8')
    for name in ['journey','scheduler','ownership']:
        widget=re.search(rf'<!-- {name}:start -->(.*?)<!-- {name}:end -->',widgets,re.S).group(1)
        body=body.replace(f'<!-- interactive-{name} -->',widget)
    def inline(m):
        file=HERE/m.group(1);mime='image/svg+xml' if file.suffix=='.svg' else 'image/png'
        return 'src="data:'+mime+';base64,'+base64.b64encode(file.read_bytes()).decode()+'"'
    body=re.sub(r'src="(figures/[^\"]+)"',inline,body)
    css=(HERE.parent/'week01/style.css').read_text(encoding='utf-8')+'\n'+(HERE/'style.css').read_text(encoding='utf-8')
    js=(HERE/'interactive.js').read_text(encoding='utf-8')
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="vLLM 请求生命周期：从源码证据到三请求逐步模拟。"><title>跟着一个请求读通 vLLM · 第三周</title><style>{css}</style></head><body>
<a class="skip" href="#content">跳到正文</a><header class="hero"><div class="eyebrow">INFERENCE STUDY / WEEK 03</div><h1>一个请求，<br>走过哪些地方？</h1><p>vLLM 请求生命周期 · 把源码、状态、数据与资源连成一条线</p><div class="badges"><span>16 小时学习周</span><span>6 张图解</span><span>3 个交互演示</span><span>45 个源码锚点</span></div></header>
<div class="layout"><aside><h2>本周导航</h2>{md.toc}<p class="aside-note">每一条箭头，<br>都能找到证据。</p><button id="print-button" type="button">打印 / 保存 PDF</button></aside><main id="content"><div class="offline-note">正文、图片与交互已内嵌，可单文件离线阅读。代码、数据和外部源码链接需保留原目录。本页含静态源码分析和教学模拟，没有 GPU 性能实测。</div>{body}</main></div><footer>2026-10-07 · 本地完整源码快照 + 可复现教学模拟 · 学习验收待用户完成</footer><script>{js}</script></body></html>'''
    (HERE/'index.html').write_text(page,encoding='utf-8',newline='\n')
    print(f'Built {len(page.encode("utf-8")):,} bytes: 2 SVG diagrams, 4 data figures, 3 interactive widgets.')


if __name__=='__main__':
    setup();diagrams()
    for f in [budget_figure,kv_figure,counter_figure,scenario_figure]:f()
    document()
