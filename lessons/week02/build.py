"""Rebuild week 2 figures and self-contained HTML from versioned teaching data."""
import base64
import csv
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
DATA=ROOT/'experiments/w02-performance/data'
FIG=HERE/'figures'
BLUE,ORANGE,GREEN,INK='#0072B2','#D55E00','#009E73','#20374c'


def rows(name):
    with (DATA/name).open(encoding='utf-8') as f:
        return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]


def setup():
    FIG.mkdir(exist_ok=True)
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    else:plt.rcParams['font.family']=['Noto Sans CJK SC','DejaVu Sans']
    plt.rcParams.update({'font.size':11,'axes.unicode_minus':False,'axes.titleweight':'bold',
        'axes.spines.top':False,'axes.spines.right':False,'text.color':INK,'axes.labelcolor':INK,
        'axes.titlecolor':INK,'legend.frameon':False,'pdf.fonttype':42})


def save(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=240,bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def phase():
    r=rows('gemm.csv');labels=[str(int(x['m'])) for x in r];x=np.arange(len(r))
    fig,ax=plt.subplots(figsize=(10.8,4.2))
    ax.bar(x,[v['intensity'] for v in r],color=[BLUE]*4+[GREEN]*2,width=.6)
    ax.set(yscale='log',xticks=x,xticklabels=labels,xlabel='参与一次线性层的 token 行数 M',ylabel='算术强度 / (FLOP/byte)',ylim=(.6,2100))
    ax.axhline(200000/1500,color=ORANGE,ls='--',label='假想硬件屋脊点 133.33')
    for a,b in zip(x,r):ax.text(a,b['intensity']*1.16,f"{b['intensity']:.2f}",ha='center')
    ax.set_title('权重复用如何改变算术强度？',loc='left',pad=18)
    ax.legend(loc='upper left');ax.grid(axis='y',alpha=.15)
    fig.text(.12,-.06,'理论计算｜N=K=4096，2-byte 元素；输入/权重各读一次、输出写一次。',fontsize=10)
    save(fig,'phase')


def roofline():
    fig,ax=plt.subplots(figsize=(10.8,4.8));intensity=np.logspace(-1,4,300)
    ax.loglog(intensity,np.minimum(200,1.5*intensity),color=BLUE,lw=3,label='min(200, 1.5×I) TFLOP/s')
    r=rows('gemm.csv')
    for i in [0,2,4,5]:
        v=r[i];ax.scatter(v['intensity'],v['roof_tflops'],color=ORANGE,zorder=3)
        ax.annotate(f"M={int(v['m'])}",(v['intensity'],v['roof_tflops']),xytext=(5,-18 if i==4 else 11),textcoords='offset points',fontsize=10)
    ax.axvline(200/1.5,ls=':',color=GREEN,label='屋脊点 133.33 FLOP/byte')
    ax.set(xlabel='HBM 层级算术强度 / (FLOP/byte)',ylabel='性能上界 / (TFLOP/s)',ylim=(.1,500),title='Roofline 上的教学落点：这些不是实测点')
    ax.legend(loc='lower right');ax.grid(which='both',alpha=.14)
    fig.text(.12,-.03,'假想硬件：200 TFLOP/s、1500 GB/s；同一种运算精度，无稀疏加成。',fontsize=10)
    save(fig,'roofline')


def batching():
    r=rows('batch.csv');fig,axs=plt.subplots(1,2,figsize=(12.6,4.5))
    for s,c,m in [(1024,BLUE,'o'),(8192,ORANGE,'s'),(32768,GREEN,'^')]:
        a=[v for v in r if v['context']==s]
        for ax,key in zip(axs,['output_tokens_s','step_ms']):
            ax.plot([v['batch'] for v in a],[v[key] for v in a],marker=m,color=c,label=f'S={s//1024}K')
    for ax in axs:ax.set_xscale('log',base=2);ax.set_xlabel('活动 decode batch B');ax.grid(alpha=.16);ax.legend()
    axs[0].set(ylabel='总输出 token/s（模型值）',title='同时产出更多 token')
    axs[1].set(ylabel='单步间隔 / ms（模型值）',title='每个请求也可能等得更久')
    fig.tight_layout();fig.text(.08,-.045,'理论假设｜14 GB 权重、200 TFLOP/s、1500 GB/s、固定开销 1 ms；未施加容量上限，无排队。',fontsize=10)
    save(fig,'batching')


def queue():
    r=rows('queue_summary.csv');rho=sorted(set(x['rho'] for x in r))
    fig,axs=plt.subplots(1,2,figsize=(12.6,4.5))
    for ax,key,color,label in [(axs[0],'mean_ms',BLUE,'模拟均值'),(axs[1],'p95_ms',ORANGE,'模拟 p95'),(axs[1],'p99_ms',GREEN,'模拟 p99')]:
        groups=[[v[key] for v in r if v['rho']==p] for p in rho]
        center=np.array([np.mean(x) for x in groups])
        errors=np.array([[center[i]-min(g) for i,g in enumerate(groups)],[max(g)-center[i] for i,g in enumerate(groups)]])
        ax.errorbar(np.array(rho)*100,center,yerr=errors,capsize=4,marker='o',color=color,label=label)
    axs[0].plot(np.array(rho)*100,[1000/(200*(1-p)) for p in rho],color=ORANGE,ls='--',label='解析均值 1/(μ−λ)')
    for ax in axs:ax.set(xlabel='ρ=λ/μ / %',ylabel='系统响应时间 / ms');ax.grid(alpha=.15);ax.legend()
    axs[0].set_title('服务均值仍为 5 ms，等待在增长');axs[1].set_title('尾部比均值更敏感')
    fig.tight_layout();fig.text(.08,-.04,'M/M/1 模拟｜点为五轮统计量的平均；线为 min–max，非置信区间；每轮 10000 个观测请求。',fontsize=10)
    save(fig,'queue')


def timeline():
    fig,ax=plt.subplots(figsize=(12,4.8));ax.set(xlim=(-8,404),ylim=(-.6,3.3),yticks=[],xlabel='同一客户端单调时钟 / ms')
    points=[(0,'计划'),(20,'发送'),(120,'输出 1'),(140,'输出 2'),(160,'输出 3'),(360,'输出 4'),(380,'HTTP 完成')]
    ax.axhline(2.3,color='#b9ccd5')
    for i,(x,label) in enumerate(points):
        ax.plot(x,2.3,'o',color=ORANGE if '输出' in label else BLUE)
        ax.text(x,2.55+(i%2)*.4,f'{label}\n{x}',ha='center',fontsize=10)
    spans=[(20,120,1.65,'TTFT=100 ms',BLUE),(120,360,1.05,'20 + 20 + 200 ms；TPOT=80 ms/token',ORANGE),
           (20,360,.4,'E2E_token=340 ms',GREEN),(20,380,-.2,'完成延迟=360 ms',BLUE)]
    for start,end,y,label,c in spans:
        ax.annotate('',xy=(end,y),xytext=(start,y),arrowprops=dict(arrowstyle='<->',color=c,lw=2))
        ax.text((start+end)/2,y+.08,label,ha='center',color=c,fontsize=10)
    ax.set_title('构造请求 A：先问清楚从哪里开始、到哪里结束',loc='left',pad=22)
    ax.spines['left'].set_visible(False);fig.text(.13,-.03,'手工构造时间戳｜客户端排队 20 ms；最后输出后还有 20 ms 的完成收尾。',fontsize=10)
    save(fig,'timeline')


def metrics():
    fig,axs=plt.subplots(1,2,figsize=(12.4,4.3))
    axs[0].bar(['每请求 TPOT 平均','汇集全部 ITL 平均'],[105,48],color=[BLUE,GREEN],width=.55)
    for i,v in enumerate([105,48]):axs[0].text(i,v+3,str(v),ha='center')
    axs[0].set(ylabel='ms（两种权重）',ylim=(0,130),title='B 与 C：同一批数据，不同的平均')
    axs[1].bar(['第 1 个间隔','第 2 个间隔','第 3 个间隔'],[20,20,200],color=[BLUE,BLUE,ORANGE],width=.6)
    axs[1].axhline(80,color=GREEN,ls='--',label='A 的 TPOT=80 ms/token')
    axs[1].axhline(100,color='#7b648f',ls=':',label='平均 TPOT SLO=100 ms/token')
    axs[1].set(ylabel='单次间隔 / ms',title='A：均值合格，仍出现 200 ms 停顿');axs[1].legend()
    for ax in axs:ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.text(.08,-.035,'构造 trace｜B: [200] ms；C: [10,10,10,10] ms；A: [20,20,200] ms。',fontsize=10)
    save(fig,'metrics')


def transfer_figure():
    r=rows('transfer.csv');fig,axs=plt.subplots(1,2,figsize=(12.7,4.6))
    for bw,c,m in [(7,BLUE,'o'),(25,ORANGE,'s'),(50,GREEN,'^')]:
        a=[v for v in r if v['requests']==1 and v['chunk_mib']==4 and v['bandwidth_gbs']==bw]
        axs[0].plot([v['payload_gib'] for v in a],[v['payload_ms'] for v in a],color=c,marker=m,label=f'假设 {bw} GB/s')
    axs[0].set(xlabel='单请求 KV 载荷 / GiB',ylabel='理想载荷时间 / ms',title='先分清容量与路径带宽');axs[0].legend()
    a=[v for v in r if v['context']==8192 and v['requests']==1 and v['bandwidth_gbs']==25];x=np.arange(len(a))
    axs[1].bar(x,[v['payload_ms'] for v in a],color=BLUE,label='载荷时间')
    axs[1].bar(x,[v['setup_ms'] for v in a],bottom=[v['payload_ms'] for v in a],color=ORANGE,label='假设串行开销')
    axs[1].set(xticks=x,xticklabels=[str(v['chunk_mib']) for v in a],xlabel='块大小 / MiB',ylabel='时间模型 / ms',title='1 GiB / 25 GB/s，每块假设 40 μs')
    axs[1].legend();fig.tight_layout()
    fig.text(.08,-.045,'理论计算｜GB=10^9 byte；GiB=2^30 byte。未测量 SSD、网络、PCIe 或 HBM。',fontsize=10)
    save(fig,'transfer')


def result_text():
    batch=[r for r in rows('batch.csv') if r['context']==8192 and r['batch'] in [1,8,32]]
    lines=['以下为脚本生成的教学结果，均非真实引擎性能：','','| 活动 batch | 单步时间模型 / ms | 总输出吞吐模型 / token·s⁻¹ | KV 载荷 / GiB |','| ---: | ---: | ---: | ---: |']
    lines += [f"| {r['batch']:.0f} | {r['step_ms']:.3f} | {r['output_tokens_s']:.2f} | {r['kv_gib']:.1f} |" for r in batch]
    near_limit=[r['mean_ms'] for r in rows('queue_summary.csv') if r['rho']==.95]
    lines += ['','固定 S=8192。上表同一假想模型中，吞吐增加同时伴随单请求间隔增加；真实结果仍须验证。','',
              f"排队模拟在 ρ=0.95 时，五轮样本均值范围为 **{min(near_limit):.2f}–{max(near_limit):.2f} ms**，解析稳态均值为 100 ms。有限运行与相关的忙期会产生显著波动；不能选择最接近解析值的一轮当作全部证据。",'',
              '**已执行的验证**：18 项 Python 检查通过，覆盖单位、GEMM 手算、batch 权衡、流块合并、单 token、失败请求、聚合权重、FCFS 与不稳定边界。']
    return '\n'.join(lines)


def build_document():
    source=HERE/'README.md';text=source.read_text(encoding='utf-8')
    text=re.sub(r'<!-- results:start -->.*?<!-- results:end -->','<!-- results:start -->\n'+result_text()+'\n<!-- results:end -->',text,flags=re.S)
    source.write_text(text,encoding='utf-8')
    md=markdown.Markdown(extensions=['tables','fenced_code','toc','md_in_html'],extension_configs={'toc':{'toc_depth':'2-2'}})
    body=md.convert(text.replace('<details>','<details markdown="1">'))
    widgets=(HERE/'widgets.html').read_text(encoding='utf-8')
    for name in ['roofline','queue','metrics','transfer']:
        match=re.search(rf'<!-- {name}:start -->(.*?)<!-- {name}:end -->',widgets,re.S)
        body=body.replace(f'<!-- interactive-{name} -->',match.group(1))
    body=re.sub(r'src="(figures/[^\"]+\.png)"',lambda m:'src="data:image/png;base64,'+base64.b64encode((HERE/m.group(1)).read_bytes()).decode()+'"',body)
    css=(HERE.parent/'week01/style.css').read_text(encoding='utf-8')+'\n'+(HERE/'style.css').read_text(encoding='utf-8')
    js=(HERE/'interactive.js').read_text(encoding='utf-8')
    output=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="推理性能分析：Roofline、排队、TTFT、TPOT、ITL 与可复现实验协议。"><title>推理性能分析 · 第二周学习手册</title><style>{css}</style></head>
<body><a class="skip" href="#content">跳到正文</a><header class="hero"><div class="eyebrow">INFERENCE STUDY / WEEK 02</div><h1>快了多少，<br>证据在哪里？</h1><p>推理性能分析 · 把算力、带宽、排队与体验放进同一张账本</p>
<div class="badges"><span>16 小时学习周</span><span>7 张数据图</span><span>4 个交互实验</span><span>18 项计算检查</span></div></header>
<div class="layout"><aside><h2>本周导航</h2>{md.toc}<p class="aside-note">先定义指标，<br>再比较结果。</p><button id="print-button" type="button">打印 / 保存 PDF</button></aside>
<main id="content"><div class="offline-note">正文、图片和交互均已内嵌，可离线阅读。数据与源码链接需保留仓库目录。带宽、算力和时间戳是教学假设，本页没有真实 GPU benchmark。</div>{body}</main></div>
<footer>2026-10-07 · 第二周备课教材 · 公式计算与 trace 模拟，真实引擎性能待测。</footer><script>{js}</script></body></html>'''
    (HERE/'index.html').write_text(output,encoding='utf-8')
    print('Built week 2 self-contained HTML.')


if __name__=='__main__':
    setup()
    for f in [phase,roofline,batching,queue,timeline,metrics,transfer_figure]:f()
    build_document()
