"""Build reproducible figures and an offline interactive edition of the lesson."""
import base64
import csv
import json
from pathlib import Path
import re

import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
from matplotlib.colors import ListedColormap
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'experiments/w01-attention/data'
FIGURES = HERE / 'figures'
FIGURES.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN = '#0072B2', '#D55E00', '#009E73'
INK, LIGHT = '#183c43', '#e9f0f1'


def read_json(name):
    return json.loads((DATA / name).read_text(encoding='utf-8'))


def save(fig, name):
    fig.savefig(FIGURES / f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURES / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def setup_style():
    cjk_font = Path('C:/Windows/Fonts/msyh.ttc')
    if cjk_font.exists():
        font_manager.fontManager.addfont(str(cjk_font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(cjk_font)).get_name()
    else:
        plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams.update({'font.size': 11, 'axes.unicode_minus': False, 'axes.titleweight': 'bold',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelcolor': INK,
        'text.color': INK, 'axes.titlecolor': INK, 'pdf.fonttype': 42, 'legend.frameon': False})


def pipeline():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set(xlim=(0, 12), ylim=(0, 4.8)); ax.axis('off')
    titles = [('token IDs', '[B,T] 整数'), ('Embedding', '[B,T,D]'), ('Decoder × L', '[B,T,D]'),
              ('Norm + LM head', '[B,T,Vocab]'), ('采样', '选择下一个 ID')]
    for i, (title, shape) in enumerate(titles):
        x = .12 + i * 2.4
        ax.add_patch(patches.FancyBboxPatch((x, 3.1), 2.15, 1.0, boxstyle='round,pad=0.04',
                     facecolor='#edf6fa' if i != 2 else '#daf2eb', edgecolor=BLUE if i != 2 else GREEN))
        ax.text(x + 1.075, 3.75, title, ha='center', weight='bold', fontsize=11)
        ax.text(x + 1.075, 3.37, shape, ha='center', fontsize=10)
        if i < 4:
            ax.annotate('', xy=(x+2.36,3.6), xytext=(x+2.17,3.6), arrowprops={'arrowstyle':'->','color':INK})
    ax.text(.1, 4.5, '从 token 到下一个 token：形状不会凭空消失', fontsize=16, weight='bold')
    ax.text(.15, 2.6, '展开一个教学用 pre-norm block', fontsize=12, weight='bold')
    blocks = ['Norm', 'Attention', '+ 残差', 'Norm', 'FFN', '+ 残差']
    for i, label in enumerate(blocks):
        x = .2 + i * 1.97
        ax.add_patch(patches.FancyBboxPatch((x,1.35),1.63,.65,boxstyle='round,pad=.03',
                     facecolor=LIGHT,edgecolor='#a7bbbe'))
        ax.text(x+.815,1.675,label,ha='center',va='center',fontsize=11)
        if i<5:
            ax.annotate('',xy=(x+1.94,1.675),xytext=(x+1.66,1.675),arrowprops={'arrowstyle':'->'})
    ax.annotate('',xy=(4.97,2.1),xytext=(.2,2.1),arrowprops={'arrowstyle':'->','color':ORANGE})
    ax.text(2.7,2.32,'跳过 Norm 与 Attention 的残差路径',ha='center',fontsize=9,color=ORANGE)
    ax.annotate('',xy=(10.88,1.15),xytext=(5.8,1.15),arrowprops={'arrowstyle':'->','color':ORANGE})
    ax.text(8.4,.76,'跳过 Norm 与 FFN 的残差路径',ha='center',fontsize=9,color=ORANGE)
    ax.text(.2,.15,'形状示意｜真实模型可使用 RMSNorm、门控 FFN 与融合算子；此图不表示 kernel 调用次数。',fontsize=10)
    save(fig,'pipeline')


def attention_figure():
    example=read_json('attention_example.json')
    scores=np.array(example['scores']); weights=np.array(example['weights'])
    fig, axes=plt.subplots(1,3,figsize=(11.8,3.6),layout='constrained')
    values=[scores,np.tril(np.ones((3,3))),weights]
    for ax, value, title in zip(axes,values,['缩放点积分数','因果允许矩阵：1=允许','归一化权重：每行和为 1']):
        ax.imshow(value,cmap='Blues',vmin=0,vmax=max(1,float(value.max())));ax.set_title(title,fontsize=12)
        ax.set_xticks(range(3),['K0','K1','K2']);ax.set_yticks(range(3),['Q0','Q1','Q2'])
        ax.set_xlabel('Key 位置');ax.set_ylabel('Query 位置')
        for i in range(3):
            for j in range(3):
                text=f'{value[i,j]:.3f}' if title!='因果允许矩阵：1=允许' else str(int(value[i,j]))
                ax.text(j,i,text,ha='center',va='center',color='white' if value[i,j]>.65*max(1,value.max()) else INK)
    save(fig,'attention')


def timeline():
    fig,ax=plt.subplots(figsize=(11,3.4),layout='constrained')
    states=np.array([[2,2,2,2,0,0,0],[1,1,1,1,2,0,0],[1,1,1,1,1,2,0]])
    ax.imshow(states,cmap=ListedColormap(['#edf0f1','#0072B2','#D55E00']),vmin=0,vmax=2,aspect='auto')
    ax.set_xticks(range(7),['p0','p1','p2','p3','g0','g1','g2'])
    ax.set_yticks(range(3),['prefill → 选出 g0','decode 1 → 选出 g1','decode 2 → 选出 g2'])
    for i in range(3):
        for j in range(7):
            ax.text(j,i,['未写入','已有 KV','新写入'][states[i,j]],ha='center',va='center',
                    fontsize=10,color=INK if states[i,j]==0 else 'white')
    ax.set_title('生成 3 个 token：最后的 g2 已输出，但尚未写入缓存',pad=16)
    ax.set_xlabel('每层缓存中的 token 位置；图中不表示物理页布局')
    save(fig,'cache_timeline')


def head_groups():
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),layout='constrained')
    palette=[BLUE,ORANGE,GREEN,'#CC79A7','#E69F00','#56B4E9','#626f80','#8759a3']
    for ax,hkv,name in zip(axes,[8,2,1],['MHA','GQA','MQA']):
        ax.set(xlim=(-.5,7.5),ylim=(-.65,1.65));ax.axis('off');ax.set_title(f'{name} · Hq=8 / Hkv={hkv}')
        positions=np.linspace(0,7,hkv) if hkv>1 else [3.5]
        for q in range(8):
            group=q//(8//hkv);color=palette[group]
            ax.plot([q,positions[group]],[1,0],color=color,alpha=.55,lw=1.4)
            ax.scatter(q,1,s=220,color=color,zorder=3)
            ax.text(q,1.32,f'Q{q}',ha='center',fontsize=9)
        for i,p in enumerate(positions):
            ax.scatter(p,0,s=330,color=palette[i],marker='s',zorder=3)
            ax.text(p,-.35,f'KV{i}',ha='center',fontsize=9)
        ax.text(3.5,-.61,f'每个 KV 头服务 {8//hkv} 个 Q 头',ha='center',fontsize=10)
    save(fig,'head_groups')


def capacity():
    with (DATA/'capacity.csv').open(encoding='utf-8') as stream: rows=list(csv.DictReader(stream))
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for mode,color,marker in [('MHA',BLUE,'o'),('GQA',ORANGE,'s'),('MQA',GREEN,'^')]:
        subset=[r for r in rows if r['mode']==mode and r['requests']=='1' and r['bytes_per_element']=='2']
        x=[int(r['tokens']) for r in subset];y=[float(r['kv_gib']) for r in subset]
        axes[0].plot(x,y,label=mode,color=color,marker=marker,lw=2)
    axes[0].set_xscale('log',base=2);axes[0].set_xticks(x,[f'{v//1024}K' for v in x])
    axes[0].set(xlabel='每请求上下文 token 数（1K=1024）',ylabel='逻辑 KV 容量 / GiB',title='单请求：L=32, d=128, s=2 bytes')
    axes[0].grid(alpha=.18);axes[0].legend()
    contexts=[1024,4096,8192,16384,32768];batches=[1,4,8,16]
    z=np.array([[2*32*b*s*8*128*2/2**30 for s in contexts] for b in batches])
    im=axes[1].imshow(z,cmap='Blues',aspect='auto',vmin=0)
    axes[1].set_xticks(range(5),[f'{s//1024}K' for s in contexts]);axes[1].set_yticks(range(4),batches)
    axes[1].set(xlabel='每请求上下文 token 数',ylabel='驻留请求数',title='GQA · Hkv=8：并发使容量相乘')
    for i in range(4):
        for j in range(5):axes[1].text(j,i,f'{z[i,j]:g}',ha='center',va='center',color='white' if z[i,j]>35 else INK)
    fig.colorbar(im,ax=axes[1],label='GiB',shrink=.85)
    save(fig,'capacity')


def work():
    with (DATA/'work.csv').open(encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    x=np.array([int(r['output_token']) for r in rows])
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for prefix,label,color in [('full','完整前缀重算',BLUE),('cached','KVCache',ORANGE)]:
        token_rows=np.cumsum([int(r[prefix+'_qkv_token_rows']) for r in rows])
        cells=np.array([int(r[prefix+'_cumulative_cells']) for r in rows])
        axes[0].plot(x,token_rows,color=color,label=label,lw=2)
        axes[1].plot(x,cells,color=color,label=label,lw=2)
    axes[0].set(ylabel='累计进入投影的 token 行数',title='投影工作量代理（含共同 prefill）')
    axes[1].set(ylabel='每头累计密集分数网格单元',title='Attention 工作量代理（不是延迟）')
    for ax in axes:ax.set_xlabel('已输出的新 token 数');ax.grid(alpha=.2);ax.legend()
    save(fig,'work')


def correctness():
    checks=read_json('checks.json')['checks']
    labels=['正确缓存\n最大误差','掩码偏移\n遗漏','RoPE 位置\n重置','权重更新\n旧 KV','前缀修改\n旧 KV']
    values=[max(c['max_abs_error'] for c in checks if c['type']=='positive')]
    values += [next(c['max_abs_error'] for c in checks if c['case']==name)
               for name in ['wrong_mask','wrong_position','stale_weight_cache','edited_prefix_stale_cache']]
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    ax.bar(range(5),np.maximum(values,1e-16),color=[GREEN]+[ORANGE]*4,width=.6)
    ax.set_yscale('log');ax.set_ylim(1e-16,10);ax.set_xticks(range(5),labels)
    ax.set(ylabel='logits 最大绝对误差（对数轴）',title='CPU / float64：正确输出一致，故意错误可被检出')
    ax.axhline(1e-10,color=BLUE,ls='--',label='正例绝对容差 1e-10（另有相对容差）')
    for i,v in enumerate(values):ax.text(i,max(v,1e-16)*2,f'{v:.2e}',ha='center',fontsize=10)
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.17));ax.grid(axis='y',alpha=.16)
    save(fig,'correctness')


def result_markdown():
    report=read_json('checks.json');checks=report['checks']
    positive=[c for c in checks if c['type']=='positive']
    lines=[f"已执行 **{len(checks)} 项检查**：{len(positive)} 项正确性检查通过，4 项故意故障均被检出。正例最大绝对误差 **{max(c['max_abs_error'] for c in positive):.3e}**。",
           '', f"运行环境：Python {report['environment']['python']}、PyTorch {report['environment']['torch']}、CPU 单线程、随机种子 {report['seed']}。",'',
           '| 检查组 | 观察值 / 最大绝对误差 | 结论 |','| --- | --- | --- |']
    names={'wrong_mask':'错误掩码偏移','wrong_position':'错误 RoPE 位置','stale_weight_cache':'新权重 + 旧缓存',
           'edited_prefix_stale_cache':'新前缀 + 旧缓存','rebuild_after_weight_change':'新权重 + 重建缓存','greedy_generation_8_tokens':'8-token 贪心生成'}
    lines.append(f"| 三种头结构 × 三组 B/T × 两种切分 | {max(c['max_abs_error'] for c in checks[:18]):.3e} | 正确性通过 |")
    for c in checks:
        if c['case'] in names:
            lines.append(f"| {names[c['case']]} | {c['max_abs_error']:.3e} | {'故障被检出' if c['type']=='negative_control' else '正确性通过'} |")
    lines += ['', '完整记录：[checks.json](../../experiments/w01-attention/data/checks.json)。']
    return '\n'.join(lines)


def build_document():
    source=HERE/'README.md'
    text=source.read_text(encoding='utf-8')
    text=re.sub(r'<!-- results:start -->.*?<!-- results:end -->',
                '<!-- results:start -->\n'+result_markdown()+'\n<!-- results:end -->',text,flags=re.S)
    source.write_text(text,encoding='utf-8')
    md=markdown.Markdown(extensions=['tables','fenced_code','toc','md_in_html'],extension_configs={'toc':{'toc_depth':'2-2'}})
    body=md.convert(text.replace('<details>', '<details markdown="1">'))
    widgets=(HERE/'widgets.html').read_text(encoding='utf-8')
    for name in ['attention','decode','capacity']:
        widget=re.search(rf'<!-- {name}:start -->(.*?)<!-- {name}:end -->',widgets,re.S).group(1)
        body=body.replace(f'<!-- interactive-{name} -->',widget)
    def inline_image(match):
        filename=match.group(1)
        encoded=base64.b64encode((HERE/filename).read_bytes()).decode('ascii')
        return 'src="data:image/png;base64,'+encoded+'"'
    body=re.sub(r'src="(figures/[^\"]+\.png)"',inline_image,body)
    css=(HERE/'style.css').read_text(encoding='utf-8')
    js=(HERE/'interactive.js').read_text(encoding='utf-8')
    offline_note='本文件的正文、图片和三个交互实验均可离线使用。源码、数据及学习日志链接需要保留仓库目录；外部参考链接需要联网。'
    output=f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="第一周 Attention 与 KVCache 图文教材：手算、交互演示和可复现实验。">
<title>Attention 与 KVCache · 第一周学习手册</title><style>{css}</style></head>
<body><a class="skip" href="#content">跳到正文</a><header class="hero"><div class="eyebrow">INFERENCE STUDY / WEEK 01</div>
<h1>让每一个 token<br>都有迹可循。</h1><p>Attention 与 KVCache 基础 · 从张量形状到可验证的缓存</p>
<div class="badges"><span>12 小时启动周</span><span>7 项核心任务</span><span>25 项实验检查</span><span>3 个交互演示</span></div></header>
<div class="layout"><aside><h2>本周导航</h2>{md.toc}<p class="aside-note">先推导，再动手。<br>用证据验收理解。</p><button id="print-button" type="button">打印 / 保存 PDF</button></aside>
<main id="content"><div class="offline-note">{offline_note}</div>{body}</main></div>
<footer>2026-10-07 · 私有学习项目 · 容量与工作量为理论值，实验误差来自 CPU 实测。</footer><script>{js}</script></body></html>'''
    (HERE/'index.html').write_text(output,encoding='utf-8')
    print('Built seven PNG/PDF figures and self-contained index.html.')


if __name__=='__main__':
    setup_style()
    for render in [pipeline,attention_figure,timeline,head_groups,capacity,work,correctness]:render()
    build_document()
