"""Per-tool outcome analysis over archived trials: within-task reward advantage of using a tool, failure signals."""
import json, os, re, sys, collections, multiprocessing as mp
sys.path.insert(0, '$W/verl')
from recipe.rlmath.tool_metrics import parse_turn_tools, exes_in, libs_in
A = sys.argv[1]
TOOLS = {'sage': ('exe', 'sage'), 'gp': ('exe', 'gp'), 'gap': ('exe', 'gap'), 'M2': ('exe', 'M2'), 'Singular': ('exe', 'Singular'),
         'py:sympy': ('lib', 'sympy'), 'py:numpy': ('lib', 'numpy'), 'py:networkx': ('lib', 'networkx'), 'py:z3': ('lib', 'z3'),
         'py:ortools': ('lib', 'ortools'), 'py:pysat': ('lib', 'pysat'), 'py:sage': ('lib', 'sage'), 'py:cypari2': ('lib', 'cypari2')}
ERR = {'sage': r'sage: (command )?not found|sage.*Error|SyntaxError', 'gp': r'gp: (command )?not found|\*\*\*', 'gap': r'gap: (command )?not found|Error,',
       'M2': r'M2: (command )?not found|error:', 'Singular': r'Singular: (command )?not found|\? error'}
def one(name):
    d = os.path.join(A, name)
    try:
        r = float(open(os.path.join(d, 'verifier', 'reward.txt')).read().strip())
    except Exception:
        return None
    try:
        tj = json.load(open(os.path.join(d, 'agent', 'trajectory.json')))
    except Exception:
        return None
    exes, libs, ncmd = set(), set(), 0
    for st in tj.get('steps', []):
        if st.get('source') != 'agent':
            continue
        texts = [st.get('message') or '']
        for tc in st.get('tool_calls') or []:
            a = tc.get('arguments') or {}
            if isinstance(a, dict) and a.get('keystrokes'):
                texts.append(json.dumps({'keystrokes': a['keystrokes']}))
        for t in texts:
            for ks in parse_turn_tools(t if isinstance(t, str) else json.dumps(t)):
                ncmd += 1; exes |= exes_in(ks); libs |= libs_in(ks)
    used = {k for k, (kind, n) in TOOLS.items() if (n in exes if kind == 'exe' else n in libs)}
    errs = set()
    if used & set(ERR):
        try:
            pane = open(os.path.join(d, 'agent', 'terminus_2.pane'), errors='ignore').read()
        except Exception:
            pane = ''
        for k in used & set(ERR):
            if re.search(ERR[k], pane):
                errs.add(k)
    task = re.sub(r'-[A-Za-z0-9]{20,}$', '', name)
    try:
        dur = json.load(open(os.path.join(d, 'result.json'))).get('agent_execution') or {}
        secs = (lambda a, b: None)(0, 0)
    except Exception:
        dur = {}
    return task, r, sorted(used), sorted(errs), ncmd
if __name__ == '__main__':
    names = [n for n in os.listdir(A)]
    with mp.Pool(24) as p:
        rows = [x for x in p.imap_unordered(one, names, chunksize=64) if x]
    by_task = collections.defaultdict(list)
    for task, r, used, errs, n in rows:
        by_task[task].append((r, set(used)))
    print(f'trials {len(rows)}  tasks {len(by_task)}')
    print(f'{"tool":12s} {"trials":>7s} {"tasks":>6s} {"reward_using":>12s} {"same-task w/o":>13s} {"within-task Δ":>13s} {"err-in-output":>13s}')
    for k in TOOLS:
        using = [(t, r) for t, r, u, e, n in rows if k in u]
        if not using:
            continue
        deltas, base_r = [], []
        for t in set(t for t, _ in using):
            w = [r for r, u in by_task[t] if k in u]; wo = [r for r, u in by_task[t] if k not in u]
            if w and wo:
                deltas.append((sum(w)/len(w) - sum(wo)/len(wo), len(w))); base_r.append(sum(wo)/len(wo))
        wsum = sum(n for _, n in deltas)
        d = sum(dd*n for dd, n in deltas)/wsum if wsum else float('nan')
        errfrac = sum(1 for t, r, u, e, n in rows if k in u and k in e) / len(using)
        print(f'{k:12s} {len(using):7d} {len(set(t for t,_ in using)):6d} {sum(r for _,r in using)/len(using):12.3f} {sum(base_r)/max(len(base_r),1):13.3f} {d:+13.3f} {errfrac:13.1%}' + ('' if k in ERR else ' (n/a)'))
    # where M2/sage are used: task kind / name prefix
    for k in ('M2', 'sage', 'gap', 'gp'):
        c = collections.Counter(t.split('-')[0] for t, r, u, e, n in rows if k in u)
        tasks = collections.Counter(t for t, r, u, e, n in rows if k in u)
        print(k, dict(c), 'top tasks:', tasks.most_common(4))
