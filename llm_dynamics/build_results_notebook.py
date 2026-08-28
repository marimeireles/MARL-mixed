"""Build results_<base>_vs_<rl>.ipynb — every statistic of every benchmark for
the base-vs-RL comparison, read live from the result artifacts.

    TAG_BASE=qwen8b_base TAG_RL=qwen8b_rl_s6 VERSION=v3 \
    JAX_PLATFORMS=cpu .venv-jaxmarl/bin/python llm_dynamics/build_results_notebook.py --execute

Sections (each degrades to "pending" when its artifact is missing):
  1. Dynamics (llm_dynamics v3): per-cell base vs RL with bootstrap CIs over
     seeds — agreement, best-response captured, welfare captured, cooperation
     rate; training-signal (rho, r1, R_std); repair; group stage; self-play;
     probed policies; reciprocity-plane stars.
  2. MACHIAVELLI: per-arm means ± 95% CI for every normalized and raw score,
     paired-by-game differences (t-test, Wilcoxon, bootstrap CI), per-game table.
  3. DiG-bench: auc_level / levels_beaten / level_reached / turns / stop reasons
     per arm, paired-by-game differences with tests, per-game table.
  4. EigenBench (Gemma-4 judge): per constitution — order-stable win rate with
     Wilson CI, tie/flip rates, per-criterion preferences, sign test, and the
     length control (net preference vs. response-length difference).
"""
import os
import subprocess
import sys
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
TB = os.environ.get("TAG_BASE", "qwen8b_base")
TR = os.environ.get("TAG_RL", "qwen8b_rl_s6").split(",")[0]
TAGS_RL = [t for t in os.environ.get("TAG_RL", "qwen8b_rl_s6").split(",") if t]
VER = os.environ.get("VERSION", "v3")
OUT_NAME = os.environ.get("OUT_NAME", f"results_{TB}_vs_{TR}.ipynb")

nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md(f"""# Results — `{TB}` vs `{TR}`

Every statistic of every benchmark, read from the artifacts on disk when this
notebook is executed (rebuild with `build_results_notebook.py --execute`).
Arms: **base** = `{TB}`, **RL** = `{TR}` (donorSim GRPO checkpoint). All
tests are two-sided; CIs are 95%; bootstrap CIs use 2000 resamples. "pending"
means the artifact does not exist yet.

| benchmark | artifact | unit of pairing |
|---|---|---|
| dynamics (llm_dynamics {VER}) | `results/{{tag}}_*_{VER}*/`, `results/probes/` | cell (strategy × w × q × memory), seeds as replicates |
| MACHIAVELLI | `external_benchmarks/machiavelli/data/machiavelli_trajectories.csv` | game (episodes averaged) |
| DiG-bench | `external_benchmarks/dig-bench/data/digbench_runs.csv` | game (reps averaged) |
| EigenBench | `external_benchmarks/EigenBench/runs/qwen8b/judgments_gemma4_step6/<constitution>.jsonl` | scenario, both presentation orders |
""")

# --- Section 0: written review (from FINDINGS_qwen8b.md, embedded at build time) ---
_f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINDINGS_qwen8b.md")
if os.path.exists(_f):
    md("## 0. Full review — where the models diverge (written analysis)\n\n"
       "*This section is the human-readable review; every number in it is computed in the sections below.*\n\n---\n\n"
       + open(_f).read())

code(r"""import os, sys, json, glob, math, itertools, collections
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.abspath(os.path.join(os.getcwd(), '..')) if os.path.basename(os.getcwd())=='llm_dynamics' else os.getcwd()
sys.path.insert(0, ROOT); os.chdir(ROOT)
from IPython.display import Markdown, Image, display
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 200)
TB, TR, VER = %r, %r, %r
TAGS_RL = %r
R = 'llm_dynamics/results'; EB = 'external_benchmarks'
rng = np.random.default_rng(0)

def boot_ci(x, n=2000, stat=np.mean):
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], float)
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    if len(x) == 1: return (x[0], x[0], x[0])
    bs = [stat(rng.choice(x, len(x))) for _ in range(n)]
    return (stat(x), np.percentile(bs, 2.5), np.percentile(bs, 97.5))

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan, np.nan)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (p, c-h, c+h)

def fmt(m, lo, hi, nd=3): return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"

def paired_tests(a, b):
    '''a, b aligned arrays (same units). Returns dict of mean diff, bootstrap CI, t-test p, Wilcoxon p, n.'''
    a, b = np.asarray(a, float), np.asarray(b, float); m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    d = b - a
    out = dict(n=len(d), mean_base=a.mean() if len(a) else np.nan, mean_rl=b.mean() if len(b) else np.nan,
               diff=d.mean() if len(d) else np.nan)
    if len(d) >= 2:
        _, lo, hi = boot_ci(d); out['diff_ci'] = (lo, hi)
        out['p_ttest'] = stats.ttest_rel(b, a).pvalue
        try: out['p_wilcoxon'] = stats.wilcoxon(b, a).pvalue if np.any(d != 0) else 1.0
        except ValueError: out['p_wilcoxon'] = np.nan
    return out

def pending(what): display(Markdown(f'**{what}: pending** (artifact not found)'))

HI_M = {'agreement','br_captured','welfare_captured','coop','ppr','points','rho','r1','mean_rho','mean_r1',
        'recovered_frac','mutual_c_rate','a_total','b_total','trajectory_scalar','mean_reward',
        'game.score','points_pct_of_max','achievements','reached_end','auc_level','levels_beaten','level_reached','max_level_seen'}
LO_M = {'recovery_rounds','mean_cfe','mean_brier','parse_failures','latency','R_std'}
def _direction(name):
    n = str(name)
    if n in HI_M: return 1
    if n in LO_M or n.startswith(('violations', 'power', 'utility', 'raw.')): return -1
    return 0
def _fmtv(v, nd):
    if isinstance(v, (int, float, np.floating)):
        return '' if v != v else f'{v:.{nd}f}'
    return str(v)
def bold_arms(df, nd=3):
    '''Render a DataFrame whose columns are (metric, arm) (or metric-only) as a
    Markdown table, bolding the best arm per row within each metric
    (direction from the metric name; directionless metrics stay unbolded).'''
    cols = list(df.columns)
    multi = isinstance(df.columns, pd.MultiIndex)
    groups = {}
    for c in cols: groups.setdefault(c[:-1] if multi else (c,), []).append(c)
    headers = [' '.join(str(x) for x in c) if multi else str(c) for c in cols]
    inames = [str(n or '') for n in (df.index.names if df.index.nlevels > 1 else [df.index.name or ''])]
    lines = ['| ' + ' | '.join(inames + headers) + ' |', '|' + '---|' * (len(inames) + len(cols))]
    for ix, row in df.iterrows():
        bold = set()
        for g, gc in groups.items():
            d = _direction(g[-1]) if len(gc) > 1 else 0
            if d:
                vals = {c: row[c] for c in gc if isinstance(row[c], (int, float, np.floating)) and row[c] == row[c]}
                if vals: bold.add(max(vals, key=vals.get) if d > 0 else min(vals, key=vals.get))
        ivals = [str(x) for x in (ix if isinstance(ix, tuple) else (ix,))]
        lines.append('| ' + ' | '.join(ivals + [('**' + _fmtv(row[c], nd) + '**') if c in bold else _fmtv(row[c], nd) for c in cols]) + ' |')
    return Markdown(chr(10).join(lines))
def bold_paired(rows, a0='base', a1='rl', nd=3):
    '''rows: list of dicts with metric plus a0/a1 columns and stats; bold the better arm.'''
    extra = [k for k in rows[0] if k not in ('metric', a0, a1)]
    lines = ['| metric | ' + a0 + ' | ' + a1 + ' | ' + ' | '.join(extra) + ' |', '|' + '---|' * (3 + len(extra))]
    for r in rows:
        d = _direction(r['metric'])
        c0, c1 = _fmtv(r[a0], nd), _fmtv(r[a1], nd)
        if d and r[a0] == r[a0] and r[a1] == r[a1]:
            if (r[a1] > r[a0]) == (d > 0): c1 = '**' + c1 + '**'
            else: c0 = '**' + c0 + '**'
        lines.append('| ' + str(r['metric']) + ' | ' + c0 + ' | ' + c1 + ' | ' + ' | '.join(_fmtv(r[k], 4) if isinstance(r[k], (int, float, np.floating)) else str(r[k]) for k in extra) + ' |')
    return Markdown(chr(10).join(lines))
print('ready')""" % (TB, TR, VER, TAGS_RL))

# ── 1. dynamics ───────────────────────────────────────────────────────────
md(r"""## 1. Cooperation dynamics (llm_dynamics)

Cells are (strategy, w, q, memory); replicates are seeds. For each cell and
metric we report base and RL means with bootstrap CIs and the RL−base
difference with its CI; the final rows aggregate over all cells (paired by cell).
Metrics: **agreement** with the reference policy, **BR captured** (fraction of
the best-response payoff), **welfare captured**, **cooperation rate**; training
signal: **mean ρ** (Term-2 reciprocation), **mean r₁**, **R_std** (across-seed
std of the trajectory scalar — GRPO's advantage denominator).""")
code(r"""from llm_dynamics import analysis as A
def per_game_metrics(dirs):
    '''one row per game file: cell key + metrics (for seed-level bootstrap)'''
    rows = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, 'rounds', '*.jsonl')):
            rr = [json.loads(l) for l in open(f) if l.strip()]
            if not rr or 'q' not in rr[0]: continue
            r0 = rr[0]; g = A.analyze_game(rr); s = A.signal_metrics(rr)
            rows.append(dict(strategy=r0['opponent_strategy'], w=r0['w'], q=r0['q'], memory=A._mem_tag(r0), seed=r0['seed'],
                             agreement=g['agreement'], br_captured=g['captured'], welfare_captured=g['welfare_captured'],
                             coop=A._coop_rate(rr), rho=s['mean_rho'], r1=s['mean_r1'], R=s['R'], latency=s['latency']))
    return pd.DataFrame(rows)
dyn = {}
for tag in (TB, TR):
    dirs = glob.glob(f'{R}/{tag}_donors_{VER}/*')
    dyn[tag] = per_game_metrics(dirs) if dirs else None
if dyn[TB] is None: pending('dynamics (base)')
if dyn[TR] is None: pending('dynamics (RL)')
METRICS = ['agreement','br_captured','welfare_captured','coop','rho','r1']
if dyn[TB] is not None:
    keys = ['strategy','w','q','memory']
    def cell_table(df):
        g = df.groupby(keys)
        out = g[METRICS].mean(); out['R_std'] = g['R'].std(ddof=0); out['n_seeds'] = g.size(); return out
    tb = cell_table(dyn[TB])
    if dyn[TR] is not None:
        tr = cell_table(dyn[TR])
        both = tb.join(tr, lsuffix='_base', rsuffix='_rl', how='inner')
        side = pd.concat({m: both[[f'{m}_base', f'{m}_rl']].set_axis(['base', 'rl'], axis=1) for m in METRICS + ['R_std']}, axis=1)
        display(Markdown(f'### Per-cell means, base vs RL side by side ({len(side)} cells) — **bold** = better arm (direction-aware; directionless metrics unbolded)'))
        display(bold_arms(side))
    else:
        display(Markdown(f'### Base — per-cell means ({len(tb)} cells)')); display(tb.round(3))
    if dyn[TR] is not None:
        display(Markdown('### RL − base per cell (memory full), with the summary over all cells'))
        diff = pd.DataFrame({m: both[f'{m}_rl'] - both[f'{m}_base'] for m in METRICS + ['R_std']})
        display(diff.xs('full', level='memory').round(3) if 'full' in diff.index.get_level_values('memory') else diff.round(3))
        rows = []
        for m in METRICS + ['R_std']:
            t = paired_tests(both[f'{m}_base'], both[f'{m}_rl'])
            rows.append(dict(metric=m, n_cells=t['n'], base=t['mean_base'], rl=t['mean_rl'], diff=t['diff'],
                             diff_ci=fmt(t['diff'], *t.get('diff_ci', (np.nan, np.nan))), p_ttest=t.get('p_ttest'), p_wilcoxon=t.get('p_wilcoxon')))
        display(Markdown('### Aggregate over cells (paired by cell) — **bold** = better arm')); display(bold_paired(rows))
        display(Markdown('### By strategy (memory full): RL − base'))
        display(diff.reset_index().query("memory=='full'").groupby('strategy')[METRICS].mean().round(3))
        display(Markdown('### By (w, q) (memory full): RL − base'))
        display(diff.reset_index().query("memory=='full'").groupby(['w','q'])[METRICS].mean().round(3))
        display(Markdown('### By memory window: RL − base'))
        display(diff.reset_index().groupby('memory')[METRICS].mean().round(3))""")

md(r"""### 1.1 c/b sweep, horizons, thinking-on — cooperation rate and BR captured per arm""")
code(r"""for label, pat in [('c/b sweep', f'{{tag}}_donors_{VER}_cb*'), ('horizon N', f'{{tag}}_donors_{VER}_N*'), ('thinking on (w=1)', f'{{tag}}_donors_{VER}_think')]:
    frames = []
    for tag in (TB, TR):
        for d in sorted(glob.glob(f'{R}/' + pat.format(tag=tag))):
            sub = glob.glob(d + '/*'); df = per_game_metrics(sub)
            if len(df): df['arm'] = 'base' if tag == TB else 'rl'; df['setting'] = os.path.basename(d).split(VER + '_')[-1]; frames.append(df)
    if not frames: pending(label); continue
    df = pd.concat(frames)
    display(Markdown(f'### {label}'))
    display(bold_arms(df.groupby(['setting','strategy','arm'])[['coop','br_captured','welfare_captured','agreement']].mean().unstack('arm')))""")

md(r"""### 1.2 Repair after a forced defection, group-selection stage, self-play""")
code(r"""rep, grp, gci, slf = {}, {}, {}, {}
for tag in (TB, TR):
    arm = 'base' if tag == TB else 'rl'
    d = f'{R}/{tag}_donors_{VER}_perturb'
    if os.path.isdir(d):
        t = A.signal_table([d])
        rep[arm] = pd.DataFrame([dict(strategy=k[0], w=k[2], q=k[1], recovery_rounds=v['recovery'], recovered_frac=v['recovered_frac']) for k, v in t.items()]).set_index(['strategy','w','q'])
    f = glob.glob(f'{R}/{tag}_group_{VER}/summary_*.csv')
    if f:
        g = pd.read_csv(f[0]); num = g.select_dtypes('number').drop(columns=[c for c in ('scenario','seed','K','G','rounds') if c in g], errors='ignore')
        gci[arm] = pd.Series({c: fmt(*boot_ci(num[c])) for c in num.columns}, name=arm)
        grp[arm] = g.groupby('partner')[['cooperation_rate','mean_r1','mean_rho','mean_cfe','mean_brier','trajectory_scalar']].mean()
        grp[arm + '_n'] = len(g)
    f = glob.glob(f'{R}/{tag}_selfplay_{VER}/summary_*.csv')
    if f:
        slf[arm] = pd.read_csv(f[0]).groupby(['w','q'])[['a_cooperation_rate','b_cooperation_rate','mutual_c_rate','a_total','b_total']].mean()
def side(dd):
    arms_ = [a for a in ('base', 'rl') if a in dd]
    return pd.concat({a: dd[a] for a in arms_}, axis=1).swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False)[dd[arms_[0]].columns.tolist()]
if rep:
    display(Markdown('### Repair after a forced defection — **bold** = better arm (fewer recovery rounds / more recovered)'))
    display(bold_arms(side(rep), nd=2))
else: pending('repair')
if gci:
    display(Markdown(f'### Group stage — mean [bootstrap CI] per arm (base n={grp.get("base_n", 0)}, rl n={grp.get("rl_n", 0)} games)'))
    display(pd.DataFrame(gci))
    display(Markdown('### Group stage by partner — **bold** = better arm (higher r₁/ρ/trajectory, lower CFE/Brier; cooperation rate is descriptive)'))
    display(bold_arms(side({a: grp[a] for a in ('base', 'rl') if a in grp})))
else: pending('group stage')
if slf:
    display(Markdown('### Self-play (means over seeds) — **bold** = better arm'))
    display(bold_arms(side(slf), nd=2))
else: pending('self-play')""")

md(r"""### 1.3 Probed conditional policies (memory 1) and the reciprocity-plane stars""")
code(r"""def probe_df(tag):
    rows = []
    for f in sorted(glob.glob(f'{R}/probes/{tag}_*_m1*.json')):
        p = json.load(open(f)); row = {'probe': os.path.basename(f).replace(tag + '_', '').replace('.json', '')}
        row.update({k: v['p_cooperate'] for k, v in sorted(p['states'].items())}); rows.append(row)
    return pd.DataFrame(rows).set_index('probe') if rows else None
pb, pr = probe_df(TB), probe_df(TR)
if pb is not None:
    display(Markdown('### base')); display(pb.round(2))
    if pr is not None:
        display(Markdown('### RL')); display(pr.round(2)); display(Markdown('### RL − base')); display((pr - pb).round(2))
else: pending('probes')
for f in sorted(glob.glob(f'{R}/reciprocity_*{TR}*.png')) + sorted(glob.glob(f'{R}/portraits/*{TB}*{TR}*.png')):
    display(Image(f, width=520))""")

# ── 2. MACHIAVELLI ────────────────────────────────────────────────────────
md(r"""### 1.4 Regret with respect to each reward term (figure format after Tennant et al., ICLR 2025)

Donors game, base vs RL, mean ± 95% CI over (strategy × seed) games; the matrix games are
covered game-by-game in 1.5. Each panel is the shortfall of the played game with respect to one
term of the training reward (Eq. reward):

* **Term 1 — individual-payoff regret** = 1 − best-response captured: the share of the maximum
  own payoff (Eq. t1, DP over the partner's strategy) the model left on the table.
* **Term 2 — reciprocation regret** = fraction of rounds with ρ_k = −1 (Eq. t2): the model's
  action did *not* match the partner's previous action. This is the direct-reciprocity
  pressure; the RL reward pays for ρ = +1 every round.
* **Term 3 / bonus — collective-payoff regret** = 1 − welfare captured: the shortfall of the
  dyad's joint payoff vs. its maximum, the dyadic analogue of the group-survival bonus
  (Eq. bonus). The forecast-error part of Term 3 (CFE, Eq. cfe) is shown from the
  group-stage evaluation below where available.

The stacked bars decompose Term 2: the model's action given the partner's previous move.
Green (C|C, D|D) = reciprocated (ρ=+1); pink/red (C|D, D|C) = not reciprocated (ρ=−1);
grey = unparseable output. Two variants: pooled over all nine opponent strategies, and a
Random-only opponent (uniform state coverage).""")
code(r"""import matplotlib.pyplot as plt
def _prev(r):
    p = r.get('prev_opp')
    if p is None and r.get('visible_state') not in (None, '', 'start'):
        p = 'COOPERATE' if r['visible_state'].split('|')[-1][1] == 'c' else 'DEFECT'
    return p
def game_rows(tag):
    # (game, strategy, file) -> rows; donors = memory full at c/b 0.5, matrix games = memory 1
    out = {}
    for f in glob.glob(f'{R}/{tag}_donors_{VER}/*/rounds/*.jsonl'):
        rr = [json.loads(l) for l in open(f) if l.strip()]
        if rr and rr[0].get('memory') is None: out[('donors', rr[0]['opponent_strategy'], f)] = rr
    for f in glob.glob(f'{R}/{tag}_matrix_{VER}/*/rounds/*_m1_*.jsonl'):
        rr = [json.loads(l) for l in open(f) if l.strip()]
        if rr: out[(rr[0]['game'], rr[0]['opponent_strategy'], f)] = rr
    return out
def regrets(rr):
    g = A.analyze_game(rr)
    after_c = [r for r in rr if _prev(r) == 'COOPERATE']
    deon = float(np.mean([r['model_action'] == 'DEFECT' for r in after_c])) if after_c else np.nan
    mism = [(_prev(r) is not None) and (r['model_action'] != _prev(r)) for r in rr if _prev(r) is not None]
    return dict(term1_payoff=1 - g['captured'], term2_reciprocation=float(np.mean(mism)) if mism else np.nan,
                term3_collective=1 - g['welfare_captured'])
def composition(rr):
    c = collections.Counter()
    for r in rr:
        p = _prev(r)
        if p is None: continue
        c[('i' if r.get('parse_failed') else r['model_action'][0]) + '|' + p[0]] += 1
    return c
GAMES = ['donors']  # matrix games: see 1.5
TITLES = {'term1_payoff': 'Term 1: individual-payoff regret (1 − BR captured)', 'term2_reciprocation': 'Term 2: reciprocation regret (P[ρ = −1])', 'term3_collective': 'Term 3/bonus: collective-payoff regret (1 − welfare captured)'}
DATA = {tag: game_rows(tag) for tag in (TB, TR)}
def moral_figure(only_random=False, title=''):
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, metric in zip(axs, ['term1_payoff', 'term2_reciprocation', 'term3_collective']):
        for k, (tag, arm) in enumerate([(TB, 'base'), (TR, 'RL')]):
            means, los, his = [], [], []
            for gname in GAMES:
                vals = [regrets(rr)[metric] for (g, s, f), rr in DATA[tag].items() if g == gname and (not only_random or s == 'random')]
                vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
                if vals: m, lo, hi = boot_ci(vals)
                else: m, lo, hi = np.nan, np.nan, np.nan
                means.append(m); los.append(0 if not vals else m - lo); his.append(0 if not vals else hi - m)
            x = np.arange(len(GAMES)) + (k - 0.5) * 0.36
            ax.bar(x, means, 0.34, yerr=[los, his], capsize=3, label=arm, color=['#7a7a7a', '#c0392b'][k])
        ax.set_xticks(range(len(GAMES))); ax.set_xticklabels(GAMES, rotation=20); ax.set_title(TITLES[metric], fontsize=10); ax.set_ylim(0, 1)
    axs[0].legend(fontsize=8); fig.suptitle(title, fontsize=10); plt.tight_layout(); plt.show()
def composition_figure(only_random=False, title=''):
    cats = ['C|C', 'D|D', 'C|D', 'D|C', 'i|C', 'i|D']; cols = ['#1b7f3b', '#5cb85c', '#f4a3b5', '#8e0d2a', '#bbbbbb', '#888888']
    fig, ax = plt.subplots(figsize=(13, 3.8)); xs, labels = [], []; pos = 0.0
    for gname in GAMES:
        for tag, arm in [(TB, 'base'), (TR, 'RL')]:
            tot = collections.Counter()
            for (g, s, f), rr in DATA[tag].items():
                if g == gname and (not only_random or s == 'random'): tot.update(composition(rr))
            n = sum(tot.values()) or 1; bottom = 0.0
            for cat, col in zip(cats, cols):
                v = 100 * tot.get(cat, 0) / n; ax.bar(pos, v, 0.8, bottom=bottom, color=col, label=cat if pos == 0 else None); bottom += v
            xs.append(pos); labels.append(f'{gname}\n{arm}'); pos += 1
        pos += 0.6
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7); ax.set_ylabel("action | partner's previous move (%)")
    ax.legend(fontsize=7, ncol=6, loc='upper center', bbox_to_anchor=(0.5, -0.22)); ax.set_title(title, fontsize=10); plt.tight_layout(); plt.show()
moral_figure(False, 'Regret per reward term, pooled over the nine opponent strategies (donors: memory full; matrix: memory 1)')
composition_figure(False, 'Term 2 decomposition — action | partner\'s previous move, pooled over the nine opponent strategies')
moral_figure(True, 'Regret per reward term vs the Random opponent (uniform state coverage)')
composition_figure(True, 'Term 2 decomposition vs the Random opponent')""")
md(r"""**Thinking-on subset (w=1) and per-opponent breakdown.** The thinking-on games are the
training mode; the per-opponent stacked bars separate reciprocators from unconditional
partners (donors game, memory full).""")
code(r"""def think_rows(tag):
    out = {}
    for f in glob.glob(f'{R}/{tag}_donors_{VER}_think/*/rounds/*.jsonl'):
        rr = [json.loads(l) for l in open(f) if l.strip()]
        if rr: out[('donors-think', rr[0]['opponent_strategy'], f)] = rr
    return out
TDATA = {tag: think_rows(tag) for tag in (TB, TR)}
if all(TDATA.values()):
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
    strats = sorted({s for d in TDATA.values() for (_, s, _) in d})
    for ax, metric in zip(axs, ['term1_payoff', 'term2_reciprocation', 'term3_collective']):
        for k, (tag, arm) in enumerate([(TB, 'base'), (TR, 'RL')]):
            means, los, his = [], [], []
            for s_ in strats:
                vals = [regrets(rr)[metric] for (g, s, f), rr in TDATA[tag].items() if s == s_]
                vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
                m, lo, hi = boot_ci(vals) if vals else (np.nan, np.nan, np.nan)
                means.append(m); los.append(0 if not vals else m - lo); his.append(0 if not vals else hi - m)
            ax.bar(np.arange(len(strats)) + (k - 0.5) * 0.36, means, 0.34, yerr=[los, his], capsize=2, label=arm, color=['#7a7a7a', '#c0392b'][k])
        ax.set_xticks(range(len(strats))); ax.set_xticklabels([s_.replace('_', ' ') for s_ in strats], rotation=60, fontsize=7); ax.set_title(TITLES[metric] + ' — thinking ON, w=1', fontsize=9); ax.set_ylim(0, 1)
    axs[0].legend(fontsize=8); plt.tight_layout(); plt.show()
else: pending('thinking-on subset (both arms)')
# per-opponent composition, donors game (memory full) and thinking-on
for label, D in [('donors game, thinking off (memory full)', DATA), ('donors game, thinking ON (w=1)', TDATA)]:
    if not all(D.values()): continue
    cats = ['C|C', 'D|D', 'C|D', 'D|C', 'i|C', 'i|D']; cols = ['#1b7f3b', '#5cb85c', '#f4a3b5', '#8e0d2a', '#bbbbbb', '#888888']
    strats = sorted({s for d in D.values() for (g, s, _) in d if g.startswith('donors')})
    fig, ax = plt.subplots(figsize=(14, 3.8)); xs, labels = [], []; pos = 0.0
    for s_ in strats:
        for tag, arm in [(TB, 'base'), (TR, 'RL')]:
            tot = collections.Counter()
            for (g, s, f), rr in D[tag].items():
                if g.startswith('donors') and s == s_: tot.update(composition(rr))
            n = sum(tot.values()) or 1; bottom = 0.0
            for cat, col in zip(cats, cols):
                v = 100 * tot.get(cat, 0) / n; ax.bar(pos, v, 0.8, bottom=bottom, color=col, label=cat if pos == 0 else None); bottom += v
            xs.append(pos); labels.append(f"{s_.replace('_', ' ')}\n{arm}"); pos += 1
        pos += 0.6
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6); ax.set_ylabel("action | partner's previous move (%)"); ax.set_title(f'Term 2 decomposition per opponent — {label}', fontsize=10)
    ax.legend(fontsize=7, ncol=6, loc='upper center', bbox_to_anchor=(0.5, -0.3)); plt.tight_layout(); plt.show()""")
md(r"""**Term 3 (forecast error) and the full training reward, from the group-stage evaluation** —
mean CFE (Eq. cfe), Brier score of the model's own PREDICT, ρ, r₁, group bonus and the
trajectory scalar R, base vs RL, mean ± 95% CI over the 100 group-stage games.""")
code(r"""gs = {}
for tag, arm in [(TB, 'base'), (TR, 'RL')]:
    f = glob.glob(f'{R}/{tag}_group_{VER}/summary_*.csv')
    if f: gs[arm] = pd.read_csv(f[0])
if len(gs) == 2:
    cols_ = ['mean_cfe', 'mean_brier', 'mean_rho', 'mean_r1', 'bonus', 'trajectory_scalar', 'cooperation_rate']
    fig, axs = plt.subplots(1, len(cols_), figsize=(2.3 * len(cols_), 3.4))
    for ax, c in zip(axs, cols_):
        for k, arm in enumerate(['base', 'RL']):
            m, lo, hi = boot_ci(gs[arm][c]); ax.bar(k, m, 0.7, yerr=[[m - lo], [hi - m]], capsize=4, color=['#7a7a7a', '#c0392b'][k])
        ax.set_xticks([0, 1]); ax.set_xticklabels(['base', 'RL']); ax.set_title(c.replace('mean_', ''), fontsize=9)
    fig.suptitle('Group-stage evaluation (training environment): reward terms, base vs RL', fontsize=10); plt.tight_layout(); plt.show()
    rows = []
    for c in cols_:
        t = paired_tests(gs['base'].sort_values(['scenario','seed'])[c].values, gs['RL'].sort_values(['scenario','seed'])[c].values) if len(gs['base']) == len(gs['RL']) else None
        rows.append(dict(metric=c, base=gs['base'][c].mean(), rl=gs['RL'][c].mean(), diff=(t or {}).get('diff'), diff_ci=fmt(t['diff'], *t['diff_ci']) if t and 'diff_ci' in t else '', p_ttest=(t or {}).get('p_ttest')))
    display(bold_paired(rows, nd=4))
else: pending('group-stage evaluation (both arms)')""")

md(r"""### 1.5 Games × models, against the most discriminating opponent

Opponent selection (transparent, data-driven): among {Random, Grim trigger, TF2T,
suspicious TFT} we pick the opponent against which the RL models differ **most** from
the base model on the reference game — the **IPD** (memory 1) for the matrix games,
the **donors game** (memory full, c/b = 0.5, pooled over w × q) for the donors
setting. Interest = mean over RL arms of |Δ cooperation| + |Δ payoff| / max payoff.
The chosen opponent is then held fixed and the models are compared across games:
x = game, bars = mean **points per round** (95% CI over seeds), dots = **cooperation
rate** (right axis). Same for the donors game across the (w, q) grid.""")
code(r"""ARMS = [(TB, 'base')] + [(t, t.replace('qwen8b_', '')) for t in TAGS_RL]
CANDS = ['random', 'grim_trigger', 'tit_for_two_tats', 'suspicious_tit_for_tat']
ARM_COL = ['#7a7a7a', '#c0392b', '#2c6e9e', '#2f6f3e', '#a14a3b']
PAYKEY = {'donors': 'payoff_raw'}
def load_games(tag, kind, memory_filter=None):
    '''kind='matrix' -> (game, strategy) -> [rows per seed]; kind='donors' -> (w, q, strategy) -> [rows per seed]'''
    out = collections.defaultdict(list)
    if kind == 'matrix':
        for f in glob.glob(f'{R}/{tag}_matrix_{VER}/*/rounds/*_m1_*.jsonl'):
            rr = [json.loads(l) for l in open(f) if l.strip()]
            if rr: out[(rr[0]['game'], rr[0]['opponent_strategy'])].append(rr)
    else:
        for f in glob.glob(f'{R}/{tag}_donors_{VER}/*/rounds/*.jsonl'):
            rr = [json.loads(l) for l in open(f) if l.strip()]
            if rr and rr[0].get('memory') is None: out[(rr[0]['w'], rr[0]['q'], rr[0]['opponent_strategy'])].append(rr)
    return out
def ppr(rr, key): return float(np.mean([r[key] for r in rr]))            # points per round
def coop(rr): return float(np.mean([r['model_action'] == 'COOPERATE' for r in rr]))
MAXPAY = {'ipd': 5, 'chicken': 5, 'staghunt': 5, 'harmony': 5}
M = {tag: load_games(tag, 'matrix') for tag, _ in ARMS}
D = {tag: load_games(tag, 'donors') for tag, _ in ARMS}
def interest_table(kind):
    rows = []
    for s_ in CANDS:
        rec = dict(opponent=s_)
        for tag, arm in ARMS:
            if kind == 'matrix':
                games = M[tag].get(('ipd', s_), []); key, mx = 'payoff', MAXPAY['ipd']
            else:
                games = [rr for (w, q, s), lst in D[tag].items() if s == s_ for rr in lst]; key, mx = 'payoff_raw', 6.0
            rec[f'coop_{arm}'] = np.mean([coop(rr) for rr in games]) if games else np.nan
            rec[f'ppr_{arm}'] = np.mean([ppr(rr, key) for rr in games]) if games else np.nan
        rec['interest'] = np.nanmean([abs(rec[f'coop_{a}'] - rec['coop_base']) + abs(rec[f'ppr_{a}'] - rec['ppr_base']) / (MAXPAY['ipd'] if kind == 'matrix' else 6.0) for _, a in ARMS[1:]]) if len(ARMS) > 1 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).set_index('opponent')
def pick(tab):
    if tab['interest'].notna().any(): return tab['interest'].idxmax()
    return 'suspicious_tit_for_tat'
it_m = interest_table('matrix'); opp_m = pick(it_m)
display(Markdown(f'#### Matrix games — candidate opponents on the IPD (chosen: **{opp_m}**)')); display(it_m.round(3))
it_d = interest_table('donors'); opp_d = pick(it_d)
display(Markdown(f'#### Donors game — candidate opponents (chosen: **{opp_d}**)')); display(it_d.round(3))
def games_figure(kind, opp):
    if kind == 'matrix':
        xs = ['ipd', 'chicken', 'staghunt', 'harmony']; getter = lambda tag, x: M[tag].get((x, opp), []); key = 'payoff'; xlabels = ["Prisoner's Dilemma", 'Chicken', 'Stag Hunt', 'Harmony']; ttl = f'Matrix games vs {opp} (memory 1): points per round (bars) and cooperation rate (dots)'
    else:
        xs = sorted({(w, q) for tag, _ in ARMS for (w, q, s) in D[tag] if s == opp}); getter = lambda tag, x: D[tag].get((x[0], x[1], opp), []); key = 'payoff_raw'; xlabels = [f'w={w:g}\nq={q:g}' for w, q in xs]; ttl = f'Donors game vs {opp} (b=4, c/b=0.5, memory full): points per round (bars) and cooperation rate (dots)'
    fig, ax = plt.subplots(figsize=(1.6 * len(xs) + 4, 4.2)); ax2 = ax.twinx(); n = len(ARMS); wdt = 0.8 / n
    for k, (tag, arm) in enumerate(ARMS):
        means, los, his, cs = [], [], [], []
        for x in xs:
            games = getter(tag, x)
            if games: vals = [ppr(rr, key) for rr in games]; m, lo, hi = boot_ci(vals); c = np.mean([coop(rr) for rr in games])
            else: m, lo, hi, c = np.nan, np.nan, np.nan, np.nan
            means.append(m); los.append(0 if np.isnan(m) else m - lo); his.append(0 if np.isnan(m) else hi - m); cs.append(c)
        pos = np.arange(len(xs)) + (k - (n - 1) / 2) * wdt
        ax.bar(pos, means, wdt * 0.95, yerr=[los, his], capsize=3, color=ARM_COL[k % len(ARM_COL)], label=arm)
        ax2.scatter(pos, cs, color=ARM_COL[k % len(ARM_COL)], edgecolor='k', zorder=5, s=45)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xlabels); ax.set_ylabel('points per round'); ax2.set_ylabel('cooperation rate (dots)'); ax2.set_ylim(0, 1.05)
    ax.set_title(ttl, fontsize=10); ax.legend(loc='upper left', fontsize=8); plt.tight_layout(); plt.show()
games_figure('matrix', opp_m)
games_figure('donors', opp_d)
# numeric companion tables
for kind, opp in [('matrix', opp_m), ('donors', opp_d)]:
    rows = []
    src = M if kind == 'matrix' else D
    keys = ['ipd', 'chicken', 'staghunt', 'harmony'] if kind == 'matrix' else sorted({(w, q) for tag, _ in ARMS for (w, q, s) in src[tag] if s == opp})
    for x in keys:
        rec = {'game' if kind == 'matrix' else '(w,q)': x}
        for tag, arm in ARMS:
            games = src[tag].get((x, opp), []) if kind == 'matrix' else src[tag].get((x[0], x[1], opp), [])
            rec[f'ppr_{arm}'] = np.mean([ppr(rr, 'payoff' if kind == 'matrix' else 'payoff_raw') for rr in games]) if games else np.nan
            rec[f'coop_{arm}'] = np.mean([coop(rr) for rr in games]) if games else np.nan
        rows.append(rec)
    display(Markdown(f'#### {kind}: vs {opp}')); display(pd.DataFrame(rows).round(3))""")

md(r"""## 2. MACHIAVELLI

Scores are the benchmark's own `get_mach_scores()`: each count divided by the
random agent's mean on the same game × 100 (100 = as harmful as random; lower
is better for violations/power; `game.score` higher is better). Raw counts
(`raw.*`) are shown too. Pairing unit = game (episodes averaged per arm).""")
code(r"""f = f'{EB}/machiavelli/data/machiavelli_trajectories.csv'
if not os.path.exists(f): pending('MACHIAVELLI aggregate')
else:
    m = pd.read_csv(f); display(Markdown(f'{len(m)} trajectories; arms: {m.arm.value_counts().to_dict()}; games per arm: {m.groupby("arm").game.nunique().to_dict()}'))
    num = [c for c in m.select_dtypes('number').columns if c not in ('episode',)]
    HI = {'game.score', 'points_pct_of_max', 'achievements', 'reached_end'}
    LO = {c for c in num if c.startswith(('violations', 'power', 'utility', 'raw.'))} | {'parse_failures', 'random_actions'}
    ARMS = [a for a in ('base', 'rl', 'rl51') if a in set(m.arm)] or sorted(m.arm.unique())
    def best_arm(c, vals):
        ok = {a: v for a, v in vals.items() if v == v}
        if not ok: return None
        if c in HI: return max(ok, key=ok.get)
        if c in LO: return min(ok, key=ok.get)
        return None
    def bold_table(cols, means, cells, arms_):
        lines = ['| metric | ' + ' | '.join(arms_) + ' |', '|---' * (len(arms_) + 1) + '|']
        for c in cols:
            b = best_arm(c, {a: means[a][c] for a in arms_})
            lines.append('| ' + c + ' | ' + ' | '.join(f'**{cells[a][c]}**' if a == b else str(cells[a][c]) for a in arms_) + ' |')
        return Markdown('\n'.join(lines))
    means = {a: m[m.arm == a][num].mean().to_dict() for a in ARMS}
    cells = {a: {c: fmt(*boot_ci(m.loc[m.arm == a, c]), nd=2) for c in num} for a in ARMS}
    display(Markdown('### Per-arm means with bootstrap CIs — **bold** = best arm on that metric (direction-aware: higher is better for score/points/achievements, lower for violations/power/utility; cost/telemetry metrics unbolded)'))
    display(bold_table(num, means, cells, ARMS))
    a0 = 'base' if 'base' in ARMS else ARMS[0]
    for a1 in [a for a in ARMS if a != a0]:
        pg = m[m.arm.isin([a0, a1])].groupby(['game', 'arm'])[num].mean().unstack('arm').dropna()
        lines = [f'| metric | n games | {a0} | {a1} | diff | p t-test | p wilcoxon |', '|---|---|---|---|---|---|---|']
        for c in num:
            t = paired_tests(pg[(c, a0)], pg[(c, a1)])
            b = best_arm(c, {a0: t['mean_base'], a1: t['mean_rl']})
            c0, c1 = f"{t['mean_base']:.2f}", f"{t['mean_rl']:.2f}"
            if b == a0: c0 = f'**{c0}**'
            if b == a1: c1 = f'**{c1}**'
            lines.append(f"| {c} | {t['n']} | {c0} | {c1} | {t['diff']:+.2f} | {t.get('p_ttest', float('nan')):.3f} | {t.get('p_wilcoxon', float('nan')):.3f} |")
        display(Markdown(f'### Paired by game: {a1} vs {a0} — **bold** = better arm on that metric'))
        display(Markdown('\n'.join(lines)))
    display(Markdown('### Episode-level extras')); display(m.groupby('arm')[[c for c in ('steps','reached_end','parse_failures','random_actions','mean_reasoning_chars','total_completion_tokens','wall_s') if c in m]].agg(['mean','std']).round(2))""")

# ── 3. DiG-bench ──────────────────────────────────────────────────────────
md(r"""## 3. DiG-bench

Hidden-rule discovery games (digbench.ai): the agent is dropped into a game
whose rules it must figure out by experimenting, for at most 200 turns per
run; each game has numbered levels. How to read the metrics:

* `level_reached` — highest level the agent was on when the run ended (every
  run starts on level 1).
* `levels_beaten` = `level_reached` − 1 — levels actually completed. A mean of
  0.25 means the agent beats even the *first* level in only ~1 run in 4; these
  games are very hard for 8B models.
* `auc_level` — the primary endpoint: the agent's current level averaged over
  the turns of the run. It rewards reaching levels *early*; a run stuck on
  level 1 for all 200 turns scores exactly 1.0, so values like 1.19 mean
  "mostly still on level 1".
* `turns` + `stop_reason` — `max_steps` = hit the 200-turn cap still playing
  (the typical outcome); `done`/`game_over` = the game itself ended.
* everything else is cost/telemetry (tokens, wall time, reasoning length).

21 public games × 10 reps per arm through the baseline harness (guided-json
move channel). Pairing unit = game (reps averaged).""")
code(r"""f = f'{EB}/dig-bench/data/digbench_runs.csv'
if not os.path.exists(f): pending('DiG-bench aggregate')
else:
    d = pd.read_csv(f); display(Markdown(f'{len(d)} runs; arms: {d.arm.value_counts().to_dict()}; games per arm: {d.groupby("arm").game.nunique().to_dict()}'))
    num = [c for c in ('auc_level','levels_beaten','level_reached','max_level_seen','turns','llm_calls','prompt_tokens','output_tokens','total_tokens','wall_s','llm_s','creative_turns','mean_reasoning_chars','zero_reasoning_turns','transport_warnings') if c in d]
    HI = {'auc_level', 'levels_beaten', 'level_reached', 'max_level_seen'}
    LO = {'transport_warnings', 'zero_reasoning_turns'}
    ARMS = [a for a in ('base', 'rl', 'rl51') if a in set(d.arm)] or sorted(d.arm.unique())
    def best_arm(c, vals):
        ok = {a: v for a, v in vals.items() if v == v}
        if not ok: return None
        if c in HI: return max(ok, key=ok.get)
        if c in LO: return min(ok, key=ok.get)
        return None
    def bold_table(cols, means, cells, arms_):
        lines = ['| metric | ' + ' | '.join(arms_) + ' |', '|---' * (len(arms_) + 1) + '|']
        for c in cols:
            b = best_arm(c, {a: means[a][c] for a in arms_})
            lines.append('| ' + c + ' | ' + ' | '.join(f'**{cells[a][c]}**' if a == b else str(cells[a][c]) for a in arms_) + ' |')
        return Markdown('\n'.join(lines))
    means = {a: d[d.arm == a][num].mean().to_dict() for a in ARMS}
    cells = {a: {c: fmt(*boot_ci(d.loc[d.arm == a, c]), nd=3) for c in num} for a in ARMS}
    display(Markdown('### Per-arm means with bootstrap CIs — **bold** = best arm on that metric (higher is better for level metrics; cost/telemetry unbolded). NOTE: raw means are not comparable across arms with unequal game coverage — trust the paired tables below.'))
    display(bold_table(num, means, cells, ARMS))
    display(Markdown('### Stop reasons / results by arm')); display(pd.crosstab(d.arm, d.stop_reason));
    if 'result' in d: display(pd.crosstab(d.arm, d.result))
    a0 = 'base' if 'base' in ARMS else ARMS[0]
    for a1 in [a for a in ARMS if a != a0]:
        pg = d[d.arm.isin([a0, a1])].groupby(['game', 'arm'])[num].mean().unstack('arm').dropna()
        if not len(pg): continue
        lines = [f'| metric | n games | {a0} | {a1} | diff | p t-test | p wilcoxon |', '|---|---|---|---|---|---|---|']
        for c in num:
            t = paired_tests(pg[(c, a0)], pg[(c, a1)])
            b = best_arm(c, {a0: t['mean_base'], a1: t['mean_rl']})
            c0, c1 = f"{t['mean_base']:.3f}", f"{t['mean_rl']:.3f}"
            if b == a0: c0 = f'**{c0}**'
            if b == a1: c1 = f'**{c1}**'
            lines.append(f"| {c} | {t['n']} | {c0} | {c1} | {t['diff']:+.3f} | {t.get('p_ttest', float('nan')):.3f} | {t.get('p_wilcoxon', float('nan')):.3f} |")
        display(Markdown(f'### Paired by game: {a1} vs {a0} — **bold** = better arm on that metric'))
        display(Markdown('\n'.join(lines)))
        display(Markdown(f'### Per-game table ({a1} vs {a0}, reps averaged) — **bold** = better arm'))
        display(bold_arms(pg[[('auc_level',a0),('auc_level',a1),('levels_beaten',a0),('levels_beaten',a1),('turns',a0),('turns',a1)]]))
    if 'tier' in d:
        display(Markdown('### By tier — **bold** = best arm'))
        display(bold_arms(d.groupby(['tier','arm'])[['auc_level','levels_beaten']].mean().unstack('arm')))""")

# ── 4. EigenBench ─────────────────────────────────────────────────────────
md(r"""## 4. EigenBench (external judge: Gemma-4-31B-it)

For each constitution, every scenario is judged in both presentation orders
(`ab` = base first, `ba` = RL first). A criterion verdict counts only if the two
orders agree in arm terms; an order-flip is a position effect and scores a tie
(EigenBench's `handle_inconsistencies_with_ties` rule). Reported: order-stable
RL win rate with Wilson CI, tie and flip rates, per-criterion preferences, a
sign test on per-scenario net preference, and the length control — net
preference regressed on the difference in visible response length (the artifact
that explained the 32B kindness effect).""")
code(r"""JD = f'{EB}/EigenBench/runs/qwen8b/judgments_gemma4_step6'
H2H = f'{EB}/EigenBench/runs/qwen8b/responses/arms_head_to_head_step6.jsonl'
lens = {}
if os.path.exists(H2H):
    for l in open(H2H):
        r = json.loads(l); b = r.get('base') or r.get('a') or {}; a = r.get('rl') or r.get('b') or {}
        lb = len((b.get('response_visible') if isinstance(b, dict) else '') or ''); la = len((a.get('response_visible') if isinstance(a, dict) else '') or '')
        lens[r['scenario_index']] = (lb, la)
def arm_pref(order, val):
    if val == 0: return 'tie'
    if order == 'ab': return 'base' if val == 1 else 'rl'
    return 'rl' if val == 1 else 'base'
files = sorted(glob.glob(f'{JD}/*.jsonl'))
if not files: pending('EigenBench judgments')
summary = []
for f in files:
    const = os.path.basename(f)[:-6]
    rows = [json.loads(l) for l in open(f) if l.strip()]
    by = collections.defaultdict(dict)
    for r in rows: by[r['scenario_index']][r['order']] = {int(k): v for k, v in r['choices'].items()}
    paired = {s: o for s, o in by.items() if 'ab' in o and 'ba' in o}
    ncrit = max((max(o['ab'].keys(), default=0) for o in paired.values()), default=0)
    wins = collections.Counter(); per_crit = collections.defaultdict(collections.Counter); net = []; flips = 0; total = 0
    for s, o in paired.items():
        n_s = 0
        for k in range(1, ncrit + 1):
            a, b = o['ab'].get(k), o['ba'].get(k)
            if a is None or b is None: continue
            pa, pb = arm_pref('ab', a), arm_pref('ba', b); total += 1
            if pa == pb and pa != 'tie': wins[pa] += 1; per_crit[k][pa] += 1; n_s += (1 if pa == 'rl' else -1)
            else:
                wins['tie'] += 1; per_crit[k]['tie'] += 1
                if pa != pb and 'tie' not in (pa, pb): flips += 1
        net.append((s, n_s / max(ncrit, 1)))
    if not total: continue
    dec = wins['base'] + wins['rl']; p, lo, hi = wilson(wins['rl'], dec) if dec else (np.nan,)*3
    nets = np.array([v for _, v in net]); pos, neg = int((nets > 0).sum()), int((nets < 0).sum())
    p_sign = stats.binomtest(pos, pos + neg).pvalue if pos + neg else np.nan
    _, nlo, nhi = boot_ci(nets)
    row = dict(constitution=const, scenarios=len(paired), criteria=ncrit, verdicts=total, rl_wins=wins['rl'], base_wins=wins['base'], ties=wins['tie'],
               order_flips=flips, rl_win_rate=fmt(p, lo, hi), net_pref=fmt(nets.mean(), nlo, nhi, 4), p_sign=p_sign, parsed_fail=sum(1 for r in rows if r['n_parsed'] < ncrit))
    if lens:
        x = np.array([lens[s][1] - lens[s][0] for s, _ in net if s in lens]); y = np.array([v for s, v in net if s in lens])
        if len(x) > 2:
            sl, ic, rr, pv, se = stats.linregress(x, y); row['rho_len'] = stats.spearmanr(x, y).correlation; row['R2_len'] = rr**2
            row['net_at_equal_len'] = fmt(ic, ic - 1.96*se*0 - 1.96*np.std(y - (sl*x+ic))/math.sqrt(len(y)), ic + 1.96*np.std(y - (sl*x+ic))/math.sqrt(len(y)), 4)
    summary.append(row)
    display(Markdown(f'### {const}: per-criterion (order-stable) preferences'))
    display(pd.DataFrame({k: dict(v) for k, v in sorted(per_crit.items())}).T.fillna(0).astype(int))
if summary:
    display(Markdown('### Summary over constitutions (rl_win_rate = RL share of decided verdicts, Wilson CI; net_pref = per-scenario (RL−base)/criteria; length control: Spearman ρ and R² of net vs Δlength, intercept = net at equal length)'))
    display(pd.DataFrame(summary).set_index('constitution'))""")

md(r"""## 5. Reading guide

* Dynamics: the RL effect is in the sign and size of the per-cell differences —
  forgiveness (`P(C|cd)` in the probes and the repair table), stranger
  exploitation (w=0 cells), suspicious-TFT derailment, Chicken/Harmony vs AllD,
  and whether `mean ρ` moved (Term 2 saturation) and `R_std` shrank (GRPO acted).
* MACHIAVELLI / DiG: paired-by-game tests are the honest ones (games differ
  enormously); the trajectory-level CIs show spread.
* EigenBench: a genuine value effect must survive the length control and be
  absent on the humour/poeticism controls.
""")

nb["cells"] = C
out = os.path.join(HERE, OUT_NAME)
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out)
if "--execute" in sys.argv:
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=3600", "--ExecutePreprocessor.allow_errors=True", out], check=True)
    print("executed", out)
