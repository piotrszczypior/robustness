import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'DejaVu Sans',
})

SEV_COLORS = {
    's1': '#43A047',
    's2': '#8BC34A',
    's3': '#FFB300',
    's4': '#F57C00',
    's5': '#E53935',
}

_SRC = {'fill': '#1C4E80', 'text': '#FFFFFF', 'edge': '#0D3561'}
_TGT = {'fill': '#EEF2F8', 'text': '#1a1a1a', 'edge': '#7A9CBF'}


def _node(ax, cx, cy, w, h, st, label, fs=8.5):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.12',
        facecolor=st['fill'], edgecolor=st['edge'],
        linewidth=0.9, zorder=3,
    ))
    ax.text(cx, cy, label, ha='center', va='center', fontsize=fs,
            color=st['text'], zorder=4, linespacing=1.35,
            multialignment='center')


def render_attractor_graph(source, targets, corruption, out_path):
    """
    source  : {'name': str, 'wn_id': str}
    targets : list of {'name': str, 'wn_id': str, 'severity': 's1'..'s5'}
    corruption : str  (e.g. 'gaussian_noise')
    out_path   : str  (.png or .pdf)
    """
    n = len(targets)
    v_sp = 1.55
    fw = 10.0
    fh = max(3.2, n * v_sp + 1.65)

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, fw)
    ax.set_ylim(0, fh)
    ax.set_facecolor('white')
    ax.axis('off')

    title = corruption.replace('_', ' ')
    ax.text(fw / 2, fh - 0.44, title,
            ha='center', va='center', fontsize=11,
            fontweight='bold', color='#1a1a1a')

    ax.axhline(y=fh - 0.70, xmin=0.05, xmax=0.95,
               color='#cccccc', linewidth=0.6, zorder=1)

    sx, sy = 2.5, fh / 2
    sw, sh = 3.6, 0.92
    _node(ax, sx, sy, sw, sh, _SRC,
          source['name'] + '\n(' + source['wn_id'] + ')')

    tx, tw, th = 7.5, 3.4, 0.80
    tys = [fh / 2 - (n - 1) * v_sp / 2 + i * v_sp for i in range(n)]
    x0 = sx + sw / 2 + 0.10
    x1 = tx - tw / 2 - 0.10
    dx = x1 - x0

    for t, ty in zip(targets, tys):
        sev = t.get('severity', 's3')
        clr = SEV_COLORS.get(sev, '#888')
        _node(ax, tx, ty, tw, th, _TGT,
              t['name'] + '\n(' + t['wn_id'] + ')')

        dy = ty - sy
        chord = float(np.hypot(dx, dy))
        rad = float(np.clip(0.30 * dy / chord, -0.42, 0.42)) if chord > 0.1 else 0.0

        ax.annotate('', xy=(x1, ty), xytext=(x0, sy),
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color=clr, lw=1.6,
                        mutation_scale=13,
                        connectionstyle=f'arc3,rad={rad:.3f}',
                    ), zorder=2)

        mx, my = (x0 + x1) / 2, (sy + ty) / 2
        if chord > 0:
            px, py = -dy / chord, dx / chord
        else:
            px, py = 0.0, 1.0

        bx = mx + 0.42 * rad * chord * px
        by = my + 0.42 * rad * chord * py
        if abs(dy) < 0.1:
            by += 0.20

        ax.text(bx, by, sev, ha='center', va='center',
                fontsize=8, color='white', fontweight='bold', zorder=6,
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor=clr, edgecolor='none', alpha=0.95))

    plt.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor='white', pad_inches=0.12)
    plt.close()
    print(f'Saved: {out_path}')



examples = [
    {
        'source': {'name': 'tiger shark', 'wn_id': 'n01491361'},
        'targets': [
            {'name': 'great white shark', 'wn_id': 'n01484850', 'severity': 's5'},
        ],
        'corruption': 'brightness',
    },
    {
        'source': {'name': 'aircraft carrier', 'wn_id': 'n02687172'},
        'targets': [
            {'name': 'liner', 'wn_id': 'n03673027', 'severity': 's4'},
        ],
        'corruption': 'elastic_transform',
    },
    {
        'source': {'name': 'radio telescope', 'wn_id': 'n04041544'},
        'targets': [
            {'name': 'lighthouse', 'wn_id': 'n03742115', 'severity': 's2'},
            {'name': 'flagpole',   'wn_id': 'n03327234', 'severity': 's4'},
            {'name': 'church',     'wn_id': 'n03028079', 'severity': 's5'},
        ],
        'corruption': 'gaussian_noise',
    },
]


os.makedirs("attractors", exist_ok=True)

for ex in examples:
    p = f"attractors/attr_{ex['corruption']}_{ex['source']['wn_id']}.png"
    render_attractor_graph(ex['source'], ex['targets'], ex['corruption'], p)

print('Done.')