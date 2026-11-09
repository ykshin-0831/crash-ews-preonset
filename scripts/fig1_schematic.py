"""Figure 1 of the manuscript: schematic of one event (illustrative; the script reads no data).

A synthetic one-minute close path, indexed to 100 at the peak, with peak p_e (minute 0), onset o_e (first close
1% below the peak, minute 60) and breach b_e (first close 4% below the H-minute high, sustained five closes,
minute 170); the judging window [b_e - 240, b_e - 30] (211 positions) split into its pre-onset part (L_e minutes)
and in-progress part. I made up the path for the illustration: a monotone cubic through hand-set points, not
market data. The script checks that the drawn path satisfies the event rule at the marked minutes.
Writes scripts/output/fig1_schematic.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

OUT = Path(__file__).resolve().parent / 'output'
PEAK, ONSET, BREACH, W_MIN, W_MAX, H = 0, 60, 170, 30, 240, 240

# hand-set points of the illustrative path (minute relative to the peak, close indexed to the peak)
KNOTS = [(-120, 99.45), (-100, 99.47), (-90, 99.55), (-80, 99.65), (-70, 99.68), (-50, 99.69), (-40, 99.80),
         (-30, 99.90), (-15, 99.875), (-10, 99.91), (0, 100.0), (10, 99.84), (20, 99.57), (30, 99.46), (40, 99.40),
         (50, 99.21), (60, 99.0), (70, 98.73), (80, 98.49), (90, 98.24), (100, 97.99), (110, 97.745), (120, 97.485),
         (130, 97.12), (140, 96.78), (150, 96.595), (160, 96.385), (165, 96.16), (170, 96.0), (180, 95.88),
         (190, 95.81), (200, 95.74), (210, 95.645), (220, 95.52), (228, 95.42)]
path = PchipInterpolator(*zip(*KNOTS))
t = np.arange(-120, 229)
close = path(t)

# the drawn path obeys the event rule at the marked minutes
look = (t >= BREACH - H) & (t < BREACH)
assert t[look][np.argmax(close[look])] == PEAK and close.max() == close[t == PEAK][0] == 100.0
assert t[(t > PEAK) & (close <= 99.0)][0] == ONSET
assert t[(t > PEAK) & (close <= 96.0)][0] == BREACH and (close[(t >= BREACH) & (t < BREACH + 5)] <= 96.0).all()
L = int(((t >= BREACH - W_MAX) & (t <= BREACH - W_MIN) & (t < ONSET)).sum())
print(f'peak {PEAK}, onset {ONSET}, breach {BREACH}, gap g_e = {BREACH - ONSET}, L_e = {L} of {W_MAX - W_MIN + 1} window minutes')

plt.rcParams.update({'font.family': ['Inter', 'DejaVu Sans'], 'font.size': 10, 'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Inter', 'mathtext.it': 'Inter:italic',
                     'mathtext.cal': 'Inter:italic'})
GREY, BLUE_TXT, ORANGE_TXT = '#888888', '#2b5c8a', '#a35a12'
fig, ax = plt.subplots(figsize=(7.6, 3.5))
fig.subplots_adjust(left=0.0928, right=0.9807, bottom=0.183, top=0.9586)   # axes box of the published figure
ax.axvspan(BREACH - W_MAX, ONSET, color='#d6e1ec', lw=0)
ax.axvspan(ONSET, BREACH - W_MIN, color='#fbe3cc', lw=0)
for yv, txt in ((100, 'peak close ($H$-minute high)'), (99, '−1% (onset threshold)'), (96, '−4% (breach threshold)')):
    ax.axhline(yv, color='#aaaaaa', lw=0.6, ls=':')
    ax.text(-117, yv + 0.05, txt, color=GREY, fontsize=8, va='bottom')
for xv, txt in ((PEAK, '$p_e$ peak'), (ONSET, '$o_e$ onset'), (BREACH, '$b_e$ breach')):
    ax.axvline(xv, color='black', lw=1, ls=(0, (3, 2)))
    ax.text(xv + 2.5, 95.005, txt, fontsize=9, va='bottom')
ax.plot(t, close, color='#333333', lw=1.8)
ax.text(-30, 97.945, 'pre-onset\npositives\n($L_e$ minutes)', color=BLUE_TXT, ha='center', va='center', fontsize=9)
ax.text(100, 99.545, 'in-progress\npositives', color=ORANGE_TXT, ha='center', va='center', fontsize=9)
arrow = dict(arrowstyle='<->', lw=0.9, color='black', shrinkA=0, shrinkB=0)
ax.annotate('', (BREACH - W_MAX, 100.945), (BREACH - W_MIN, 100.945), arrowprops=arrow)
ax.text((2 * BREACH - W_MAX - W_MIN) / 2, 101.18, 'judging window [$b_e$ \u2212 240,  $b_e$ \u2212 30] (211 minutes)', ha='center',
        va='center', fontsize=9, bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
ax.annotate('', (BREACH - W_MIN, 100.545), (BREACH, 100.545), arrowprops=dict(arrow, color=GREY, lw=0.8))
ax.text(BREACH - W_MIN / 2, 100.6, 'last 30 min\nnot judged', color=GREY, ha='center', va='bottom', fontsize=7.5)
ax.annotate('', (ONSET, 95.595), (BREACH, 95.595), arrowprops=arrow)
ax.text((ONSET + BREACH) / 2, 95.66, r'gap $g_e = b_e - o_e$', ha='center', va='bottom', fontsize=9.5)
ax.set_xlim(-120, 228); ax.set_ylim(94.847, 101.503)
ax.set_yticks([96, 97, 98, 99, 100])
ax.set_xlabel('minutes relative to the peak')
ax.set_ylabel('close (index, peak = 100)')
ax.spines[['top', 'right']].set_visible(False)
fig.savefig(OUT / 'fig1_schematic.png', dpi=200)
print('Wrote', OUT / 'fig1_schematic.png')
