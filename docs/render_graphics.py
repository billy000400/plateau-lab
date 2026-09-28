"""Build the README's SVG illustrations from source and saved measured data.

Run from the repository root: .venv/bin/python docs/render_graphics.py
Uses only the standard library. The JSON contains an actual Engine.run result
reduced to curves and reproducibility metadata; no model inference happens here.
"""
from html import escape
import json
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / 'assets'


def text(x, y, value, size=16, fill='#242435', weight=400, **attrs):
    extra = ' '.join(f'{key.replace("_", "-")}="{escape(str(value))}"' for key, value in attrs.items())
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" {extra}>{escape(str(value))}</text>'


def svg(width, height, title, description, parts):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>'
            '<g font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">'
            + ''.join(parts) + '</g></svg>\n')


def workflow():
    parts = ['<rect width="1200" height="476" rx="24" fill="#242330"/>',
             '<path d="M1042 0h158v174" fill="#302b45"/>',
             text(42, 47, 'MARS V  /  ACTIVATION EXPLORER', 13, '#b7a4ee', 600, letter_spacing=2),
             text(42, 104, 'Two prompts. One hidden-state path.', 36, '#ffffff', 600),
             text(42, 139, 'Compare how a model’s internal representations move as you interpolate.', 19, '#c5c2d2')]
    for x, number, label in [(42, '01', 'COMPARE'), (430, '02', 'INTERVENE'), (818, '03', 'MEASURE')]:
        parts.extend([f'<rect x="{x}" y="180" width="340" height="230" rx="16" fill="#32313f" stroke="#484556"/>',
                      text(x+20, 215, number, 13, '#b8a2f5', 600),
                      text(x+55, 215, label, 13, '#f0edf6', 600, letter_spacing=1.5)])
    for y, letter, ending, color in [(269, 'A', 'big', '#c7b2ff'), (315, 'B', 'in', '#9ecce4')]:
        parts.extend([text(65, y, letter, 14, color, 600), text(91, y, 'The house was', 18, '#eceaf1'),
                      f'<rect x="231" y="{y-22}" width="61" height="32" rx="7" fill="#4b415f"/>',
                      text(242, y, ending, 18, color, 600)])
    parts.extend([text(64, 374, 'Choose two different token sequences.', 14, '#bbb7cb'),
                  text(452, 251, 'A states', 15, '#c7b2ff', 500), text(672, 251, 'B states', 15, '#9ecce4', 500),
                  '<path d="M471 284 C524 258 569 311 625 282 S698 272 725 284" fill="none" stroke="#b8a2f5" stroke-width="3"/>',
                  '<circle cx="471" cy="284" r="7" fill="#c7b2ff"/><circle cx="600" cy="289" r="8" fill="#f3dbac"/><circle cx="725" cy="284" r="7" fill="#9ecce4"/>',
                  text(464, 321, 't = 0', 13, '#c5c2d2'), text(580, 329, 't', 16, '#f3dbac', 600), text(686, 321, 't = 1', 13, '#c5c2d2'),
                  text(452, 374, 'Patch at a chosen layer; continue forward.', 14, '#bbb7cb'),
                  text(840, 263, 'c(t)', 29, '#c7b2ff', 600), text(909, 263, 'Path progress', 19, '#f1eef8', 500),
                  text(840, 309, 'd(t)', 29, '#9ecce4', 600), text(909, 309, 'Endpoint distance', 19, '#f1eef8', 500),
                  text(840, 374, 'Read residuals + logits at the final token.', 14, '#bbb7cb'),
                  text(390, 303, '→', 28, '#a39bb7'), text(778, 303, '→', 28, '#a39bb7'),
                  text(42, 447, 'LOCAL INFERENCE', 12, '#d4c1fc', 600, letter_spacing=1),
                  text(220, 447, 'GPT-2 · Pythia · Qwen', 14, '#bdb8cb'),
                  text(818, 447, 'Workflow illustration · not a measured trajectory', 12, '#a7a1b8')])
    (ASSETS/'workflow.svg').write_text(svg(1200, 476, 'Plateau Lab: compare, intervene, measure',
        'A and B prompts provide hidden states. Interpolate selected token states at a chosen block, run the remaining model, and measure c(t) and d(t) at residual layers and logits. The illustrated path is schematic.', parts))


def measured():
    record = json.loads((ASSETS/'measured-example.json').read_text())
    parts = ['<rect width="1200" height="790" rx="20" fill="#f7f7fa"/>',
             text(32, 43, 'A real run, two views of the same trajectory', 26, weight=600),
             text(32, 75, f'{record["model_label"]} · after layer 0 · SLERP · 41 samples · final-token measurement', 16, '#646176'),
             text(32, 104, f'A  “{record["sequence_a"]}”     B  “{record["sequence_b"]}”', 16, '#646176'),
             text(32, 153, 'c(t)  ·  Cumulative path progress', 21, '#7154b1', 600),
             '<line x1="865" y1="147" x2="895" y2="147" stroke="#a4a4b3" stroke-dasharray="4 4"/>',
             text(905, 152, 'Uniform path progress: c = t', 13, '#747183'),
             text(32, 451, 'd(t)  ·  Relative endpoint distance', 21, '#386d92', 600),
             '<line x1="998" y1="445" x2="1028" y2="445" stroke="#a4a4b3" stroke-dasharray="4 4"/>',
             text(1038, 450, 'd = t reference', 13, '#747183')]
    for row, metric in enumerate(('c', 'd')):
        top = 176 + row*298
        color = '#8063c6' if metric == 'c' else '#427eaa'
        for i, curve in enumerate(record['curves']):
            left = 32+i*292
            parts.append(f'<rect x="{left}" y="{top}" width="272" height="244" rx="12" fill="#ffffff" stroke="#e2dfe9"/>')
            label = 'Logits' if curve['key']=='logits' else f'Layer {curve["key"]} · resid_post'
            parts.append(text(left+15, top+27, label, 15, weight=600))
            x0,x1,y0,y1 = left+38,left+254,top+51,top+182
            xx = lambda v: x0+v*(x1-x0)
            yy = lambda v: y1-v*(y1-y0)
            for value in (0, .5, 1):
                parts.extend([f'<line x1="{x0}" x2="{x1}" y1="{yy(value)}" y2="{yy(value)}" stroke="#efedf4"/>',
                              text(x0-8, yy(value)+4, f'{value:g}', 11, '#95909f', text_anchor='end'),
                              text(xx(value), y1+19, f'{value:g}', 11, '#95909f', text_anchor='middle')])
            parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y0}" stroke="#b6b4c1" stroke-dasharray="3 4"/>')
            points=' '.join(f'{xx(t):.3f},{yy(value):.3f}' for t,value in zip(curve['t'],curve[metric]))
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
            for t,value in zip(curve['t'],curve[metric]):
                parts.append(f'<circle cx="{xx(t):.3f}" cy="{yy(value):.3f}" r="1.5" fill="{color}"/>')
            parts.append(text(left+146, y1+35, 'Interpolation t', 11, '#8b8596', text_anchor='middle'))
            if metric=='c':
                parts.append(text(left+15, top+231, f'Total path L2  {curve["total_length"]:.4f}', 12, '#746b83'))
    parts.extend([text(32, 752, 'Same samples. Separate normalization for every readout. Raw total path length appears under c(t).', 14, '#6d6579'),
                  text(32, 776, 'Measured example, not a general plateau claim. Source values and model revision are included in measured-example.json.', 12, '#8b8496')])
    (ASSETS/'measured-curves.svg').write_text(svg(1200,790,'Measured c(t) and d(t) for Pythia 70M',
        'Four columns show residual layers 1, 3, 5 and logits. The first row plots cumulative arc length c(t), the second relative endpoint distance d(t), on matching zero-to-one axes. Every point comes from an actual 41-sample SLERP run.',parts))


if __name__ == '__main__':
    ASSETS.mkdir(parents=True, exist_ok=True)
    workflow()
    measured()
    print('Wrote docs/assets/workflow.svg and docs/assets/measured-curves.svg')
