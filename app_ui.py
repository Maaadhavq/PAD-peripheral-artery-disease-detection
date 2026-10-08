"""Presentation layer for the copilot.

The visual language is borrowed from the vascular lab rather than from
dashboards. PAD is diagnosed by reading a ratio — the ankle-brachial index —
against a marked scale, so the central element here is a calibrated readout
strip: the score, the clinically meaningful bands, and the cohort distribution
it is being compared against, all on one axis.

That last part is the honest bit. The cohort is matched 1:1 by construction, so
a probability from this model only means anything relative to the distribution
it came from. Drawing the distribution puts that caveat in the reader's eye
instead of in a footnote.

Palette is taken from perfusion: arterial red carries every positive signal,
a desaturated slate carries its opposite, and nothing else competes with them.
Lab values are set in mono because that is how every chart renders them.
"""

import html

import altair as alt
import numpy as np

INK = "#15202B"
SURFACE = "#FBFAF8"
RULE = "#DBDDD7"
MUTED = "#6B7780"
ARTERIAL = "#B8372A"
VENOUS = "#3F5F7A"

BAND_EDGES = [0.33, 0.66]
BAND_NAMES = ["Low", "Moderate", "High"]

DISPLAY = "'Bricolage Grotesque'"
BODY = "'Source Serif 4'"
MONO = "'IBM Plex Mono'"


def stylesheet():
    """Global CSS. Loaded once, before anything renders."""
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
  --ink: {INK};
  --surface: {SURFACE};
  --rule: {RULE};
  --muted: {MUTED};
  --arterial: {ARTERIAL};
  --venous: {VENOUS};
}}

html, body, [class*="st-"], .stApp {{
  font-family: {BODY}, Georgia, serif;
  color: var(--ink);
}}
.stApp {{ background: var(--surface); }}

/* Streamlit's own chrome competes with the header; keep the toolbar reachable
   but stop it announcing itself. */
#MainMenu, footer, div[data-testid="stToolbar"] {{ visibility: hidden; height: 0; }}
header[data-testid="stHeader"] {{ background: transparent; }}

h1, h2, h3, h4 {{
  font-family: {DISPLAY}, 'Helvetica Neue', sans-serif;
  color: var(--ink);
  letter-spacing: -0.02em;
  font-weight: 700;
}}

/* Masthead ------------------------------------------------------------- */
.masthead {{
  border-bottom: 2px solid var(--ink);
  padding-bottom: 0.65rem;
  margin-bottom: 1.6rem;
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}}
.masthead-title {{
  font-family: {DISPLAY}, sans-serif;
  font-weight: 800;
  font-size: clamp(1.5rem, 3.4vw, 2.3rem);
  line-height: 1;
  letter-spacing: -0.035em;
  margin: 0;
}}
.masthead-title em {{
  font-style: normal;
  color: var(--arterial);
}}
.masthead-meta {{
  font-family: {MONO}, monospace;
  font-size: 0.66rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  text-align: right;
}}

/* Small caps label used to head every block ---------------------------- */
.eyebrow {{
  font-family: {MONO}, monospace;
  font-size: 0.66rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--muted);
  border-top: 1px solid var(--rule);
  padding-top: 0.45rem;
  margin: 1.8rem 0 0.7rem;
  display: block;
}}

/* Readout --------------------------------------------------------------- */
.readout-value {{
  font-family: {DISPLAY}, sans-serif;
  font-weight: 800;
  font-size: clamp(2.6rem, 6vw, 4.6rem);
  line-height: 0.9;
  letter-spacing: -0.045em;
  margin: 0;
  white-space: nowrap;   /* the percent sign was wrapping onto its own line */
}}
.readout-band {{
  font-family: {MONO}, monospace;
  font-size: 0.78rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-top: 0.5rem;
}}
.readout-note {{
  font-size: 0.86rem;
  color: var(--muted);
  margin-top: 0.3rem;
  line-height: 1.45;
}}
.readout-note strong {{ color: var(--ink); }}

/* Clinical prose -------------------------------------------------------- */
.prose {{
  font-family: {BODY}, Georgia, serif;
  font-size: 1.0rem;
  line-height: 1.62;
  max-width: 64ch;
}}
.prose p {{ margin: 0 0 0.85rem; }}

/* Reference list -------------------------------------------------------- */
.ref {{
  font-size: 0.84rem;
  line-height: 1.5;
  padding: 0.4rem 0;
  border-top: 1px solid var(--rule);
  display: flex;
  gap: 0.6rem;
}}
.ref-n {{
  font-family: {MONO}, monospace;
  font-weight: 600;
  color: var(--arterial);
  flex: 0 0 1.6rem;
}}
.ref-score {{
  font-family: {MONO}, monospace;
  color: var(--muted);
  font-size: 0.74rem;
}}

/* Passage card ---------------------------------------------------------- */
.passage {{
  border-left: 2px solid var(--rule);
  padding: 0.1rem 0 0.1rem 0.85rem;
  margin-bottom: 1.1rem;
}}
.passage-head {{
  font-family: {MONO}, monospace;
  font-size: 0.7rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.3rem;
}}
.passage-body {{ font-size: 0.88rem; line-height: 1.55; color: #37424C; }}

/* Notices --------------------------------------------------------------- */
.notice {{
  font-family: {MONO}, monospace;
  font-size: 0.74rem;
  line-height: 1.5;
  letter-spacing: 0.02em;
  padding: 0.55rem 0.7rem;
  margin-bottom: 0.8rem;
  border-left: 3px solid var(--arterial);
  background: rgba(184, 55, 42, 0.05);
}}
.notice-quiet {{
  border-left-color: var(--venous);
  background: rgba(63, 95, 122, 0.05);
}}

/* Form: lab values are monospaced, as they are on every chart ----------- */
section[data-testid="stSidebar"] {{
  background: #F1EFEA;
  border-right: 1px solid var(--rule);
}}
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] label span {{
  font-family: {MONO}, monospace !important;
  font-size: 0.68rem !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted) !important;
}}

/* Streamlit's widgets are painted from its own palette, which ignores the
   injected one: inputs came out near-black on near-black. Forced globally,
   not just in the sidebar, because the main-area inputs had the same fault. */
input,
textarea,
div[data-baseweb="input"],
div[data-baseweb="base-input"],
div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] {{
  background: var(--surface) !important;
  color: var(--ink) !important;
  border-radius: 0 !important;
  font-family: {MONO}, monospace !important;
  font-size: 0.88rem !important;
}}
div[data-baseweb="input"],
div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] {{
  border: 1px solid var(--rule) !important;
}}
input::placeholder {{ color: var(--muted) !important; opacity: 1; }}

/* The step buttons on a number input sit inside the field. */
div[data-testid="stNumberInput"] button {{
  background: transparent !important;
  color: var(--muted) !important;
  border: none !important;
}}
div[data-testid="stNumberInput"] button:hover {{ color: var(--arterial) !important; }}

/* Tabs: rules, not pills */
button[data-baseweb="tab"] {{
  font-family: {MONO}, monospace !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}}
div[data-baseweb="tab-border"] {{ background: var(--rule); }}
div[data-baseweb="tab-highlight"] {{ background: var(--arterial); }}

.stButton button {{
  border-radius: 0 !important;
  border: 1px solid var(--ink) !important;
  background: var(--ink) !important;
  padding: 0.55rem 1rem !important;
}}
/* Streamlit wraps the label in its own <p>, which carries the body colour and
   left the text ink-on-ink. Inherit from the button instead. */
.stButton button p {{
  font-family: {MONO}, monospace !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--surface) !important;
  margin: 0;
}}
.stButton button:hover {{
  background: var(--arterial) !important;
  border-color: var(--arterial) !important;
}}
.stButton button:hover p {{ color: #fff !important; }}

/* Code blocks: st.code renders Streamlit's own dark block, which reads as a
   hole punched in the page here. */
.stCode, pre, code {{
  background: #F1EFEA !important;
  border: 1px solid var(--rule) !important;
  border-radius: 0 !important;
}}
.stCode *, pre *, code {{
  color: var(--ink) !important;
  font-family: {MONO}, monospace !important;
  font-size: 0.78rem !important;
}}

/* Tables sit inside the same paper, not on a floating card. */
div[data-testid="stDataFrame"] {{ border: 1px solid var(--rule); }}

@media (prefers-reduced-motion: reduce) {{
  * {{ animation: none !important; transition: none !important; }}
}}
</style>
"""


def masthead(model_name, index_embedder):
    return f"""
<div class="masthead">
  <h1 class="masthead-title">Peripheral artery disease<br><em>risk copilot</em></h1>
  <div class="masthead-meta">
    {html.escape(str(model_name))}<br>
    index · {html.escape(str(index_embedder) or "none")}<br>
    research demo · not a medical device
  </div>
</div>
"""


def band_for(risk):
    for edge, name in zip(BAND_EDGES, BAND_NAMES):
        if risk < edge:
            return name
    return BAND_NAMES[-1]


def readout(risk, distribution=None, width=620, height=170):
    """The score on a marked scale, over the distribution it is compared against.

    An ankle-brachial index means nothing without the reference values printed
    beside it, and neither does this. Drawing the cohort underneath makes the
    comparison visible rather than implied.
    """
    pad_left, pad_right = 6, 6
    axis_y = height - 44
    span = width - pad_left - pad_right

    def x_of(value):
        return pad_left + span * min(max(value, 0.0), 1.0)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'role="img" aria-label="Score {risk:.0%} on a zero to one scale" '
        'style="display:block;margin:0.2rem 0 0.4rem">'
    ]

    # Cohort distribution as a density of fine ticks under the axis.
    if distribution is not None and len(distribution):
        values = np.asarray(distribution, dtype=float)
        counts, edges = np.histogram(values, bins=40, range=(0.0, 1.0))
        peak = counts.max() or 1
        bar_w = span / len(counts)
        for count, left in zip(counts, edges[:-1]):
            if not count:
                continue
            bar_h = (count / peak) * 52
            parts.append(
                f'<rect x="{x_of(left):.1f}" y="{axis_y - bar_h:.1f}" '
                f'width="{max(bar_w - 0.8, 0.8):.1f}" height="{bar_h:.1f}" '
                f'fill="{VENOUS}" opacity="0.3"/>'
            )

    # Axis and band boundaries.
    parts.append(
        f'<line x1="{pad_left}" y1="{axis_y}" x2="{width - pad_right}" y2="{axis_y}" '
        f'stroke="{INK}" stroke-width="2"/>'
    )
    for edge in BAND_EDGES:
        parts.append(
            f'<line x1="{x_of(edge):.1f}" y1="{axis_y - 5}" x2="{x_of(edge):.1f}" '
            f'y2="{axis_y + 7}" stroke="{MUTED}" stroke-width="1.5"/>'
        )

    # Band names, each centred in its own zone.
    zones = list(zip([0.0] + BAND_EDGES, BAND_EDGES + [1.0], BAND_NAMES))
    for low, high, name in zones:
        parts.append(
            f'<text x="{x_of((low + high) / 2):.1f}" y="{axis_y + 18}" '
            f'text-anchor="middle" fill="{MUTED}" '
            f'font-family="IBM Plex Mono, monospace" font-size="12" '
            f'letter-spacing="2">{name.upper()}</text>'
        )

    # The needle.
    marker_x = x_of(risk)
    parts.append(
        f'<line x1="{marker_x:.1f}" y1="{axis_y - 72}" x2="{marker_x:.1f}" '
        f'y2="{axis_y + 9}" stroke="{ARTERIAL}" stroke-width="3"/>'
    )
    parts.append(
        f'<polygon points="{marker_x - 7:.1f},{axis_y - 72} '
        f'{marker_x + 7:.1f},{axis_y - 72} {marker_x:.1f},{axis_y - 60}" '
        f'fill="{ARTERIAL}"/>'
    )
    parts.append("</svg>")
    return "".join(parts)


def factor_chart(frame, height):
    """Contributing factors, in the same two colours as everything else."""
    return (
        alt.Chart(frame)
        .mark_bar(height=13)
        .encode(
            x=alt.X("effect:Q", title="EFFECT ON THE MODEL'S SCORE",
                    axis=alt.Axis(grid=False, tickCount=5, labelFontSize=9,
                                  titleFontSize=9, labelFont="IBM Plex Mono",
                                  titleFont="IBM Plex Mono", titleColor=MUTED,
                                  labelColor=MUTED, domainColor=RULE,
                                  tickColor=RULE)),
            y=alt.Y("factor:N", sort="-x", title=None,
                    axis=alt.Axis(grid=False, labelFontSize=11,
                                  labelFont="Source Serif 4", labelColor=INK,
                                  domainColor=RULE, tickColor=RULE)),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["raises the score", "lowers the score"],
                                range=[ARTERIAL, VENOUS]),
                legend=alt.Legend(title=None, orient="bottom", labelFontSize=9,
                                  labelFont="IBM Plex Mono", labelColor=MUTED,
                                  symbolType="square"),
            ),
            tooltip=["factor", "value", "effect", "direction"],
        )
        .properties(height=height)
        .configure_view(strokeWidth=0)
        .configure(background="transparent")
    )


def eyebrow(text):
    return f'<span class="eyebrow">{html.escape(text)}</span>'


def notice(text, quiet=False):
    cls = "notice notice-quiet" if quiet else "notice"
    return f'<div class="{cls}">{html.escape(text)}</div>'
