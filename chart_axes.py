"""Independent price/time axis navigation for Plotly candlestick charts.

Plotly allows axis-specific drag zoom on the axis tick-label region. These
buttons provide accessible explicit X-only and Y-only zoom operations too.
All loaded candles remain in the underlying trace; only the viewport changes.
"""
import pandas as pd


def _scaled_numeric(lo, hi, factor):
    middle = (float(lo) + float(hi)) / 2
    half = max((float(hi) - float(lo)) / 2, abs(middle) * 1e-9, 1e-9)
    return [middle - half * factor, middle + half * factor]


def _scaled_dates(lo, hi, factor):
    start, end = pd.Timestamp(lo), pd.Timestamp(hi)
    middle = start + (end - start) / 2
    half = (end - start) / 2 * factor
    return [(middle - half).isoformat(), (middle + half).isoformat()]


def add_independent_axis_controls(fig, x_range, y_range):
    """Add independent time/price zoom buttons; keep pan and axis drag enabled.

    Buttons calculate from the initial view, while direct axis dragging gives
    continuous interactive control. Plotly's modebar also provides box zoom.
    """
    x0, x1 = x_range
    y0, y1 = y_range
    buttons = []
    for name, axis, factor in (
        ("Time +", "xaxis", 0.5), ("Time −", "xaxis", 2.0),
        ("Price +", "yaxis", 0.5), ("Price −", "yaxis", 2.0),
    ):
        limits = (_scaled_dates(x0, x1, factor) if axis == "xaxis"
                  else _scaled_numeric(y0, y1, factor))
        buttons.append(dict(label=name, method="relayout",
                            args=[{f"{axis}.range": limits,
                                   f"{axis}.autorange": False}]))
    buttons.append(dict(label="Reset", method="relayout", args=[{
        "xaxis.range": [pd.Timestamp(x0).isoformat(), pd.Timestamp(x1).isoformat()],
        "xaxis.autorange": False,
        "yaxis.range": [float(y0), float(y1)],
        "yaxis.autorange": False,
    }]))
    fig.update_layout(
        dragmode="pan",
        xaxis_fixedrange=False,
        yaxis_fixedrange=False,
        updatemenus=[dict(type="buttons", direction="right", buttons=buttons,
                          x=0, xanchor="left", y=1.12, yanchor="bottom",
                          showactive=False, pad=dict(t=0, b=0))],
    )
    return fig
