import pandas as pd
import plotly.graph_objects as go
from chart_axes import add_independent_axis_controls


def test_independent_axis_buttons_do_not_mutate_other_axis():
    fig = go.Figure()
    x = [pd.Timestamp('2026-01-01', tz='UTC'), pd.Timestamp('2026-01-10', tz='UTC')]
    add_independent_axis_controls(fig, x, [100, 200])
    buttons = list(fig.layout.updatemenus[0].buttons)
    assert [b.label for b in buttons] == ['Time +', 'Time −', 'Price +', 'Price −', 'Reset']
    for b in buttons[:2]:
        assert set(b.args[0]) == {'xaxis.range', 'xaxis.autorange'}
    for b in buttons[2:4]:
        assert set(b.args[0]) == {'yaxis.range', 'yaxis.autorange'}
    assert fig.layout.xaxis.fixedrange is False
    assert fig.layout.yaxis.fixedrange is False


def test_axis_zoom_ranges_expand_and_contract():
    fig = go.Figure()
    add_independent_axis_controls(fig, [pd.Timestamp('2026-01-01'), pd.Timestamp('2026-01-11')], [100, 200])
    buttons = fig.layout.updatemenus[0].buttons
    assert buttons[2].args[0]['yaxis.range'] == [125, 175]
    assert buttons[3].args[0]['yaxis.range'] == [50, 250]
    assert pd.Timestamp(buttons[0].args[0]['xaxis.range'][1]) - pd.Timestamp(buttons[0].args[0]['xaxis.range'][0]) == pd.Timedelta(days=5)
