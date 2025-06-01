import numpy as np

def calculate_cagr(start_value, end_value, periods):
    """Compounded Annual Growth Rate."""
    return (end_value / start_value) ** (1 / periods) - 1

def calculate_max_drawdown(series):
    """Maximum Drawdown from a return series."""
    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()
