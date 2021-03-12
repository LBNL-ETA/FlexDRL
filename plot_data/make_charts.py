import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot, iplot

def plot_df(df, filename=None):
    data = []
    for col in df.columns:
        data+= [
            go.Scatter(
                x = df.index,
                y = df[col],
                name=col
            )
        ]
    fig = go.Figure(data=data)
    plot(fig, filename=filename)


