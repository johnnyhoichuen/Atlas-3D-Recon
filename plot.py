import matplotlib.pyplot as plt
import click
import csv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot(data, comparison, layout):
    # plot graphs
    fig = go.Figure(layout=layout)
    for i in range(len(data['metrics'])):
        metrics = pd.DataFrame(data['metrics'][i]['data'])
        xdata = comparison['xdata_name']
        ydata = comparison['ydata_name']

        # plot graph of current metrics
        fig.add_trace(go.Scatter(x=metrics[xdata], y=metrics[ydata],
                                 mode='markers+lines',
                                 name=data['metrics'][i]['legend']))

    fig.show()


if __name__ == "__main__":
    iphone_metrics = pd.read_csv('iphone_metrics.csv')
    ust_conf3_metrics = pd.read_csv('ust_conf3_metrics.csv')
    ust_conf3_filtered_metrics = pd.read_csv('ust_conf3_filtered_metrics.csv')

    fscore_comparison = {
        "xlabel": "Number of Frames",
        "ylabel": "F-score",
        "xdata_name": 'num_frames',
        "ydata_name": 'fscore',
    }

    runtime_comparison = {
        "xlabel": "Number of Frames",
        "ylabel": "Runtime",
        "xdata_name": 'num_frames',
        "ydata_name": 'time',
    }

    fscore_runtime_comparison = {
        "xlabel": "Runtime",
        "ylabel": "F-score",
        "xdata_name": 'time',
        "ydata_name": 'fscore',
    }

    data = {
        "metrics": [
            {
                "data": iphone_metrics,
                "legend": "iPhone"
            },
            {
                "data": ust_conf3_metrics,
                "legend": "360 camera + OpenVSLAM pose"
            },
            {
                "data": ust_conf3_filtered_metrics,
                "legend": "360 camera + OpenVSLAM pose (photographer filtered)"
            },
        ]
    }

    # fscore comparison
    layout = go.Layout(
        title=f'{fscore_comparison["xlabel"]} vs {fscore_comparison["ylabel"]}',
        xaxis=dict(title=f'{fscore_comparison["xlabel"]}'),
        yaxis=dict(title=f'{fscore_comparison["ylabel"]}'),
        legend=dict(x=0.6, y=0.05, orientation='v')
    )
    plot(data=data, comparison=fscore_comparison, layout=layout)

    # zoomed fscore comparison
    layout = go.Layout(
        title=f'{fscore_comparison["xlabel"]} vs {fscore_comparison["ylabel"]}',
        xaxis=dict(title=f'{fscore_comparison["xlabel"]}', range=[0, 500]),
        yaxis=dict(title=f'{fscore_comparison["ylabel"]}'),
        legend=dict(x=0.45, y=0.05, orientation='v')
    )
    plot(data=data, comparison=fscore_comparison, layout=layout)

    # runtime comparison
    layout = go.Layout(
        title=f'{runtime_comparison["xlabel"]} vs {runtime_comparison["ylabel"]}',
        xaxis=dict(title=f'{runtime_comparison["xlabel"]}'),
        yaxis=dict(title=f'{runtime_comparison["ylabel"]} (s)'),
        legend=dict(x=0.6, y=0.05, orientation='v')
    )
    plot(data=data, comparison=runtime_comparison, layout=layout)

    # runtime vs fscore
    layout = go.Layout(
        title=f'{fscore_runtime_comparison["xlabel"]} vs {fscore_runtime_comparison["ylabel"]}',
        xaxis=dict(title=f'{fscore_runtime_comparison["xlabel"]} (s)'),
        yaxis=dict(title=f'{fscore_runtime_comparison["ylabel"]}'),
        legend=dict(x=0.6, y=0.05, orientation='v')
    )
    plot(data=data, comparison=fscore_runtime_comparison, layout=layout)



# @click.command()
# @click.option('--name', prompt='file name', default='iphone')
# def plot(name):
# 	click.echo(f'{name}')

# 	num_frames = []
# 	runtime = []
# 	fscore = []

# 	# get data from csv
# 	with open(f'{name}_metrics.csv', 'r') as f:
# 		reader = csv.reader(f)

# 		next(reader)
# 		for row in reader:
# 			num_frames.append(row[0])
# 			runtime.append(row[1])
# 			fscore.append(row[2])

# 	print(f'num_frames: {len(num_frames)}, \n{num_frames}')
# 	# print(f'runtime: {len(runtime)}, \n{runtime}')
# 	print(f'fscore: {len(fscore)}, \n{fscore}')

# 	# plot graph
# 	plt.plot(num_frames, fscore)

# 	plt.xlim(0, int(num_frames[-1]) + 50)
# 	plt.ylim(0, 1)
# 	plt.xlabel('Number of frames')
# 	plt.ylabel('F-score')
# 	plt.show()
