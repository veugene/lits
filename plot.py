import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import os
import sys
import seaborn
from collections import OrderedDict
import numpy as np
import argparse
import re

def parse():
    parser = argparse.ArgumentParser(description="Plot from log file.")
    parser.add_argument('--max_x',
                        help="maximum x value on the plot",
                        required=True, type=int)
    parser.add_argument('--max_y',
                        help="maximum y value on the plot",
                        required=False, default='auto')
    parser.add_argument('--min_y',
                        help="minimum y value on the plot",
                        required=False, default='auto')
    parser.add_argument('--keys',
                        help="the metric keys to scrub from the logs and plot",
                        required=True, nargs='+', type=str)
    parser.add_argument('--title',
                        help="title of the plot",
                        required=True, type=str)
    parser.add_argument('--source',
                        help="source log text file",
                        required=True, type=str)
    parser.add_argument('--dest',
                        help="write the plot to this file",
                        required=False, default="plot.png", type=str)
    return parser.parse_args()

    
def scrub(path, keys):
    history = OrderedDict()
    history_summary = OrderedDict()
    for key in keys:
        history[key] = OrderedDict()
        history_summary[key] = []
    with open(path, 'rt') as f:
        for line in f:
            # Split by ' - ' and ':' plus any surrounding non alpha-num chars.
            split_line = re.split('[: ]+', line)
            epoch = split_line[1]
            for key in keys:
                if epoch not in history[key]:
                    history[key][epoch] = []
                if key in split_line:
                    val_idx = split_line.index(key)+1
                    history[key][epoch].append(float(split_line[val_idx]))
                    
    # Average keys over each epoch.
    for key in keys:
        for epoch in history[key]:
            history_summary[key].append( np.mean(history[key][epoch]) )
    
    # Print the highest Dice score
    max_value = 0
    for i,value in enumerate(history_summary['val_masked_dice_loss']):
       value = -value # Reverse Dice loss
       if value > max_value:
           max_value = value
           best_idx = i
    print('Best val dice : ', history_summary['val_masked_dice_loss'][best_idx])
    print('Average train dice at best val dice : ', history_summary['masked_dice_loss'][best_idx])

    return history_summary


if __name__=='__main__':
    # Get all arguments
    try:
        args = parse()
    except:
        print('Error')
        sys.exit()
    
    # Read log file
    history = scrub(args.source, args.keys)
    
     # Color generator for the plots
    def gen_colors(num_colors):
        for c in seaborn.color_palette('hls', n_colors=num_colors):
            yield c
    
    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1)
    color_generator = gen_colors(num_colors=len(args.keys))
    for i, key in enumerate(args.keys):
        ax = ax
        if args.min_y=='auto':
            min_y = min([min(history[key]) for key in history.keys() \
                                           for ID in history.keys()])*1.1
        else:
            min_y = float(args.min_y)
        if args.max_y=='auto':
            max_y = max([max(history[key]) for key in history.keys() \
                                           for ID in history.keys()])*1.1
        else:
            max_y = float(args.max_y)
        title = args.title
        ax.set_title(title)
        ax.set_xlabel("number of epochs")
        ax.axis([0, args.max_x, min_y, max_y])
        ax.plot(history[key][:args.max_x],
                color=next(color_generator), label=key)
            
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(top=1.5)
    fig.savefig(args.dest, bbox_inches='tight')
    #fig.savefig(args.dest)
    
