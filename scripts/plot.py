import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import os
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
        last_epoch, epoch = None, 0
        for line in f:
            # Split by ' - ' and ':' plus any surrounding non alpha-num chars.
            split_line = re.split('[: ]+', line)
            if last_epoch is None:
                last_epoch = split_line[1]
            if split_line[1] != last_epoch:
                epoch += 1
            last_epoch = split_line[1]
            for key in keys:
                if epoch not in history[key]:
                    history[key][epoch] = []
                if key in split_line:
                    val_idx = len(split_line)-split_line[::-1].index(key)
                    history[key][epoch].append(float(split_line[val_idx]))
            #epoch += 1
                    
    # Average keys over each epoch.
    for key in keys:
        #window = [0]*2000
        #window = [0]*1
        for epoch in history[key]:
            if history[key][epoch]:
                #window.append(np.mean(history[key][epoch]))
                #window.pop(0)
                history_summary[key].append( np.mean(history[key][epoch]) )
                #history_summary[key].append( np.mean(window) )
                    
    return history_summary


if __name__=='__main__':
    # Get all arguments
    args = parse()
    
    # Read log file
    history = scrub(args.source, args.keys)
    
    # TEMP
    # Print best score
    key_train, key_val = args.keys
    if np.all(np.greater_equal(history[key_val], 0)):
        idx = np.argmax(history[key_val])
    elif np.all(np.less_equal(history[key_val], 0)):
        idx = np.argmin(history[key_val])
    else:
        raise ValueError
    print("Best validation {}: {}"
          "".format(key_val, history[key_val][idx]))
    print("-- training {}: {}".format(key_train, history[key_train][idx]))
    print("-- epoch {} of {}".format(idx+1, len(history[key_train])))
    # /TEMP
    
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
        ax.yaxis.set_ticks(np.arange(min_y, max_y, (max_y-min_y)/20.))
            
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(top=1.5)
    fig.savefig(args.dest, bbox_inches='tight')
    #fig.savefig(args.dest)
    
