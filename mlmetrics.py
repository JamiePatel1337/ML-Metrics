
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License Version 3 as
#published by the Free Software Foundation.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>.

#IMPORTS

import os
import sys
import time
import warnings

import numpy as np
import argparse as ap
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn import show_versions
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning

#CONSTANTS

MAX_ITERS = 100000
MAX_NODES = 1000
MAX_FACTOR = 0.8
MAX_VERBOSE = 3
MAX_STEPS = 10

BAR_WIDTH = 0.25

#VARIABLES

G_D = './exp.data'

G_N = 0
G_I = [10000]
G_L = [[5]]
G_F = 0.5
G_V = 0
G_S = 3

#FUNCTIONS

def get_user_args():
    parser = ap.ArgumentParser()
    parser.add_argument("DATA_SIZE", type=int, help="Number of samples in dataset, selected sequentially from start of data file")
    parser.add_argument("-d", "--file", help=f"Path to data file, default = {G_D}", default=G_D)
    parser.add_argument("-i", "--iters", type=get_arg_iters, help=f"List of iterations, default = {G_I}, max. = {MAX_ITERS}", default=G_I)
    parser.add_argument("-l", "--layers", type=get_arg_layers, help=f"List of hidden layer configurations, default = {G_L}, max. = {MAX_NODES}", default=G_L)
    parser.add_argument("-f", "--factor", type=float, help=f"Factor of samples to use for training, default = {G_F}, max. = {MAX_FACTOR}", default=G_F)
    parser.add_argument("-s", "--steps", type=int, help=f"Sample steps to use for training, default = {G_S}, max. = {MAX_STEPS}", default=G_S)
    parser.add_argument("-v", "--verbose", action="count", help=f"Increase verbosity, default = {G_V}, max. = {MAX_VERBOSE}", default=G_V)
    args = parser.parse_args()
    path = args.file
    if not os.path.isfile(path) or not path.endswith('.data'):
        print(f"Invalid data file! Exiting...")
        sys.exit(1)
    return load(path), args
def parse_user_args(N, args):
    n = len(N)
    if args.DATA_SIZE < len(N):
        n = args.DATA_SIZE
    t = args.iters
    for i in range(len(t)):
        if t[i] > MAX_ITERS:
            t[i] = MAX_ITERS
    l = args.layers
    for i in range(len(l)):
        for j in range(len(l[i])):
            if l[i][j] > MAX_NODES:
                l[i][j] = MAX_NODES
    f = args.factor
    if f > MAX_FACTOR:
        f = MAX_FACTOR
    v = args.verbose
    if v > MAX_VERBOSE:
        v = MAX_VERBOSE
    s = args.steps
    if s > MAX_STEPS:
        s = MAX_STEPS
    return n, t, l, f, s, v
def get_arg_iters(value):
    try:
        values = [int(i) for i in value.split(',')]
        assert len(values) < 6
    except (ValueError, AssertionError):
        raise ap.ArgumentTypeError('Max. supported iterations is 5')
    return values
def get_arg_layers(value):
    ret = []
    try:
        tuples = [i for i in value.split(':')]
        assert len(tuples) < 6
    except (AssertionError):
        raise ap.ArgumentTypeError('Max. supported layer sets is 5')
    for tuple in tuples:
        try:
            values = [int(i) for i in tuple.split(',')]
            assert len(values) < 6
            tup = []
            for value in values:
                tup.append(value)
            ret.append(tup)
        except (ValueError, AssertionError):
            raise ap.ArgumentTypeError('Max. supported hidden layers is 5')
    return ret
def split_data(df):
    c = df.columns
    d = []
    for i in range(len(c)):
        j = np.array(df[c[i]])
        d.append(j.reshape(-1,1))
    X = np.hstack((d[0], d[1], d[2], d[3], d[4]))
    y = np.hstack((d[5], d[6]))
    v = np.hstack((d[7], d[8]))
    X_t = X[0:int(G_N * G_F)]
    y_t = y[0:int(G_N * G_F)]
    v_t = v[0:int(G_N * G_F)]
    X_v = X[int(G_N * G_F):int(G_N)]
    y_v = y[int(G_N * G_F):int(G_N)]
    v_v = v[int(G_N * G_F):int(G_N)]
    return X_t, y_t, v_t, X_v, y_v, v_v
def train_all_models(X, y, X_val, y_val):
    ret = {}
    r_i_a = []
    c_i_a = []
    t_i_a = []
    s_i_a = []
    for i in range(len(G_L)):
        r_j_a = []
        c_j_a = []
        t_j_a = []
        s_j_a = []
        for j in range(len(G_I)):
            r_k_a = []
            c_k_a = []
            t_k_a = []
            s_k_a = []
            for k in range(G_S):
                X_t_a = X[0:int(float((k+1)/G_S) * len(X))]
                y_t_a = y[0:int(float((k+1)/G_S) * len(y))]
                c = 1
                start_time = time.perf_counter()
                try:
                    r_k_a.append(MLPRegressor(max_iter=G_I[j], hidden_layer_sizes=tuple(G_L[i])).fit(X_t_a, y_t_a))
                except Exception as e:
                    if type(e) == ConvergenceWarning:
                        c = 0
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        r_k_a.append(MLPRegressor(max_iter=G_I[j], hidden_layer_sizes=tuple(G_L[i])).fit(X_t_a, y_t_a))
                        warnings.resetwarnings()
                        warnings.filterwarnings("error", category=ConvergenceWarning)
                end_time = time.perf_counter()
                score = r_k_a[k].score(X_val, y_val)
                t_k_a.append(end_time - start_time)
                c_k_a.append(c)
                s_k_a.append(score)
                if G_V > 0:
                    print(f"Model {i},{j},{k} trained!")
            r_j_a.append(r_k_a)
            c_j_a.append(c_k_a)
            t_j_a.append(t_k_a)
            s_j_a.append(s_k_a)
        r_i_a.append(r_j_a)
        c_i_a.append(c_j_a)
        t_i_a.append(t_j_a)
        s_i_a.append(s_j_a)
    ret.update({'regr': r_i_a})
    ret.update({'conv': c_i_a})
    ret.update({'time': t_i_a})
    ret.update({'score': s_i_a})
    return ret
def create_plot_steps(x, steps):
    x_p = []
    for k in range(steps):
        x_p.append(int(float((k+1)/steps) * len(x)))
    X_p = np.array(x_p)
    X_p = X_p.reshape(-1,1)
    return X_p
def plot_all_graphs(x, data, plt):
    rows = len(G_L)
    cols = len(G_I)
    if rows == 1 and cols == 1:
        plt.plot(x, data['score'][0][0], 'g.--')
        plt.plot(x, data['time'][0][0], 'r.-')
        plt.plot(x, data['conv'][0][0], 'bx:')
        plt.xlabel('Training set size')
        plt.grid()
        plt.legend(['Scores', 'Times', 'Converged'])
        plt.title(f"Layers: {G_L[0]}, Iterations: {G_I[0]}")
        plt.tight_layout()
        plt.show()
    elif rows == 1 and cols > 1:
        fig_luz, axes = plt.subplots(rows, cols, figsize=(9, 7), dpi = 100, layout='constrained')
        for c in range(cols):
            axes[c].plot(x, data['score'][0][c], 'g.--')
            axes[c].plot(x, data['time'][0][c], 'r.-')
            axes[c].plot(x, data['conv'][0][c], 'bx:')
            axes[c].legend(['Scores', 'Times', 'Converged'])
            axes[c].set_xlabel('Training set size')
            axes[c].grid()
            axes[c].set_title(f"Layers: {G_L[0]}, Iterations: {G_I[c]}")
        plt.show()
    elif cols == 1 and rows > 1:
        fig_luz, axes = plt.subplots(rows, cols, figsize=(9, 7), dpi = 100, layout='constrained')
        for r in range(rows):
            axes[r].plot(x, data['score'][r][0], 'g.--')
            axes[r].plot(x, data['time'][r][0], 'r.-')
            axes[r].plot(x, data['conv'][r][0], 'bx:')
            axes[r].legend(['Scores', 'Times', 'Converged'])
            axes[r].set_xlabel('Training set size')
            axes[r].grid()
            axes[r].set_title(f"Layers: {G_L[r]}, Iterations: {G_I[0]}")
        plt.show()
    else:
        fig_luz, axes = plt.subplots(rows, cols, figsize=(9, 7), dpi = 100, layout='constrained')
        for r in range(rows):
            for c in range(cols):
                axes[r,c].plot(x, data['score'][r][c], 'g.--')
                axes[r,c].plot(x, data['time'][r][c], 'r.-')
                axes[r,c].plot(x, data['conv'][r][c], 'bx:')
                axes[r,c].legend(['Scores', 'Times', 'Converged'])
                axes[r,c].set_xlabel('Training set size')
                axes[r,c].grid()
                axes[r,c].set_title(f"Layers: {G_L[r]}, Iterations: {G_I[c]}")
        plt.show()
def plot_all_bars(x, data, plt):
    rows = len(G_L)
    cols = len(G_I)
    xb = np.arange(len(x))
    xl = tuple(x.ravel())
    if rows == 1 and cols == 1:
        s = tuple(data['score'][0][0])
        t = tuple(data['time'][0][0])
        co = tuple(data['conv'][0][0])
        o = {}
        o.update({'Scores': s})
        o.update({'Times': t})
        o.update({'Converged': co})
        multiplier = 0
        for attribute, measurement in o.items():
            offset = BAR_WIDTH * multiplier
            rects = plt.bar(xb + offset, measurement, BAR_WIDTH, label=attribute)
            plt.bar_label(rects, padding=3, fmt="%.3f")
            multiplier += 1
        plt.title(f"Layers: {G_L[0]}, Iterations: {G_I[0]}")
        plt.xlabel('Training set size')
        plt.xticks(xb + BAR_WIDTH, xl)
        plt.legend(ncols=3)
        plt.tight_layout()
        plt.show()
    elif rows == 1 and cols > 1:
        fig_luz, axes = plt.subplots(rows, cols, figsize=(9, 7), dpi = 100, layout='constrained')
        for c in range(cols):
            s = tuple(data['score'][0][c])
            t = tuple(data['time'][0][c])
            co = tuple(data['conv'][0][c])
            o = {}
            o.update({'Scores': s})
            o.update({'Times': t})
            o.update({'Converged': co})
            multiplier = 0
            for attribute, measurement in o.items():
                offset = BAR_WIDTH * multiplier
                width = BAR_WIDTH
                rects = axes[c].bar(xb + offset, measurement, width, label=attribute)
                axes[c].bar_label(rects, padding=3, fmt="%.3f")
                multiplier += 1
            axes[c].set_title(f"Layers: {G_L[0]}, Iterations: {G_I[c]}")
            axes[c].set_xlabel('Training set size')
            axes[c].set_xticks(xb + BAR_WIDTH, xl)
            axes[c].legend(ncols=3)
        plt.show()
    elif cols == 1 and rows > 1:
        fig_luz, axes = plt.subplots(rows, cols, figsize=(9, 7), dpi = 100, layout='constrained')
        for r in range(rows):
            s = tuple(data['score'][r][0])
            t = tuple(data['time'][r][0])
            co = tuple(data['conv'][r][0])
            o = {}
            o.update({'Scores': s})
            o.update({'Times': t})
            o.update({'Converged': co})
            multiplier = 0
            for attribute, measurement in o.items():
                offset = BAR_WIDTH * multiplier
                width = BAR_WIDTH
                rects = axes[r].bar(xb + offset, measurement, width, label=attribute)
                axes[r].bar_label(rects, padding=3, fmt="%.3f")
                multiplier += 1
            axes[r].set_title(f"Layers: {G_L[r]}, Iterations: {G_I[0]}")
            axes[r].set_xlabel('Training set size')
            axes[r].set_xticks(xb + BAR_WIDTH, xl)
            axes[r].legend(ncols=3)
        plt.show()
    else:
        fig_luz, axes = plt.subplots(rows, cols, figsize=(9, 7), dpi = 100, layout='constrained')
        for r in range(rows):
            for c in range(cols):
                s = tuple(data['score'][r][c])
                t = tuple(data['time'][r][c])
                co = tuple(data['conv'][r][c])
                o = {}
                o.update({'Scores': s})
                o.update({'Times': t})
                o.update({'Converged': co})
                multiplier = 0
                for attribute, measurement in o.items():
                    offset = BAR_WIDTH * multiplier
                    width = BAR_WIDTH
                    rects = axes[r,c].bar(xb + offset, measurement, width, label=attribute)
                    axes[r,c].bar_label(rects, padding=3, fmt="%.3f")
                    multiplier += 1
                axes[r,c].set_title(f"Layers: {G_L[r]}, Iterations: {G_I[c]}")
                axes[r,c].set_xlabel('Training set size')
                axes[r,c].set_xticks(xb + BAR_WIDTH, xl)
                axes[r,c].legend(ncols=3)
        plt.show()

#MAIN

df, args = get_user_args()
G_N, G_I, G_L, G_F, G_S, G_V = parse_user_args(df, args)
X_t, y_t, v_t, X_v, y_v, v_v = split_data(df)

if(G_V > 0):
    print(f"DATA = {len(df)}")
    print(f"SAMPLES = {G_N}")
    print(f"ITERS = {G_I}")
    print(f"LAYERS = {G_L}")
    print(f"FACTOR = {G_F}")
    print(f"STEPS = {G_S}")
    print(f"VERBOSE = {G_V}")
    print(f"{df.head(1)}")
    if(G_V > 1):
        print(f"X_t[0]: {X_t[0]}, shape(X): {X_t.shape}")
        print(f"y_t[0]: {y_t[0]}, shape(y): {y_t.shape}")
        print(f"v_t[0]: {v_t[0]}, shape(v): {v_t.shape}")
        print(f"X_v[0]: {X_v[0]}, shape(X_v): {X_v.shape}")
        print(f"y_v[0]: {y_v[0]}, shape(y_v): {y_v.shape}")
        print(f"v_v[0]: {v_v[0]}, shape(v_v): {v_v.shape}")

ic = input(f"Start training? [Y/n]\n")
if ic != 'Y':
    sys.exit(0)
if(G_V > 0):
    print(f"Starting training...")

warnings.resetwarnings()
warnings.filterwarnings("error", category=ConvergenceWarning)

#{'regr': [[rows/layers][columns/iters][samples/steps]], 'conv' = [], 'time' = [], 'score' = []}
data_struct = train_all_models(X_t, y_t, X_v, y_v)

if(G_V > 0):
    if(G_V > 1):
        print(f"Regr: {len(data_struct['regr'])}")
        print(f"Conv: {(data_struct['conv'])}")
        print(f"Score: {(data_struct['score'])}")
        if(G_V > 2):
            print(f"Data: {data_struct}")
        ro = 0
        co = 0
        sam = G_S - 1
        print(f"score: {data_struct['score'][ro][co][sam]}, time: {data_struct['time'][ro][co][sam]}, conv:{data_struct['conv'][ro][co][sam]}, regr: {data_struct['regr'][ro][co][sam]}")
    print(f"Training complete!")

warnings.resetwarnings()
warnings.filterwarnings("ignore", category=UserWarning)

X_p = create_plot_steps(X_t, G_S)
plot_all_graphs(X_p, data_struct, plt)
plot_all_bars(X_p, data_struct, plt)

warnings.resetwarnings()

ic = input(f"Save data? [Y/n]\n")

if ic == 'Y':
    ts = int(time.time())
    a = dump(X_p, f"{ts}.sample.data")
    b = dump(data_struct, f"{ts}.model.data")
    if G_V > 0:
        vs = show_versions()
        print(f"Saved files: {a[0]} & {b[0]}")
else:
    if G_V > 0:
        ic = input(f"Last chance! Save data? [Y/n]\n")
        if ic == 'Y':
            ts = int(time.time())
            a = dump(X_p, f"{ts}.sample.data")
            b = dump(data_struct, f"{ts}.model.data")
            vs = show_versions()
            print(f"Saved files: {a[0]} & {b[0]}")

sys.exit(0)

