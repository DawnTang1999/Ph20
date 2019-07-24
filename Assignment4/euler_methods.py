import numpy as np
import matplotlib.pyplot as plt
import argparse


def explicit_euler(h, N, x0, v0):
    time = np.arange(N+1) * h
    x = np.zeros(N+1)
    v = np.zeros(N+1)
    diff_x = np.zeros(N+1)
    diff_v = np.zeros(N+1)
    x[0] = x0
    v[0] = v0
    for i in range(1, N+1):
        x[i] = x[i-1] + h * v[i-1]
        v[i] = v[i-1] - h * x[i-1]
    fig = plt.figure()
    plotx = fig.add_subplot(2, 2, 1)
    plotx.plot(time, x)
    plotx.title.set_text('Displacement vs Time')
    plotx.set_ylabel('x')
    plotx.set_xlabel('time')

    plotv = fig.add_subplot(2, 2, 2)
    plotv.plot(time, v)
    plotv.title.set_text('Velocity vs Time')
    plotv.set_xlabel('time')
    plotv.set_ylabel('v')

    # Difference with analytic function
    for a in range(N+1):
        diff_x[a] = x0 * np.cos(time[a]) + v0 * np.sin(time[a]) - x[a]
        diff_v[a] = v0 * np.cos(time[a]) - x0 * np.sin(time[a]) - v[a]

    plotdiffx = fig.add_subplot(2, 2, 3)
    plotdiffx.plot(time, diff_x)
    plotdiffx.title.set_text('Displacement Difference vs Time')
    plotdiffx.set_ylabel('diff_x')
    plotdiffx.set_xlabel('time')

    plotdiffv = fig.add_subplot(2, 2, 4)
    plotdiffv.plot(time, diff_v)
    plotdiffv.title.set_text('Velocity Difference vs Time')
    plotdiffv.set_ylabel('diff_v')
    plotdiffv.set_xlabel('time')

    plt.tight_layout()
    plt.show()
    fig.savefig('explicit.pdf')


def implicit_euler(h, N, x0, v0):
    time = np.arange(N+1) * h
    x = np.zeros(N+1)
    v = np.zeros(N+1)
    diff_x = np.zeros(N+1)
    diff_v = np.zeros(N+1)
    x[0] = x0
    v[0] = v0
    for i in range(1, N+1):
        x[i] = (x[i-1] + h * v[i-1]) / (1 + h**2)
        v[i] = (1 - h**2) * v[i-1] / (1 + h**2) - (h * x[i-1]) / (1 + h**2)
    fig = plt.figure()
    plotx = fig.add_subplot(2, 2, 1)
    plotx.plot(time, x)
    plotx.title.set_text('Displacement vs Time')
    plotx.set_ylabel('x')
    plotx.set_xlabel('time')

    plotv = fig.add_subplot(2, 2, 2)
    plotv.plot(time, v)
    plotv.title.set_text('Velocity vs Time')
    plotv.set_xlabel('time')
    plotv.set_ylabel('v')

    # Difference with analytic function
    for a in range(N+1):
        diff_x[a] = x0 * np.cos(time[a]) + v0 * np.sin(time[a]) - x[a]
        diff_v[a] = v0 * np.cos(time[a]) - x0 * np.sin(time[a]) - v[a]

    plotdiffx = fig.add_subplot(2, 2, 3)
    plotdiffx.plot(time, diff_x)
    plotdiffx.title.set_text('Displacement Difference vs Time')
    plotdiffx.set_ylabel('diff_x')
    plotdiffx.set_xlabel('time')

    plotdiffv = fig.add_subplot(2, 2, 4)
    plotdiffv.plot(time, diff_v)
    plotdiffv.title.set_text('Velocity Difference vs Time')
    plotdiffv.set_ylabel('diff_v')
    plotdiffv.set_xlabel('time')

    plt.tight_layout()
    plt.show()
    fig.savefig('implicit.pdf')


def symplectic_euler(h, N, x0, v0):
    time = np.arange(N+1) * h
    x = np.zeros(N+1)
    v = np.zeros(N+1)
    diff_x = np.zeros(N+1)
    diff_v = np.zeros(N+1)
    x[0] = x0
    v[0] = v0
    for i in range(1, N+1):
        x[i] = x[i-1] + h * v[i-1]
        v[i] = v[i-1] - h * x[i]
    fig = plt.figure()
    plotx = fig.add_subplot(2, 2, 1)
    plotx.plot(time, x)
    plotx.title.set_text('Displacement vs Time')
    plotx.set_ylabel('x')
    plotx.set_xlabel('time')

    plotv = fig.add_subplot(2, 2, 2)
    plotv.plot(time, v)
    plotv.title.set_text('Velocity vs Time')
    plotv.set_xlabel('time')
    plotv.set_ylabel('v')

    # Difference with analytic function
    for a in range(N+1):
        diff_x[a] = x0 * np.cos(time[a]) + v0 * np.sin(time[a]) - x[a]
        diff_v[a] = v0 * np.cos(time[a]) - x0 * np.sin(time[a]) - v[a]

    plotdiffx = fig.add_subplot(2, 2, 3)
    plotdiffx.plot(time, diff_x)
    plotdiffx.title.set_text('Displacement Difference vs Time')
    plotdiffx.set_ylabel('diff_x')
    plotdiffx.set_xlabel('time')

    plotdiffv = fig.add_subplot(2, 2, 4)
    plotdiffv.plot(time, diff_v)
    plotdiffv.title.set_text('Velocity Difference vs Time')
    plotdiffv.set_ylabel('diff_v')
    plotdiffv.set_xlabel('time')

    plt.tight_layout()
    plt.show()
    fig.savefig('symplectic.pdf')


parser = argparse.ArgumentParser(description='Takes in an argument and plots the Euler methods')
parser.add_argument('--Method', metavar='Method', type=str, help='e is explicit, i is implicit, s is symplectic', required=True)
args = parser.parse_args()
method = args.Method

if method == 'e':
    explicit_euler(0.001, 100000, 0.1, 0.1)
elif method == 'i':
    implicit_euler(0.001, 100000, 0.1, 0.1)
elif method == 's':
    symplectic_euler(0.001, 100000, 0.1, 0.1)
else:
    print("Input is invalid. Choose e, i or s.")
