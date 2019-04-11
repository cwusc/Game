import argparse

def getargs():
    parser = argparse.ArgumentParser(description='Set variant algos')
    parser.add_argument('--n', type=int, default=0, dest='nrand')
    parser.add_argument('--lr', type=float, default=-1, dest='lr')
    parser.add_argument('--t', type=int, default=100000, dest='T')
    parser.add_argument('--iter', type=int, default=1000, dest='itermin')
    parser.add_argument('--p', action='store_true', default=False, dest='policy')
    parser.add_argument('--b', action='store_true', default=False, dest='bandits')
    parser.add_argument('--nopt', action='store_true', default=False, dest='nopt')
    parser.add_argument('--nonz', action='store_true', default=False, dest='nonz')
    parser.add_argument('--swap', action='store_true', default=False, dest='swap')
    parser.add_argument('--dylr', action='store_true', default=False, dest='dylr')
    parser.add_argument('--seed', type=int, default=-1, dest='seed')
    parser.add_argument('--stint', type=int, default=5, dest='stint')
    parser.add_argument('--step', type=int, default=1000, dest='step')
    return parser.parse_args()
