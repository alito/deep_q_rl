#!/usr/bin/env python
""" This script runs a pre-trained network with the game
visualization turned on.

Specify the network file first, then any other options you want
"""
import subprocess
import sys
import argparse

DefaultTestEpsilon = 0.01

def run_watch(args):

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-n', '--network', dest='networkfile', required=True,
                        help='Network file. Use "none" to test a newly created (ie random) network')
    parser.add_argument('--test-epsilon', dest="test_epsilon", default=DefaultTestEpsilon, 
                        type=float,
                        help=('Set the rate of random actions (epsilon) to be used during ' +
                        'testing epochs'))        
    parser.add_argument('--no-screen', dest="screen", default=True, action="store_false",
                        help="Don't show the screen. Only option that should come before the network")        
    parser.add_argument('--nips', dest="nips", default=False, action="store_true",
                        help="Use the NIPS network architecture")    
    parser.add_argument('--record', dest="record", default=False, action="store_true",
                        help="Record stats for the run")
    parameters, unknown = parser.parse_known_args(args)

    command = []
    if parameters.nips:
        command.append('./run_nips.py')
    else:
        command.append('./run_double.py')

    command.extend(['--steps-per-epoch', '0', '--test-epsilon', str(parameters.test_epsilon)])
    if parameters.networkfile.lower() != 'none':
        command.extend(['--nn-file', parameters.networkfile])
    if parameters.screen:
        command.append('--display-screen')
    if not parameters.record:
        command.append("--no-record")

    command += unknown
    subprocess.call(command)

    return 0

if __name__ == "__main__":
    sys.exit(run_watch(sys.argv[1:]))
