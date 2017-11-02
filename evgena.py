#!/usr/bin/env python

import sys
import argparse


def create_config():
    NotImplemented


def show_off():
    NotImplemented


def tamper():
    NotImplemented


def benchmark():
    NotImplemented


def dummy(argv):
    parser = argparse.ArgumentParser(
        prog='evgena.py dummy',
        description='nothing interesting here',
        epilog='nothing interesting here either'
    )

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    _, _ = parser.parse_known_args(argv)

if __name__ == '__main__':
    description = '''
        Unleash the power of deception of Inception
    (or any other machine learning model of your choice)
        
          |\__/|                          |\__/|
         (_ ^-^)                         (_ ^-^)
          )   (                           )   (  
         /     \  )        ---->         /     \ (   
        (       )(                      (       ) )
         \__/__/\_)                      \__\__/_/
         
           cat                            panda
           
       well, its a bit exaggerated, only small images
          of alphanumeric characters are supported'''

    epilog = '''
For detailed info about each action, call:
    ./evgena.sh <action> [-h | --help]
    '''

    main_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=description,
        epilog=epilog
    )

    activity_group = main_parser.add_mutually_exclusive_group(required=True)

    activity_group.add_argument(
        '--create_config',
        action='store_const',
        const=create_config,
        dest='activity',
        help='Walks you through creation of config'
    )

    activity_group.add_argument(
        '--show_off',
        action='store_const',
        const=show_off,
        dest='activity',
        help='Basically runs advertisement for this project'
    )

    activity_group.add_argument(
        '--tamper',
        action='store_const',
        const=tamper,
        dest='activity',
        help='Modify image to be classified with given label with respect to given model'
    )

    activity_group.add_argument(
        '--benchmark',
        action='store_const',
        const=benchmark,
        dest='activity',
        help='Benchmark performance of algorithm'
    )

    activity_group.add_argument(
        '--dummy',
        action='store_const',
        const=dummy,
        dest='activity',
        help='Dummy run, just for meaningless use of hardware resources'
    )

    if len(sys.argv) == 1:
        main_parser.print_help()
        sys.exit(1)

    parsed_args, unknown = main_parser.parse_known_args()

    parsed_args.activity(unknown)
