import argparse
import sys

def x():
    
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline")
    parser.add_argument('-r', help='Will set the pipeline to execute the pipline fully, if not set will be executed in test mode', action = 'store_true')
    print(parser.parse_args().r)

if __name__ == "__main__":
    x()
