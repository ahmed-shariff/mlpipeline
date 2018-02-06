import argparse
import sys

def x():
    
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline")
    parser.add_argument('-r','--run', help='Will set the pipeline to execute the pipline fully, if not set will be executed in test mode', action = 'store_true')
    parser.add_argument('-u','--use-history', help='If set will use the history log to determine if a model script has been executed.', action = 'store_true')
    print(parser.parse_args())
    

if __name__ == "__main__":
    x()
