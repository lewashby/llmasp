from llm import LLMASP, LLMHandler
from asp import Solver
import argparse
from examples import marketplace_example

def main():

    parser = argparse.ArgumentParser(
        prog='LLMASP Engine',
        description='Prototype system for smooth interaction between an LLM and an ASP solver',
        epilog='Hope you get the best answer!!!'
    )
    parser.add_argument("behavior_file", help="behavior file",  nargs='?', default=None)
    parser.add_argument("application_file", help="application file",  nargs='?', default=None)
    parser.add_argument("-e", "--example", action="store_true", help="show an example")
    parser.add_argument("-m", "--model", type=str, help="model name", required=True)
    parser.add_argument("-s", "--server", type=str, help="hostname", required=True)
    parser.add_argument("-v", "--verbose", type=int, choices=[0,1], default=0, help="print every step result")
    args = parser.parse_args()
    model = args.model
    server = args.server
    if (args.example):
        marketplace_example(model, server)
        return
    elif args.behavior_file is None or args.application_file is None:
        parser.error("behavior_file and application_file are required if --example is not set")

    else:
        behavior = args.behavior_file
        application = args.application_file
        verbose = args.verbose
        user_input = input("input: ")
        llm = LLMHandler(model, server)
        solver = Solver()
        llmasp = LLMASP(application, behavior, llm, solver)
        response = llmasp.run(user_input, verbose=verbose)
        if (verbose==0):
            print(response)


if __name__ == "__main__":
    main()