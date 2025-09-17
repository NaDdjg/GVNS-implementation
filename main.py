import argparse
from gvns.instance import parse_vrptws_txt
from gvns.local_search import gvns
from gvns.evaluate import evaluate_solution

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("instance")
    p.add_argument("--tmax", type=int, default=60)
    args = p.parse_args()
    inst = parse_vrptws_txt(args.instance)
    best, history = gvns(inst, Tmax_seconds=args.tmax)
    stats = evaluate_solution(inst, best)
    print("Result F=", stats['F'], "f=", stats['f'])
