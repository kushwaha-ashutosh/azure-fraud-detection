
import json
import time
import argparse
from simulator.generator import generate_transaction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--fraud-rate", type=float, default=0.02)
    parser.add_argument("--output", choices=["stdout", "servicebus"], default="stdout")
    args = parser.parse_args()
    print(f"Starting: {args.count} transactions at {args.rate}/sec")
    for i in range(args.count):
        txn = generate_transaction(args.fraud_rate)
        if args.output == "stdout":
            print(json.dumps(txn.to_dict()))
        time.sleep(1.0 / args.rate)
    print(f"Done. Generated {args.count} transactions.")

if __name__ == "__main__":
    main()
