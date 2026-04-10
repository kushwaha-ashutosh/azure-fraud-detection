import json
import time
import argparse
from dotenv import load_dotenv
from simulator.generator import generate_transaction

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--fraud-rate", type=float, default=0.02)
    parser.add_argument("--output", choices=["stdout", "servicebus"], default="stdout")
    args = parser.parse_args()

    print(f"Starting: {args.count} transactions at {args.rate}/sec, output={args.output}")
    interval = 1.0 / args.rate

    sender = None
    if args.output == "servicebus":
        from simulator.servicebus_sender import get_sender, send_transaction

        sender = get_sender()
        print("Connected to Service Bus queue: raw-transactions")

    for i in range(args.count):
        txn = generate_transaction(args.fraud_rate)
        txn_dict = txn.to_dict()

        if args.output == "stdout":
            print(json.dumps(txn_dict))
        else:
            send_transaction(sender, txn_dict)
            if (i + 1) % 10 == 0:
                print(f"Sent {i + 1}/{args.count} transactions")

        time.sleep(interval)

    if sender:
        sender.close()

    print(f"Done. Generated {args.count} transactions.")


if __name__ == "__main__":
    main()
