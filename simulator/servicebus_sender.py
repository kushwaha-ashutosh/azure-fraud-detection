import json
import os
from azure.servicebus import ServiceBusClient, ServiceBusMessage


def get_sender():
    conn_str = os.environ.get("SERVICEBUS_CONNECTION_STRING")
    if not conn_str:
        raise ValueError("SERVICEBUS_CONNECTION_STRING not set in environment")
    client = ServiceBusClient.from_connection_string(conn_str)
    return client.get_queue_sender(queue_name="raw-transactions")


def send_transaction(sender, transaction: dict):
    message = ServiceBusMessage(json.dumps(transaction))
    sender.send_messages(message)


def send_batch(sender, transactions: list):
    messages = [ServiceBusMessage(json.dumps(t)) for t in transactions]
    sender.send_messages(messages)