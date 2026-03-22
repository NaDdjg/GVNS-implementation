import os
import re

INPUT_FOLDER = "instances/Solomon/100"
OUTPUT_FOLDER = "instances/New_synch/100"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def parse_solomon_instance(filepath):

    with open(filepath, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # number of vehicles
    for i, line in enumerate(lines):
        if line.startswith("NUMBER"):
            vehicles_line = lines[i+1]
            nb_vehicles = int(vehicles_line.split()[0])
            break

    # find where customers start
    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("CUST NO"):
            start_idx = i + 1
            break

    customers = []

    for line in lines[start_idx:]:
        parts = line.split()
        cust_id = int(parts[0])
        x = int(parts[1])
        y = int(parts[2])
        demand = int(parts[3])
        ready = int(parts[4])
        due = int(parts[5])
        service = int(parts[6])

        customers.append({
            "id": cust_id,
            "x": x,
            "y": y,
            "ready": ready,
            "due": due,
            "service": service,
            "window": due - ready
        })

    return nb_vehicles, customers


def choose_synch_customers(customers):

    n_customers = len(customers) - 1  # excluding depot

    sync_count = max(1, n_customers // 10)

    # exclude depot
    cust_only = [c for c in customers if c["id"] != 0]

    # sort by time window size
    cust_only.sort(key=lambda x: x["window"], reverse=True)

    synch_ids = [c["id"] for c in cust_only[:sync_count]]

    return sync_count, synch_ids


def write_instance(output_path, vehicles, customers, sync_count, sync_ids):

    horizon = customers[0]["due"]
    n_customers = len(customers) - 1

    with open(output_path, "w") as f:

        f.write(f"{vehicles}\n")
        f.write(f"{n_customers}\n")
        f.write(f"{horizon}\n")
        f.write(f"{sync_count}\n")

        f.write(" ".join(map(str, sync_ids)) + "\n")

        for c in customers:
            f.write(
                f"{c['id']} {c['x']} {c['y']} {c['ready']} {c['due']} {c['service']}\n"
            )


def process_instances():

    for file in os.listdir(INPUT_FOLDER):

        if not file.lower().endswith(".txt"):
            continue

        input_path = os.path.join(INPUT_FOLDER, file)

        vehicles, customers = parse_solomon_instance(input_path)

        sync_count, sync_ids = choose_synch_customers(customers)

        instance_name = os.path.splitext(file)[0]
        n_customers = len(customers) - 1

        # if there are more than 200 customers, omit the customer count from filename
        if n_customers > 200:
            new_filename = f"{instance_name}_{sync_count}.txt"
        else:
            new_filename = f"{instance_name}_{n_customers}_{sync_count}.txt"

        output_path = os.path.join(OUTPUT_FOLDER, new_filename)

        write_instance(output_path, vehicles, customers, sync_count, sync_ids)

        print(f"Processed {file}")


if __name__ == "__main__":
    process_instances()