import os
import random
import math

INPUT_FOLDER = "instances/New_synch/200"
OUTPUT_FOLDER = "instances/New_synch/300"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def read_instance(filepath):

    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]

    vehicles = int(lines[0])
    customers_n = int(lines[1])
    horizon = int(lines[2])
    sync_n = int(lines[3])
    sync_ids = list(map(int, lines[4].split()))

    customers = []

    for line in lines[5:]:
        p = line.split()

        customers.append({
            "id": int(p[0]),
            "x": int(p[1]),
            "y": int(p[2]),
            "ready": int(p[3]),
            "due": int(p[4]),
            "service": int(p[5])
        })

    return vehicles, customers_n, horizon, sync_n, sync_ids, customers


def replicate_customers(customers, target_n):

    depot = customers[0]
    original = customers[1:]

    current_n = len(original)

    needed = target_n - current_n

    new_customers = []

    for i in range(needed):

        base = random.choice(original)

        new = base.copy()

        new["id"] = current_n + i + 1

        new["x"] = base["x"] + random.randint(-3,3)
        new["y"] = base["y"] + random.randint(-3,3)

        new_customers.append(new)

    return [depot] + original + new_customers


def choose_sync(customers, sync_count):

    cust = customers[1:]

    windows = [(c["id"], c["due"] - c["ready"]) for c in cust]

    windows.sort(key=lambda x: x[1], reverse=True)

    return [cid for cid,_ in windows[:sync_count]]


def write_instance(path, vehicles, customers, horizon, sync_ids):

    n_customers = len(customers) - 1
    sync_count = len(sync_ids)

    with open(path,"w") as f:

        f.write(f"{vehicles}\n")
        f.write(f"{n_customers}\n")
        f.write(f"{horizon}\n")
        f.write(f"{sync_count}\n")

        f.write(" ".join(map(str,sync_ids))+"\n")

        for c in customers:
            f.write(f"{c['id']} {c['x']} {c['y']} {c['ready']} {c['due']} {c['service']}\n")


def generate():

    for file in os.listdir(INPUT_FOLDER):

        if not file.endswith(".txt"):
            continue

        path = os.path.join(INPUT_FOLDER,file)

        vehicles,n_customers,horizon,sync_n,sync_ids,customers = read_instance(path)

        if n_customers == 100:
            target = 150
        elif n_customers == 200:
            target = 300
        else:
            continue

        new_customers = replicate_customers(customers,target)

        new_vehicles = math.ceil(vehicles * target / n_customers)

        new_sync = target // 10

        new_sync_ids = choose_sync(new_customers,new_sync)

        name = file.split(".")[0].split("_")[0]

        new_name = f"{name}_{target}_{new_sync}.txt"

        out = os.path.join(OUTPUT_FOLDER,new_name)

        write_instance(out,new_vehicles,new_customers,horizon,new_sync_ids)

        print("Generated",new_name)


if __name__ == "__main__":
    generate()