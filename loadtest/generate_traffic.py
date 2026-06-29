"""Generate traffic against the Iris API to populate the Grafana dashboards.

Usage:
    python loadtest/generate_traffic.py --url http://localhost:8000 \
        --requests 500 --rate 20
"""
import argparse
import random
import time

import requests


def random_iris():
    return {
        "sepal_length": round(random.uniform(4.3, 7.9), 1),
        "sepal_width": round(random.uniform(2.0, 4.4), 1),
        "petal_length": round(random.uniform(1.0, 6.9), 1),
        "petal_width": round(random.uniform(0.1, 2.5), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--rate", type=float, default=20.0, help="requests per second")
    args = ap.parse_args()

    delay = 1.0 / args.rate if args.rate > 0 else 0.0
    latencies, errors = [], 0
    started = time.perf_counter()

    for i in range(args.requests):
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{args.url}/predict", json=random_iris(), timeout=5)
            latencies.append(time.perf_counter() - t0)
            if r.status_code != 200:
                errors += 1
        except requests.RequestException:
            errors += 1
        if (i + 1) % 50 == 0:
            print(f"sent {i + 1}/{args.requests}")
        time.sleep(delay)

    elapsed = time.perf_counter() - started
    latencies.sort()
    n = len(latencies)
    mean_ms = (sum(latencies) / n * 1000) if n else 0.0
    p95_ms = (latencies[int(0.95 * n) - 1] * 1000) if n else 0.0

    print(f"requests : {args.requests}")
    print(f"errors   : {errors}")
    print(f"duration : {elapsed:.1f}s  ({args.requests / elapsed:.1f} req/s)")
    print(f"latency  : mean {mean_ms:.1f} ms   p95 {p95_ms:.1f} ms")


if __name__ == "__main__":
    main()
