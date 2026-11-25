"""Upload the raw sales CSV to the S3 bucket on LocalStack (never real AWS).

    python scripts/upload_raw_data.py                       # data/raw/sales.csv -> s3://retailpulse-raw-data/raw/sales.csv
    python scripts/upload_raw_data.py --file other.csv --bucket my-bucket

Generates the synthetic dataset first if the local file does not exist.
LocalStack (community) does not persist data across restarts: re-run after `docker compose down`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_generation import generate_or_load  # noqa: E402
from src.storage import DEFAULT_KEY, get_s3_client, raw_bucket, upload_raw_file  # noqa: E402


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", default="data/raw/sales.csv")
    p.add_argument("--bucket", default=None, help="default: $RETAILPULSE_RAW_BUCKET or retailpulse-raw-data")
    p.add_argument("--key", default=DEFAULT_KEY)
    p.add_argument("--endpoint-url", default=os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566"))
    args = p.parse_args(argv)

    path = generate_or_load(args.file)  # uses the existing file, or generates it
    client = get_s3_client(args.endpoint_url)
    info = upload_raw_file(client, path, args.bucket or raw_bucket(), args.key)
    print(f"Uploaded {path} -> s3://{info['bucket']}/{info['key']} "
          f"({info['bytes']} bytes, etag {info['etag']}) via {args.endpoint_url}")
    return info


if __name__ == "__main__":
    main()
