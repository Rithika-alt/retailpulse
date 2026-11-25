"""S3 access for raw data, pointed at LocalStack (never real AWS by default).

Safety rail: get_s3_client() refuses to build a client unless AWS_ENDPOINT_URL is set to a
non-AWS endpoint (e.g. LocalStack), or ALLOW_REAL_AWS=1 is set explicitly. This project makes
no real AWS calls; the rail makes an accidental one (and its cost) impossible by default.

RETAILPULSE_RAW_SOURCE controls the ETL's raw-data step:
  auto  (default) use S3 if reachable and the object exists, else WARN and fall back to the
                  local data/raw/ folder / synthetic generator (Phase 1-3 workflow keeps working)
  s3              strict: fail loudly if S3 or the object is unavailable
  local           never touch S3
boto3 is imported lazily so the rest of the project does not need it.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "retailpulse-raw-data"
DEFAULT_KEY = "raw/sales.csv"
DEFAULT_REGION = "us-east-1"
MODES = ("auto", "s3", "local")


class StorageError(RuntimeError):
    pass


def raw_bucket() -> str:
    return os.environ.get("RETAILPULSE_RAW_BUCKET", DEFAULT_BUCKET)


def resolve_mode(mode: str | None = None) -> str:
    mode = (mode or os.environ.get("RETAILPULSE_RAW_SOURCE", "auto")).strip().lower()
    if mode not in MODES:
        raise StorageError(f"RETAILPULSE_RAW_SOURCE must be one of {MODES}, got {mode!r}")
    return mode


def check_endpoint_is_safe(endpoint: str | None) -> str | None:
    allow_real = os.environ.get("ALLOW_REAL_AWS") == "1"
    if not endpoint:
        if allow_real:
            return None
        raise StorageError(
            "AWS_ENDPOINT_URL is not set; refusing to create an S3 client that could reach real AWS. "
            "Point it at LocalStack (http://localhost:4566) or set ALLOW_REAL_AWS=1 to override."
        )
    if "amazonaws.com" in endpoint and not allow_real:
        raise StorageError(f"Endpoint {endpoint!r} looks like real AWS; set ALLOW_REAL_AWS=1 to override.")
    return endpoint


def get_s3_client(endpoint_url: str | None = None):
    import boto3
    from botocore.config import Config

    endpoint = check_endpoint_is_safe(endpoint_url or os.environ.get("AWS_ENDPOINT_URL"))
    kwargs = {}
    if endpoint:  # LocalStack accepts any credentials; never use real ones here
        kwargs["aws_access_key_id"] = os.environ.get("AWS_ACCESS_KEY_ID", "test")
        kwargs["aws_secret_access_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "test")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=os.environ.get("AWS_DEFAULT_REGION", DEFAULT_REGION),
        config=Config(connect_timeout=3, read_timeout=10, retries={"max_attempts": 1}),
        **kwargs,
    )


def ensure_bucket(client, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        client.create_bucket(Bucket=bucket)  # us-east-1: no LocationConstraint


def upload_raw_file(client, path, bucket: str | None = None, key: str = DEFAULT_KEY) -> dict:
    bucket = bucket or raw_bucket()
    ensure_bucket(client, bucket)
    client.upload_file(str(path), bucket, key)
    head = client.head_object(Bucket=bucket, Key=key)
    return {"bucket": bucket, "key": key, "bytes": head["ContentLength"], "etag": head["ETag"].strip('"')}


def fetch_raw_from_s3(dest, bucket: str | None = None, key: str = DEFAULT_KEY,
                      mode: str | None = None, client=None) -> str | None:
    """Download the raw CSV to `dest`. Returns the path if downloaded, None if skipped/fell back."""
    mode = resolve_mode(mode)
    if mode == "local":
        logger.info("RETAILPULSE_RAW_SOURCE=local: not using S3.")
        return None
    bucket = bucket or raw_bucket()
    dest = Path(dest)
    try:
        client = client or get_s3_client()
        head = client.head_object(Bucket=bucket, Key=key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        client.download_file(bucket, key, str(tmp))
        os.replace(tmp, dest)
    except Exception as e:  # noqa: BLE001 - any S3/network/config failure
        if mode == "s3":
            raise StorageError(
                f"RETAILPULSE_RAW_SOURCE=s3 but s3://{bucket}/{key} could not be fetched: {e}. "
                "Did you run scripts/upload_raw_data.py (and is LocalStack up)?"
            ) from e
        logger.warning(
            "S3 raw data unavailable (s3://%s/%s: %s). Falling back to local data/raw/ "
            "(or synthetic generation).", bucket, key, e,
        )
        return None
    logger.info(
        "Fetched s3://%s/%s -> %s (%s bytes, etag %s)",
        bucket, key, dest, head["ContentLength"], head["ETag"].strip('"'),
    )
    return str(dest)
