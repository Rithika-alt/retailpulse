import logging
import os
import urllib.request

import pytest

from src import storage
from src.storage import StorageError, fetch_raw_from_s3, resolve_mode


class FakeS3:
    def __init__(self, objects=None, fail=None):
        self.objects, self.fail, self.calls = objects or {}, fail, []

    def head_object(self, Bucket, Key):
        self.calls.append("head")
        if self.fail:
            raise self.fail
        if (Bucket, Key) not in self.objects:
            raise RuntimeError("404 Not Found")
        return {"ContentLength": len(self.objects[(Bucket, Key)]), "ETag": '"abc"'}

    def download_file(self, Bucket, Key, Filename):
        self.calls.append("download")
        with open(Filename, "wb") as f:
            f.write(self.objects[(Bucket, Key)])


def test_default_mode_is_auto(monkeypatch):
    monkeypatch.delenv("RETAILPULSE_RAW_SOURCE", raising=False)
    assert resolve_mode() == "auto"


def test_invalid_mode_raises(monkeypatch):
    monkeypatch.setenv("RETAILPULSE_RAW_SOURCE", "cloud")
    with pytest.raises(StorageError):
        resolve_mode()


def test_fetch_downloads_file(tmp_path):
    s3 = FakeS3({("b", "raw/sales.csv"): b"item_id,sales\nA,1\n"})
    dest = tmp_path / "raw" / "sales.csv"
    assert fetch_raw_from_s3(dest, bucket="b", mode="s3", client=s3) == str(dest)
    assert dest.read_bytes() == b"item_id,sales\nA,1\n"
    assert not (tmp_path / "raw" / "sales.csv.part").exists()


def test_strict_mode_fails_loudly_when_object_missing(tmp_path):
    with pytest.raises(StorageError, match="upload_raw_data"):
        fetch_raw_from_s3(tmp_path / "x.csv", bucket="b", mode="s3", client=FakeS3())
    assert not (tmp_path / "x.csv").exists()


def test_strict_mode_fails_when_s3_unreachable(tmp_path):
    with pytest.raises(StorageError):
        fetch_raw_from_s3(tmp_path / "x.csv", bucket="b", mode="s3", client=FakeS3(fail=ConnectionError("refused")))


def test_auto_mode_falls_back_with_warning_and_leaves_local_file(tmp_path, caplog):
    dest = tmp_path / "sales.csv"
    dest.write_text("local")
    with caplog.at_level(logging.WARNING, logger="src.storage"):
        out = fetch_raw_from_s3(dest, bucket="b", mode="auto", client=FakeS3(fail=ConnectionError("refused")))
    assert out is None and dest.read_text() == "local"
    assert any("Falling back" in r.message for r in caplog.records)


def test_local_mode_never_touches_s3(tmp_path):
    s3 = FakeS3({("b", "raw/sales.csv"): b"x"})
    assert fetch_raw_from_s3(tmp_path / "x.csv", bucket="b", mode="local", client=s3) is None
    assert s3.calls == []


# ---- safety rail: never reach real AWS by accident ----------------------------
def test_client_refused_without_endpoint(monkeypatch):
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("ALLOW_REAL_AWS", raising=False)
    with pytest.raises(StorageError, match="real AWS"):
        storage.get_s3_client()


def test_client_refused_for_amazonaws_endpoint(monkeypatch):
    monkeypatch.delenv("ALLOW_REAL_AWS", raising=False)
    with pytest.raises(StorageError):
        storage.get_s3_client("https://s3.us-east-1.amazonaws.com")


def test_auto_mode_without_endpoint_falls_back_instead_of_calling_aws(monkeypatch, tmp_path):
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("ALLOW_REAL_AWS", raising=False)
    assert fetch_raw_from_s3(tmp_path / "x.csv", mode="auto") is None


def test_localstack_endpoint_builds_a_client(monkeypatch):
    pytest.importorskip("boto3")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://localhost:4566")
    assert storage.get_s3_client().meta.endpoint_url == "http://localhost:4566"


# ---- optional integration test against a running LocalStack --------------------
def _localstack_up():
    try:
        urllib.request.urlopen("http://localhost:4566/_localstack/health", timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _localstack_up(), reason="LocalStack not running on :4566")
def test_roundtrip_against_localstack(tmp_path, monkeypatch):
    pytest.importorskip("boto3")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://localhost:4566")
    client = storage.get_s3_client()
    src = tmp_path / "in.csv"
    src.write_text("item_id,sales\nA,1\n")
    info = storage.upload_raw_file(client, src, bucket="retailpulse-test-roundtrip", key="raw/t.csv")
    assert info["bytes"] == src.stat().st_size
    out = fetch_raw_from_s3(tmp_path / "out.csv", bucket="retailpulse-test-roundtrip", key="raw/t.csv",
                            mode="s3", client=client)
    assert open(out).read() == src.read_text()
