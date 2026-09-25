"""Reject any accidental real-data access in this synthetic CPU test."""
def ParquetFile(*args, **kwargs):
    raise RuntimeError("L3 synthetic CPU test: real Parquet access is forbidden")
