"""Fail closed on accidental Parquet access in synthetic CPU tests."""
def ParquetFile(*args, **kwargs):
    raise RuntimeError("FP8 synthetic CPU test forbids real-data reads")
