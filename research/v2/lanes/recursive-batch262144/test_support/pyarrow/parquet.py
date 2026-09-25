"""Synthetic self-checks may not read real Parquet data."""

def ParquetFile(*args, **kwargs):
    raise RuntimeError('Batch synthetic check forbids real Parquet reads')
