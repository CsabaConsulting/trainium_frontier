"""Fail closed if a synthetic self-check accidentally requests real data."""
def ParquetFile(*args, **kwargs):
    raise RuntimeError('Recursive stack synthetic check forbids real Parquet reads')
