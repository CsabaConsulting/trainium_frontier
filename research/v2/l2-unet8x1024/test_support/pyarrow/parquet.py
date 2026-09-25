"""Any accidental Parquet read fails closed in the synthetic model test."""

def ParquetFile(*args, **kwargs):
    raise RuntimeError('L2 synthetic check forbids real Parquet reads')
