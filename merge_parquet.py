import glob
import pyarrow as pa
import pyarrow.parquet as pq

files = sorted(glob.glob("data/raw_parquet/*.parquet"))
print(f"Found {len(files)} files")
tables = []
for f in files:
    t = pq.read_table(f)
    print(f"{f}: {t.num_rows:,} rows, {t.num_columns} cols")
    tables.append(t)

names = tables[0].schema.names
for f, t in zip(files, tables):
    if t.schema.names != names:
        raise SystemExit(f"Column mismatch in {f}")

full = pa.concat_tables(tables, promote_options="permissive")
pq.write_table(full, "data/cicids2017_full.parquet", compression="snappy")
print(f"Saved data/cicids2017_full.parquet with {full.num_rows:,} rows")
