import glob
import tensorflow as tf

PATTERN = "data/4_out/deepvariant/HG003/Illumina/chr20/cov48x_sorig/intermediate/make_examples.tfrecord-*-of-00014.gz"

paths = sorted(glob.glob(PATTERN))
print("pattern:", PATTERN)
print("matches:", len(paths))
if not paths:
    raise SystemExit("No TFRecords matched. Check path / run with -w /data.")

p = paths[0]
print("TFRecord:", p)

ds = tf.data.TFRecordDataset([p], compression_type="GZIP").take(1)
for rec in ds:
    ex = tf.train.Example.FromString(rec.numpy())
    keys = sorted(ex.features.feature.keys())
    print("Has image/encoded:", "image/encoded" in keys)
    print("Has locus:", "locus" in keys)
    print("Has variant/encoded:", "variant/encoded" in keys)
    print("Has alt_allele_indices:", "alt_allele_indices" in keys)
    if "locus" in keys:
        print("locus example:", ex.features.feature["locus"].bytes_list.value[0].decode("utf-8"))
    print("num_keys:", len(keys))
    print("first_30_keys:", keys[:30])