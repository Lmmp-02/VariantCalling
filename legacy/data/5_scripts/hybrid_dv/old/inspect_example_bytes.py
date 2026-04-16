import glob
import binascii
import tensorflow as tf

PATTERN = "data/4_out/deepvariant/HG003/Illumina/chr20/cov48x_sorig/intermediate/make_examples.tfrecord-*-of-00014.gz"
paths = sorted(glob.glob(PATTERN))
print("pattern:", PATTERN)
print("matches:", len(paths))
assert paths, "No tfrecords matched"

p = paths[0]
print("Using:", p)

def hx(b, n=32):
    return binascii.hexlify(b[:n]).decode()

ds = tf.data.TFRecordDataset([p], compression_type="GZIP")

for idx, rec in enumerate(ds.take(5)):
    ex = tf.train.Example.FromString(rec.numpy())
    f = ex.features.feature

    locus = f["locus"].bytes_list.value[0].decode() if "locus" in f else ""
    vtype = int(f["variant_type"].int64_list.value[0]) if "variant_type" in f else None
    stype = int(f["sequencing_type"].int64_list.value[0]) if "sequencing_type" in f else None

    print("\n--- example", idx, "---")
    print("locus:", locus, "variant_type:", vtype, "sequencing_type:", stype)

    vb = f["variant/encoded"].bytes_list.value[0] if "variant/encoded" in f else b""
    ab = f["alt_allele_indices/encoded"].bytes_list.value[0] if "alt_allele_indices/encoded" in f else b""

    print("variant/encoded: len", len(vb), "hex[:16]", hx(vb, 16))
    print("alt_allele_indices/encoded: len", len(ab), "hex[:16]", hx(ab, 16))

    # prueba “parse_tensor” en modo no-ruidoso: capturamos excepción y seguimos
    for out_type, name in [(tf.int32, "int32"), (tf.int64, "int64"), (tf.string, "string")]:
        if not ab:
            continue
        try:
            t = tf.io.parse_tensor(ab, out_type=out_type).numpy()
            print(f"alt_idx parse_tensor as {name}:", t)
        except Exception as e:
            print(f"alt_idx parse_tensor as {name}: FAIL ({type(e).__name__})")

    for out_type, name in [(tf.string, "string")]:
        if not vb:
            continue
        try:
            t = tf.io.parse_tensor(vb, out_type=out_type).numpy()
            # si fuese tensor string, el contenido real estaría aquí
            inner = t if isinstance(t, (bytes, bytearray)) else (t.item() if getattr(t, "shape", None)==() else None)
            if isinstance(inner, (bytes, bytearray)):
                print("variant parse_tensor as string: OK inner_len", len(inner), "inner_hex[:16]", hx(inner, 16))
            else:
                print("variant parse_tensor as string: OK but unexpected shape/type")
        except Exception as e:
            print(f"variant parse_tensor as string: FAIL ({type(e).__name__})")