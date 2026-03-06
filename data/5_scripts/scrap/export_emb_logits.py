"""
export_emb_logits.py

Objetivo
--------
Dado:
- un SavedModel de DeepVariant (p.ej /opt/models/hybrid_pacbio_illumina)
- examples TFRecord GZIP generados con make_examples

exporta un NPZ con:
- embeddings      (N, 2048)  : salida penúltima capa
- teacher_logits  (N, 3)     : logits reales (pre-softmax)
- teacher_probs   (N, 3)     : softmax (lo que devuelve serving normalmente)
- label           (N,)       : ground truth de make_examples (0/1/2)
- locus           (N,)       : string locus

Ejecución esperada (dentro del Docker de DeepVariant)
-----------------------------------------------------
docker run --rm -v "$HOME/dv_poc":/data google/deepvariant:1.9.0 \
  python3 -u /data/scripts/export_emb_logits.py \
    --model_dir /opt/models/hybrid_pacbio_illumina \
    --tfrecord_glob "/data/examples/chr20.training_set.with_label.tfrecord-0000*-of-00004.gz" \
    --out_prefix /data/datasets/chr20_emb2048_logits \
    --max_records 20000 \
    --batch_size 64
"""

import argparse, glob
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# (DV 1.9.0 + hybrid_pacbio_illumina) — nombres validados
DEFAULT_EMB_NAME    = "StatefulPartitionedCall/inceptionv3/dropout/Identity:0"
DEFAULT_PROBS_NAME  = "Identity:0"
DEFAULT_LOGITS_NAME = "StatefulPartitionedCall/inceptionv3/classification/BiasAdd:0"

def parse_example(rec_bytes: bytes):
    """Parse TF.train.Example de DeepVariant -> (img float32, label int, locus str)."""
    ex = tf.train.Example.FromString(rec_bytes)
    f = ex.features.feature

    img_bytes = f["image/encoded"].bytes_list.value[0]
    shape = list(f["image/shape"].int64_list.value)  # e.g. [100,221,6]
    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(shape).astype(np.float32)

    # Normalización típica DV: uint8 -> float en [-1,1)
    img = img / 128.0 - 1.0

    y = int(f["label"].int64_list.value[0]) if "label" in f else -1
    locus = f["locus"].bytes_list.value[0].decode("utf-8") if "locus" in f else ""
    return img, y, locus

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tfrecord_glob", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--chunk_size", type=int, default=20000)
    ap.add_argument("--max_records", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=2000)

    ap.add_argument("--emb_tensor", default=DEFAULT_EMB_NAME)
    ap.add_argument("--probs_tensor", default=DEFAULT_PROBS_NAME)
    ap.add_argument("--logits_tensor", default=DEFAULT_LOGITS_NAME)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.tfrecord_glob))
    if not paths:
        raise SystemExit(f"No matches: {args.tfrecord_glob}")

    print("TF:", tf.__version__, "Eager:", tf.executing_eagerly())

    # 1) Cargar SavedModel y congelar
    loaded = tf.saved_model.load(args.model_dir)
    fn = loaded.signatures["serving_default"]
    frozen = convert_variables_to_constants_v2(fn)
    g = frozen.graph

    # 2) Tensores de interés
    in_t = frozen.inputs[0]  # input_1:0
    emb_t = g.get_tensor_by_name(args.emb_tensor)
    logits_t = g.get_tensor_by_name(args.logits_tensor)
    probs_t = g.get_tensor_by_name(args.probs_tensor)

    print("Input         :", in_t.name, in_t.shape)
    print("Embedding     :", emb_t.name, emb_t.shape)
    print("Teacher logits:", logits_t.name, logits_t.shape, "op=", logits_t.op.type)
    print("Teacher probs :", probs_t.name, probs_t.shape, "op=", probs_t.op.type)

    # 3) Prune: input -> [embeddings, logits, probs]
    pruned = frozen.prune(in_t, [emb_t, logits_t, probs_t])

    # 4) Dataset TFRecord (GZIP) en eager
    ds = tf.data.TFRecordDataset(paths, compression_type="GZIP")

    X, Lg, Pb, Y, Loc = [], [], [], [], []
    bx, by, bloc = [], [], []

    chunk = 0
    total = 0

    def flush():
        nonlocal chunk, X, Lg, Pb, Y, Loc
        if not X:
            return
        Xnp   = np.stack(X).astype(np.float32)
        Lgnp  = np.stack(Lg).astype(np.float32)
        Pbnp  = np.stack(Pb).astype(np.float32)
        Ynp   = np.asarray(Y, dtype=np.int64)
        Locnp = np.asarray(Loc, dtype=np.str_)
        out = f"{args.out_prefix}_{chunk:04d}.npz"
        np.savez_compressed(
            out,
            embeddings=Xnp,
            teacher_logits=Lgnp,
            teacher_probs=Pbnp,
            label=Ynp,
            locus=Locnp,
        )
        print("Saved", out, "emb", Xnp.shape, "logits", Lgnp.shape, "probs", Pbnp.shape)
        chunk += 1
        X, Lg, Pb, Y, Loc = [], [], [], [], []

    for rec in ds:
        img, y, locus = parse_example(rec.numpy())
        bx.append(img); by.append(y); bloc.append(locus)

        if len(bx) >= args.batch_size:
            batch = tf.convert_to_tensor(np.stack(bx).astype(np.float32))
            emb, logits, probs = pruned(batch)
            emb = emb.numpy(); logits = logits.numpy(); probs = probs.numpy()

            if total == 0:
                print("Sanity mean(sum(probs)) =", float(probs.sum(axis=1).mean()))
                e = np.max(np.abs(tf.nn.softmax(logits, axis=1).numpy() - probs))
                print("Sanity max|softmax(logits)-probs| =", float(e))

            for i in range(emb.shape[0]):
                X.append(emb[i]); Lg.append(logits[i]); Pb.append(probs[i])
                Y.append(by[i]); Loc.append(bloc[i])

            total += emb.shape[0]
            if total % args.log_every == 0:
                print("Processed:", total)

            bx, by, bloc = [], [], []
            if len(X) >= args.chunk_size:
                flush()

        if args.max_records is not None and total >= args.max_records:
            break

    # último batch parcial
    if bx and (args.max_records is None or total < args.max_records):
        batch = tf.convert_to_tensor(np.stack(bx).astype(np.float32))
        emb, logits, probs = pruned(batch)
        emb = emb.numpy(); logits = logits.numpy(); probs = probs.numpy()
        for i in range(emb.shape[0]):
            X.append(emb[i]); Lg.append(logits[i]); Pb.append(probs[i])
            Y.append(by[i]); Loc.append(bloc[i])
        total += emb.shape[0]

    flush()
    print("Done. Total:", total)

if __name__ == "__main__":
    main()
