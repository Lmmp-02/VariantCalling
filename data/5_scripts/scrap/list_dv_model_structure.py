"""
list_dv_model_structure.py

Objetivo
--------
Listar de forma estructurada un modelo de DeepVariant exportado como TF SavedModel
(ej: /opt/models/hybrid_pacbio_illumina) e identificar CLARAMENTE:

- INPUT tensor
- OUTPUT "serving" (normalmente probs)
- PENULTIMATE: embedding (N,2048)
- FINAL: logits (pre-softmax) (N,3)
- FINAL: softmax (N,3)

Ejecución esperada (dentro del Docker de DeepVariant)
-----------------------------------------------------
docker run --rm -v "$HOME/dv_poc":/data google/deepvariant:1.9.0 \
  python3 -u /data/scripts/list_dv_model_structure.py \
    --model_dir /opt/models/hybrid_pacbio_illumina --show_ops
"""

import argparse
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--show_ops", action="store_true")
    args = ap.parse_args()

    print("TF:", tf.__version__, "Eager:", tf.executing_eagerly())

    loaded = tf.saved_model.load(args.model_dir)
    fn = loaded.signatures["serving_default"]
    frozen = convert_variables_to_constants_v2(fn)
    g = frozen.graph

    in_t = frozen.inputs[0]
    out_t = frozen.outputs[0]

    print("\n== Signature ==")
    print("Input :", in_t.name, in_t.shape, in_t.dtype)
    print("Output:", out_t.name, out_t.shape, out_t.dtype, "op=", out_t.op.type)

    # 1) Encontrar MatMul que produzca (N,3) en la cabeza de clasificación
    matmuls = []
    for op in g.get_operations():
        if op.type != "MatMul":
            continue
        for o in op.outputs:
            shp = o.shape.as_list()
            if shp is not None and len(shp) == 2 and shp[1] == 3:
                matmuls.append(o)

    chosen_mm = None
    for t in matmuls:
        if "classification" in t.name:
            chosen_mm = t
            break
    if chosen_mm is None and matmuls:
        chosen_mm = matmuls[0]
    if chosen_mm is None:
        print("\n[WARN] No encontré MatMul (N,3). Modelo distinto?")
        return

    mm_op = chosen_mm.op
    mm_in_a = mm_op.inputs[0]  # típicamente embedding (N,2048)
    mm_in_b = mm_op.inputs[1]  # kernel

    # 2) Logits: BiasAdd consumiendo el MatMul
    logits_t = None
    softmax_t = None

    for c in chosen_mm.consumers():
        if c.type == "BiasAdd":
            logits_t = c.outputs[0]
            for cc in logits_t.consumers():
                if cc.type == "Softmax":
                    softmax_t = cc.outputs[0]
            break

    # Fallback: si Softmax consume MatMul directo (raro)
    if logits_t is None:
        for c in chosen_mm.consumers():
            if c.type == "Softmax":
                softmax_t = c.outputs[0]
                logits_t = chosen_mm
                break

    # 3) Penúltima: buscamos (N,2048). Normalmente mm_in_a.
    embedding_t = None
    if mm_in_a.shape.rank == 2 and mm_in_a.shape.as_list()[1] == 2048:
        embedding_t = mm_in_a
    else:
        for op in g.get_operations():
            for o in op.outputs:
                shp = o.shape.as_list()
                if shp is not None and len(shp) == 2 and shp[1] == 2048:
                    embedding_t = o
                    break
            if embedding_t is not None:
                break

    print("\n== Identificación de capas ==")
    print("Chosen MatMul (N,3):", chosen_mm.name, chosen_mm.shape)
    print("MatMul input A:", mm_in_a.name, mm_in_a.shape)
    print("MatMul input B:", mm_in_b.name, mm_in_b.shape)

    print("\n---- RESULTADO CLARO ----")
    if embedding_t is not None:
        print("PENULTIMATE (embedding):", embedding_t.name, embedding_t.shape, "op=", embedding_t.op.type)
    else:
        print("PENULTIMATE (embedding): [NO ENCONTRADO]")

    if logits_t is not None:
        print("FINAL (logits):        ", logits_t.name, logits_t.shape, "op=", logits_t.op.type)
    else:
        print("FINAL (logits):        [NO ENCONTRADO]")

    if softmax_t is not None:
        print("FINAL (softmax):       ", softmax_t.name, softmax_t.shape, "op=", softmax_t.op.type)
    else:
        print("FINAL (softmax):       [NO ENCONTRADO]")

    print("Serving output (prob): ", out_t.name, out_t.shape, "op=", out_t.op.type)

    # Opcional: listado compacto
    if args.show_ops:
        print("\n== Ops compactas (Conv2D/MatMul/BiasAdd/Softmax/Pool) ==")
        keep = {"Conv2D", "MaxPool", "AvgPool", "Relu", "MatMul", "BiasAdd", "Softmax", "Mean"}
        for op in g.get_operations():
            if op.type not in keep:
                continue
            out = op.outputs[0] if op.outputs else None
            shp = out.shape.as_list() if out is not None else None
            print(f"{op.name:70s} | {op.type:8s} | {shp}")

if __name__ == "__main__":
    main()