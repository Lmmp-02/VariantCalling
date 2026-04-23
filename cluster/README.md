# Variant Calling — runners de cluster

Este documento resume cómo lanzar los tres runners canónicos del repo y cuál es el flujo esperado.

## Flujo canónico

1. **`cluster/run_dv_make_examples.sh`**  
   Parte de un **BAM** y genera los **TFRecords de DeepVariant**.

2. **`cluster/run_dv_export_features.sh`**  
   Parte del **directorio de examples** generado por DeepVariant y exporta **datasets unimodales** (`.npz`) con embeddings, logits y metadatos.

3. **`cluster/run_dv_build_join_datasets.sh`**  
   Parte de los **datasets unimodales** de Illumina y ONT y construye el **dataset multimodal** unido por locus.

---

## 1) `run_dv_make_examples.sh`

### Qué consume
- `TECH=illumina|ont`
- `BAM=/ruta/al.bam`
- `SAMPLE=HG002|HG003|HG004|HG005|...`
- `REGIONS_ID=...` o `DATASET_ID=...`
- opcionalmente `MODE=training|calling` (default: `training`)

### Qué crea
Directorio canónico:

`data/4_out/deepvariant/examples/<dataset_id>/<dv_preset>/<mode>/`

### Lanzamiento típico
```bash
sbatch \
  --export=ALL,TECH=illumina,BAM=/ruta/al.bam,SAMPLE=HG005,MODE=calling,DATASET_ID=hg005_chr20_chr21_harmonized_40x \
  cluster/run_dv_make_examples.sh
```

---

## 2) `run_dv_export_features.sh`

### Qué consume
- `DATASET_ID=...`
- `TECH=illumina|ont`
- opcionalmente `MODE=training|calling` (default del runner: `training`)
- opcionalmente `EXAMPLES_DIR=...`

### Regla importante
`EXAMPLES_DIR` debe apuntar al **directorio** que contiene los TFRecords, no al shard `.tfrecord.gz` individual.

Correcto:

`data/4_out/deepvariant/examples/hg005_chr20_chr21_harmonized_40x/illumina_wgs/calling`

Incorrecto:

`data/4_out/deepvariant/examples/hg005_chr20_chr21_harmonized_40x/illumina_wgs/calling/make_examples.tfrecord-00000-of-00001.gz`

### Qué crea
Directorio canónico:

`data/4_out/datasets/unimodal/<dataset_id>/<dv_preset>/<mode>/`

### Lanzamiento típico
```bash
sbatch \
  --export=ALL,DATASET_ID=hg005_chr20_chr21_harmonized_40x,TECH=illumina,MODE=calling \
  cluster/run_dv_export_features.sh
```

### Lanzamiento explícito con `EXAMPLES_DIR`
```bash
sbatch \
  --export=ALL,DATASET_ID=hg005_chr20_chr21_harmonized_40x,TECH=illumina,MODE=calling,EXAMPLES_DIR=data/4_out/deepvariant/examples/hg005_chr20_chr21_harmonized_40x/illumina_wgs/calling \
  cluster/run_dv_export_features.sh
```

---

## 3) `run_dv_build_join_datasets.sh`

### Qué consume
- `DATASET_ID=...`
- opcionalmente `MODE=training|calling` (default: `training`)
- opcionalmente `UNIMODAL_ROOT=...`
- opcionalmente `OUT_ROOT=...`
- opcionalmente:
  - `BUILD_INNER=0|1`
  - `BUILD_OUTER=0|1`
  - `CHUNK_SIZE=40000`
  - `DROP_AMBIGUOUS_BIMODAL=0|1`
  - `INCLUDE_TEACHER=0|1`
  - `WRITE_META_CSV=0|1`
  - `FORCE=0|1`

### Qué crea
Directorio canónico:

`data/4_out/datasets/multimodal/by_subject/<dataset_id>/`

con salidas en:
- `inner/`
- `outer/`
- `logs/`

### Lanzamiento típico
```bash
sbatch \
  --export=ALL,DATASET_ID=hg005_chr20_chr21_harmonized_40x,MODE=calling \
  cluster/run_dv_build_join_datasets.sh
```

### Lanzamiento con opciones explícitas
```bash
sbatch \
  --export=ALL,DATASET_ID=hg005_chr20_chr21_harmonized_40x,MODE=calling,BUILD_INNER=0,BUILD_OUTER=1,DROP_AMBIGUOUS_BIMODAL=1,INCLUDE_TEACHER=1 \
  cluster/run_dv_build_join_datasets.sh
```

---

## Resumen rápido de contratos

- **make_examples**: BAM → TFRecords DeepVariant  
- **export_features**: examples DeepVariant → unimodal `.npz`  
- **build_join**: unimodales Illumina + ONT → multimodal unido por locus
