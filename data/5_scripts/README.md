# Scripts & Runbooks

Este directorio agrupa scripts auxiliares y “runbooks” ejecutables para el caso de uso de Variant Calling.

## Convenciones

- Todos los comandos se ejecutan **desde el root del repo**:
  ```bash
  cd ~/VariantCalling
  ```

- Los outputs (VCF, logs, métricas, etc.) se escriben en `data/4_out/` (no aquí).

- Este README asume que ya existen los recortes a chr20 (BAM + REF chr20 + TRUTH chr20).

> Se verifica con mini-checks, pero no se explica cómo generarlos (está documentado en los README de `data/1_input_bams`,`data/2_refrence` y `data/3_truth` del repo).

---

# Parte 1: DeepVariant pipeline completo + validación con hap.py

Objetivo: fijar el baseline “canónico” end-to-end para HG003 chr20:

**BAM (chr20) + REF (chr20) → DeepVariant 1.9 (pipeline completo) → VCF/gVCF → hap.py → métricas**

> NOTA: en este repositorio, DV se ejecuta con Docker (imagen oficial).
hap.py se recomienda instalar en local vía conda/mamba (Bioconda), por reproducibilidad y simplicidad.

---

## 1.1 Inputs esperados (paths del repo)

> NOTA: Este ejemplo toma de referencia el **BAM de Illumina**, para el de ONT el proceso es muy similar pero cambiando directorios. 

- BAM chr20:
    - `data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam`
    - `data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam.bai`

- Referencia chr20:
    - `data/2_reference/GRCh38_no_alt_plus_hs38d1/GRCh38_no_alt_plus_hs38d1.chr20.fa`
    - `data/2_reference/GRCh38_no_alt_plus_hs38d1/GRCh38_no_alt_plus_hs38d1.chr20.fa.fai`

- Truth chr20:
    - `data/3_truth/HG003/HG003_chr20.truth.vcf.gz`
    - `data/3_truth/HG003/HG003_chr20.truth.vcf.gz.tbi`
    - `data/3_truth/HG003/HG003_chr20.confident.bed`

---

## 1.2 Mini-checks rápidos (antes de ejecutar)

Ejecutar desde el root:

```bash
cd ~/VariantCalling
```

### 1) Existencia de ficheros e índices

```bash
ls -lh \
  data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam \
  data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam.bai \
  data/2_reference/GRCh38_no_alt_plus_hs38d1/GRCh38_no_alt_plus_hs38d1.chr20.fa \
  data/2_reference/GRCh38_no_alt_plus_hs38d1/GRCh38_no_alt_plus_hs38d1.chr20.fa.fai \
  data/3_truth/HG003/HG003_chr20.truth.vcf.gz \
  data/3_truth/HG003/HG003_chr20.truth.vcf.gz.tbi \
  data/3_truth/HG003/HG003_chr20.confident.bed
```

### 2) Contigs coherentes (`chr20` vs `20`)
Queremos que BAM/REF/TRUTH usen el mismo naming (idealmente `chr20`).

```bash
# BAM header: ¿aparece chr20?
samtools view -H data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam | grep '^@SQ' | head

# REF: primer contig (debería ser >chr20)
grep -m1 '^>' data/2_reference/GRCh38_no_alt_plus_hs38d1/GRCh38_no_alt_plus_hs38d1.chr20.fa

# TRUTH: contigs presentes (debería listar chr20)
tabix -l data/3_truth/HG003/HG003_chr20.truth.vcf.gz
```

Si aquí hay mismatch (`20` vs `chr20`), hap.py y/o DV pueden fallar o dar métricas incoherentes.

---

## 1.3 DeepVariant 1.9: pipeline completo (Docker)

Outputs esperados:
- `data/4_out/deepvariant/HG003/Illumina/chr20/HG003.dv1.9.0.chr20.vcf.gz` (+ `.tbi`)
- `data/4_out/deepvariant/HG003/Illumina/chr20/HG003.dv1.9.0.chr20.g.vcf.gz` (+ `.tbi`)
- logs: `data/4_out/deepvariant/HG003/Illumina/chr20/logs/`

### Ejecutar

Primero, creamos las rutas y referencias correspondientes para nuestra run: 

```bash
cd ~/VariantCalling

DV_VERSION="1.9.0"
SAMPLE="HG003"
TECH="Illumina"
NUM_SHARDS="$(nproc)"

BAM="data/1_input_bams/${SAMPLE}/${TECH}/${SAMPLE}.GRCh38.2x250.chr20.bam"
REF="data/2_reference/GRCh38_no_alt_plus_hs38d1/GRCh38_no_alt_plus_hs38d1.chr20.fa"

OUT_DIR="data/4_out/deepvariant/${SAMPLE}/${TECH}/chr20"
LOG_DIR="${OUT_DIR}/logs"
INT_DIR="${OUT_DIR}/intermediate"
mkdir -p "${LOG_DIR}" "${INT_DIR}"

OUT_VCF="${OUT_DIR}/${SAMPLE}.${TECH}.dv${DV_VERSION}.chr20.vcf.gz"
OUT_GVCF="${OUT_DIR}/${SAMPLE}.${TECH}.dv${DV_VERSION}.chr20.g.vcf.gz"
```

Seguido, lanzamos el pipeline:

```bash
docker pull google/deepvariant:${DV_VERSION}

docker run --rm \
  -v "$PWD":/work \
  google/deepvariant:${DV_VERSION} \
  /opt/deepvariant/bin/run_deepvariant \
  --model_type=WGS \
  --ref="/work/${REF}" \
  --reads="/work/${BAM}" \
  --output_vcf="/work/${OUT_VCF}" \
  --output_gvcf="/work/${OUT_GVCF}" \
  --num_shards=${NUM_SHARDS} \
  --intermediate_results_dir="/work/${INT_DIR}" \
  --logging_dir="/work/${LOG_DIR}" \
  --vcf_stats_report=true
```

> **IMPORTANTE**: En el caso de usar el **BAM de ONT** el `--model_type` deberá ser asignado como `ONT_R104`.

> **IMPORTANTE**: Por defecto, el número de shards (`NUM_SHARDS`) es igual a la cantidad de cores (máximo rendimiento). Si esto da problemas del tipo: `parallel: This job failed`, entonces se recomienda reducir el valor de los shards a 2 o 4 para asegurar estabilidad. 

### Monitorizar mientras corre

```bash
tail -n 80 -f data/4_out/deepvariant/HG003/Illumina/chr20/logs/make_examples.log
```

---

## 1.4 Validación con hap.py (local, con conda/mamba)

### Instalar hap.py (una vez)

Recomendado con Bioconda:

```bash
# con mamba
mamba create -n happy -c conda-forge -c bioconda hap.py
mamba activate happy

# (alternativa con conda)
# conda create -n happy -c conda-forge -c bioconda hap.py
# conda activate happy

hap.py --help
```

### Ejecutar hap.py contra el VCF de DV

Outputs esperados (prefijo ejemplo):
- `data/4_out/happy/HG003/Illumina/chr20/HG003.Illumina/dv1.9.0.chr20.summary.csv` (principal)

```bash
cd ~/VariantCalling
conda activate happy

DV_VERSION="1.9.0"
SAMPLE="HG003"
TECHNOLOGY="Illumina"

REF="data/2_reference/GRCh38_no_alt_plus_hs38d1/GRCh38_no_alt_plus_hs38d1.chr20.fa"
TRUTH_VCF="data/3_truth/${SAMPLE}/${SAMPLE}_chr20.truth.vcf.gz"
TRUTH_BED="data/3_truth/${SAMPLE}/${SAMPLE}_chr20.confident.bed"

QUERY_VCF="data/4_out/deepvariant/${SAMPLE}/${TECHNOLOGY}/chr20/${SAMPLE}.${TECHNOLOGY}.dv${DV_VERSION}.chr20.vcf.gz"

OUT_DIR="data/4_out/happy/${SAMPLE}/${TECHNOLOGY}/chr20"
OUT_PREFIX="${OUT_DIR}/${SAMPLE}.${TECHNOLOGY}.dv${DV_VERSION}.chr20"
mkdir -p "${OUT_DIR}"

hap.py \
  "${TRUTH_VCF}" \
  "${QUERY_VCF}" \
  -f "${TRUTH_BED}" \
  -r "${REF}" \
  -o "${OUT_PREFIX}" \
  --threads "$(nproc)"
```

### Ver resultados

```bash
ls -lh data/4_out/happy/HG003/Illumina/chr20 | head
head -n 30 data/4_out/happy/HG003/Illumina/chr20/HG003.Illumina.dv1.9.0.chr20.summary.csv
```

---

# Parte 2:  DeepVariant “make_examples only” (placeholder)

TODO: Documentar ejecución de DeepVariant para generar únicamente `make_examples` (sin `call_variants` ni `postprocess`).

Incluirá:
- comando(s) y outputs esperados (tfrecords/examples)
- cómo guardar en data/4_out/deepvariant/<SAMPLE>/chr20/examples/
- cómo enlaza con scripts de este directorio (p. ej. export_emb_logits.py)

---
