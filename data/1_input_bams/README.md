# BAMs (Aligned reads)

Un **BAM** (Binary Alignment/Map) es un archivo binario comprimido que contiene **lecturas de secuenciación (reads)** alineadas contra un **genoma de referencia**. Es la versión binaria de un SAM, e incluye (entre otras cosas):

- Posición alineada, CIGAR, flags
- Base qualities / mapping qualities
- Información de read groups (RG), sample, plataforma, etc.
- Opcionalmente tags auxiliares (p. ej. NM, MD…)

En **Variant Calling**, el BAM (junto con su índice) es el input principal para herramientas como **DeepVariant**, porque representa la evidencia observada del individuo (o muestra) frente a la referencia.

## Requisitos prácticos para este repo

Para que un BAM sea usable en la mayoría de pipelines:
- Debe estar **ordenado por coordenadas** (coordinate-sorted).
- Debe tener un **índice**: normalmente `.bai` (o `.csi` en algunos casos).
- El **build** de referencia debe ser consistente con el resto de assets:
  - BAM + referencia FASTA + truth VCF/BED deben ser del **mismo build** (p. ej. GRCh38 con GRCh38).
- El naming de contigs debe ser consistente:
-   chr20 vs 20 (y lo mismo para chr1, etc.).


## Fuente de datos: GIAB (Genome in a Bottle)

Usamos datasets públicos de **GIAB**, un consorcio liderado por NIST que publica muestras “benchmark” (p. ej. trios familiares) para validar pipelines y modelos.

Caso base del repo:
- Trio Ashkenazim
- **HG003 (NA24149, father)**
- Illumina **2x250** / Oxford Nanopore **(ONT)**
- Build **GRCh38**

Los alineamientos de HG003 están indexados públicamente aquí (BAM + BAI + MD5): 

https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG003_NA24149_father/

---

## Instalación de herramientas (WSL / Ubuntu)

Recomendación: un entorno con `samtools` (y opcionalmente `bcftools`, `tabix`, `mosdepth`).

Con conda/miniforge:

```bash
conda create -n hts -c conda-forge -c bioconda -y \
  samtools htslib bcftools tabix bedtools mosdepth
conda activate hts
```
---

## Estructura recomendada en este repo

Ejemplo (mínimo) para HG003:

```
data/1_input_bams/HG003/
  Illumina/
  ONT/
```

> Nota: para ONT es muy recomendable que el nombre del fichero incluya (**basecalling preset + run/flowcell**) para trazabilidad.

---

# 1) Illumina (NCBI FTP)

> Para las próximas secciones usamos como ejemplo el **BAM de Illumina** desde NCBI FTP.

> Para ONT el proceso es idéntico conceptualmente, cambiando la fuente/URL.

## Descargar el BAM completo (GRCh38, HG003)

> Ojo: el BAM completo puede ser muy grande (decenas o >100 GB). Usa `-c` para reanudar.

```bash
mkdir -p data/1_input_bams/HG003/Illumina

BASE="https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG003_NA24149_father/NIST_Illumina_2x250bps/novoalign_bams"
BAM_URL="$BASE/HG003.GRCh38.2x250.bam"
BAI_URL="$BASE/HG003.GRCh38.2x250.bam.bai"

wget -c -O data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.bam     "$BAM_URL"
wget -c -O data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.bam.bai "$BAI_URL"
```

## Descargar SOLO `chr20` (sin bajar el BAM entero)

Esto funciona porque el BAM remoto tiene índice `.bai`, lo que permite pedir regiones y streaming.

```bash
mkdir -p data/1_input_bams/HG003/Illumina

BASE="https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG003_NA24149_father/NIST_Illumina_2x250bps/novoalign_bams"
BAM_URL="$BASE/HG003.GRCh38.2x250.bam"

samtools view -b -@ 8 "$BAM_URL" chr20 \
  | samtools sort -@ 8 -o data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam

samtools index -@ 8 data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam
```

### (Opcional) extraer un intervalo concreto dentro de chr20

```bash
samtools view -b -@ 8 "$BAM_URL" "chr20:1000000-2000000" \
  | samtools sort -@ 8 -o data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20_1-2Mb.bam

samtools index -@ 8 data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20_1-2Mb.bam
```

---

# 2) ONT (ONT Open Data, AWS S3)

Para ONT, usamos el dataset público **GIAB 2025.01** publicado por ONT en AWS S3 (ONT Open Data).
Aquí hay dos conceptos clave:

- `hac` vs `sup`:
  - `hac` *(high accuracy)*: más rápido, algo menos preciso.
  - `sup` *(super accurate)*: más preciso (recomendado para baseline de calidad).
- Varias ejecuciones por muestra **(flowcells/runs)**:
  - HG003 suele tener 2 runs (p. ej. `PAY87794`, `PAY87954`).
  - Puedes usar 1 run (más rápido) o mergear ambos (baseline más estable por cobertura).

## URL del BAM remoto (SUP, HG003, PAY87794)

```bash
SAMPLE="HG003"
BC="sup"
RUN="PAY87794"

BAM_URL="https://ont-open-data.s3.amazonaws.com/giab_2025.01/basecalling/${BC}/${SAMPLE}/${RUN}/calls.sorted.bam"
```

## Detectar el nombre del contig (`chr20` vs `20`)

```bash
samtools view -H "$BAM_URL" 2>/dev/null | grep -q 'SN:chr20' && REGION="chr20" || REGION="20"
echo "REGION=$REGION"
```

## Descargar SOLO `chr20` (sin bajar el BAM entero)

Requiere que el BAM remoto tenga índice accesible (en este caso: `calls.sorted.bam.bai` existe).

```bash
OUT_BAM="data/1_input_bams/${SAMPLE}/ONT/${SAMPLE}.GRCh38.ONT_R104_${BC}_${RUN}.chr20.bam"
mkdir -p "$(dirname "$OUT_BAM")"

samtools view -b -@ 8 "$BAM_URL" "$REGION" \
  | samtools sort -@ 8 -o "$OUT_BAM"

samtools index -@ 8 "$OUT_BAM"
```

### Check rápido (asegurar que es “chr20-only”)

```bash
samtools idxstats "$OUT_BAM" | awk '$3>0{print}'
# Esperado: solo una línea con chr20 y reads>0
```

### (Opcional) Merge de dos runs (PAY87794 + PAY87954)

Esto suele dar un baseline más estable (mejor cobertura).

1. Repite el recorte para el segundo run (`PAY87954`) generando `OUT_BAM_2`.
2. Merge:

```bash
OUT_BAM_1="data/1_input_bams/HG003/ONT/HG003.GRCh38.ONT_R104_sup_PAY87794.chr20.bam"
OUT_BAM_2="data/1_input_bams/HG003/ONT/HG003.GRCh38.ONT_R104_sup_PAY87954.chr20.bam"

MERGED="data/1_input_bams/HG003/ONT/HG003.GRCh38.ONT_R104_sup_PAY87794_PAY87954.chr20.bam"

samtools merge -@ 8 -O BAM "$MERGED" "$OUT_BAM_1" "$OUT_BAM_2"
samtools index -@ 8 "$MERGED"

samtools idxstats "$MERGED" | awk '$3>0{print}'
```

---

# Inspeccionar BAMs (checks rápidos)

## Header y contigs disponibles

```bash
samtools view -H data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam | head
samtools view -H data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam | grep '^@SQ' | head -n 30
```

## Estadísticas por contig (muy útil para verificar “solo chr20”)

```bash
samtools idxstats data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam | head -n 30
```

> Lo esperado: reads > 0 en chr20 y 0 en el resto.

## Conteo de reads en un contig

```bash
samtools view -c data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam chr20
samtools view -c data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam chr1
```

> El primer conteo debería devolver valores, el segundo debería ser 0

## Sanity checks (integridad / formato)

```bash
samtools quickcheck -v data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam
samtools flagstat data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam | head -n 30
```

---

# Datasets legacy (solo para pruebas, NO baseline)

## ONT ultra-long R9.4.1 (UCSC 2020) --> NO compatible con `ONT_R104`

En pruebas iniciales se usó un BAM ONT ultra-long antiguo (R9.4.1).
No debe usarse como baseline con DeepVariant `--model_type=ONT_R104`, ya que puede degradar mucho el rendimiento (especialmente INDELs).

Naming recomendado:
```bash
data/1_input_bams/HG003/ONT_legacy_r9_4_1_ultralong_ucsc2020/
  HG003.GRCh38.ONT_R9_4_1_UL.UCSC_20200508.chr20.bam
  HG003.GRCh38.ONT_R9_4_1_UL.UCSC_20200508.chr20.bam.bai
```
---

## To be added (pendiente)

- Medir cobertura con mosdepth (WGS vs recortes, y comparativa por región).
- Downsampling (p. ej. para simular distintas coberturas).
- Recortar regiones con BED (subconjuntos “difíciles” / “fáciles”).

---
