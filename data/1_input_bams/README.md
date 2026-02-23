# Índice
- [BAMs (Aligned reads)](#bams-a reads))
- [1) Illumina (NCBI FTP)](#1-illumina-ncbi-ftp)
- [2) ONT (ONT Open Data, AWS S3)](#2-ont-ont-open-data-aws-s3)
- [Inspeccionar BAMs (checks rápidos)](#inspeccionar-bams-checks-rápidos)
- [Cobertura (coverage) y mosdepth](#cobertura-coverage-y-mosdepth)
- [Downsampling de BAMs (simular menor cobertura)](#downsampling-de-bams-simular-menor-cobertura)

---

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

# Cobertura (coverage) y mosdepth

## ¿Qué es la cobertura en un BAM?

La **cobertura** (coverage / depth) mide **cuántas lecturas cubren cada posición** del genoma.
- **Depth en un punto**: nº de reads que cubren esa base.
- **Cobertura media (mean coverage)**: promedio del depth a lo largo de una región (en nuestro caso, `chr20`).

Esto es útil porque:
- nos da contexto para interpretar métricas (p. ej. comparar Illumina vs ONT con coberturas distintas),
- nos permite detectar problemas (zonas con coverage muy bajo o muy irregular),
- nos deja una “foto” rápida del dataset que estamos usando.

Saber el coverage también nos permitirá generar versiones "con menos cobertura" (downsampling) de los BAMs. Estas versiones nos permitirán evaluar la **degradación que sufre Deep Variant** al disponer de **menos informació**n con la que **generar ejemplos** y **clasificar variantes**. 

### ¿Por qué mosdepth?

**mosdepth** es una herramienta que calcula cobertura de forma muy rápida a partir de un BAM indexado, generando:
- `*.mosdepth.summary.txt` → coverage medio por contig (y otros stats básicos).
- `*.regions.bed.gz` (si usas `--by`) → coverage por ventanas (ideal para plot).

## Cobertura media en chr20 (mosdepth summary)

### Inputs (BAMs chr20-only)

Ejemplo de paths (ajustar si difieren):

```bash
ILL_BAM="data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam"

# ONT parcial: 1 run (ej: PAY87794)
ONT_PARTIAL_BAM="data/1_input_bams/HG003/ONT/HG003.GRCh38.ONT_R104_sup_PAY87794.chr20.bam"

# ONT completo: merge PAY87794+PAY87954
ONT_FULL_BAM="data/1_input_bams/HG003/ONT/HG003.GRCh38.ONT_R104_sup_PAY87794_PAY87954.chr20.bam"
```

### Ejecutar mosdepth (solo summary)
> NOTA: No necesitamos explicitar un BEDs para el **mean coverage**: mosdepth ya calcula el summary del BAM completo (en este caso, como el BAM es `chr20-only`, el summary de chr20 es lo que nos interesa).

```bash
conda activate hts
cd ~/VariantCalling

THREADS=4
OUT_BASE="data/4_out/mosdepth/HG003/chr20"

mkdir -p "$OUT_BASE/Illumina" "$OUT_BASE/ONT_PARTIAL" "$OUT_BASE/ONT_FULL"

# Illumina
mosdepth -t "$THREADS" -n \
  "$OUT_BASE/Illumina/HG003.Illumina.chr20" \
  "$ILL_BAM"

# ONT (parcial)
mosdepth -t "$THREADS" -n \
  "$OUT_BASE/ONT_PARTIAL/HG003.ONT.partial.chr20" \
  "$ONT_PARTIAL_BAM"

# ONT (full/merged)
mosdepth -t "$THREADS" -n \
  "$OUT_BASE/ONT_FULL/HG003.ONT.full.chr20" \
  "$ONT_FULL_BAM"
```

Flags usados:

- `-n` → no genera per-base (más rápido y realiza menos opearciones en disco).

- `-t` → threads (principalmente descompresión; 4 suele ser suficiente).

### Leer el mean coverage de chr20

```bash
echo "Illumina chr20:"
grep -w '^chr20' "$OUT_BASE/Illumina/HG003.Illumina.chr20.mosdepth.summary.txt"

echo "ONT partial chr20:"
grep -w '^chr20' "$OUT_BASE/ONT_PARTIAL/HG003.ONT.partial.chr20.mosdepth.summary.txt"

echo "ONT full chr20:"
grep -w '^chr20' "$OUT_BASE/ONT_FULL/HG003.ONT.full.chr20.mosdepth.summary.txt"
```

En nuestro caso (ejemplo real de esta PoC):
- Illumina: **48.61×**
- ONT partial: **40.67×**
- ONT full: **74.35×**

## Cobertura a lo largo de chr20 (ventanas 10kb)

Para visualizar cómo cambia la cobertura a lo largo de chr20 (y detectar “dropouts”), generamos cobertura por **ventanas de 10kb**. Esto produce el fichero:  `HG003.<name>.chr20.win10000.regions.bed.gz` que luego usamos para plot.

### Generar ventanas 10kb (mosdepth --by 10000)

```bash
WIN=10000

# Illumina
mosdepth -t "$THREADS" -n --by "$WIN" \
  "$OUT_BASE/Illumina/HG003.Illumina.chr20.win${WIN}" \
  "$ILL_BAM"

# ONT partial
mosdepth -t "$THREADS" -n --by "$WIN" \
  "$OUT_BASE/ONT_PARTIAL/HG003.ONT.partial.chr20.win${WIN}" \
  "$ONT_PARTIAL_BAM"

# ONT full
mosdepth -t "$THREADS" -n --by "$WIN" \
  "$OUT_BASE/ONT_FULL/HG003.ONT.full.chr20.win${WIN}" \
  "$ONT_FULL_BAM"
```

## Plot de cobertura (script)

Tenemos un script para plotear la cobertura por ventanas 10kb:

- Script: `data/4_out/mosdepth/plot_coverage.py`

- Inputs esperados:
`data/4_out/mosdepth/HG003/chr20/<FOLDER>/HG003.<name>.chr20.win10000.regions.bed.gz`

- Output generado:
`data/4_out/mosdepth/HG003/chr20/<FOLDER>/coverage_<name>.png`

Ejemplos:

```bash
python3 data/4_out/mosdepth/plot_coverage.py --folder Illumina
python3 data/4_out/mosdepth/plot_coverage.py --folder ONT_PARTIAL
python3 data/4_out/mosdepth/plot_coverage.py --folder ONT_FULL
```

---

# Downsampling de BAMs (simular menor cobertura)

## Objetivo

Queremos medir **cuánto se degrada DeepVariant** cuando baja la cobertura del BAM.  
Para ello generamos versiones “recortadas” (**downsampled**) de los BAMs `chr20-only` manteniendo:

- un **set base** de coberturas: **30× / 15× / 10× / 5×** (seed base = `42`)
- **replicados opcionales** (seeds extra) para estimar variabilidad por muestreo: p. ej. `123` y `2026` en **10× y 5×**.

Este downsampling lo aplicamos tanto a:
- **Illumina chr20**
- **ONT partial chr20** (1 run)

> Nota importante: el **mean coverage** usado para calcular el downsampling se lee siempre desde el `*.mosdepth.summary.txt` (cálculo “correcto”, consistente con nuestros summaries actuales).  
> El script genera ese summary si no existe.

---

## Idea técnica (cómo funciona)

1) Medimos la cobertura original del BAM con **mosdepth** y leemos el valor `total` del `*.mosdepth.summary.txt`.

2) Para una cobertura objetivo `T` (por ejemplo `15×`), calculamos la fracción:

```math
\mathrm{FRAC} = \frac{\mathrm{TARGET}}{\mathrm{mean\_orig}}
```

3) Hacemos downsample con **samtools** usando: `samtools view -s SEED.FRAC`

donde `SEED` fija el muestreo (reproducible) y `FRAC` es la fracción de lecturas/templates.

5) Ordenamos, indexamos y validamos:  `samtools sort`, `samtools index` y `samtools quickcheck`

Volvemos a ejecutar **mosdepth** sobre el BAM downsampleado para verificar que el mean queda cerca del target.

---

## Script: `downsample.sh`

- **Ubicación:** `data/1_input_bams/downsample.sh`
- **Requisitos:** entorno con `samtools` + `mosdepth` (en este repo usamos conda env `hts`).

### Preparación

```bash
cd ~/VariantCalling
chmod +x data/1_input_bams/downsample.sh
```

### Ejecutar (set base con seed 42)

Genera para **Illumina** y **ONT_partial**:

- 30× / 15× / 10× / 5× con `seed=42`
- y recalcula/valida cobertura con mosdepth

```bash
./data/1_input_bams/downsample.sh
```

## Ejecutar con replicados (seeds extra)

Además de lo anterior, genera seeds extra (por defecto `123` y `2026`) para `10×` y `5×`:

```bash
RUN_EXTRA=1 ./data/1_input_bams/downsample.sh
```
---

## Outputs (convención de nombres)
### BAMs downsampleados

Se guardan aquí:
- `data/1_input_bams/HG003/downsampled/ILLUMINA_chr20/`
- `data/1_input_bams/HG003/downsampled/ONT_partial_chr20/`

Naming:
- `HG003.ILLUMINA.chr20.<TARGET>x.s<SEED>.bam`
- `HG003.ONT_partial.chr20.<TARGET>x.s<SEED>.bam`

Cada BAM se genera junto con su índice `.bai`.

### mosdepth (validación de coverage)

Se guardan aquí:
- `data/4_out/mosdepth/HG003/chr20/downsampled/ILLUMINA/`
- `data/4_out/mosdepth/HG003/chr20/downsampled/ONT_partial/`

Incluye:
- coverage original (si no existía): `ILLUMINA_orig.mosdepth.summary.txt` y `ONT_partial_orig.mosdepth.summary.txt`
- y para cada run downsampleado: `<TECH>_<TARGET>x_s<SEED>.mosdepth.summary.txt`

---

## To be added (pendiente)

- Recortar regiones con BED (subconjuntos “difíciles” / “fáciles”).

---
