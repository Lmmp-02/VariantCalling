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

## Fuente de datos: GIAB (Genome in a Bottle)

Usamos datasets públicos de **GIAB**, un consorcio liderado por NIST que publica muestras “benchmark” (p. ej. trios familiares) para validar pipelines y modelos.

En este proyecto, **nuestro caso base** es:
- Trio Ashkenazim
- **HG003 (NA24149, father)**
- Illumina **2x250**
- Build **GRCh38**

Los alineamientos “novoalign” de HG003 están indexados públicamente aquí (BAM + BAI + MD5):

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

## Descargar el BAM completo (GRCh38, HG003)

> Ojo: el BAM completo puede ser muy grande (decenas o >100 GB). Usa -c para reanudar.

```bash
mkdir -p data/1_input_bams/HG003

BASE="https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG003_NA24149_father/NIST_Illumina_2x250bps/novoalign_bams"
BAM_URL="$BASE/HG003.GRCh38.2x250.bam"
BAI_URL="$BASE/HG003.GRCh38.2x250.bam.bai"

wget -c -O data/1_input_bams/HG003/HG003.GRCh38.2x250.bam     "$BAM_URL"
wget -c -O data/1_input_bams/HG003/HG003.GRCh38.2x250.bam.bai "$BAI_URL"
```

---

## Descargar SOLO chr20 (sin bajar el BAM entero)

Esto funciona porque el BAM remoto tiene índice .bai, lo que permite pedir regiones y streaming.

### 1) Extraer chr20 desde el BAM remoto y crear BAM local ordenado + indexado

```bash
mkdir -p data/1_input_bams/HG003

BASE="https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG003_NA24149_father/NIST_Illumina_2x250bps/novoalign_bams"
BAM_URL="$BASE/HG003.GRCh38.2x250.bam"

samtools view -b -@ 8 "$BAM_URL" chr20 \
  | samtools sort -@ 8 -o data/1_input_bams/HG003/HG003.GRCh38.2x250.chr20.bam

samtools index -@ 8 data/1_input_bams/HG003/HG003.GRCh38.2x250.chr20.bam
```

> Importante: el nombre del contig debe existir tal cual en el BAM (chr20 vs 20). Verifícalo con el header (sección “Inspección”).

### 2) (Opcional) Extraer un intervalo concreto dentro de chr20

```bash
samtools view -b -@ 8 "$BAM_URL" "chr20:1000000-2000000" \
  | samtools sort -@ 8 -o data/1_input_bams/HG003/HG003.GRCh38.2x250.chr20_1-2Mb.bam

samtools index -@ 8 data/1_input_bams/HG003/HG003.GRCh38.2x250.chr20_1-2Mb.bam
```

---

## Inspeccionar BAMs (checks rápidos)

### Header y contigs disponibles

```bash
samtools view -H data/1_input_bams/HG003/HG003.GRCh38.2x250.chr20.bam | head
samtools view -H data/1_input_bams/HG003/HG003.GRCh38.2x250.chr20.bam | grep '^@SQ' | head -n 30
```

### Estadísticas por contig (muy útil para verificar “solo chr20”)

```bash
samtools idxstats data/1_input_bams/HG003/HG003.GRCh38.2x250.chr20.bam | head -n 30
```

> Lo esperado: reads > 0 en chr20 y 0 en el resto.

### Conteo de reads en un contig

```bash
samtools view -c data/1_input_bams/HG003/HG003.GRCh38.2x250.chr20.bam chr20
samtools view -c data/1_input_bams/HG003/HG003.GRCh38.2x250.chr20.bam chr1
```

> El primer conteo debería devolver valores, el segundo debería ser 0

### Sanity checks (integridad / formato)

```bash
samtools quickcheck -v data/1_input_bams/HG003/HG003.GRCh38.2x250.chr20.bam
samtools flagstat data/1_input_bams/HG003/HG003.GRCh38.2x250.chr20.bam | head -n 30
```

---

## To be added (pendiente)

- Medir cobertura con mosdepth (WGS vs recortes, y comparativa por región).
- Downsampling (p. ej. para simular distintas coberturas).
- Recortar regiones con BED (subconjuntos “difíciles” / “fáciles”).
