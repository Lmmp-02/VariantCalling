# Reference genome (FASTA)

La **referencia** es un genoma “modelo” en formato **FASTA** (texto) contra el que se **alinean** las lecturas (BAM) y sobre el que se expresan las coordenadas de variantes (VCF).  
En práctica, para que un pipeline tipo DeepVariant funcione de forma reproducible, necesitamos:

- FASTA (p. ej. `.fna` / `.fa`)
- Índice FASTA: `*.fai` (generado con `samtools faidx`)
- (Opcional) Diccionario `*.dict` (algunos tools lo usan; DeepVariant normalmente no lo exige)

## Por qué esta referencia (GRCh38 “analysis set”)

Los BAMs de GIAB en GRCh38 suelen estar alineados contra una referencia del estilo **GRCh38 no_alt + hs38d1** (a veces verás pistas en headers / documentación), que:

- **omite ALT contigs** (alternate loci) para evitar incompatibilidades en muchos pipelines de alineamiento.
- incluye **secuencias decoy hs38d1** para “absorber” lecturas que mapean a regiones problemáticas/repetitivas, mejorando el comportamiento del alineamiento en la práctica.
- usa naming con prefijo **`chr`** (importante para que coincida con BAM/VCF/BED cuando están en UCSC-style). 
- se conoce como **“analysis set”** (un empaquetado de GRCh38 pensado para alineamiento y análisis).

En este proyecto usamos:

**`GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.gz`** 

> Regla de oro: BAM + referencia FASTA + truth VCF/BED deben ser del **mismo build** y con el **mismo naming** (p. ej. `chr20` vs `20`).

---

## Nota: Estructura de `ftp.ncbi.nlm.nih.gov/genomes/all/` 

El FTP de NCBI bajo `genomes/all/` es un repositorio masivo con **ensamblados (assemblies) de genomas** de miles de organismos (humanos, ratones, bacterias, plantas, etc.), incluyendo versiones actuales y versiones históricas. Cada ensamblado tiene su propio directorio con un “core set” de ficheros (secuencias, reportes del ensamblado, etc.) y, según el organismo, ficheros adicionales relevantes para pipelines.

### División `GCA` vs `GCF`
NCBI organiza los ensamblados en dos grandes ramas:
- `GCA/` → ensamblados de **GenBank**
- `GCF/` → ensamblados de **RefSeq**

Esta estructura se introdujo (y se migró en 2016) para estandarizar rutas y evitar dependencias de paths antiguos. 

### ¿Cómo funciona la estructura de subcarpetas?
NCBI evita meter millones de carpetas en un único directorio creando un árbol de 3 niveles a partir del **número del assembly accession**.

Ejemplo:
- Accession: `GCF_000001405.40`
- Parte numérica (9 dígitos): `000001405`
- Se separa en grupos de 3: `000/001/405`

Por eso el ensamblado humano GRCh38 (el que queremos) vive bajo:
`.../genomes/all/GCF/000/001/405/`

Dentro de ese directorio verás subcarpetas del tipo:
`GCF_000001405.40_GRCh38.p14/`, `GCF_000001405.39_GRCh38.p13/`, etc., que representan **distintas versiones** del mismo ensamblado.

### ¿Y el subdirectorio de “alignment pipelines”?
Para humanos (y algunos otros organismos muy usados), NCBI publica ficheros “preparados para pipelines de alineamiento” (los conocidos **analysis sets**), y UCSC también referencia explícitamente esta idea y su ubicación en NCBI. 

---

## Descargar la referencia (FASTA `.gz`), descomprimir y crear `.fai`

Estructura recomendada (para poder convivir con otras referencias en el futuro):

```bash
data/2_reference/
└── GRCh38_no_alt_plus_hs38d1/
```

### 1) Descargar

```bash
mkdir -p data/2_reference/GRCh38_no_alt_plus_hs38d1
cd data/2_reference/GRCh38_no_alt_plus_hs38d1

REF_GZ="GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.gz"

# URL (NCBI, GRCh38.p14 alignment pipelines dir)
REF_URL="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GRCh38_major_release_seqs_for_alignment_pipelines/${REF_GZ}"

curl -L -o "${REF_GZ}" "${REF_URL}"
```

(El concepto “analysis set” y su procedencia NCBI/UCSC están documentados públicamente.)

### 2) Descomprimir (manteniendo el .gz)

```bash
gunzip -k "${REF_GZ}"

```

Esto deja el FASTA como:

```bash
REF="GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna"
```

### 3) Indexar (crear .fai)

```bash
samtools faidx "${REF}"
```

Resultado esperado:
- ${REF} (FASTA)
- ${REF}.fai (índice)

---

## Sanity checks (contigs y chr20)

### 1) Ver contigs iniciales

```bash
cut -f1 "${REF}.fai" | head -n 30
```

### 2) Confirmar que existe chr20

```bash
grep -n $'^chr20\t' "${REF}.fai" || echo "No veo chr20 en el .fai"
```

### 3) Leer un trocito de secuencia (prueba definitiva)

```bash
samtools faidx "${REF}" chr20:10000000-10000020
```

---

## Check de compatibilidad con BAM/Truth (muy recomendado)

Antes de lanzar DeepVariant, verifica que el naming coincide:

### BAM 

```bash
samtools view -H data/1_input_bams/HG003/HG003.GRCh38.2x250.chr20.bam | grep '^@SQ' | head -n 20
```

### Reference

```bash
cut -f1 "${REF}.fai" | head -n 20
```

Si el BAM tiene chr20 y la referencia tiene 20 (o viceversa), algo no cuadra y te saldrán errores/VCFs incompatibles.

---

## (Opcional) Generar un FASTA “solo chr20” para pruebas rápidas

Útil para debugging y prototipos regionales; para pipelines generales lo normal es usar la referencia completa.

```bash
samtools faidx "${REF}" chr20 > GRCh38_no_alt_plus_hs38d1.chr20.fa
samtools faidx GRCh38_no_alt_plus_hs38d1.chr20.fa
```
