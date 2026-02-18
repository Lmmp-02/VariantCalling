# Truth set (GIAB benchmark): VCF + BED

En benchmarking de Variant Calling, llamamos **truth set** al conjunto de “verdad” (ground truth) con el que comparamos las variantes llamadas por nuestro método.

En GIAB, el truth set para *small variants* (SNPs e indels) se publica como:

- **VCF (benchmark variants)**: las variantes consideradas “benchmark” para esa muestra.
- **BED (benchmark/confident regions)**: las regiones del genoma donde GIAB considera que la caracterización es suficientemente robusta para **evaluar** métodos.

> Nota terminológica: GIAB cambió el término de “high-confidence” a “benchmark” para dejar claro que su objetivo es **benchmarking**, no decir “aquí todo el mundo debería confiar en sus variantes”. 

## Por qué necesitamos VCF + BED (los dos)

El **VCF** contiene variantes, pero **no define por sí solo** dónde evaluar. El **BED** delimita las regiones en las que calculamos métricas (precision/recall/F1, etc.). Es estándar usarlos en conjunto (por ejemplo, con `hap.py`).

Además, GIAB incluye algunas variantes del VCF fuera del BED para evitar problemas con variantes complejas que cruzan bordes de intervalos.

## Nuestro caso base

- Dataset: **Ashkenazim Trio**
- Individuo: **HG003 (NA24149, father)**
- Release: **NIST v4.2.1**
- Build: **GRCh38**
- Tipo: small variants benchmark (VCF) + benchmark regions (BED)

Los archivos están publicados en el FTP/HTTPS de GIAB (NIST/NCBI).

---

## Descargar truth set completo (HG003, GRCh38, v4.2.1)

```bash
mkdir -p data/truth/HG003
cd data/truth/HG003

BASE="https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG003_NA24149_father/NISTv4.2.1/GRCh38"

curl -L -O "${BASE}/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
curl -L -O "${BASE}/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz.tbi"
curl -L -O "${BASE}/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed"
```

Archivos esperados:
- `*.vcf.gz` : truth variants
- `*.vcf.gz.tbi` : índice para consultas por región (imprescindible para recortes y accesos rápidos)
- `*.bed` : regiones benchmark/confident

---

## (Opcional) Quedarnos solo con chr20

Requisitos: bcftools + tabix.

### A) Recortar VCF local a chr20 (manteniendo indexado)
```bash
bcftools view --threads 8 -r chr20 \
  HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz -Oz \
  -o HG003_chr20.truth.vcf.gz

tabix -p vcf HG003_chr20.truth.vcf.gz
```

### B) Alternativa: recortar sin descargar el VCF completo

Si tienes acceso HTTP al VCF remoto + su .tbi, bcftools puede leerlo remoto y escribir solo chr20:

```bash
VCF_URL="${BASE}/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"

bcftools view --threads 8 -r chr20 "$VCF_URL" -Oz \
  -o HG003_chr20.truth.vcf.gz

tabix -p vcf HG003_chr20.truth.vcf.gz
```

## Recortar BED (confident/benchmark regions) a chr20
```bash
awk '$1=="chr20"' HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed \
  > HG003_chr20.confident.bed
```

---

## Sanity checks rápidos

### Datos del VCF (deberían salir chr20 si está recortado)
```bash
tabix -l HG003_chr20.truth.vcf.gz
```

### Número de variantes en el VCF
```bash
bcftools index -n HG003_chr20.truth.vcf.gz
```

### Número de intervalos en el BED recortado
```bash
wc -l HG003_chr20.confident.bed
```

---

## Notas prácticas

- Asegúrate de que el naming de contigs coincide con tus BAMs y referencia (chr20 vs 20).
- Mantén consistencia de build: GRCh38 con GRCh38 (BAM, FASTA, truth VCF/BED).

## To be added

- Subsets por intervalos específicos (BEDs “difíciles”, “fáciles”, etc.).


---
