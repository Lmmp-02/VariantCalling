# Output directory

Este directorio reúne las salidas de las etapas de datos. `deepvariant/` contiene ejemplos TFRecord y llamadas completas VCF/gVCF; `datasets/unimodal/` contiene shards `.npz` separados por tecnología; `datasets/multimodal/by_subject/` contiene las uniones Illumina/ONT; `happy/`, `mosdepth/` y `plots/` contienen métricas y gráficos. Las rutas concretas dependen de `<dataset_id>`, tecnología y modo (`training`/`calling`).

Se conservan metadatos ligeros y resultados resumidos de ejecuciones previas. Los BAM, referencia, truth, TFRecords y shards `.npz` deben prepararse por separado: la presencia de `SUCCESS`, `export_manifest.json` o `join_manifest.json` **no** garantiza que el archivo binario asociado esté presente en esta copia. Al volver a ejecutar un paso, inspecciona las rutas de entrada y salida antes de usar `--force 1`, que puede borrar una salida previa.

Para una lectura correcta de `datasets/multimodal/by_subject/<dataset_id>/outer/`, consulta [las políticas de unión](../../docs/join_policies.md).
