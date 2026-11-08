# Workload documents

Put only the documentation and non-secret configuration that Lumin should use
under this directory. The container mounts it read-only and indexes supported
`.md`, `.txt`, `.rst`, `.adoc`, `.yaml`, `.yml`, `.json`, `.toml`, `.ini`,
`.cfg`, `.conf`, and `.properties` files locally at UI startup.

For predictable resource use, indexing is capped at 500 files, 1 MB per file,
and 10 MB total. Results are limited to three chunks of at most 1,200
characters each. Citations use paths relative to this mounted directory and
include the source line range.

Do not put credentials, private keys, `.env` files, or other secrets here.
Workload files are ignored by Git; this README is the only tracked file in
this directory. Restart the UI/container after changing documents to rebuild
the in-memory index.
