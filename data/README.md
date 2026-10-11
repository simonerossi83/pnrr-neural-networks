# Reproducible OpenPNRR input snapshot

The unchanged input CSVs used by this project are available in
[`pnrr-input-data.zip` in release v1.0.0](https://github.com/simonerossi83/pnrr-neural-networks/releases/tag/v1.0.0).
Extract the archive into the repository root; it contains paths under `data/`.
The snapshot's source date is unknown. File modification, project execution and
release dates must not be interpreted as the date of the underlying records.

| File | Purpose |
|---|---|
| `progetti.csv` | Project titles and programme identifiers; required by the notebook |
| `misure.csv` | Measure hierarchy; required by the notebook |
| `temi.csv` | Theme hierarchy; required by the notebook |
| `progetti_territori.csv` | Recorded locations; needed only to rebuild the geographic companion |

Verify the four CSVs from the repository root with `shasum -a 256 -c data/SHA256SUMS`.
The release also provides an archive-level `SHA256SUMS` file. The notebook checks
its inputs independently and records their hashes in each new run.

For Google Colab, open the notebook using the README link, run all cells, and
upload `progetti.csv`, `misure.csv`, and `temi.csv` from the extracted archive when
the notebook asks. No repository modules or prior models are needed.

## Source and attribution

These data are from **OpenPNRR, Fondazione Openpolis**:

- [Open-data catalogue](https://openpnrr.openpolis.io/opendata/)
- [Metadata and field definitions](https://openpnrr.openpolis.io/metadati/)
- [Open Database Licence 1.0](https://opendatacommons.org/licenses/odbl/1-0/)

The publisher states that the exposed data are released under ODbL 1.0.
The distributed CSV bytes are unchanged; packaging into a ZIP does not add
independently reviewed labels. Retain the source attribution and applicable data
licence when redistributing. The licence notice for these inputs is not a new
licence for the project's code. Official ISTAT boundary attribution is documented
separately in [geography/README.md](geography/README.md).
