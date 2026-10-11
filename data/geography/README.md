# Geographic reference

`istat_boundaries_2025.zip` is the unmodified ISTAT **generalized administrative boundaries for 1 January 2025**, downloaded on 6 October 2026.

- [Source page](https://www.istat.it/notizia/confini-delle-unita-amministrative-a-fini-statistici-al-1-gennaio-2018-2/)
- [Exact archive](https://www.istat.it/storage/cartografia/confini_amministrativi/generalizzati/2025/Limiti01012025_g.zip)
- [ISTAT explanation of municipality codes](https://www.istat.it/it/files/2011/01/nota-informativa.pdf): the first three digits identify the province.

The map uses the region polygons and province-to-region attributes from this archive. The 2025 reference resolves every province code in the local project-territory file. This choice does not date the project snapshot. The region display is projected to EPSG:3035 and simplified by 900 metres for the interactive map; source geometries and assignments are preserved. No municipality names are geocoded or matched approximately.

The source ZIP and local input hashes are recorded in `pnrr_outputs/geography/geography_audit.json` whenever the map is rebuilt.
