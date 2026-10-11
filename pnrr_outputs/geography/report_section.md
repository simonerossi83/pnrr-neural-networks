<!-- PNRR_GEOGRAPHY_START -->
## Geographic distribution of projects

[Open the interactive Italy map](pnrr_outputs/geography/italy_projects.html). Select a recorded programme theme to recolour all 20 regions; hover or tap a region for its count, or expand the regional table for exact values and theme shares.

![PNRR projects by Italian region](pnrr_outputs/geography/italy_projects.png)

This descriptive companion counts **all 298,338 raw project IDs**, rather than the deduplicated titles used for model training. **297,500 projects** have a regional location, **767** are nationwide only, and **71** have no location record. The highest raw counts are **Lombardia (44,487)**, **Veneto (26,763)**, **Campania (26,506)**. These counts describe the supplied OpenPNRR snapshot, whose date is unknown; they are not current live totals.

**Counting rule.** Join `progetti_territori.csv` to `progetti.csv` by project ID. Use ISTAT region codes directly for regional records and the official province-to-region lookup for municipality, province, and metropolitan-city records. Count a project once per region, removing repeated locations in the same region. All non-national location records resolve. The **806 multi-region projects** generate **300,652 project-region memberships** overall, so regional counts must not be summed as a count of unique national projects. Nationwide records are kept separate instead of being replicated over the map.

**Theme comparison.** Themes are inherited from `misure.csv` and the root hierarchy in `temi.csv`, not predicted by either model. Low-support categories are retained for this descriptive map. **406 projects** have no matched theme and remain visible under “Theme unavailable”. The table's theme share divides the selected theme's count by all mapped projects in that same region; it is not a population rate.

![Italy choropleths for each recorded theme](pnrr_outputs/geography/italy_projects_by_theme.png)

Each theme panel and interactive selection has its **own linear colour scale**. Compare the legends and exact counts, not colour intensity between panels. These are project counts, not funding amounts, completion rates, population-adjusted access, or evidence of regional fairness. More projects can reflect a larger region or different programme structures.

Boundaries and province-to-region attributes: [ISTAT generalized administrative boundaries, 1 January 2025](https://www.istat.it/notizia/confini-delle-unita-amministrative-a-fini-statistici-al-1-gennaio-2018-2/). Municipality-to-province mapping follows the [ISTAT code convention](https://www.istat.it/it/files/2011/01/nota-informativa.pdf). The 2025 lookup resolves every supplied code; it does not establish the project snapshot date. Source hashes and coverage checks are saved in `pnrr_outputs/geography/geography_audit.json`, and exact counts in `regional_project_counts.csv`.
<!-- PNRR_GEOGRAPHY_END -->
