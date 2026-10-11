"""Count distinct project-region pairs and render the recorded PNRR geography.

This is a descriptive companion to the title-classification experiment. It uses
all raw projects and inherited themes, never title deduplication or predictions.
"""
from pathlib import Path
import hashlib
import json
import os
import textwrap

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from prepare import ancestor

BOUNDARY_URL = 'https://www.istat.it/storage/cartografia/confini_amministrativi/generalizzati/2025/Limiti01012025_g.zip'
BOUNDARY_PAGE = 'https://www.istat.it/notizia/confini-delle-unita-amministrative-a-fini-statistici-al-1-gennaio-2018-2/'
OUT = ROOT / 'pnrr_outputs/geography'
UNKNOWN = 'Theme unavailable'
ALL = 'All projects'


def load_geography(data_dir):
    archive = data_dir / 'geography/istat_boundaries_2025.zip'
    layers = {}
    for layer in ['Reg', 'ProvCM']:
        path = f'zip://{archive.resolve()}!{layer}01012025_g/{layer}01012025_g_WGS84.shp'
        layers[layer] = gpd.read_file(path, encoding='utf-8')
    regions = layers['Reg'][['COD_REG', 'DEN_REG', 'geometry']].sort_values('COD_REG')
    assert set(regions.COD_REG) == set(range(1, 21))
    province_region = layers['ProvCM'].set_index('COD_PROV').COD_REG
    assert province_region.index.is_unique
    return regions.to_crs(3035), province_region


def aggregate_projects(data_dir, province_region, regions):
    projects = pd.read_csv(data_dir / 'progetti.csv', usecols=['progetto_id', 'misura_id'])
    territories = pd.read_csv(data_dir / 'progetti_territori.csv',
                              usecols=['progetto_id', 'istat_id', 'tipologia'],
                              dtype={'istat_id': 'string'})
    measures = pd.read_csv(data_dir / 'misure.csv', usecols=['id', 'tema'])
    themes = pd.read_csv(data_dir / 'temi.csv')
    assert projects.progetto_id.is_unique and measures.id.is_unique and themes.id.is_unique
    assert territories.progetto_id.isin(projects.progetto_id).all()
    assert territories.tipologia.isin(['C', 'CM', 'P', 'R', 'N']).all()
    projects = projects.merge(measures, left_on='misura_id', right_on='id',
                              how='left', validate='many_to_one')
    parents = themes.set_index('id').parent_id.to_dict()
    names = themes.set_index('id').descrizione.to_dict()
    projects['theme'] = projects.tema.map(
        lambda value: UNKNOWN if pd.isna(value) else names[ancestor(int(value), parents)])

    territories['region_code'] = pd.Series(pd.NA, index=territories.index, dtype='Int64')
    local = territories.tipologia.isin(['C', 'CM', 'P'])
    # Official ISTAT convention: first 3 digits of a municipal code identify its province.
    territories.loc[local, 'region_code'] = (
        territories.loc[local, 'istat_id'].str[:3].astype(int).map(province_region).astype('Int64'))
    regional = territories.tipologia.eq('R')
    territories.loc[regional, 'region_code'] = territories.loc[regional, 'istat_id'].astype('Int64')
    assert territories.loc[territories.tipologia.ne('N'), 'region_code'].notna().all(), 'Unmapped location'
    located = territories.dropna(subset=['region_code'])
    assert located.region_code.isin(regions.COD_REG).all()
    pairs = located[['progetto_id', 'region_code']].drop_duplicates()
    pairs = pairs.merge(projects[['progetto_id', 'theme']], on='progetto_id', validate='many_to_one')
    regional_ids = set(pairs.progetto_id)
    national_ids = set(territories.loc[territories.tipologia.eq('N'), 'progetto_id'])
    absent_ids = set(projects.progetto_id) - set(territories.progetto_id)
    multiplicity = pairs.groupby('progetto_id').size()
    projects['has_region'] = projects.progetto_id.isin(regional_ids)
    projects['national_only'] = projects.progetto_id.isin(national_ids - regional_ids)
    projects['no_location'] = projects.progetto_id.isin(absent_ids)
    assert (projects[['has_region', 'national_only', 'no_location']].sum(axis=1) == 1).all()
    labels = [ALL] + sorted(projects.theme.unique(), key=lambda name: (name == UNKNOWN, name))
    series, coverage, records = {}, {}, []
    for label in labels:
        selected_pairs = pairs if label == ALL else pairs[pairs.theme.eq(label)]
        selected_projects = projects if label == ALL else projects[projects.theme.eq(label)]
        counts = selected_pairs.groupby('region_code').progetto_id.nunique().reindex(regions.COD_REG, fill_value=0)
        series[label] = [int(value) for value in counts]
        coverage[label] = {
            'projects': len(selected_projects), 'mapped': int(selected_projects.has_region.sum()),
            'national_only': int(selected_projects.national_only.sum()),
            'no_location': int(selected_projects.no_location.sum()),
            'region_memberships': int(counts.sum())}
        assert sum(coverage[label][key] for key in ['mapped', 'national_only', 'no_location']) == len(selected_projects)
        for code, name, count in zip(regions.COD_REG, regions.DEN_REG, counts):
            regional_total = series[ALL][list(regions.COD_REG).index(code)]
            records.append({'theme': label, 'region_code': int(code), 'region': name,
                            'projects': int(count),
                            'share_of_region_projects': float(count / regional_total) if regional_total else None})
    assert np.array_equal(np.sum([series[label] for label in labels[1:]], axis=0), series[ALL])
    assert sum(series[ALL]) == len(pairs)
    audit = {'raw_projects': len(projects), 'location_rows': len(territories),
             'mapped_projects': len(regional_ids), 'national_only_projects': len(national_ids - regional_ids),
             'projects_without_location': len(absent_ids), 'national_and_regional_projects': len(national_ids & regional_ids),
             'project_region_pairs': len(pairs), 'multi_region_projects': int((multiplicity > 1).sum()),
             'redundant_location_rows_removed_for_region_counting': len(located) - len(pairs),
             'projects_without_theme': int(projects.theme.eq(UNKNOWN).sum()),
             'unmapped_non_national_location_rows': 0,
             'counting_unit': 'Distinct progetto_id within region; one project can occur in several regions',
             'theme_source': 'Root tema inherited via misure.csv and temi.csv, not model predictions',
             'project_snapshot_date': None, 'boundary_reference_date': '2025-01-01',
             'boundary_url': BOUNDARY_URL, 'projection': 'EPSG:3035',
             'source_sha256': {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                               for name in ['progetti.csv', 'progetti_territori.csv', 'misure.csv', 'temi.csv',
                                            'geography/istat_boundaries_2025.zip']}}
    return pd.DataFrame(records), series, coverage, audit


def draw_static(regions, series, audit, out):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    counts = np.array(series[ALL])
    fig, (ax, ranking) = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [1.1, 1]},
                                    layout='constrained')
    regions.assign(projects=counts).plot(column='projects', cmap='Blues', vmin=0, vmax=counts.max(),
                                        linewidth=.65, edgecolor='#ffffff', ax=ax)
    ax.set_axis_off()
    ax.set_title('Italy · regional project counts', loc='left', fontsize=15)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, counts.max()), cmap='Blues'),
                           ax=ax, orientation='horizontal', shrink=.85, pad=.02)
    colorbar.set_label('Distinct projects in each region · linear scale')
    order = np.argsort(counts)
    ranking.barh(np.arange(20), counts[order], color='#28699d')
    ranking.set_yticks(np.arange(20), regions.DEN_REG.iloc[order])
    ranking.set_xlim(0, counts.max() * 1.2)
    ranking.set_xlabel('Projects')
    ranking.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    for row, value in enumerate(counts[order]):
        ranking.text(value + counts.max() * .015, row, f'{value:,}', va='center', fontsize=9)
    ranking.set_title('All recorded themes', loc='left', fontsize=13)
    fig.suptitle('PNRR project distribution', fontsize=19, fontweight='medium')
    fig.supxlabel(f"{audit['mapped_projects']:,} projects mapped · {audit['national_only_projects']:,} nationwide · "
                  f"{audit['projects_without_location']:,} without location\n"
                  'Projects spanning several regions count once in each. Counts are not funding or per-capita measures.\n'
                  'Source: local OpenPNRR CSVs (snapshot date unknown); ISTAT boundaries, 1 January 2025.', fontsize=9)
    fig.savefig(out / 'italy_projects.png', dpi=170)
    plt.close(fig)

    labels = list(series)[1:]
    columns = 4
    rows = (len(labels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(16, 4.5 * rows), layout='constrained')
    for ax, label in zip(axes.flat, labels):
        values = np.array(series[label])
        vmax = max(1, int(values.max()))
        regions.assign(projects=values).plot(column='projects', cmap='Blues', vmin=0, vmax=vmax,
                                             linewidth=.4, edgecolor='#ffffff', ax=ax)
        ax.set_axis_off()
        ax.set_title(textwrap.fill(label, 28), fontsize=11)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, vmax), cmap='Blues'),
                          ax=ax, orientation='horizontal', shrink=.7, pad=.01, ticks=[0, vmax])
        cb.ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    for ax in list(axes.flat)[len(labels):]:
        ax.set_axis_off()
    fig.suptitle('PNRR projects by recorded programme theme', fontsize=19)
    fig.supxlabel('Distinct projects per region · each panel has its own linear colour scale\n'
                  'Nationwide and unlocated projects excluded from shading. Source: local OpenPNRR CSVs; ISTAT 2025.', fontsize=10)
    fig.savefig(out / 'italy_projects_by_theme.png', dpi=150)
    plt.close(fig)


def svg_regions(regions):
    # Project first; simplify only display geometry, never counts or assignments.
    shapes = regions.geometry.simplify(900, preserve_topology=True)
    xmin, ymin, xmax, ymax = shapes.total_bounds
    width, height, padding = 660, 700, 12
    scale = min((width - 2 * padding) / (xmax - xmin), (height - 2 * padding) / (ymax - ymin))
    offset_x = (width - scale * (xmax - xmin)) / 2
    offset_y = (height - scale * (ymax - ymin)) / 2
    paths = []
    for row, geometry in zip(regions.itertuples(), shapes):
        parts = list(geometry.geoms) if geometry.geom_type == 'MultiPolygon' else [geometry]
        pieces = []
        for part in parts:
            for ring in [part.exterior, *part.interiors]:
                coords = [(offset_x + (x - xmin) * scale, offset_y + (ymax - y) * scale)
                          for x, y in ring.coords]
                pieces.append('M' + 'L'.join(f'{x:.2f},{y:.2f}' for x, y in coords) + 'Z')
        paths.append(f'<path data-region="{row.COD_REG}" fill-rule="evenodd" d="{"".join(pieces)}"></path>')
    return '\n'.join(paths)


def report_section(table, audit):
    top = table[table.theme.eq(ALL)].sort_values('projects', ascending=False).head(3)
    leaders = ', '.join(f"**{row.region} ({row.projects:,})**" for row in top.itertuples())
    return f'''<!-- PNRR_GEOGRAPHY_START -->
## Geographic distribution of projects

[Open the interactive Italy map](pnrr_outputs/geography/italy_projects.html). Select a recorded programme theme to recolour all 20 regions; hover or tap a region for its count, or expand the regional table for exact values and theme shares.

![PNRR projects by Italian region](pnrr_outputs/geography/italy_projects.png)

This descriptive companion counts **all {audit['raw_projects']:,} raw project IDs**, rather than the deduplicated titles used for model training. **{audit['mapped_projects']:,} projects** have a regional location, **{audit['national_only_projects']:,}** are nationwide only, and **{audit['projects_without_location']:,}** have no location record. The highest raw counts are {leaders}. These counts describe the supplied OpenPNRR snapshot, whose date is unknown; they are not current live totals.

**Counting rule.** Join `progetti_territori.csv` to `progetti.csv` by project ID. Use ISTAT region codes directly for regional records and the official province-to-region lookup for municipality, province, and metropolitan-city records. Count a project once per region, removing repeated locations in the same region. All non-national location records resolve. The **{audit['multi_region_projects']:,} multi-region projects** generate **{audit['project_region_pairs']:,} project-region memberships** overall, so regional counts must not be summed as a count of unique national projects. Nationwide records are kept separate instead of being replicated over the map.

**Theme comparison.** Themes are inherited from `misure.csv` and the root hierarchy in `temi.csv`, not predicted by either model. Low-support categories are retained for this descriptive map. **{audit['projects_without_theme']:,} projects** have no matched theme and remain visible under “Theme unavailable”. The table's theme share divides the selected theme's count by all mapped projects in that same region; it is not a population rate.

![Italy choropleths for each recorded theme](pnrr_outputs/geography/italy_projects_by_theme.png)

Each theme panel and interactive selection has its **own linear colour scale**. Compare the legends and exact counts, not colour intensity between panels. These are project counts, not funding amounts, completion rates, population-adjusted access, or evidence of regional fairness. More projects can reflect a larger region or different programme structures.

Boundaries and province-to-region attributes: [ISTAT generalized administrative boundaries, 1 January 2025]({BOUNDARY_PAGE}). Municipality-to-province mapping follows the [ISTAT code convention](https://www.istat.it/it/files/2011/01/nota-informativa.pdf). The 2025 lookup resolves every supplied code; it does not establish the project snapshot date. Source hashes and coverage checks are saved in `pnrr_outputs/geography/geography_audit.json`, and exact counts in `regional_project_counts.csv`.
<!-- PNRR_GEOGRAPHY_END -->
'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    regions, provinces = load_geography(ROOT / 'data')
    table, series, coverage, audit = aggregate_projects(ROOT / 'data', provinces, regions)
    table.to_csv(OUT / 'regional_project_counts.csv', index=False)
    (OUT / 'geography_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    payload = {'regions': [{'code': int(code), 'name': name} for code, name in zip(regions.COD_REG, regions.DEN_REG)],
               'series': series, 'coverage': coverage, 'audit': audit}
    (OUT / 'map_data.json').write_text(json.dumps(payload, ensure_ascii=False))
    draw_static(regions, series, audit, OUT)
    fragment = (ROOT / 'src/templates/italy-projects.html').read_text()
    fragment = fragment.replace('<!-- REGION_PATHS -->', svg_regions(regions))
    fragment = fragment.replace('/* MAP_DATA */null', json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c'))
    assert '<!-- REGION_PATHS -->' not in fragment and '/* MAP_DATA */null' not in fragment
    (OUT / 'italy-projects.html').write_text(fragment)
    standalone = (ROOT / 'src/templates/map-page.html').read_text().replace('<!-- MAP -->', fragment)
    (OUT / 'italy_projects.html').write_text(standalone)
    section = report_section(table, audit)
    (OUT / 'report_section.md').write_text(section)
    report_path = ROOT / 'REPORT.md'
    report = report_path.read_text()
    if '<!-- PNRR_GEOGRAPHY_START -->' in report:
        start = report.index('<!-- PNRR_GEOGRAPHY_START -->')
        end = report.index('<!-- PNRR_GEOGRAPHY_END -->', start) + len('<!-- PNRR_GEOGRAPHY_END -->')
        report = report[:start] + section.rstrip() + report[end:]
    else:
        report = report.rstrip() + '\n\n' + section
    report_path.write_text(report)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(table[table.theme.eq(ALL)].sort_values('projects', ascending=False).head().to_string(index=False))
    print(f'Maps saved in {OUT}')


if __name__ == '__main__':
    main()
