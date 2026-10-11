"""Check geographic counting where location rows must not equal project counts."""
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from map_projects import aggregate_projects, ALL, UNKNOWN


class GeographyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        (self.data / 'geography').mkdir()
        # The aggregator hashes this input but does not read geometry itself.
        (self.data / 'geography/istat_boundaries_2025.zip').write_bytes(b'test boundary reference')
        pd.DataFrame({'progetto_id': range(1, 8), 'misura_id': [1, 1, 1, 2, 99, 2, 1]}).to_csv(
            self.data / 'progetti.csv', index=False)
        pd.DataFrame({'id': [1, 2], 'tema': [12, 11]}).to_csv(self.data / 'misure.csv', index=False)
        pd.DataFrame({'id': [10, 11, 12], 'parent_id': [None, None, 10],
                      'descrizione': ['Theme A', 'Theme B', 'Child A']}).to_csv(self.data / 'temi.csv', index=False)
        self.locations = pd.DataFrame([
            [1, '001001', 'C'], [1, '001001', 'C'], [1, '001', 'P'], [1, '01', 'R'],
            [2, '001002', 'C'], [2, '002', 'CM'], [3, None, 'N'],
            [5, '002003', 'C'], [6, '02', 'R'], [7, None, 'N'], [7, '01', 'R']],
            columns=['progetto_id', 'istat_id', 'tipologia'])
        self.locations.to_csv(self.data / 'progetti_territori.csv', index=False)
        self.lookup = pd.Series({1: 1, 2: 2})
        self.regions = pd.DataFrame({'COD_REG': [1, 2, 3], 'DEN_REG': ['One', 'Two', 'No projects']})

    def test_distinct_regions_nationwide_missing_locations_and_unknown_themes(self):
        table, series, coverage, audit = aggregate_projects(self.data, self.lookup, self.regions)
        self.assertEqual(series[ALL], [3, 3, 0])
        self.assertEqual(series['Theme A'], [3, 1, 0])
        self.assertEqual(series['Theme B'], [0, 1, 0])
        self.assertEqual(series[UNKNOWN], [0, 1, 0])
        self.assertEqual(coverage[ALL], {'projects': 7, 'mapped': 5, 'national_only': 1,
                                        'no_location': 1, 'region_memberships': 6})
        self.assertEqual(audit['multi_region_projects'], 1)
        self.assertEqual(audit['national_and_regional_projects'], 1)
        self.assertEqual(audit['redundant_location_rows_removed_for_region_counting'], 3)
        self.assertTrue(table.loc[table.region_code.eq(3), 'share_of_region_projects'].isna().all())

    def test_repeated_location_rows_leave_project_counts_unchanged(self):
        expected = aggregate_projects(self.data, self.lookup, self.regions)[1]
        pd.concat([self.locations] * 3).to_csv(self.data / 'progetti_territori.csv', index=False)
        actual = aggregate_projects(self.data, self.lookup, self.regions)[1]
        self.assertEqual(actual, expected)

    def test_unknown_province_is_not_silently_discarded(self):
        self.locations.loc[0, 'istat_id'] = '999001'
        self.locations.to_csv(self.data / 'progetti_territori.csv', index=False)
        with self.assertRaisesRegex(AssertionError, 'Unmapped location'):
            aggregate_projects(self.data, self.lookup, self.regions)


if __name__ == '__main__':
    unittest.main()
