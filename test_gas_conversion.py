"""Tests for dry-air mass-ratio to moist-air mole-fraction conversion."""

import unittest

import numpy as np

import frtmip_aux_functions as fax


class GasConversionTest(unittest.TestCase):
    def test_major_gases_and_trace_gas_use_the_same_moist_air_total(self):
        water = np.array([0.0, 0.00347599806])
        dry_ratios = np.array([0.755006194, 0.230907321, 0.000605300011])
        masses = np.array([fax.molecular_mass(name)[0]
                           for name in ("N2", "O2", "CO2")])
        actual = fax.mixingratio2vmr(dry_ratios[:, None], masses[:, None], water)
        expected = ((dry_ratios[:, None] / masses[:, None])
                    / (1.0 / fax.M_d + water / fax.M_W))
        np.testing.assert_allclose(actual, expected, rtol=1e-14)
        self.assertAlmostEqual(actual[0, 1], 0.7763, places=4)
        self.assertAlmostEqual(actual[1, 1], 0.2079, places=4)
        self.assertGreater(actual[0, 1], 0.7)

    def test_water_round_trip(self):
        water = np.array([0.0, 1e-7, 0.01, 0.025])
        vmr = fax.mixingratio2vmr(water, fax.M_W, water)
        np.testing.assert_allclose(
            fax.vmr2mixingratio(vmr, fax.M_W), water, rtol=1e-14)

    def test_cfc_species_use_molecular_formulas(self):
        labels = fax.format_species_list(["CFC11", "CFC12", "CFC14"])
        masses = fax.get_species_masses(labels)
        for label, formula in (("CFC11", "CCl3F"), ("CFC12", "CCl2F2"),
                               ("CF4", "CF4")):
            self.assertEqual(masses[label], fax.molecular_mass(formula)[0])


if __name__ == "__main__":
    unittest.main()
