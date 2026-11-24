"""Focused tests for IASI viewing geometry and product propagation."""

import numpy as np
import pytest
import xarray as xr
from scipy import constants

import frtmip_aux_functions as fax
import merge_radiance_chunks as merge
import convolve_sensor_response as convolve
import convolve_sensor_response_bt as convolve_bt
import export_convolved_netcdf as export
import fmip_radiance_batch as batch
import fmip_radiance_slurm_sub as worker
import compare_convolved_with_obs as comparison


def test_observation_los_matches_idx_not_position(tmp_path):
    path = tmp_path / "sat.nc"
    xr.Dataset(coords={"idx": [42, 7, 91], "los": [-12., 20., 0.]}).to_netcdf(path)
    np.testing.assert_array_equal(fax.load_iasi_los(path, [91, 42]), [0., -12.])
    with pytest.raises(ValueError, match="no observation"):
        fax.load_iasi_los(path, [2])
    with pytest.raises(ValueError, match="unique"):
        fax.load_iasi_los(path, [42, 42])


def test_cosine_interpolation_per_column_and_reversed_grid():
    za = np.array([90., 120., 150., 180.])
    los = np.array([0., -20., 30., 20.])
    mu = np.cos(np.deg2rad(za))
    # A field linear in cos(za), with distinct columns and frequencies.
    field = np.arange(8).reshape(4, 2, 1, 1) + mu[None, None, :, None]
    expected = np.arange(8).reshape(4, 2) + np.cos(
        np.deg2rad(180 - np.abs(los)))[:, None]
    np.testing.assert_allclose(
        fax.interpolate_to_iasi_los(field, za, [0.], los), expected)
    np.testing.assert_allclose(
        fax.interpolate_to_iasi_los(field[:, :, ::-1, :], za[::-1], [0.], los),
        expected)


@pytest.mark.parametrize("za,aa,los", [
    ([120., 150., 150., 180.], [0.], [10.]),
    ([120., 150., 170.], [0.], [0.]),
    ([90., 150., 180.], [0., 180.], [10.]),
    ([90., 150., 180.], [0.], [np.nan]),
    ([90., 150., 180.], [0.], [90.]),
])
def test_invalid_geometry_fails(za, aa, los):
    with pytest.raises(ValueError):
        fax.interpolate_to_iasi_los(
            np.ones((1, 2, len(za), len(aa))), za, aa, los)


def make_chunk(wvn, los):
    ds = xr.Dataset(
        {"spectral_radiance_TOA": (("frequency", "idx"),
                                   np.full((len(wvn), 2), 1e-12)),
         "altitude_min": ("idx", [0., 0.]),
         "altitude_max": ("idx", [30000., 30000.])},
        coords={"idx": [7, 42],
                "frequency": ("frequency", np.asarray(wvn) * 100 * constants.c),
                "wavenumber": ("frequency", wvn)})
    fax.add_iasi_los_metadata(ds, los)
    return ds


def test_merge_convolution_and_export_preserve_geometry(tmp_path, monkeypatch):
    run = "test_iasi_los_Nf6_Nza20"
    folder = tmp_path / run
    folder.mkdir()
    chunks = []
    for i, wvn in enumerate(([999., 1000., 1001.], [1002., 1003., 1004.])):
        path = folder / f"Reference_radiance_TOA_Nf3_Nza20_test_iasi_los_fchunk{i}.nc"
        make_chunk(wvn, [-20., 0.]).to_netcdf(path)
        chunks.append(path)
    monkeypatch.setattr(merge, "results_folder", folder)
    monkeypatch.setattr(merge, "run_name", run)
    monkeypatch.setattr(merge, "chunk_pattern", "Reference_radiance_TOA_Nf*.nc")
    merge.main()
    merged = folder / f"Reference_radiance_TOA_merged_{run}.nc"
    with xr.open_dataset(merged) as ds:
        assert ds["spectral_radiance_TOA"].shape == (6, 2)
        np.testing.assert_array_equal(ds["iasi_los"], [-20., 0.])
        assert fax.viewing_geometry_signature(ds)[0] == "iasi_los"

    srf = tmp_path / "srf.nc"
    sat = tmp_path / "sat.nc"
    xr.Dataset({"response": ("wvn", [0., 1., 0.])},
               coords={"wvn": [-0.5, 0., 0.5]}).to_netcdf(srf)
    xr.Dataset(coords={"wvn": [1000., 1001., 1002., 1003.]}).to_netcdf(sat)
    for module in (convolve, convolve_bt):
        monkeypatch.setattr(module, "results_folder", folder)
        monkeypatch.setattr(module, "run_name", run)
        monkeypatch.setattr(module, "sensor_response_file", srf)
        monkeypatch.setattr(module, "satdata_file", sat)
        module.main()
    for prefix in ("Reference_convolved_", "Reference_BTconvolved_"):
        path = next(folder.glob(f"{prefix}*.nc"))
        with xr.open_dataset(path) as ds:
            assert ds["brightness_temperature"].shape == (4, 2)
            np.testing.assert_array_equal(ds["iasi_los"], [-20., 0.])
            assert "zenith_angle" not in ds["brightness_temperature"].attrs
            assert fax.viewing_geometry_signature(ds)[0] == "iasi_los"

    monkeypatch.setattr(export, "results_folder", folder)
    monkeypatch.setattr(export, "run_name", run)
    monkeypatch.setattr(export, "out_file", folder / f"Reference_convolved_{run}_ALL.nc")
    monkeypatch.setattr(export, "atmdata_file", tmp_path / "missing.nc")
    monkeypatch.setattr(export, "configs_root", tmp_path / "configs")
    export.main()
    with xr.open_dataset(export.out_file) as ds:
        np.testing.assert_array_equal(ds["viewing_zenith_angle"], [160., 180.])
        assert fax.viewing_geometry_signature(ds)[0] == "iasi_los"

    make_chunk([1002., 1003., 1004.], [-21., 0.]).to_netcdf(chunks[1], mode="w")
    with pytest.raises(ValueError, match="different iasi_los"):
        merge.main()


def test_legacy_and_los_geometry_are_distinct():
    legacy = xr.Dataset({"spectral_radiance_TOA": (("frequency", "idx"),
                        [[1.]], {"zenith_angle": 180.})},
                        coords={"idx": [0], "frequency": [1.]})
    assert fax.viewing_geometry_signature(legacy) == ("legacy", 180.)
    los = make_chunk([1000.], [0., 0.])
    assert fax.viewing_geometry_signature(los)[0] == "iasi_los"
    los["viewing_zenith_angle"].values[0] = 170.
    with pytest.raises(ValueError, match="inconsistent"):
        fax.viewing_geometry_signature(los)


@pytest.mark.parametrize("module,prefix", [
    (convolve, "Reference_convolved_"),
    (convolve_bt, "Reference_BTconvolved_"),
])
def test_legacy_convolution_and_mixed_geometry_rejection(
        tmp_path, monkeypatch, module, prefix):
    run = "test_Nf6_Nza20"
    folder = tmp_path / run
    folder.mkdir()
    wvn = np.arange(999., 1005.)
    legacy = make_chunk(wvn, [-20., 0.]).drop_vars(
        ["iasi_los", "viewing_zenith_angle"])
    legacy.attrs.clear()
    legacy["spectral_radiance_TOA"].attrs["zenith_angle"] = 180.
    legacy.to_netcdf(folder / "Reference_radiance_TOA_Nf6_Nza20_test_chunk000.nc")
    srf = tmp_path / "srf.nc"
    sat = tmp_path / "sat.nc"
    xr.Dataset({"response": ("wvn", [0., 1., 0.])},
               coords={"wvn": [-0.5, 0., 0.5]}).to_netcdf(srf)
    xr.Dataset(coords={"wvn": [1000., 1001., 1002., 1003.]}).to_netcdf(sat)
    monkeypatch.setattr(module, "results_folder", folder)
    monkeypatch.setattr(module, "run_name", run)
    monkeypatch.setattr(module, "sensor_response_file", srf)
    monkeypatch.setattr(module, "satdata_file", sat)
    module.main()
    with xr.open_dataset(next(folder.glob(f"{prefix}*.nc"))) as ds:
        assert ds["brightness_temperature"].attrs["zenith_angle"] == 180.
        assert "iasi_los" not in ds
    make_chunk(wvn, [-20., 0.]).to_netcdf(
        folder / "Reference_radiance_TOA_Nf6_Nza20_test_chunk001.nc")
    with pytest.raises(ValueError, match="inconsistent viewing geometry"):
        module.main()


def test_comparison_matches_reordered_observation_indices(tmp_path):
    wvn = np.array([1000., 1001.])
    ds = xr.Dataset(
        {"brightness_temperature": (("frequency", "idx"),
                                    [[271., 292.], [271., 292.]])},
        coords={"idx": [7, 42], "frequency": ("frequency", wvn * 100 * constants.c),
                "wavenumber": ("frequency", wvn)})
    fax.add_iasi_los_metadata(ds, [20., -10.])
    path = tmp_path / "convolved.nc"
    ds.to_netcdf(path)
    stats = comparison.run_comparison(
        [path], np.array([[290., 290.], [270., 270.]]), wvn, "test",
        obs_idx=[42, 7])
    np.testing.assert_allclose(stats["bt"]["mean"], [1.5, 1.5])
    assert stats["viewing_geometry"] == "iasi_los"
    with pytest.raises(ValueError, match="no observation"):
        comparison.run_comparison(
            [path], np.array([[290., 290.]]), wvn, "test", obs_idx=[42])


def test_standalone_and_worker_select_same_subset(tmp_path, monkeypatch):
    idx = np.array([10, 20, 30])
    atm = tmp_path / "atm.nc"
    sfc = tmp_path / "sfc.nc"
    em = tmp_path / "em.nc"
    sat = tmp_path / "sat.nc"
    xr.Dataset(
        {"p": (("idx", "lev"), np.ones((3, 2)) * 1e5),
         "t": (("idx", "lev"), [[250., 251.], [260., 261.], [270., 271.]]),
         "alt": (("idx", "lev"), np.ones((3, 2))),
         "CO2": (("idx", "lev"), np.ones((3, 2)) * 0.0004)},
        coords={"idx": idx, "lev": [0, 1]}).to_netcdf(atm)
    xr.Dataset(coords={"idx": idx, "temperature": [280., 281., 282.],
                       "lat": [0., 1., 2.], "lon": [0., 1., 2.]}).to_netcdf(sfc)
    xr.Dataset({"emissivity": (("idx", "wvn"), np.ones((3, 2)))},
               coords={"idx": idx, "wvn": [999., 1002.]}).to_netcdf(em)
    xr.Dataset(coords={"idx": [30, 10, 20], "los": [-30., 0., 20.]}).to_netcdf(sat)
    wvn = np.array([999., 1000., 1001.])
    frequency = wvn * 100 * constants.c
    za = np.array([90., 120., 150., 180.])

    def make_atm(p, t, z, gases, names):
        return np.full((2, 2, 1, 1), t[0])

    class Simulator:
        def __init__(self, setup):
            pass

        def set_frequency_grid(self, f_grid):
            self.f_grid = np.asarray(f_grid)

        def add_species(self, species, verbose=False):
            pass

        def get_lookuptableBatch(self, atms, recalc=False):
            pass

        def radiance_simulator_batch(self, atms, *args, **kwargs):
            angular = np.cos(np.deg2rad(za))
            radiances = [
                np.broadcast_to(a[0, 0, 0, 0] + angular[None, None, :, None],
                                (len(self.f_grid), 2, len(za), 1)).copy()
                for a in atms]
            return {"f_grid": self.f_grid, "zenith_angle": za,
                    "azimuth_angle": [0.],
                    "array_of_spectral_radiance_clearsky": radiances,
                    "array_of_pressure": [np.ones(2) for _ in atms],
                    "array_of_altitude": [np.ones(2) for _ in atms]}

    monkeypatch.setattr(batch.fsm, "FluxSimulator", Simulator)
    for module in (batch, worker):
        for key, path in (("atmdata_file", atm), ("sfcdata_file", sfc),
                          ("surface_emissivity_file", em), ("satdata_file", sat)):
            monkeypatch.setattr(module, key, path)
        monkeypatch.setattr(module, "make_gridded_field", make_atm)
        monkeypatch.setattr(module, "make_reflectivities",
                            lambda eps, em_wvn, w: [np.zeros(len(w)) for _ in eps])
        monkeypatch.setattr(module, "define_abs_species", lambda *args: [])
    output = tmp_path / "standalone"
    monkeypatch.setattr(batch, "results_folder", output)
    monkeypatch.setattr(batch, "define_frequency_grid", lambda *a, **k: (frequency, wvn))
    for key, value in (("start_column", 1), ("n_columns", 2), ("chunk_size", 1),
                       ("N_wvn", 3), ("N_za", 4), ("N_aa", 1),
                       ("refine", False), ("export_results", True),
                       ("export_full_field", False)):
        monkeypatch.setattr(batch, key, value)
    batch.main()

    monkeypatch.setattr(worker, "script_dir", tmp_path)
    monkeypatch.setattr(worker, "results_folder", tmp_path / "worker")
    monkeypatch.chdir(tmp_path)
    worker.simulate_frequency_chunk(
        {"identifier": "fchunk000", "flxsim_setup": "test_fchunk000",
         "output_setup_name": "test_iasi_los", "viewing_geometry": "iasi_los",
         "run_name": "test_iasi_los_Nf3_Nza4", "f_grid": frequency.tolist(),
         "start_column": 1, "n_columns": 2, "chunk_size": 1, "N_za": 4,
         "N_aa": 1, "refine": False, "refinement_steps": 1,
         "points_added_per_step": 3, "recalc_lut": False, "export_full_field": False})
    standalone = sorted(output.glob("*.nc"))
    slurm = sorted((tmp_path / "worker" / "test_iasi_los_Nf3_Nza4").glob("*.nc"))
    assert len(standalone) == len(slurm) == 2
    for expected_idx, a, b in zip([20, 30], standalone, slurm):
        with xr.open_dataset(a) as left, xr.open_dataset(b) as right:
            assert left["idx"].item() == right["idx"].item() == expected_idx
            np.testing.assert_allclose(left["spectral_radiance_TOA"],
                                       right["spectral_radiance_TOA"])
            np.testing.assert_array_equal(left["iasi_los"], right["iasi_los"])
