#!/usr/bin/env python3
"""Recalculate the add-3 CO2-window test using corrected dry-air inputs."""

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pyarts import arts

import FluxSimulator as fsm
import fmip_radiance_batch as frb
import frtmip_aux_functions as fax


# =============================================================================
# %% run constants
# =============================================================================

script_dir = pathlib.Path(__file__).parent.resolve()
results_folder = script_dir.parent / "results" / "CO2_REFINEMENT"
plot_folder = script_dir.parent / "plot" / "co2_refinement"
old_result = results_folder / "co2_profile_refinement_idx12345_Nf489.nc"

atm_index = 12345
wvn_min, wvn_max, dwvn = 2398.0, 2520.0, 0.25
refinement_steps, points_added_per_step = 1, 3
N_za, N_aa = 20, 1
wvn_grid = np.arange(wvn_min, wvn_max + dwvn / 2, dwvn)
setup_name = (f"co2_dry_air_idx{atm_index}_add{points_added_per_step}"
              f"_Nf{wvn_grid.size}")


# =============================================================================
# %% single-column calculation
# =============================================================================

def main():
    if not np.isclose(wvn_grid[-1], wvn_max):
        raise ValueError("wavenumber interval is not divisible by the spacing")
    f_grid = arts.convert.kaycm2freq(wvn_grid)

    with xr.open_dataset(frb.atmdata_file) as atmdata, \
            xr.open_dataset(frb.sfcdata_file) as sfcdata, \
            xr.open_dataset(frb.surface_emissivity_file) as emissivity:
        if atmdata.attrs.get("source_gas_mass_ratio_basis") != \
                "assumed kg of gas per kg of dry air":
            raise ValueError("atmospheric data do not have the corrected dry-air basis")
        matches = np.flatnonzero(atmdata["idx"].values == atm_index)
        if matches.size != 1:
            raise ValueError(f"expected exactly one atmosphere with idx {atm_index}")
        j = int(matches[0])
        if int(sfcdata["idx"].values[j]) != atm_index or \
                int(emissivity["idx"].values[j]) != atm_index:
            raise ValueError("surface and atmospheric column indices do not match")
        column = atmdata.isel(idx=slice(j, j + 1)).load()
        species = [name for name in column.data_vars if name not in ("p", "t", "alt")]
        p, t, z, gases, n_levels = frb.refine_all_profiles(
            column["p"].values, column["t"].values, column["alt"].values,
            {name: column[name].values for name in species},
            refinement_steps, points_added_per_step)
        T_surf = float(sfcdata["temperature"].values[j])
        geo = [float(sfcdata["lat"].values[j]),
               float(sfcdata["lon"].values[j])]
        reflectivity = frb.make_reflectivities(
            emissivity["emissivity"].isel(idx=j).values[None, :],
            emissivity["wvn"].values, wvn_grid)[0]

    atm = frb.make_gridded_field(
        p[0, ::-1], t[0, ::-1], z[0, ::-1],
        {name: values[0, ::-1] for name, values in gases.items()}, species)

    with xr.open_dataset(old_result) as previous:
        if int(previous.attrs["atm_index"]) != atm_index or \
                int(previous.attrs["refinement_steps"]) != refinement_steps:
            raise ValueError("previous result does not match the requested atmosphere")
        old = previous.sel(variant="add_3")
        if int(old["points_added_per_step"]) != points_added_per_step or \
                int(old["profile_level_count"]) != n_levels or \
                not np.allclose(old["wavenumber"].values, wvn_grid,
                                rtol=0.0, atol=1e-10):
            raise ValueError("previous result uses a different profile or spectral grid")
        old_rad = old["spectral_radiance_TOA"].values.copy()
        old_bt = old["brightness_temperature_TOA"].values.copy()

    def new_simulator():
        flx = fsm.FluxSimulator(setup_name)
        flx.set_frequency_grid(f_grid)
        flx.add_species(frb.define_abs_species(flx, species))
        flx.emission = 1
        flx.gas_scattering = False
        return flx

    print(f"idx {atm_index}: {n_levels} levels, {wvn_grid.size} frequencies; "
          f"rebuilding LUT {setup_name}", flush=True)
    started = time.time()
    flx = new_simulator()
    flx.get_lookuptableBatch([atm], recalc=True, cutoff=True)
    print(f"LUT built in {time.time() - started:.1f} s", flush=True)

    flx = new_simulator()
    result = flx.radiance_simulator_batch(
        [atm], [T_surf], [float(atm[1, 0, 0, 0])], [reflectivity], [geo], [[]],
        N_za=N_za, N_aa=N_aa, end_index=-1, recalc=False, cutoff=True)
    f_out = np.asarray(result["f_grid"])
    if f_out.shape != f_grid.shape or not np.allclose(f_out, f_grid, rtol=1e-12):
        raise ValueError("simulator frequency grid differs from requested grid")
    za = np.asarray(result["zenith_angle"])
    za_index = int(np.argmin(np.abs(za - 180.0)))
    if not np.isclose(za[za_index], 180.0):
        raise ValueError("simulator angular grid does not contain 180 degrees")
    radiance = np.asarray(result["array_of_spectral_radiance_clearsky"][0])[
        :, -1, za_index, 0]
    los = fax.load_iasi_los(frb.satdata_file, [atm_index])
    radiance_los = fax.interpolate_to_iasi_los(
        np.asarray(result["array_of_spectral_radiance_clearsky"][0])[None, :, -1, :, :],
        za, result["azimuth_angle"], los)[0]
    bt = fax.radiance2brightness_temp(radiance, wvn_grid)
    if not np.all(np.isfinite(radiance)) or not np.all(np.isfinite(bt)) or \
            not np.all(np.isfinite(old_rad)) or not np.all(np.isfinite(old_bt)):
        raise ValueError("radiance or brightness temperature contains non-finite values")

    results_folder.mkdir(parents=True, exist_ok=True)
    output = results_folder / f"{setup_name}.nc"
    ds = xr.Dataset(
        data_vars={
            "radiance_corrected": ("frequency", radiance, {"units": "W m-2 Hz-1 sr-1"}),
            "radiance_old": ("frequency", old_rad, {"units": "W m-2 Hz-1 sr-1"}),
            "brightness_temperature_corrected": ("frequency", bt, {"units": "K"}),
            "brightness_temperature_old": ("frequency", old_bt, {"units": "K"}),
            "brightness_temperature_difference": (
                "frequency", bt - old_bt,
                {"units": "K", "description": "corrected minus old"}),
            "pressure": ("level", p[0], {"units": "Pa"}),
            "temperature": ("level", t[0], {"units": "K"}),
            "altitude": ("level", z[0], {"units": "m"}),
        },
        coords={"frequency": ("frequency", f_grid, {"units": "Hz"}),
                "wavenumber": ("frequency", wvn_grid, {"units": "cm-1"}),
                "level": np.arange(n_levels)},
        attrs={"atm_index": atm_index, "refinement_steps": refinement_steps,
               "points_added_per_step": points_added_per_step,
               "gas_mass_ratio_basis": column.attrs["source_gas_mass_ratio_basis"],
               "old_result": str(old_result), "lut_setup": setup_name,
               "line_cutoff": "FluxSimulator cutoff=True",
               "angular_grid": f"N_za={N_za}, N_aa={N_aa}, TOA zenith=180 degrees"})
    ds.to_netcdf(output)
    los_ds = ds.drop_vars(["radiance_old", "brightness_temperature_old",
                           "brightness_temperature_difference"]).copy(deep=True)
    los_ds["radiance_corrected"].values[:] = radiance_los
    los_ds["brightness_temperature_corrected"].values[:] = \
        fax.radiance2brightness_temp(radiance_los, wvn_grid)
    los_ds.attrs.update({
        "viewing_geometry": "iasi_los",
        "observation_view_angle_deg": float(los[0]),
        "arts_zenith_angle_deg": float(180 - abs(los[0])),
        "viewing_interpolation": "linear in cosine of ARTS zenith angle",
        "viewing_azimuth": "azimuth-independent, single simulated azimuth",
        "angular_grid": f"N_za={N_za}, N_aa={N_aa}, IASI LOS",
    })
    los_output = results_folder / f"{setup_name}_iasi_los.nc"
    los_ds.to_netcdf(los_output)
    print(f"Saved IASI LOS result {los_output}")

    plot_folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    axes[0].plot(wvn_grid, old_bt, label="Old gas conversion")
    axes[0].plot(wvn_grid, bt, label="Corrected dry-air conversion")
    axes[0].set_ylabel("Brightness temperature / K")
    axes[0].legend()
    axes[1].plot(wvn_grid, bt - old_bt)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_ylabel("Corrected − old / K")
    axes[1].set_xlabel("Wavenumber / cm$^{-1}$")
    fig.tight_layout()
    figure = plot_folder / f"fig_{setup_name}.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print(f"Saved {output}\nSaved {figure}")
    print(f"Corrected minus old BT: mean {np.mean(bt - old_bt):.4f} K, "
          f"range [{np.min(bt - old_bt):.4f}, {np.max(bt - old_bt):.4f}] K")


if __name__ == "__main__":
    main()
