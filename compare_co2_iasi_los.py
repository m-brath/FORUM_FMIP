#!/usr/bin/env python3
"""Compare old and corrected single-column spectra at the IASI viewing angle."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pyarts import arts

import FluxSimulator as fsm
import compare_convolved_with_obs as ccw
import fmip_radiance_batch as frb
import frtmip_aux_functions as fax


# =============================================================================
# %% paths / constants
# =============================================================================

script_dir = pathlib.Path(__file__).parent.resolve()
data_dir = script_dir.parent / "data"
raw_file = script_dir.parent / "rawdata" / "atm_prof_arsa_cams_28gas_COlmd_43lev.H5"
satdata_file = data_dir / "satdata.nc"
previous_file = (script_dir.parent / "results" / "CO2_REFINEMENT"
                 / "co2_dry_air_idx12345_add3_Nf489.nc")
output_file = (script_dir.parent / "results" / "CO2_REFINEMENT"
               / "co2_dry_air_idx12345_add3_direct_iasi_los.nc")
figure_file = (script_dir.parent / "plot" / "co2_refinement"
               / "fig_co2_dry_air_idx12345_add3_direct_iasi_los.png")

atm_index = 12345
refinement_steps, points_added_per_step = 1, 3
N_za, N_aa = 20, 1
setups = {
    "old": "co2_profile_refinement_idx12345_add3_Nf489",
    "corrected": "co2_dry_air_idx12345_add3_Nf489",
}


# =============================================================================
# %% old and corrected atmospheres
# =============================================================================

def load_profiles():
    with xr.open_dataset(frb.atmdata_file) as dataset, \
            xr.open_dataset(raw_file) as raw:
        if dataset.attrs.get("source_gas_mass_ratio_basis") != \
                "assumed kg of gas per kg of dry air":
            raise ValueError("atmospheric data do not have the corrected dry-air basis")
        matches = np.flatnonzero(dataset["idx"].values == atm_index)
        if matches.size != 1:
            raise ValueError(f"expected exactly one atmosphere with idx {atm_index}")
        j = int(matches[0])
        column = dataset.isel(idx=slice(j, j + 1)).load()
        names = [name for name in column.data_vars if name not in ("p", "t", "alt")]
        corrected = {name: column[name].values for name in names}

        old = {}
        for name in ("co2", "h2o", "ch4", "o3"):
            key = name.upper()
            mass, _ = fax.molecular_mass(key)
            ratio = raw[f"{name} [kg_kg]"].values[j:j + 1]
            old[key] = ratio / (ratio + mass / fax.M_d)
        raw_gases = raw["28gas [kg_kg]"].isel(
            {raw["28gas [kg_kg]"].dims[0]: slice(j, j + 1)}).values
        for i, label in enumerate(raw["28gas_label"].values):
            original_name = fax.format_molecular_formula(str(label).replace("-", ""))
            mass, _ = fax.molecular_mass(original_name)
            name = fax.format_species_list([original_name])[0]
            ratio = raw_gases[:, i, :]
            old[name] = ratio / (ratio + mass / fax.M_d)
        if set(old) != set(corrected):
            raise ValueError("reconstructed old species do not match corrected input")

    atmospheres = {}
    for label, gases in (("old", old), ("corrected", corrected)):
        p, t, z, refined, n_levels = frb.refine_all_profiles(
            column["p"].values, column["t"].values, column["alt"].values,
            gases, refinement_steps, points_added_per_step)
        if n_levels != 169:
            raise ValueError(f"expected 169 refined levels, got {n_levels}")
        atmospheres[label] = frb.make_gridded_field(
            p[0, ::-1], t[0, ::-1], z[0, ::-1],
            {name: refined[name][0, ::-1] for name in names}, names)
    return j, names, atmospheres


def run_at_angle(label, atm, species, f_grid, T_surf, geo, reflectivity,
                 target_za, previous_nadir):
    flx = fsm.FluxSimulator(setups[label])
    flx.set_frequency_grid(f_grid)
    flx.add_species(frb.define_abs_species(flx, species))
    flx.emission = 1
    flx.gas_scattering = False
    # Fail rather than silently rebuild or substitute a different lookup table.
    flx.readLUT()
    result = flx.radiance_simulator_batch(
        [atm], [T_surf], [float(atm[1, 0, 0, 0])], [reflectivity], [geo], [[]],
        N_za=N_za, N_aa=N_aa, end_index=-1, recalc=False, cutoff=True)
    za = np.asarray(result["zenith_angle"]).copy()
    if not np.allclose(result["f_grid"], f_grid, rtol=1e-12):
        raise ValueError(f"{label} frequency grid differs from the reference")
    nadir_index = int(np.argmin(np.abs(za - 180.0)))
    if not np.isclose(za[nadir_index], 180.0):
        raise ValueError("simulator angular grid does not contain nadir")
    nadir = np.asarray(result["array_of_spectral_radiance_clearsky"][0])[
        :, -1, nadir_index, 0]
    np.testing.assert_allclose(nadir, previous_nadir, rtol=1e-6, atol=0,
                               err_msg=f"{label} nadir does not reproduce prior result")

    angle_index = int(np.argmin(np.abs(za[:-1] - target_za)))
    za[angle_index] = target_za
    if not np.all(np.diff(za) > 0):
        raise ValueError("requested angle does not fit the DISORT angular grid")
    # The batch call prepared the atmosphere and DISORT agenda; evaluate its
    # same setup again with the requested angle in the output grid.
    flx.ws.za_grid.value = za
    flx.ws.DOBatchCalc(robust=0)
    if not np.isclose(flx.ws.za_grid.value[angle_index], target_za,
                      rtol=0.0, atol=1e-10):
        raise ValueError("DISORT output grid does not contain the requested angle")
    field = np.asarray(flx.ws.dobatch_cloudbox_field.value[0])
    at_los = field[:, -1, 0, 0, angle_index, 0, 0].copy()
    check_nadir = field[:, -1, 0, 0, nadir_index, 0, 0]
    np.testing.assert_allclose(check_nadir, nadir, rtol=1e-6, atol=0,
                               err_msg=f"{label} nadir changed with the output grid")
    if not np.all(np.isfinite(at_los)) or np.any(at_los <= 0):
        raise ValueError(f"{label} LOS radiance is invalid")
    return at_los


# =============================================================================
# %% simulation and direct observation comparison
# =============================================================================

def main():
    j, species, atmospheres = load_profiles()
    with xr.open_dataset(previous_file) as previous, \
            xr.open_dataset(satdata_file) as satellite, \
            xr.open_dataset(frb.sfcdata_file) as surface, \
            xr.open_dataset(frb.surface_emissivity_file) as emissivity:
        if any(int(ds["idx"].values[j]) != atm_index
               for ds in (satellite, surface, emissivity)):
            raise ValueError("atmospheric, surface and observation indices differ")
        los = float(satellite["los"].values[j])
        if not np.isfinite(los) or abs(los) >= 90:
            raise ValueError(f"invalid signed local off-nadir angle {los}")
        target_za = 180.0 - abs(los)
        wvn = previous["wavenumber"].values
        f_grid = previous["frequency"].values
        if int(previous.attrs["atm_index"]) != atm_index or wvn.size != 489:
            raise ValueError("previous result does not match the requested test")
        channels = satellite["wvn"].values
        if not np.all(np.diff(channels) > 0):
            raise ValueError("IASI channels are not increasing")
        indices = np.searchsorted(channels, wvn)
        if np.any(indices == channels.size) or not np.allclose(
                channels[indices], wvn, rtol=0.0, atol=1e-8):
            raise ValueError("simulation grid does not match IASI channel centres")
        obs = satellite["T_b"].isel(idx=j).values[indices]
        previous_nadir = {label: previous[f"radiance_{label}"].values.copy()
                          for label in setups}
        T_surf = float(surface["temperature"].values[j])
        geo = [float(surface["lat"].values[j]), float(surface["lon"].values[j])]
        reflectivity = frb.make_reflectivities(
            emissivity["emissivity"].isel(idx=j).values[None, :],
            emissivity["wvn"].values, wvn)[0]

    print(f"idx {atm_index}: signed local off-nadir {los:.5f}°, "
          f"ARTS zenith {target_za:.5f}°", flush=True)
    radiance = {}
    for label in setups:
        print(f"Simulating {label} at observation LOS", flush=True)
        radiance[label] = run_at_angle(
            label, atmospheres[label], species, f_grid, T_surf, geo,
            reflectivity, target_za, previous_nadir[label])
    bt = {label: fax.radiance2brightness_temp(value, wvn)
          for label, value in radiance.items()}
    valid = (np.isfinite(obs) & (obs > ccw.bt_valid_range[0])
             & (obs < ccw.bt_valid_range[1])
             & np.isfinite(bt["old"]) & np.isfinite(bt["corrected"]))
    if not valid.any():
        raise ValueError("no valid collocated observations and simulations")
    obs = np.where(valid, obs, np.nan)
    diff = {label: np.where(valid, value - obs, np.nan)
            for label, value in bt.items()}

    output_file.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        data_vars={
            "brightness_temperature_iasi": ("wavenumber", obs, {"units": "K"}),
            **{f"radiance_{label}": ("wavenumber", radiance[label],
                                     {"units": "W m-2 Hz-1 sr-1"})
               for label in setups},
            **{f"brightness_temperature_{label}": ("wavenumber", bt[label],
                                                    {"units": "K"})
               for label in setups},
            **{f"{label}_minus_iasi": ("wavenumber", diff[label], {"units": "K"})
               for label in setups},
        },
        coords={"wavenumber": ("wavenumber", wvn, {"units": "cm-1"})},
        attrs={"atm_index": atm_index,
               "comparison": "direct channel-centre comparison; no sensor convolution",
               "observation_angle_convention": "signed degrees from local nadir, user confirmed",
               "observation_view_angle_deg": los,
               "arts_zenith_angle_deg": target_za,
               "lut_setups": str(setups), "previous_nadir_source": str(previous_file)})
    ds.to_netcdf(output_file)

    ccw.apply_plot_style()
    figure_file.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    axes[0].plot(wvn, obs, color=ccw.palette["obs"], label="IASI observation")
    axes[0].plot(wvn, bt["old"], color=ccw.palette["sim"], label="Old simulation")
    axes[0].plot(wvn, bt["corrected"], color="#1a7f37", label="Corrected simulation")
    axes[0].set_ylabel("$T_B$ / K")
    axes[0].legend()
    axes[0].set_title(f"idx {atm_index}: LOS {los:.2f}° from nadir\n"
                      "Monochromatic simulations vs IASI; no sensor convolution")
    axes[1].plot(wvn, diff["old"], color=ccw.palette["sim"], label="Old − IASI")
    axes[1].plot(wvn, diff["corrected"], color="#1a7f37",
                 label="Corrected − IASI")
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_ylabel("$T_B$ difference / K")
    axes[1].set_xlabel("Wavenumber / cm$^{-1}$")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figure_file, dpi=180)
    plt.close(fig)
    for label in setups:
        print(f"{label} − IASI: mean {np.nanmean(diff[label]):+.3f} K, "
              f"RMSE {np.sqrt(np.nanmean(diff[label] ** 2)):.3f} K")
    print(f"Saved {output_file}\nSaved {figure_file}")


if __name__ == "__main__":
    main()
