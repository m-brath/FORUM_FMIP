#!/usr/bin/env python3
"""Compare monochromatic CO2-window simulations directly with IASI channels."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import compare_convolved_with_obs as ccw


# =============================================================================
# %% paths / constants
# =============================================================================

script_dir = pathlib.Path(__file__).parent.resolve()
viewing_geometry = "iasi_los"  # "iasi_los" or "nadir"
if viewing_geometry not in ("iasi_los", "nadir"):
    raise ValueError(f"unsupported viewing geometry {viewing_geometry}")
suffix = "_iasi_los" if viewing_geometry == "iasi_los" else ""
result_file = (script_dir.parent / "results" / "CO2_REFINEMENT"
               / f"co2_dry_air_idx12345_add3_Nf489{suffix}.nc")
satdata_file = script_dir.parent / "data" / "satdata.nc"
output_file = (script_dir.parent / "results" / "CO2_REFINEMENT"
               / f"co2_dry_air_idx12345_add3_direct_iasi{suffix}.nc")
figure_file = (script_dir.parent / "plot" / "co2_refinement"
               / f"fig_co2_dry_air_idx12345_add3_direct_iasi{suffix}.png")


# =============================================================================
# %% comparison and plot
# =============================================================================

def main():
    with xr.open_dataset(result_file) as simulation, \
            xr.open_dataset(satdata_file) as satellite:
        idx = int(simulation.attrs["atm_index"])
        matches = np.flatnonzero(satellite["idx"].values == idx)
        if matches.size != 1:
            raise ValueError(f"expected exactly one IASI observation for idx {idx}")
        j = int(matches[0])
        wvn = simulation["wavenumber"].values
        channel_wvn = satellite["wvn"].values
        if not np.all(np.diff(channel_wvn) > 0):
            raise ValueError("IASI channel wavenumbers are not strictly increasing")
        channel_index = np.searchsorted(channel_wvn, wvn)
        if np.any(channel_index == channel_wvn.size) or not np.allclose(
                channel_wvn[channel_index], wvn, rtol=0.0, atol=1e-8):
            raise ValueError("simulation frequencies do not exactly match IASI channels")
        obs = satellite["T_b"].isel(idx=j).values[channel_index]
        is_los = simulation.attrs.get("viewing_geometry") == "iasi_los"
        if is_los != (viewing_geometry == "iasi_los"):
            raise ValueError("simulation viewing geometry does not match the selected mode")
        old = (simulation["brightness_temperature_old"].values
               if "brightness_temperature_old" in simulation else None)
        corrected = simulation["brightness_temperature_corrected"].values
        los = float(satellite["los"].values[j])
        if is_los and not np.isclose(simulation.attrs["observation_view_angle_deg"],
                                     los, rtol=0, atol=1e-10):
            raise ValueError("simulation LOS differs from the IASI observation")

    valid = (np.isfinite(obs) & (obs > ccw.bt_valid_range[0])
             & (obs < ccw.bt_valid_range[1])
             & np.isfinite(corrected))
    if old is not None:
        valid &= np.isfinite(old)
    if not valid.any():
        raise ValueError("no valid collocated observations and simulations")
    obs = np.where(valid, obs, np.nan)
    old_diff = np.where(valid, old - obs, np.nan) if old is not None else None
    corrected_diff = np.where(valid, corrected - obs, np.nan)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        data_vars={
            "brightness_temperature_iasi": ("wavenumber", obs, {"units": "K"}),
            "brightness_temperature_corrected": ("wavenumber", corrected, {"units": "K"}),
            "corrected_minus_iasi": ("wavenumber", corrected_diff, {"units": "K"}),
        },
        coords={"wavenumber": ("wavenumber", wvn, {"units": "cm-1"})},
        attrs={
            "atm_index": idx,
            "comparison": "direct channel-centre comparison; no sensor convolution",
            "simulation": f"monochromatic {viewing_geometry} brightness temperatures",
            "observation": "IASI channel brightness temperatures",
            "iasi_view_angle_degrees": los,
            "simulation_source": str(result_file),
            "observation_source": str(satdata_file),
        })
    if old is not None:
        ds["brightness_temperature_old"] = ("wavenumber", old, {"units": "K"})
        ds["old_minus_iasi"] = ("wavenumber", old_diff, {"units": "K"})
    ds.to_netcdf(output_file)

    ccw.apply_plot_style()
    figure_file.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    axes[0].plot(wvn, obs, color=ccw.palette["obs"], label="IASI observation")
    if old is not None:
        axes[0].plot(wvn, old, color=ccw.palette["sim"], label="Old simulation")
    axes[0].plot(wvn, corrected, color="#1a7f37", label="Corrected simulation")
    axes[0].set_ylabel("$T_B$ / K")
    axes[0].legend()
    axes[0].set_title(f"idx {idx}: monochromatic simulations vs IASI channels\n"
                      "Direct comparison; no sensor convolution")
    if old_diff is not None:
        axes[1].plot(wvn, old_diff, color=ccw.palette["sim"], label="Old − IASI")
    axes[1].plot(wvn, corrected_diff, color="#1a7f37", label="Corrected − IASI")
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_ylabel("$T_B$ difference / K")
    axes[1].set_xlabel("Wavenumber / cm$^{-1}$")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figure_file, dpi=180)
    plt.close(fig)

    print(f"idx {idx}: {valid.sum()} valid IASI channels, "
          f"{wvn[0]:.2f}–{wvn[-1]:.2f} cm-1; IASI view angle {los:.2f}°")
    for label, difference in (("Old", old_diff), ("Corrected", corrected_diff)):
        if difference is None:
            continue
        print(f"{label} − IASI: mean {np.nanmean(difference):+.3f} K, "
              f"RMSE {np.sqrt(np.nanmean(difference ** 2)):.3f} K")
    print(f"Saved {output_file}\nSaved {figure_file}")


if __name__ == "__main__":
    main()
