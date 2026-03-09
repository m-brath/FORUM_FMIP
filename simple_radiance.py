#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:50:05 2026

@author: u242031


"""

import os
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import pyarts3 as pa
import frtmip_aux_functions as fax


# =============================================================================
# %% path and constants
# =============================================================================

#atmospheric data
atmdata_file='../data/atmdata.nc'

# satellite data
satellite_data_file='../data/satdata.nc'

#sensor data
sensor_data_file='../data/sensor_response.nc'

#surface data
surface_data_file='../data/sfcdata.nc'

# surface emissivity data
surface_emissivity_file='../data/surface_emissivity.nc'

# Set up geometry of observation
pos = [100e3, 0, 0]
los = [180.0, 0.0]


# =============================================================================
# %% open data
# =============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
atmospheric_data_path = os.path.join(script_dir, atmdata_file)
satellite_data_path = os.path.join(script_dir, satellite_data_file)   
sensor_data_path = os.path.join(script_dir, sensor_data_file)
surface_data_path = os.path.join(script_dir, surface_data_file)
surface_emissivity_path = os.path.join(script_dir, surface_emissivity_file)

atmdata = xr.open_dataset(atmospheric_data_path)
satdata = xr.open_dataset(satellite_data_path)
sensor_data = xr.open_dataset(sensor_data_path)
surfdata = xr.open_dataset(surface_data_path)
emissivity_data = xr.open_dataset(surface_emissivity_path)

# =============================================================================
#  %% start with RT
# =============================================================================

# Download ARTS catalogs if they are not already present
pa.data.download()

# ARTS workspace
ws = pa.workspace.Workspace()

# Set up frequency grid
kayser_grid = np.linspace(1, 2000, 200)  # in Kayser (cm^-1)
ws.freq_grid = pa.arts.convert.kaycm2freq(kayser_grid)  # in Hz

index=111
ds = fax.fmipdata2arts(atmdata, index)
ws.atm_field=pa.data.to_atm_field(ds)



#set up absorption species
abs_species =[]

for spc in ds.data_vars:
    
    if spc in ['p','t']:
        continue

    if spc == 'H2O':
        abs_species.append('H2O')
        abs_species.append('H2O-ForeignContCKDMT430')
        abs_species.append('H2O-SelfContCKDMT430')
    elif spc == 'CO2':
        abs_species.append('CO2')
        abs_species.append('CO2-CKDMT252')
    elif spc == 'O2':
        abs_species.append('O2')
        abs_species.append('O2-CIAfunCKDMT100')
    elif spc == 'N2':
        abs_species.append('N2')
        abs_species.append('N2-CIAfunCKDMT252')
        abs_species.append('N2-CIArotCKDMT252')
    elif spc in ['CFC11', 'CFC12']:
        abs_species.append(spc+'-XFIT')


    
# # Select absorption species and continuum model
# # This example uses a reduced set of species to speed up the calculation.
# # Use the second line for a more realistic setup.
# ws.abs_speciesSet(
#     species=["H2O-161", "H2O-ForeignContCKDMT400", "H2O-SelfContCKDMT400", "CO2-626"]
# )
# # ws.abs_speciesSet(
# #     species=["H2O", "H2O-ForeignContCKDMT400", "H2O-SelfContCKDMT400", "CO2", "O3"]
# # )

# # Read spectral line data from ARTS catalog
# ws.ReadCatalogData()

# # Apply a frequency cutoff. To be consistent with the CKD water vapor continuum,
# # a cutoff of 25 Kayser is necessary. We set it here for all species, because it
# # also speeds up the calculation.
# cutoff = pa.arts.convert.kaycm2freq(25)
# for band in ws.abs_bands:
#     ws.abs_bands[band].cutoff = "ByLine"
#     ws.abs_bands[band].cutoff_value = cutoff

# # Remove 90% of the lines to speed up the calculation
# ws.abs_bands.keep_hitran_s(approximate_percentile=90)

# # Automatically set up the methods to compute absorption coefficients
# ws.spectral_propmat_agendaAuto()

# # Set up a simple atmosphere
# ws.surf_fieldPlanet(option="Earth")
# ws.surf_field[pa.arts.SurfaceKey("t")] = 295.0
# ws.atm_fieldRead(
#     toa=100e3, basename="planets/Earth/afgl/tropical/", missing_is_zero=1
# )


# ws.ray_pathGeometric(pos=pos, los=los, max_stepsize=1000.0)
# ws.spectral_radClearskyEmission()

# # %% Show results

# fig, ax = pa.plot(ws.spectral_rad, freqs=kayser_grid, component="I")
# ax.set_xlabel("Frequency / Kayser (cm$^{-1}$)")
# ax.set_ylabel("Spectral radiance /")
# ax.set_title("Clear sky outgoing radiance")

# if "ARTS_HEADLESS" not in os.environ:
#     plt.show()