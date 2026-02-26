#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 10:53:26 2026

@author: u242031
"""

import os
import numpy as np
import xarray as xr


import frtmip_aux_functions as fax


# =============================================================================
# paths/constants
# =============================================================================

# atmospheric data
atmdata_name='../rawdata/atm_prof_arsa_cams_28gas_COlmd_43lev.H5'

# surface emissivity data
emissivitydata_name='../rawdata/IREMIS_IASI_ARSAv2.H5'

# sensor specifications
sensor_specs_name='../rawdata/IASI_SRF.H5'

# path to store the data
data_dir='../data/'

os.makedirs(data_dir, exist_ok=True)
# =============================================================================
# %% open data set
# =============================================================================

script_dir = os.path.dirname(os.path.realpath(__file__))
data = xr.open_dataset(os.path.join(script_dir, atmdata_name))

emissivity = xr.open_dataset(os.path.join(script_dir, emissivitydata_name))
sensor_specs = xr.open_dataset(os.path.join(script_dir, sensor_specs_name))


variable_gases = ['co2 [kg_kg]', 'h2o [kg_kg]','ch4 [kg_kg]', 'o3 [kg_kg]']


# =============================================================================
# %% prepare atm data
# =============================================================================
species_name = []
# get list of all the species
for spc in data['28gas_label']:
    species_name.append(str(spc.values).replace('-', ''))

# Convert each label to proper molecular-formula capitalisation
species_name = fax.format_species_list(species_name)

# Molar masses in g mol^-1 for each species
species_mass = fax.get_species_masses(species_name)

atmdata = {    
    'idx': np.arange(len(data['arsalat'])),
    'p': data['p [hPa]'].values*100,
    'T': data['t [K]'].values,
    'lat': data['arsalat'].values,
    'lon': data['arsalon'].values,    
    'time': fax.convert_time(data, 'arsa')
}

for var in variable_gases:

    spc_name = var.split()[0].upper()
    spc_name = spc_name.replace('-', '')
    spc_name = fax.format_molecular_formula(spc_name)

    #Molecular mass of the species in kg mol^-1
    spc_molmass, _ = fax.molecular_mass(spc_name)
    
    atmdata[spc_name] = fax.mixingratio2vmr(data[var].values, spc_molmass)

for i, spc in enumerate(data['28gas_label']):

    spc_name = str(spc.values).replace('-', '')
    spc_name = fax.format_molecular_formula(spc_name)
    #Molecular mass of the species in kg mol^-1
    spc_molmass, _ = fax.molecular_mass(spc_name)

    atmdata[spc_name] = fax.mixingratio2vmr(np.mean(data['28gas [kg_kg]'].values[:,i,:],0), spc_molmass)


r_v=data['h2o [kg_kg]'].values
atmdata['alt'] = fax.get_altitude(atmdata['p'], atmdata['T'], r_v)



# =============================================================================
# %%  prepae surface data
# =============================================================================

sfcdata = {
    'idx': atmdata['idx'],
    'lat': data['arsalat'].values,
    'lon': data['arsalon'].values,    
    'time': fax.convert_time(data, 'arsa'),
    'temperature': data['tsurf'].values,
    'elevation': np.zeros_like(atmdata['lat'])
}

surface_emissivity={
    'idx': atmdata['idx'],
    'lat': data['arsalat'].values,
    'lon': data['arsalon'].values,    
    'time': fax.convert_time(data, 'arsa'),
    'emissivity': emissivity['RTTOV_SURF_EMIS_IREMIS'].values,
    'wvn': emissivity['wavenumbers'].values
}

# =============================================================================
# %% prepare satellite data
# =============================================================================

satdata= {
    'idx': atmdata['idx'],
    'lat': data['latiasi'].values,
    'lon': data['loniasi'].values,    
    'time': fax.convert_time(data, 'iasi'),
    'T_b': data['BTobsiasi'].values,
    'los': data['view_angle'].values,
    'wvn': sensor_specs['INTERF_WV'].values
}

sensor_response = {
    'wvn': sensor_specs['INTERF_ISRF_WV'].values,
    'response': sensor_specs['INTERF_ISRF_FL'].values
}

# =============================================================================
# %% export data as netCDF
# =============================================================================

def dict_to_dataset(d):
    """Convert a dictionary of numpy arrays to an xarray Dataset."""
    data_vars = {}
    coords = {}
    for key, val in d.items():
        val = np.array(val)
        if val.ndim == 1:
            coords[key] = xr.Variable(key, val)
        else:
            data_vars[key] = xr.Variable([d['idx'].size, val.shape[-1]] and
                                          ['idx', 'lev'] if val.ndim == 2 else ['idx'], val)
    return xr.Dataset(data_vars, coords=coords)


def save_dict_as_netcdf(d, filename):
    """Save a dictionary of arrays as a netCDF file using xarray."""
    data_vars = {}
    for key, val in d.items():
        val = np.array(val)
        if val.ndim == 0:
            data_vars[key] = xr.Variable([], val)
        elif val.ndim == 1:
            data_vars[key] = xr.Variable([key], val)
        elif val.ndim == 2:
            data_vars[key] = xr.Variable(['idx', 'lev'], val)
        else:
            data_vars[key] = xr.Variable([f'dim{i}' for i in range(val.ndim)], val)
    ds = xr.Dataset(data_vars)
    ds.to_netcdf(filename)
    print(f"Saved: {filename}")


output_dir = os.path.join(script_dir, data_dir)
os.makedirs(output_dir, exist_ok=True)

save_dict_as_netcdf(atmdata,           os.path.join(output_dir, 'atmdata.nc'))
save_dict_as_netcdf(sfcdata,           os.path.join(output_dir, 'sfcdata.nc'))
save_dict_as_netcdf(surface_emissivity,os.path.join(output_dir, 'surface_emissivity.nc'))
save_dict_as_netcdf(satdata,           os.path.join(output_dir, 'satdata.nc'))
save_dict_as_netcdf(sensor_response,   os.path.join(output_dir, 'sensor_response.nc'))