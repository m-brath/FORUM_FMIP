#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 11:24:07 2026

@author: u242031
"""
import os
import re
import numpy as np
import periodictable as pt
from scipy import constants as scc
from scipy.interpolate import make_interp_spline
import xarray as xr
import pyarts3 as pa




# =============================================================================
# %% paths/constants
# =============================================================================

#universal gas constant
R=scc.R

#gravity
g=scc.g  # m s^{-2}

#Avogadro constant
N_A=scc.Avogadro

#atomic mass
m_a=scc.atomic_mass

# Radius of Earth
R_e = 6371229  # m

# Molar mass of dry air# Donald P. Gatley, Sebastian Herrmann & Hans-Joachim Kretzschmar
# (2008) A Twenty-First Century Molar Mass for Dry Air, HVAC&R Research, 14:5, 655-662, DOI:
# 10.1080/10789669.2008.10391032
M_d = 28.965369e-3  # kg mol^-1

R_D = R / M_d  # J K^-1 kg^-1



# =============================================================================
# Helper: molecular formula formatter
# =============================================================================

# Build lookup sets from the periodic table.
_ONE_LETTER_ELEMENTS = {el.symbol.lower() for el in pt.elements if len(el.symbol) == 1}
_ALL_TWO_LETTER_ELEMENTS = {el.symbol.lower() for el in pt.elements if len(el.symbol) == 2}

# Use a 2-letter match ONLY when its second character is NOT itself a valid
# 1-letter element. This avoids ambiguities like:
#   "co" → Co (cobalt) vs C + O (carbon monoxide) — prefer C+O  ✓
#   "no" → No (nobelium) vs N + O (nitric oxide)  — prefer N+O  ✓
#   "ho" → Ho (holmium)  vs H + O (hydroxyl)      — prefer H+O  ✓
# but keeps unambiguous 2-letter symbols:
#   "cl" → Cl  ('l' is not an element)  ✓
#   "br" → Br  ('r' is not an element)  ✓
#   "si" → Si  ('i' is Iodine, so this will fall through to S+I — acceptable)
_SAFE_TWO_LETTER = {
    sym for sym in _ALL_TWO_LETTER_ELEMENTS
    if sym[1] not in _ONE_LETTER_ELEMENTS
}


def format_molecular_formula(formula: str) -> str:
    """Convert a molecular formula string to the correct chemical notation
    with proper upper- and lowercase letters.

    The parser is case-insensitive on input and resolves element boundaries
    by trying 2-letter symbols first (only for common, unambiguous elements),
    then falling back to single-letter symbols.

    Examples
    --------
    >>> format_molecular_formula('h2o')   # 'H2O'
    >>> format_molecular_formula('HCL')   # 'HCl'
    >>> format_molecular_formula('CH3CL') # 'CH3Cl'
    >>> format_molecular_formula('HOCL')  # 'HOCl'
    >>> format_molecular_formula('co2')   # 'CO2'
    """
    s = formula.lower()
    result = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isalpha():
            # Try 2-letter element symbol first (only "safe" ones where the
            # 2nd char is not itself a standalone 1-letter element, to avoid
            # Co vs C+O, No vs N+O, Ho vs H+O, etc.).
            two = s[i: i + 2] if i + 1 < len(s) and s[i + 1].isalpha() else ''
            if two and two in _SAFE_TWO_LETTER:
                result.append(two.capitalize())
                i += 2
            else:
                result.append(c.upper())
                i += 1
        else:
            result.append(c)
            i += 1
    return ''.join(result)


def format_species_list(species: list[str]) -> list[str]:
    """Apply :func:`format_molecular_formula` to every entry in *species*."""
    species_list = [format_molecular_formula(s) for s in species]

    # now we need to replace CFC-14 with its molecular formula CF4
    species_list = [s.replace('CFC14', 'CF4') for s in species_list]
    return species_list


def molecular_mass(formula: str) -> float:
    """Calculate the molar mass of a molecule from its molecular formula.

    The formula must use correct capitalisation (e.g. 'H2O', 'CO2', 'HCl').
    Use :func:`format_molecular_formula` first if the input case is uncertain.

    Parameters
    ----------
    formula : str
        Properly capitalised molecular formula (e.g. 'CH3Cl', 'HCl', 'N2O').

    Returns
    -------
    float
        Molar mass in g mol⁻¹.

    Examples
    --------
    >>> molecular_mass('H2O')   # 18.015
    >>> molecular_mass('CO2')   # 44.010
    >>> molecular_mass('HCl')   # 36.461
    """
    mass = 0.0
    for symbol, count in re.findall(r'([A-Z][a-z]?)(\d*)', formula):        
        if not symbol:
            continue
        n = int(count) if count else 1
        mass += pt.elements.symbol(symbol).mass * n
    return mass*N_A*m_a, mass


def get_species_masses(species: list[str]) -> dict[str, float]:
    """Return a dict mapping each species formula to its molar mass in g mol⁻¹.

    Parameters
    ----------
    species : list[str]
        List of properly capitalised molecular formula strings, e.g. as
        returned by :func:`format_species_list`.

    Returns
    -------
    dict[str, float]
        ``{formula: molar_mass_g_per_mol}`` for every entry in *species*.

    Examples
    --------
    >>> get_species_masses(['H2O', 'CO2', 'O3'])
    {'H2O': 18.015, 'CO2': 44.010, 'O3': 47.998}
    """

    output={}
    for s in species:

        # hardcode some special cases for CFC-11 and CFC-12
        if s == 'CFC11':
            output[s], _ = molecular_mass('CCl3F')
        elif s == 'CFC12':
            output[s], _ = molecular_mass('CCl2F2')
        else:
            output[s], _ = molecular_mass(s)

    return output


# =============================================================================
# %% aux
# =============================================================================

def convert_time(data, obs_type):
    """
    Convert time variables from a dataset to an array of numpy datetime64 objects.
    This function extracts time and date information from a dataset containing
    observation data in either 'arsa' or 'iasi' format, and converts them into
    numpy datetime64 objects.
    Parameters
    ----------
    data : xarray.Dataset or dict-like
        The dataset containing the time variables. It must include variables
        named 'hhmmss{obs_type}' and 'yyyymmdd{obs_type}', where {obs_type}
        is either 'arsa' or 'iasi'.
        - 'hhmmss{obs_type}': Time in HHMMSS format (e.g., 123456 for 12:34:56).
        - 'yyyymmdd{obs_type}': Date in YYYYMMDD format (e.g., 230101 for 2023-01-01).
          Note: The year is stored as a 2-digit value and 2000 is added to it.
    obs_type : str
        The type of observation data. Must be either 'arsa' or 'iasi'.
    Returns
    -------
    numpy.ndarray
        An array of numpy datetime64 objects representing the converted timestamps
        in the format 'YYYY-MM-DDTHH:MM:SS'.
    Raises
    ------
    ValueError
        If `obs_type` is not 'arsa' or 'iasi'.
    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> data = xr.Dataset({
    ...     'hhmmssarsa': ('time', [123456, 235959]),
    ...     'yyyymmddarsa': ('time', [230101, 231231])
    ... })
    >>> convert_time(data, 'arsa')
    array(['2023-01-01T12:34:56', '2023-12-31T23:59:59'], dtype='datetime64[s]')
    Notes
    -----
    There is a variable shadowing issue in the function: `mm` is first assigned
    from the time variable ('hhmmss') but is then overwritten by the month value
    from the date variable ('yyyymmdd'). As a result, the minutes in the output
    datetime will always reflect the month value, not the actual minutes.
    """


    if not obs_type in ['arsa', 'iasi']:
        raise ValueError("obs_type must be either 'arsa' or 'iasi'")
        
    var_name1 = "hhmmss" + obs_type

    hhmmss = data[var_name1].values.astype(int)
    hh = hhmmss // 10000
    mm = (hhmmss % 10000) // 100
    ss = hhmmss % 100


    var_name2 = "yyyymmdd" + obs_type

    yyyymmdd = data[var_name2].values.astype(int)
    yyyy = yyyymmdd // 10000 +2000
    mm = (yyyymmdd % 10000) // 100
    dd = yyyymmdd % 100

    # put them together to a datetime object
    time = np.array(
        [
            np.datetime64(f"{y:04d}-{m:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:02d}")
            for y, m, d, h, mi, s in zip(yyyy, mm, dd, hh, mm, ss)
        ]
    )

    return time

# =============================================================================
# %% arts helpers
# =============================================================================


def fmipdata2arts(atmdata, index):
    """
    Convert FMIP atmospheric data to ARTS format for a specific index.
    This function takes an atmospheric dataset and a specific index, then 
    reformats the data into an xarray Dataset compatible with the ARTS 
    (Atmospheric Radiative Transfer Simulator) format. The altitude array 
    is reversed to ensure ascending order as required by ARTS.
    Parameters
    ----------
    atmdata : xr.Dataset
        Input atmospheric dataset containing multiple variables with dimensions
        including 'idx', and coordinate variables 'lat', 'lon', and 'alt'.
        All variables are expected to have an 'idx' dimension for selection.
    index : int
        The index along the 'idx' dimension to select from the dataset.
    Returns
    -------
    xr.Dataset
        A new xarray Dataset with:
        - Coordinates: 'alt' (ascending), 'lat' (single value), 'lon' (single value)
        - All non-coordinate variables reshaped to dimensions (alt, lat, lon)
        - Altitude values reversed to ascending order compared to input
    Notes
    -----
    - The altitude array is reversed (ascending order) as required by ARTS.
    - Coordinate variables ('alt', 'lat', 'lon') are handled separately and 
      not reshaped into 3D arrays.
    - All other variables are reshaped to (n_alt, 1, 1) to include 
      lat and lon dimensions.
    Examples
    --------
    >>> arts_ds = fmipdata2arts(atmdata, index=0)
    >>> print(arts_ds)
    <xarray.Dataset>
    Dimensions:  (alt: n, lat: 1, lon: 1)
    Coordinates:
      * alt      (alt) float64 ...
      * lat      (lat) float64 ...
      * lon      (lon) float64 ...
    Data variables:
        ...
    """


    #Select for index all variables in the dataset
    variables=[var for var in atmdata.data_vars]

    lat=atmdata['lat'].values[index]
    lon=atmdata['lon'].values[index]
    altitude=atmdata['alt'].values[index][::-1] # reverse the altitude array to have it in ascending order, which is required by ARTS
    ds = xr.Dataset(
        coords={
            "alt": altitude,
            "lat": [lat],
            "lon": [lon],
        },
    )
    for var in variables:
    
        temp = atmdata[var].isel(idx=index).values[::-1]
        
        if var != "alt" and var != "lat" and var != "lon":
            temp = temp.reshape((len(temp), 1, 1))

            ds[var] = xr.DataArray(
                temp,
                dims=["alt", "lat", "lon"],
                coords={
                    "alt": altitude,
                    "lat": [lat],
                    "lon": [lon],
                },
            )

    return ds


def get_species_per_type(cat_folder, type_name):

    type_path=os.path.join(cat_folder, type_name)
    type_files = os.listdir(type_path)
    if type_name == 'predef' or type_name == 'cia':
        type_files = [f.split('.')[0] for f in type_files if f.endswith('.xml')]
    else:
        type_files = [f.split('-')[0] for f in type_files if f.endswith('.xml')]

    #remove duplicates
    type_files = list(set(type_files))

    return type_files

def get_species(cat_folder):

    abs_types=['lines','cia','predef','xsec']

    species_list = {}

    for abs_type in abs_types:
        species = get_species_per_type(cat_folder, abs_type)
        # print(f'{abs_type}: {species}')

        species_list[abs_type] = species

    return species_list

def get_species_from_catalog():

    #get species in the catatlog
    cat_folder = pa.data.parameters.datapath
    cat_folder = [str(folder) for folder in cat_folder if 'arts-cat-data' in str(folder)][0]

    all_species = get_species(cat_folder)

    return all_species

# =============================================================================
# %% atmospheric 
# =============================================================================

M_W, _ = molecular_mass("H2O")
R_W = R / M_W


def T_virtual(T,r_v):
    """
    Calculate the virtual temperature.
    The virtual temperature is the temperature that dry air would need to have 
    to match the density of a sample of moist air at the same pressure.
    Parameters
    ----------
    T : float or array-like
        Actual temperature [K].
    r_v : float or array-like
        Water vapor mixing ratio [kg/kg].
    Returns
    -------
    Tv : float or array-like
        Virtual temperature [K].
    Notes
    -----
    The virtual temperature is calculated using the formula:
        Tv = T * (1 + r_v / epsilon) / (1 + r_v)
    where epsilon = R_D / R_W is the ratio of the gas constant for dry air (R_D) 
    to the gas constant for water vapor (R_W), with typical values of 
    R_D = 287.05 J/(kg·K) and R_W = 461.51 J/(kg·K), giving epsilon ≈ 0.622.
    
    """

    
    epsilon=R_D/R_W
    
    Tv=T*(1+r_v/epsilon)/(1+r_v)
    
    return Tv

def mixingratio2vmr(r,Mx):
    """
    Convert mixing ratio to volume mixing ratio (VMR).
    Parameters
    ----------
    r : float or array-like
        Mixing ratio of the gas [kg/kg].
    Mx : float
        Molar mass of the gas [kg/mol].
    Returns
    -------
    float or array-like
        Volume mixing ratio (VMR) of the gas [mol/mol].
    Notes
    -----
    The conversion uses the molar mass of dry air (M_d) defined as a global
    constant in the module. The formula used is:
        VMR = r / (r + Mx / Md)
    where Md is the molar mass of dry air [kg/mol].
    Examples
    --------
    >>> # Convert water vapor mixing ratio to VMR
    >>> r = 0.01  # kg/kg
    >>> Mx = 18.015  # g/mol (water vapor)
    >>> vmr = mixingratio2vmr(r, Mx)
    """
   
    Md=M_d
    
    return r/(r+Mx/Md)


def vmr2mixingratio(vmr, Mx):
    """
    Convert volume mixing ratio (VMR) to mixing ratio. Inverse of
    :func:`mixingratio2vmr`.
    Parameters
    ----------
    vmr : float or array-like
        Volume mixing ratio of the gas [mol/mol].
    Mx : float
        Molar mass of the gas, in the same units as M_d [kg/mol].
    Returns
    -------
    float or array-like
        Mixing ratio of the gas [kg/kg].
    Notes
    -----
    Inverting VMR = r / (r + Mx / Md) gives
        r = vmr * Mx / (Md * (1 - vmr))
    """

    Md = M_d

    return vmr * Mx / (Md * (1.0 - vmr))


def get_altitude(p, T, r_v,z0=0):
    """
    Calculate altitude at each model level using the hypsometric equation.
    This function integrates the hypsometric equation from the surface upwards
    to compute the geometric height at each model level, accounting for water
    vapor through the virtual temperature.
    Parameters
    ----------
    p : np.ndarray
        Pressure at each model level, shape (n_profiles, n_levels) [Pa].
        The array is expected to start at the Top of Atmosphere (TOA).
    T : np.ndarray
        Temperature at each model level, shape (n_profiles, n_levels) [K].
        The array is expected to start at the Top of Atmosphere (TOA).
    r_v : np.ndarray
        Water vapor mixing ratio at each model level,
        shape (n_profiles, n_levels) [kg/kg].
    z0 : float or np.ndarray, optional
        Surface altitude (starting height for integration) [m].
        Can be a scalar (applied to all profiles) or a 1D array of length
        n_profiles (one value per profile). Default is 0.
    Returns
    -------
    Z : np.ndarray
        Altitude at each model level, shape (n_profiles, n_levels) [m].
    Raises
    ------
    ValueError
        If `z0` is an array and its length does not match the first dimension
        of `p`.
    Notes
    -----
    The altitude is computed using the hypsometric equation:
        dz = (R_D / g) * T_v * d(ln(p))
    where:
        - R_D is the specific gas constant for dry air [J/(kg·K)]
        - g is the gravitational acceleration [m/s²]
        - T_v is the virtual temperature [K], accounting for water vapor
        - d(ln(p)) is the natural log of the pressure ratio between adjacent levels
    The integration starts at the surface (last index) and proceeds upward
    toward the TOA (first index). For the topmost level (level == 1), the
    pressure at the lower boundary is assumed to be 0.1 Pa.
    Examples
    --------
    >>> import numpy as np
    >>> p = np.array([[101325, 85000, 70000]])  # Pressure in Pa
    >>> T = np.array([[288.15, 278.15, 268.15]])  # Temperature in K
    >>> r_v = np.zeros_like(T)  # No water vapor
    >>> Z = get_altitude(p, T, r_v, z0=0)
    """
       

    if not np.isscalar(z0):
        # check that z0 has the samme length as the first dim of p
        if len(z0) != p.shape[0]:
            raise ValueError("z0 must be a scalar or have the same length as the first dimension of p")

    # allocate
    Z = np.zeros_like(T)
    
    Z[:, -1] = z0

    # model levels
    # ml=atm_rawdata.model_level.to_numpy().astype('int')
    ml = np.arange(0, np.size(T, 1)) + 1

    z_h =Z[:,-1]

    # here we need to invert the direction since the data is starting at TOA and we
    # have to start the integration at the surface
    for level in ml[::-1][1:]:

        # compute virtual temperature
        # t_level = T[:, level] * (1.0 + 0.609133 * Q[:, level])
        t_level = T_virtual(T[:, level], r_v[:,level])

        # Calculate pressure differential for half and full level
        if level == 1:
            dlog_p = np.log(p[:,level] / 0.1)
        else:
            dlog_p = np.log(p[:,level] / p[:,level - 1])
        
        z_h = z_h + (t_level * dlog_p)*R_D/g

        Z[:, level - 1] = z_h


    return Z


def refine_pgrid(p, refinement_steps=3, points_added_per_step=2):
    """
    Refine a 1-D pressure grid by inserting interpolated points.

    The grid is refined `refinement_steps` times; at every step
    `points_added_per_step` points are inserted between each pair of
    neighbouring grid points in log10 space, so that adjacent new levels
    are equally spaced in log pressure. Same construction as
    opacity_rule/scripts/opacity_helper_function.refine_pgrid.

    Parameters
    ----------
    p : np.ndarray
        1D monotonically ordered pressure grid [Pa]. Order (ascending or
        descending) is preserved.
    refinement_steps : int, optional
        Number of refinement passes (default: 3).
    points_added_per_step : int, optional
        Number of points inserted between each pair of current grid points
        at every pass (default: 2).

    Returns
    -------
    np.ndarray
        1D refined pressure grid. Has the same start and end points as the
        input, same order. Total grid points after refinement:
        n_final = (n_initial - 1) * (points_added_per_step + 1) **
        refinement_steps + 1, so keep the factors small.
    Examples
    --------
    >>> p = np.array([101325., 70000., 10000.])
    >>> p_ref = refine_pgrid(p, refinement_steps=1, points_added_per_step=3)
    >>> p_ref[0] == p[0] and p_ref[-1] == p[-1]
    True
    """

    log_p = np.log10(p)

    for _ in range(refinement_steps):
        n = len(log_p)
        new_log_p = np.zeros((n - 1) * (points_added_per_step + 1) + 1)
        new_log_p[0] = log_p[0]
        idx = 1
        for j in range(n - 1):
            span = log_p[j:j + 2]
            for k in range(1, points_added_per_step + 1):
                new_log_p[idx] = span[0] + (span[1] - span[0]) \
                    * k / (points_added_per_step + 1)
                idx += 1
            new_log_p[idx] = span[1]
            idx += 1

        log_p = new_log_p

    # pin the endpoints to the exact input values (the log10/10** round trip
    # is not exact in floating point)
    out = np.power(10, log_p)
    out[0] = p[0]
    out[-1] = p[-1]
    return out


def refine_profile(p_prof, t_prof, spc_profiles, z0=0.0,
                   refinement_steps=3, points_added_per_step=2):
    """
    Interpolate / subdivide a single atmospheric profile onto a refined
    pressure grid.

    Mirrors the refinement used in
    opacity_rule/scripts/realistic_atmosphere_plots.py:
    - the pressure grid is refined in log10 space (:func:`refine_pgrid`),
    - temperature and all species VMRs are interpolated with a cubic spline
      in log10(p) space,
    - the altitude profile is recomputed on the new grid from the
      hypsometric equation via :func:`get_altitude` (with the virtual
      temperature, i.e. the refined H2O mixing ratio if present).

    Parameters
    ----------
    p_prof : np.ndarray
        Pressure profile [Pa], 1D, surface-first (descending).
    t_prof : np.ndarray
        Temperature profile [K], 1D, same length as p_prof.
    spc_profiles : dict of np.ndarray
        {species: VMR profile [1]}, each 1D, same length as p_prof.
        If an 'H2O' entry is present, its VMR is used for the virtual
        temperature in :func:`get_altitude`.
    z0 : float, optional
        Surface altitude (starting height for the hypsometric integration)
        [m]. Default is 0.
    refinement_steps : int, optional
        Number of refinement passes, passed to :func:`refine_pgrid`
        (default: 3).
    points_added_per_step : int, optional
        Number of points inserted between each pair of current grid points
        at every pass, passed to :func:`refine_pgrid` (default: 2).

    Returns
    -------
    p_new : np.ndarray
        Refined pressure grid [Pa], descending.
    t_new : np.ndarray
        Temperature on the refined grid [K].
    spc_new : dict of np.ndarray
        {species: refined VMR profile}, may contain tiny negative values
        from spline overshoot (clip them before use, e.g. in
        make_gridded_field).
    z_new : np.ndarray
        Altitude on the refined grid [m], computed from the hypsometric
        equation.
    Examples
    --------
    >>> import numpy as np
    >>> p = np.array([101325., 70000., 10000.])
    >>> t = np.array([285., 260., 230.])
    >>> p_new, t_new, spc_new, z_new = refine_profile(
    ...     p, t, {'CO2': np.full(3, 4e-4)}, z0=0.0)
    >>> len(p_new) > len(p)
    True
    """

    p_prof = np.asarray(p_prof, dtype=float)
    t_prof = np.asarray(t_prof, dtype=float)

    if refinement_steps <= 0:
        raise ValueError("refinement_steps must be a positive integer")

    p_new = refine_pgrid(p_prof, refinement_steps, points_added_per_step)

    # interpolate in log10(p) space. make_interp_spline needs increasing x,
    # hence the ascending view (reversed surface-first = TOA-first) and the
    # flip of the result back to surface-first order.
    log_p = np.log10(p_prof)[::-1]
    log_p_new = np.log10(p_new)[::-1]

    t_new = np.array(make_interp_spline(log_p, t_prof[::-1], k=3)(log_p_new))[::-1]

    spc_new = {}
    for spc, prof in spc_profiles.items():
        spc_new[spc] = np.array(
            make_interp_spline(log_p, np.asarray(prof, dtype=float)[::-1], k=3)
            (log_p_new)
        )[::-1]

    # altitude from the hypsometric equation on the refined grid.
    # get_altitude expects TOA-first order (surface at the last index) and
    # starts the integration at the surface with height z0, hence the flips.
    if 'H2O' in spc_new:
        M_W, _ = molecular_mass('H2O')
        r_v = vmr2mixingratio(spc_new['H2O'], M_W)
    else:
        r_v = np.zeros_like(t_new)

    Z = get_altitude(
        p_new[::-1][None, :], t_new[::-1][None, :], r_v[::-1][None, :],
        z0=z0
    )

    return p_new, t_new, spc_new, Z[0][::-1]


# =============================================================================
# %% Spectral convolution and Planck inversion
# =============================================================================

def convolve_with_srf(spectral_radiance, f_grid, srf_wvn, srf_response):
    """
    Apply spectral response function (SRF) to spectral radiance via convolution.

    Convolves the high-resolution spectral radiance with the SRF to simulate
    instrument filtering. The SRF is defined as (wavenumber offset, response)
    pairs and is assumed to be the same at every channel center. Because the
    simulation grid is uniform in wavenumber, the SRF sampled at the grid
    offsets is one stationary kernel, and the convolution reduces to a single
    1-D filter pass along the frequency axis (vectorized over all remaining
    dimensions).

    Parameters
    ----------
    spectral_radiance : ndarray, shape (N_wvn, ...)
        High-resolution spectral radiance [W m^-2 Hz^-1 sr^-1]. The frequency
        axis must be axis 0; any number of trailing dimensions (columns,
        levels, angles) is convolved independently.
    f_grid : ndarray, shape (N_wvn,)
        Frequency grid [Hz] corresponding to spectral_radiance rows. Must be
        (nearly) uniform in wavenumber, as built by np.linspace in the batch
        scripts.
    srf_wvn : ndarray, shape (N_srf,)
        SRF wavenumber offsets [cm^-1] relative to channel center.
        Should be centered near zero and symmetric if possible.
    srf_response : ndarray, shape (N_srf,)
        SRF response values (normalized, typically integral ≈ 1.0).
        Should be non-negative.

    Returns
    -------
    ndarray
        Convolved spectral radiance, same shape and units as input.

    Notes
    -----
    * The kernel weights are renormalized to unit sum, so the convolution is
      a weighted average in wavenumber space (energy conserving away from the
      grid edges).
    * Grid edges: points within half the SRF width of the array boundary are
      computed with a truncated kernel (zero outside). The radiance batch
      extends the simulation range by the SRF width
      (include_sensor_response=True), so the satellite range itself is
      unaffected.
    """
    from scipy.interpolate import interp1d
    from scipy.ndimage import convolve1d

    # Convert frequency grid to wavenumber for SRF application
    # f [Hz] = c * wvn [cm^-1] * 100, so wvn = f / (c * 100)
    wvn_grid = np.asarray(f_grid, dtype=float) / (scc.c * 100.0)  # cm^-1

    dwvn = np.diff(wvn_grid)
    if not np.allclose(dwvn, dwvn[0], rtol=1e-6):
        raise ValueError("convolve_with_srf requires a uniform wavenumber grid")
    dwvn = float(np.mean(dwvn))

    # Sample the SRF at the grid offsets: kernel[k] = SRF(k * dwvn). On a
    # uniform grid the offsets relative to any center are the same set, so a
    # single stationary kernel replaces the old per-point loop.
    srf_sorted_idx = np.argsort(srf_wvn)
    srf_interp = interp1d(
        srf_wvn[srf_sorted_idx], srf_response[srf_sorted_idx],
        kind='linear', bounds_error=False, fill_value=0.0
    )
    half_width = int(np.ceil(np.ptp(srf_wvn) / 2.0 / abs(dwvn)))
    kernel = srf_interp(np.arange(-half_width, half_width + 1) * dwvn)

    # Unit-sum weights (weighted average in wavenumber space)
    weights = kernel * abs(dwvn)
    if weights.sum() < 1e-10:
        raise ValueError("SRF kernel integrates to zero, cannot convolve")
    weights /= weights.sum()

    return convolve1d(
        np.asarray(spectral_radiance, dtype=float),
        weights, axis=0, mode='constant', cval=0.0,
    )


def radiance2brightness_temp(spectral_radiance, wavenumber):
    """
    Convert spectral radiance to brightness temperature via Planck inversion.

    Analytic inversion of the frequency-domain Planck function (exact, fully
    vectorized):
        L_nu = 2 h nu^3 / c^2 / (exp(h nu / k_B T) - 1)
        T    = h nu / (k_B * ln(1 + 2 h nu^3 / (c^2 L_nu)))
    with nu [Hz] = c * wavenumber. The radiance is per Hz, matching the
    ARTS/DISORT output units of the batch scripts; the previous fsolve loop
    inverted the per-wavenumber Planck function against per-Hz radiance,
    which is inconsistent by a factor of c and gave grossly too low
    brightness temperatures.

    Parameters
    ----------
    spectral_radiance : ndarray, shape (N_wvn, ...)
        Spectral radiance [W m^-2 Hz^-1 sr^-1], frequency along axis 0.
    wavenumber : ndarray, shape (N_wvn,)
        Wavenumber grid [cm^-1] matching axis 0 of spectral_radiance.

    Returns
    -------
    ndarray
        Brightness temperature [K], same shape as spectral_radiance.

    Notes
    -----
    * Negative, NaN or infinite radiances are clipped to a small positive
      floor (1e-30) with a warning; the inverted temperatures there are
      very low.
    * Very small radiances (strong absorption bands, emission from the
      coldest model levels) legitimately invert to low temperatures.
    """
    # Physical constants
    h = scc.h          # Planck constant [J s]
    c = scc.c          # Speed of light [m/s]
    k_B = scc.k        # Boltzmann constant [J/K]

    L = np.asarray(spectral_radiance, dtype=float)

    # Check for problematic radiances
    bad_mask = (L <= 1e-30) | np.isnan(L) | np.isinf(L)
    if np.any(bad_mask):
        print(f"WARNING: Found {np.sum(bad_mask)} negative/NaN/Inf radiances. "
              f"Clipping to 1e-30 W m^-2 Hz^-1 sr^-1.")
        L = np.where(bad_mask, 1e-30, L)

    # Frequency grid [Hz] from wavenumber [cm^-1]
    freq = np.asarray(wavenumber, dtype=float) * 100.0 * c

    # broadcast the per-frequency constants against the radiance array
    shape = (freq.size,) + (1,) * (L.ndim - 1)
    h_nu = (h * freq).reshape(shape)
    c2_nu = (2.0 * h * freq**3 / c**2).reshape(shape)

    return h_nu / (k_B * np.log1p(c2_nu / L))