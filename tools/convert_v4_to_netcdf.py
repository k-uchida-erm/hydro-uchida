import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path


def main():
    data_dir = Path(__file__).resolve().parents[1] / 'data' / 'vertical_1d_infiltration'
    out_path = data_dir / 'dataset.nc'

    # model_results.csv: index=time, 99 columns depth grid -> psi grid (pressure head)
    df_model = pd.read_csv(data_dir / 'model_results.csv', index_col=0)
    # depths used in v4: 10.0, 9.9, ..., 0.1 (99 points)
    z_vals = np.array([10.0 - i * 0.1 for i in range(99)], dtype=float)
    t_vals = df_model.index.astype(float).to_numpy()
    psi_grid = df_model.iloc[:, :99].to_numpy(dtype=float)  # shape (Nt, 99) if times as rows
    # Ensure shape is (Nz, Nt)
    psi_grid = psi_grid.T  # (99, Nt)

    # Initial condition ic.csv: columns x,y,z,h0 (h0 likely total head). Store both h0 and psi0.
    df_ic = pd.read_csv(data_dir / 'ic.csv')
    z_ic = df_ic['z'].to_numpy(dtype=float)
    h0_ic = df_ic['h0'].to_numpy(dtype=float)
    psi0_ic = h0_ic - z_ic

    # Observations obs.csv: columns z,t,h (here h is actually psi in v4). Store both.
    df_obs = pd.read_csv(data_dir / 'obs.csv')
    z_obs = df_obs['z'].to_numpy(dtype=float)
    t_obs = df_obs['t'].to_numpy(dtype=float)
    psi_obs = df_obs['h'].to_numpy(dtype=float)
    h_obs = psi_obs + z_obs

    # Boundary conditions bc.csv: keep as table variables
    df_bc = pd.read_csv(data_dir / 'bc.csv')
    # Normalize columns presence
    if 'nx' not in df_bc.columns:
        df_bc['nx'] = 0.0
    if 'ny' not in df_bc.columns:
        df_bc['ny'] = 0.0
    if 'nz' not in df_bc.columns:
        df_bc['nz'] = 0.0
    for c in ['x', 'y', 'z', 't']:
        if c not in df_bc.columns:
            df_bc[c] = 0.0

    # Build xarray Dataset
    ds = xr.Dataset(
        {
            'psi': (('z', 't'), psi_grid),
            'ic_h0': (('z_ic',), h0_ic),
            'ic_psi0': (('z_ic',), psi0_ic),
            'obs_z': (('n_obs',), z_obs),
            'obs_t': (('n_obs',), t_obs),
            'obs_psi': (('n_obs',), psi_obs),
            'obs_h': (('n_obs',), h_obs),
            'bc_x': (('n_bc',), df_bc['x'].to_numpy(dtype=float)),
            'bc_y': (('n_bc',), df_bc['y'].to_numpy(dtype=float)),
            'bc_z': (('n_bc',), df_bc['z'].to_numpy(dtype=float)),
            'bc_t': (('n_bc',), df_bc['t'].to_numpy(dtype=float) if 't' in df_bc.columns else np.zeros(len(df_bc))),
            'bc_value': (('n_bc',), df_bc['value'].to_numpy(dtype=float)),
            'bc_nx': (('n_bc',), df_bc['nx'].to_numpy(dtype=float)),
            'bc_ny': (('n_bc',), df_bc['ny'].to_numpy(dtype=float)),
            'bc_nz': (('n_bc',), df_bc['nz'].to_numpy(dtype=float)),
        },
        coords={
            'z': (('z',), z_vals, {'units': 'cm'}),
            't': (('t',), t_vals, {'units': 'h'}),
            'z_ic': (('z_ic',), z_ic, {'units': 'cm'}),
            'n_obs': np.arange(len(df_obs)),
            'n_bc': np.arange(len(df_bc)),
        },
        attrs={
            'convention': 'custom-hydro-1d-v1',
            'primary_quantity': 'psi',
            'psi_units': 'cm',
            'z_units': 'cm',
            't_units': 'h',
            'note': 'psi is pressure head; total head h = psi + z',
        }
    )

    # Add simple string columns for bc/obs types if present
    if 'type' in df_bc.columns:
        ds['bc_type'] = (('n_bc',), df_bc['type'].astype(str))
    if 'location' in df_bc.columns:
        ds['bc_location'] = (('n_bc',), df_bc['location'].astype(str))

    ds.to_netcdf(out_path)
    print(f'saved: {out_path}')


if __name__ == '__main__':
    main()


