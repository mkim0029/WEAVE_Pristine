import h5py

with h5py.File('../data/weave_nlte_grids.h5', 'r') as f:
    grid = f['interpolated/wavelength_grid'][:]
    flux = f['interpolated/flux'][:]
    print("Common grid shape:", grid.shape)
    print("Interpolated flux shape:", flux.shape)
    if flux.shape[1] == grid.shape[0]:
        print("All spectra are rescaled to the same wavelength grid.")
    else:
        print("Mismatch: spectra are not all on the same grid!")

with h5py.File('weave_nlte_grids.h5', 'r') as f:
    orig_waves = f['spectra/wavelength'][:]
    print("Original wavelength array shape:", orig_waves.shape)
    # Check if all rows are identical
    identical = all((orig_waves[0] == row).all() for row in orig_waves)
    print("All original wavelength arrays identical?", identical)