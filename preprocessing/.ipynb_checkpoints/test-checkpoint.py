from hdf5_spectrum_reader import HDF5SpectrumReader
import cont_norm

# Load the spectrum data
spectrum_file = "home/minjihk/projects/def-sfabbro/minjihk/WEAVE_Pristine/data/weave_nlte_grids.h5"
reader = HDF5SpectrumReader(spectrum_file)

# Extract wavelength and flux data for a specific spectrum (e.g., first one)
spectrum_index = 0  # Change this index to plot a different spectrum
wavelength, flux = reader.get_wavelength_flux(spectrum_index)
info = reader.get_spectrum_info(spectrum_index)
params = info['stellar_parameters']
headers = {k: params.get(k, 'N/A') for k in ['teff', 'log_g', 'metallicity']}

# median filtering
flux2 = cont_norm.median_filter(flux, len(flux)//2)
flux3 = cont_norm.contnorm_2stage(flux, len(flux)//2)

# write output 
with open('../data/test.txt', 'w') as f:
    for i in range(len(flux2)):
        f.write(f"{flux2[i],flux3[i]}\n")