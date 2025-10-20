from astropy.stats import sigma_clip
#from specutils.manipulation import median_smooth
#from specutils import Spectrum1D
from scipy.ndimage import median_filter as scipy_median_filter
import numpy as np

def a_sigclip(flux, sigma_lower = 0.5, sigma_upper = 2.0):
    """
    Continuum normalization using asymmetric-sigma clipping.
    ------------
    Best For: Spectra with a relatively flat continuum and moderate line density.

    This method estimates the continuum by masking outliers (e.g., absorption lines) 
    using sigma-clipping, then normalizes the spectrum by dividing by the median 
    of the unmasked (continuum) values.
    Asymmetric sigma-clipping allows different thresholds for the lower and upper sides.

    Parameters
    ----------
    flux : array-like
        Flux values of the spectrum.
    sigma_lower : float, optional
        Lower sigma threshold for clipping (default: 0.5).
    sigma_upper : float, optional
        Upper sigma threshold for clipping (default: 2.0).

    Returns
    -------
    norm : ndarray
        The continuum-normalized flux array.
    """

    # Sigma-clip to estimate the continuum value
    res = sigma_clip(flux, sigma_lower=sigma_lower, sigma_upper=sigma_upper, maxiters=None)
    clip_vals = res.compressed()

    # Divide flux by the median estimated continuum value to bring
    # the continuum closer to 1
    if clip_vals.size == 0:
        # fallback to avoid division by zero / empty mean
        loc = np.median(flux)
    else:
        loc = np.median(clip_vals) 
    norm = flux / loc

    return norm


def median_filter(flux, width):
    """
    Continuum normalization using median filtering.
    ------------
    Best For: "Raw" spectra which have not been continuum normalized yet, 
    especially those with broad or sloped continua.

    Estimates the continuum by applying a median filter of specified width 
    to the spectrum, suppressing narrow features (e.g., absorption lines). 
    The spectrum is normalized by dividing by the filtered continuum estimate.

    Parameters
    ----------
    flux : array-like
        Flux values of the spectrum.
    width : int
        Length of the median filter kernel in array indices (pixels). width = len(flux) // 100 or 50 might be a good starting point. 
        If width is too large (and try to process too much memory at once), the kernel may crash.

    Returns
    -------
    norm : ndarray
        The continuum-normalized flux array.
    """
    # Convert input into a Spectrum1D object
    #spectrum = Spectrum1D(flux=flux * np.ones_like(flux), spectral_axis=wavelength)

    # Apply median smoothing
    filt = scipy_median_filter(flux, size=width) # or median_smooth(flux, width)

    # Avoid divide by zero
    filt = np.where(filt == 0, 1, filt)
    norm = flux / filt

    return norm

def contnorm_2stage(flux, width, sigma_lower = 0.5, sigma_upper = 2.0):
    """
    Two-Stage Continuum Normalization Routine
    ------------
    This function combines median filtering and sigma-clipping for robust continuum normalization.

    Stage 1: Median Filter Normalization
        - Estimates the continuum using a median filter (see median_filter).
    Stage 2: k-sigma Clipping Normalization
        - Refines the continuum estimate by masking outliers and normalizing again (see a_sigclip).

    Parameters
    ----------
    flux : array-like
        Flux values of the spectrum.
    width : int
        Length of the median filter kernel in array indices (pixels).
    sigma_lower : float, optional
        Lower sigma threshold for clipping (default: 0.5).
    sigma_upper : float, optional
        Upper sigma threshold for clipping (default: 2.0).

    Returns
    -------
    norm2 : ndarray
        The continuum-normalized spectrum after two-stage normalization.
    """

    # ------- Step 1: Smooth the flux to get the first estimate of the continuum, then norm
    norm = median_filter(flux, width)

    # ------ Step 2: Improve the estimate of the continuum. Sigma clip the normalized flux. Then norm again
    norm2 = a_sigclip(norm, sigma_lower, sigma_upper)

    return norm2