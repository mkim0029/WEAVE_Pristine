from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline
from astropy.stats import sigma_clip
#from specutils.manipulation import median_smooth
#from specutils import Spectrum1D
from scipy.ndimage import median_filter as scipy_median_filter
import numpy as np

def add_noise(flux, noise=0.07, inplace=False, seed=2025):
    """
    Add Gaussian noise to a flux array.
    """
    flux = np.asarray(flux)
    if not inplace:
        flux = flux.copy()

    rng = np.random.default_rng(seed)
    noise_factor = noise * np.median(flux)
    flux += noise_factor * rng.normal(loc=0.0, scale=1.0, size=flux.shape)
    return flux

def robust_polyfit_huber(flux, wavelength, degree=3, sigma_lower=None, sigma_upper=None, huber_f=1.0):
    """
    Robust polynomial continuum fitting using least squares with Huber loss.
    ------------
    This method fits a polynomial to the spectrum by minimizing a robust Huber loss function, 
    which behaves like least squares for small residuals but like absolute error for large residuals, 
    reducing the influence of outliers (e.g., deep lines, cosmic rays). 
    Polynomial fitting is possible because the continuum is assumed to be a smooth, slowly-varying function, 
    and least squares finds the best-fit coefficients by minimizing the sum of (robust) residuals. 
    Optionally, sigma clipping can be applied before fitting to further mask outliers.

    Parameters
    ----------
    flux : array-like
        Flux values of the spectrum.
    wavelength : array-like
        Wavelength values (in Å) corresponding to flux.
    degree : int, optional
        Degree of the polynomial to fit (default: 3).
    sigma_lower : float or None, optional
        Lower sigma threshold for clipping. If None, no clipping.
    sigma_upper : float or None, optional
        Upper sigma threshold for clipping. If None, no clipping.
    huber_f : float, optional
        Huber loss parameter (default: 1.0). Lower values are more robust to outliers.

    Returns
    -------
    norm : ndarray
        The continuum-normalized flux array.
    continuum : ndarray
        The fitted polynomial continuum.
    """
    flux = np.asarray(flux)
    wavelength = np.asarray(wavelength)
    # Optionally sigma-clip to mask outliers
    if sigma_lower is not None and sigma_upper is not None:
        res = sigma_clip(flux, sigma_lower=sigma_lower, sigma_upper=sigma_upper, maxiters=None)
        mask = ~res.mask
    else:
        mask = np.ones_like(flux, dtype=bool)
    n_fit = np.sum(mask)
    print(f"Number of points used for robust fit: {n_fit}")
    # Normalize wavelength for numerical stability
    wl_norm = 2 * (wavelength - wavelength.min()) / (wavelength.max() - wavelength.min()) - 1
    wl_fit = wl_norm[mask]
    flux_fit = flux[mask]
    # Polynomial model function
    def poly_model(params, x):
        return np.polyval(params, x)
    # Residuals for least squares
    def residuals(params, x, y):
        return poly_model(params, x) - y
    # Initial guess: standard least squares fit
    p0 = np.polyfit(wl_fit, flux_fit, degree)
    # Robust least squares fit with Huber loss
    result = least_squares(residuals, p0, args=(wl_fit, flux_fit), loss='huber', f_scale=huber_f)
    params = result.x
    continuum = np.polyval(params, wl_norm)
    continuum = np.where(continuum == 0, 1, continuum)
    norm = flux / continuum
    return norm, continuum

def asym_sigclip_spline_fit(flux, wavelength, sigma_lower=0.5, sigma_upper=2.0, spline_s=1e-2):
    """
    Continuum normalization using asymmetric sigma clipping and cubic spline fitting.
    ------------
    This method first masks outliers (such as absorption lines) using asymmetric sigma clipping, 
    then fits a cubic spline—a piecewise third-degree polynomial with continuous first and second 
    derivatives—to the remaining continuum points. Cubic splines are flexible and avoid the oscillations 
    of high-degree polynomials, making them ideal for modeling broad or complex continuum shapes in spectra. 
    The fitted spline is evaluated at all wavelengths and used to normalize the spectrum.

    Parameters
    ----------
    flux : array-like
        Flux values of the spectrum.
    wavelength : array-like
        Wavelength values (in Å) corresponding to flux.
    sigma_lower : float, optional
        Lower sigma threshold for clipping (default: 0.5).
    sigma_upper : float, optional
        Upper sigma threshold for clipping (default: 2.0).
    spline_s : float, optional
        Smoothing factor for UnivariateSpline (default: 1e-2). Increase for smoother fit, decrease for tighter fit.

    Returns
    -------
    norm : ndarray
        The continuum-normalized flux array.
    continuum : ndarray
        The fitted cubic spline continuum.
    """
    flux = np.asarray(flux)
    wavelength = np.asarray(wavelength)
    # Sigma-clip to mask outliers (e.g., absorption lines)
    res = sigma_clip(flux, sigma_lower=sigma_lower, sigma_upper=sigma_upper, maxiters=None)
    mask = ~res.mask  # True for continuum points, False for clipped outliers
    n_unmasked = np.sum(mask)
    print(f"Number of unmasked (continuum) points: {n_unmasked}")
    if n_unmasked < 10:
        print(f"Warning: Too few unmasked points ({n_unmasked}) for stable spline fit. Fit may be unstable.")
    # Fit cubic spline to unmasked (continuum) points
    spline = UnivariateSpline(wavelength[mask], flux[mask], k=3, s=spline_s)
    continuum = spline(wavelength)
    continuum = np.where(continuum == 0, 1, continuum)
    norm = flux / continuum
    return norm, continuum

def local_asymmetric_sigclip(flux, wavelength, window_width=10.0, sigma_lower=0.5, sigma_upper=2.0):
    """
    Local asymmetric k-sigma clipping continuum normalization in moving wavelength windows.
    ------------
    The spectrum is normalized locally by estimating the continuum in moving windows using 
    asymmetric sigma clipping, which masks outliers and adapts to local continuum variations. 
    This is effective for crowded or variable spectra.

    Parameters
    ----------
    flux : array-like
        Flux values of the spectrum.
    wavelength : array-like
        Wavelength values (in Å) corresponding to flux.
    window_width : float, optional
        Width of the moving window in Å (default: 10.0).
    sigma_lower : float, optional
        Lower sigma threshold for clipping (default: 0.5).
    sigma_upper : float, optional
        Upper sigma threshold for clipping (default: 2.0).

    Returns
    -------
    norm : ndarray
        Locally continuum-normalized flux array.
    """
    # Ensure input arrays are numpy arrays
    flux = np.asarray(flux)
    wavelength = np.asarray(wavelength)
    norm = np.empty_like(flux)
    n = len(flux)
    for i in range(n):
        # Define the center of the window for the current wavelength point
        wl_center = wavelength[i]
        wl_min = wl_center - window_width/2
        wl_max = wl_center + window_width/2
        # Find indices of points within the window
        idx = np.where((wavelength >= wl_min) & (wavelength <= wl_max))[0]
        if idx.size == 0:
            # If no points in window, fallback to original flux
            norm[i] = flux[i]
            continue
        window_flux = flux[idx]
        # Apply asymmetric sigma clipping to window flux
        res = sigma_clip(window_flux, sigma_lower=sigma_lower, sigma_upper=sigma_upper, maxiters=None)
        clip_vals = res.compressed()
        # Estimate local continuum as median of clipped values
        if clip_vals.size == 0:
            loc = np.median(window_flux)
        else:
            loc = np.median(clip_vals)
        # Normalize current flux value by local continuum estimate
        norm[i] = flux[i] / loc if loc != 0 else flux[i]
    return norm

def asym_sigclip_legendre_fit(flux, wavelength, degree=3, sigma_lower=0.5, sigma_upper=2.0):
    """
    Continuum normalization using asymmetric sigma clipping and Legendre polynomial fitting.
    ------------
    Outliers are masked using asymmetric sigma clipping, then a Legendre polynomial is fit to 
    the continuum points to model broad or sloped continuum shapes. The spectrum is normalized by 
    dividing by the polynomial fit.

    Parameters
    ----------
    flux : array-like
        Flux values of the spectrum.
    wavelength : array-like
        Wavelength values (in Å) corresponding to flux.
    degree : int, optional
        Degree of the Legendre polynomial to fit (default: 3).
    sigma_lower : float, optional
        Lower sigma threshold for clipping (default: 0.5).
    sigma_upper : float, optional
        Upper sigma threshold for clipping (default: 2.0).

    Returns
    -------
    norm : ndarray
        The continuum-normalized flux array.
    continuum : ndarray
        The fitted Legendre polynomial continuum.
    """
    # Ensure input arrays are numpy arrays
    flux = np.asarray(flux)
    wavelength = np.asarray(wavelength)
    # Sigma-clip to mask outliers (e.g., absorption lines)
    res = sigma_clip(flux, sigma_lower=sigma_lower, sigma_upper=sigma_upper, maxiters=None)
    mask = ~res.mask  # True for continuum points, False for clipped outliers
    n_unmasked = np.sum(mask)
    print(f"Number of unmasked (continuum) points: {n_unmasked}")
    if n_unmasked < degree + 2:
        print(f"Warning: Too few unmasked points ({n_unmasked}) for Legendre fit of degree {degree}. Fit may be unstable.")
    # Normalize wavelength to [-1, 1] for Legendre polynomial fitting
    wl_norm = 2 * (wavelength - wavelength.min()) / (wavelength.max() - wavelength.min()) - 1
    # Fit Legendre polynomial to unmasked (continuum) points
    coeffs = np.polynomial.legendre.legfit(wl_norm[mask], flux[mask], degree)
    # Evaluate fitted polynomial at all wavelength points
    continuum = np.polynomial.legendre.legval(wl_norm, coeffs)
    # Avoid divide by zero in normalization
    continuum = np.where(continuum == 0, 1, continuum)
    # Normalize flux by fitted continuum
    norm = flux / continuum
    return norm, continuum

def a_sigclip(flux, sigma_lower = 0.5, sigma_upper = 2.0):
    """
    Continuum normalization using asymmetric-sigma clipping.
    ------------
    The continuum is estimated by masking outliers with asymmetric sigma clipping, then 
    normalizing the spectrum by the median of the remaining continuum values. 
    This technique is robust to moderate line density and allows different thresholds for each side.

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
    A median filter is applied to the spectrum to estimate the continuum by suppressing narrow 
    features and noise. The spectrum is then normalized by dividing by the filtered continuum estimate, 
    which is effective for broad or sloped continua.

    Parameters
    ----------
    flux : array-like
        A median filter is applied to the spectrum to estimate the continuum by suppressing narrow features and noise. The spectrum is then normalized by dividing by the filtered continuum estimate, which is effective for broad or sloped continua.
    width : int
        Length of the median filter kernel in array indices (pixels). 
        width = len(flux) // 2 may crash the kernel (oom). 
        // 100 or 50 will flatten the spectrum more aggressively 
        (but affects broad features).
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
    This approach first applies a median filter to estimate and normalize the continuum, 
    then applies sigma clipping to the result to mask outliers and normalizes again. 
    Median filtering provides local smoothing and removes narrow features, 
    while sigma clipping robustly rejects outliers and absorption lines. 
    By combining these two techniques, this method improves continuum estimation for raw 
    or complex spectra, especially when neither method alone is sufficient.

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