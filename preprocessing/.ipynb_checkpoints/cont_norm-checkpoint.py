from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline
from astropy.stats import sigma_clip
#from specutils.manipulation import median_smooth
#from specutils import Spectrum1D
from scipy.ndimage import median_filter as scipy_median_filter
import numpy as np

def generate_noise(shape, noise=0.05, seed=2025):
    """
    Generate Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    #noise_factor = noise * np.median(flux)
    #flux += noise_factor * rng.normal(loc=0.0, scale=1.0, size=flux.shape)
    noise_vector = noise * rng.normal(loc=0.0, scale=1.0, size=shape)
    return noise_vector

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
    # Convert inputs to NumPy arrays
    # flux = np.asarray(flux)
    # wavelength = np.asarray(wavelength)
    
    # Optionally sigma-clip to mask outliers
    # not used atm
    if sigma_lower is not None and sigma_upper is not None:
        res = sigma_clip(flux, sigma_lower=sigma_lower, sigma_upper=sigma_upper, maxiters=None)
        mask = ~res.mask
    else:
        mask = np.ones_like(flux, dtype=bool) # no clipping → use all points
        
    #n_fit = np.sum(mask)
    #print(f"Number of points used for robust fit: {n_fit}")
    
    # Rescales wavelength to the interval [-1, 1], improves numerical stability for polynomial fitting
    wl_norm = 2 * (wavelength - wavelength.min()) / (wavelength.max() - wavelength.min()) - 1

    # apply the mask
    wl_fit = wl_norm[mask] 
    flux_fit = flux[mask]
    
    # Polynomial model function: evaluates a polynomial with coeff. given by params at x  
    def poly_model(params, x):
        return np.polyval(params, x)
        
    # Residuals for least squares
    def residuals(params, x, y):
        return poly_model(params, x) - y
        
    # Initial guess: standard least squares fit
    p0 = np.polyfit(wl_fit, flux_fit, degree)
    
    # Robust least squares fit with Huber loss: minimize the sum of the Huber loss of residuals
    # Residuals with |r| <= huber_f are treated quadratically; larger residuals are treated linearly.
    result = least_squares(residuals, p0, args=(wl_fit, flux_fit), loss='huber', f_scale=huber_f)

    # Extract fitted polynomial coefficients and evaluate the polynomial on the full wavelength grid
    params = result.x
    continuum = np.polyval(params, wl_norm)
    
    # Guard against division by zero in case continuum has exact zeros (unlikely but safe)
    continuum = np.where(continuum == 0, 1, continuum)
    
    norm = flux / continuum # Final continuum-normalized spectrum
    
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
    Local Asymmetric Sigma Clipping Continuum Normalization (Moving Window)
    ----------------------------------------------------------------------
    This algorithm normalizes a spectrum by estimating the local continuum in a moving window
    around each pixel. Within each window, it applies asymmetric sigma clipping to mask outliers
    (e.g., absorption lines, cosmic rays) and then estimates the continuum as the median of the
    remaining (unmasked) flux values. The current pixel is divided by this local continuum estimate.

    - Asymmetric sigma clipping allows different thresholds for lower and upper outliers, making it
      robust to deep absorption features while ignoring occasional emission spikes.
    - The window slides across the spectrum, adapting to local continuum variations and handling
      crowded or variable spectra well.
    - Uses an optimized searchsorted approach for O(n log n) window finding (much faster than naive O(n²)).

    Parameters
    ----------
    flux : array-like
        Flux values of the spectrum.
    wavelength : array-like
        Wavelength values (in Å) corresponding to flux. Must be sorted in ascending order.
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
    n = len(flux)
    norm = np.empty_like(flux)
    
    for i in range(n):
        # Define window boundaries
        wl_center = wavelength[i]
        wl_min = wl_center - window_width/2
        wl_max = wl_center + window_width/2
        
        # Use searchsorted for O(log n) window finding (assumes sorted wavelength)
        left_idx = np.searchsorted(wavelength, wl_min, side='left')
        right_idx = np.searchsorted(wavelength, wl_max, side='right')
        
        # Extract window flux
        window_flux = flux[left_idx:right_idx]
        
        if window_flux.size == 0:
            # If no points in window, fallback to original flux
            norm[i] = flux[i]
            continue
            
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

def local_asymmetric_sigclip_super_fast(flux, wavelength, window_width=10.0, sigma_lower=0.5, sigma_upper=2.0, stride=10, continuum_percentile=None, smooth_continuum=False, robust_percentile=False):
    """
    Ultra-Fast Local Continuum Normalization (Sparse Grid + Interpolation)
    ---------------------------------------------------------------------
    This algorithm normalizes a spectrum by estimating the continuum only at sparse grid points
    (every `stride` pixels), then interpolating these estimates across the entire spectrum.
    At each sparse grid point, the local continuum is estimated using a robust statistic:
    - By default, the mean of the interquartile range (middle 50% of sorted flux values).
    - If `continuum_percentile` is set, uses the specified upper percentile (e.g., 90th, 98th) to
      bias the continuum estimate toward higher flux values, which helps preserve broad absorption features.
    - If `robust_percentile` is True, averages the top N% of flux values (instead of a single percentile)
      for improved noise robustness.

    The sparse continuum estimates are then linearly interpolated to all pixels. Optionally, a light
    median filter can be applied to the interpolated continuum to further reduce artifacts.

    This method is extremely fast and works well for batch processing, but may introduce small artifacts
    (e.g., staircase effects, slight continuum deviations) in very sparse or low-resolution spectra.

    Parameters
    ----------
    flux : array-like
        Flux values of the spectrum.
    wavelength : array-like
        Wavelength values (in Å) corresponding to flux.
    window_width : float, optional
        Width of the local window in Å (default: 10.0).
    sigma_lower : float, optional
        Lower sigma threshold for clipping (default: 0.5).
    sigma_upper : float, optional
        Upper sigma threshold for clipping (default: 2.0).
    stride : int, optional
        Number of pixels between sparse continuum estimates (default: 10).
    continuum_percentile : float, optional
        If provided (e.g., 80-98), use this percentile as continuum estimate instead of interquartile mean.
    smooth_continuum : bool, optional
        If True, apply light median filtering to interpolated continuum to reduce artifacts.
    robust_percentile : bool, optional
        If True and continuum_percentile is set, use mean of top percentile values instead of single percentile.

    Returns
    -------
    norm : ndarray
        Continuum-normalized flux array.
    """
    flux = np.asarray(flux)
    wavelength = np.asarray(wavelength)
    n = len(flux)
    
    # Compute continuum estimates at every 'stride' points
    sparse_indices = np.arange(0, n, stride)
    sparse_continuum = np.empty(len(sparse_indices))
    
    delta_wl = np.median(np.diff(wavelength))
    window_pixels = int(window_width / delta_wl)
    
    for j, i in enumerate(sparse_indices):
        left_idx = max(0, i - window_pixels // 2)
        right_idx = min(n, i + window_pixels // 2 + 1)
        
        window_flux = flux[left_idx:right_idx]
        
        # Simple robust estimator
        if window_flux.size > 3:
            if continuum_percentile is not None:
                if robust_percentile and window_flux.size > 10:
                    # Use mean of top few percent instead of single percentile (more robust to noise)
                    n_top = max(2, int(window_flux.size * (100 - continuum_percentile) / 100))
                    sorted_flux = np.sort(window_flux)
                    sparse_continuum[j] = np.mean(sorted_flux[-n_top:])
                else:
                    sparse_continuum[j] = np.percentile(window_flux, continuum_percentile)
            else:
                sorted_flux = np.sort(window_flux)
                # Use interquartile range for robust estimate
                q1_idx = len(sorted_flux) // 4
                q3_idx = 3 * len(sorted_flux) // 4
                sparse_continuum[j] = np.mean(sorted_flux[q1_idx:q3_idx])
        else:
            if continuum_percentile is not None:
                sparse_continuum[j] = np.percentile(window_flux, continuum_percentile) if window_flux.size > 0 else flux[i]
            else:
                sparse_continuum[j] = np.median(window_flux) if window_flux.size > 0 else flux[i]
    
    # Interpolate continuum to all points
    continuum = np.interp(np.arange(n), sparse_indices, sparse_continuum)
    
    # Optional smoothing to reduce interpolation artifacts
    if smooth_continuum:
        # Light median filter to smooth without losing broad structure
        smooth_width = max(3, len(continuum) // 1000)  # Adaptive width
        continuum = scipy_median_filter(continuum, size=smooth_width)
    
    # Normalize
    continuum = np.where(continuum == 0, 1, continuum)
    norm = flux / continuum
    
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