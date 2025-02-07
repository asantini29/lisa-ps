import jax
import jax.numpy as jnp
from functools import partial

import numpy as np

jax.config.update("jax_enable_x64", True)

# ----------------------------------------- #
            # smooting routines #
# ----------------------------------------- #

def log_bin(freqs, power, bins_per_decade=10):
    """
    JAX-optimized logarithmic binning for frequency data.
    
    Args:
        freqs: Array of frequencies
        power: Array of power values
        bins_per_decade: Number of bins per decade of frequency
    
    Returns:
        tuple: (bin_centers, binned_power)
    """
    log_freqs = jnp.log10(freqs)
    
    # Create bins
    bins = jnp.linspace(
        log_freqs.min(), 
        log_freqs.max(),
        ((log_freqs.max() - log_freqs.min()) * bins_per_decade).astype(jnp.int32)
    )
    
    # Digitize using searchsorted
    digits = jnp.searchsorted(bins, log_freqs)

    #compute bin counts
    bin_counts = jnp.bincount(digits, minlength=len(bins))
    
    # Calculate bin centers
    bin_centers = 10 ** ((bins[1:] + bins[:-1]) / 2)
    
    # Compute means for each bin using a vectorized approach
    @jax.vmap
    def bin_mean(i):
        mask = digits == i
        return jnp.where(
            mask.sum() > 0,
            (power * mask[:, None]).sum() / mask.sum(),
            0.0
        )
    
    binned_power = bin_mean(jnp.arange(1, len(bins)))
    
    return bin_centers, binned_power, bin_counts

def adaptive_log_bin(freqs, power, f_min=None, f_max=None, min_bpd=5, max_bpd=50, order='increasing'):
    """
    Adaptive logarithmic binning with varying bins per decade.
    
    Args:
        freqs: Array of frequencies (must be sorted)
        power: Array of power values
        f_min, f_max: Frequency range of interest
        min_bpd, max_bpd: Minimum and maximum bins per decade
        order: Order of bin density variation ('increasing' or 'decreasing')

    Returns:
        tuple: (bin_centers, binned_power, bin_statistics)
    """
    # Mask frequencies in range of interest

    f_min = f_min or freqs.min()
    f_max = f_max or freqs.max()

    mask = (freqs >= f_min) & (freqs <= f_max)
    freqs = freqs[mask]
    power = power[mask, ...] 
    
    # Convert to log space
    log_freqs = jnp.log10(freqs)
    log_fmin = jnp.log10(f_min)
    log_fmax = jnp.log10(f_max)
    
    # Create adaptive bin edges
    # We'll use a quadratic scaling for bins per decade
    decades = log_fmax - log_fmin
    n_sections = 50  # number of sections to divide the range into
    
    # Create array of local bins per decade that varies with frequency
    positions = jnp.linspace(0, 1, n_sections)
    if order == 'increasing':
        local_bpd = min_bpd + (max_bpd - min_bpd) * positions**2
    elif order == 'decreasing':
        local_bpd = max_bpd - (max_bpd - min_bpd) * positions**2
    else:
        raise ValueError("Invalid order argument")
    
    # Create bin edges with varying density
    bin_edges = []
    current_freq = log_fmin
    
    for i in range(n_sections):
        # Calculate local decade fraction
        decade_fraction = decades / n_sections
        # Number of bins in this section
        n_bins = int(np.ceil(local_bpd[i] * decade_fraction))
        # Create local bins
        local_bins = jnp.linspace(
            current_freq,
            current_freq + decade_fraction,
            n_bins + 1
        )[:-1]  # exclude last point to avoid duplicates
        bin_edges.append(local_bins)
        current_freq += decade_fraction
    
    # Add final edge
    bin_edges.append(jnp.array([log_fmax]))
    bin_edges = jnp.concatenate(bin_edges)
    
    # Digitize using searchsorted
    digits = jnp.searchsorted(bin_edges, log_freqs)
    
    # Calculate bin centers
    bin_centers = 10 ** ((bin_edges[1:] + bin_edges[:-1]) / 2)
    
    # Compute means for each bin using a vectorized approach
    @jax.vmap
    def bin_mean(i):
        mask = digits == i
        # Add extra dimension(s) to mask to match the length of power.shape

        return jnp.where(
            mask.sum() > 0,
            jnp.sum(power * mask.reshape(mask.shape + (1,) * (len(power.shape) - 1)), axis=0) / mask.sum(),
            0.0
        )
    
    # do this in chunks to avoid memory issues
    
    all_bins = jnp.arange(1, len(bin_edges)) 
    chunk_size = 100

    binned_power = jnp.concatenate([bin_mean(all_bins[i:i+chunk_size]) for i in range(0, len(all_bins), chunk_size)], axis=0)

    
    return bin_centers, binned_power, get_bin_statistics(freqs, bin_edges)['points_per_bin']

def get_bin_statistics(freqs, log_bin_edges):
    """
    Compute statistics about the binning to verify its behavior
    """
    log_freqs = jnp.log10(freqs)
    digits = jnp.searchsorted(log_bin_edges, log_freqs)

    
    
    # Count points per bin
    unique_digits, counts = jnp.unique(digits, return_counts=True)
    
    # Compute bin widths in Hz
    bin_widths = 10**log_bin_edges[1:] - 10**log_bin_edges[:-1]
    
    return {
        'points_per_bin': counts,
        'bin_widths': bin_widths,
        'mean_points': counts.mean(),
        'std_points': counts.std(),
        'bin_edges_hz': 10**log_bin_edges,
    }

def average_periodogram_static(freqs, power, f_min=None, f_max=None, f_seg=1e-5):
        """
        Average the periodogram matrix over segments. Snippet credits: Nikolaos Karnesis.
        
        Parameters
        ----------
        freqs : array
            The frequencies of the periodogram matrix.
         power : array
            The periodogram matrix to average.
        f_seg : float or array_like
            The segment frequency or an array of bin edges.
            
        Returns
        -------
        freqs_h : array
            The frequencies where the averaged periodogram is computed.
        power_avg : array
            The averaged periodogram matrix.
        segment_sizes : array
            The sizes of the frequency segments.
        """

        f_min = f_min or freqs.min()
        f_max = f_max or freqs.max()

        mask = (freqs >= f_min) & (freqs <= f_max)
        freqs = freqs[mask]
    
        power = power[mask.reshape(mask.shape + (1,) * (len(power.shape) - 1))] #power can be a 2D or 3D array
    

        df = (freqs[1] - freqs[0])
        if isinstance(f_seg, float):
            # Smoothing bandwidth
            bandwidth = int(f_seg / df)
            # Segment frequencies
            f_seg_arr = freqs[0::bandwidth]
            # Add the last frequency if it is not included
            if freqs[-1] not in f_seg_arr:
                f_seg_arr = jnp.concatenate((f_seg_arr, jnp.atleast_1d(freqs[-1])))
        elif hasattr(f_seg, '__array__') or isinstance(f_seg, list):
            f_seg_arr = jnp.asarray(f_seg)
        else:
            raise TypeError("f0 should be a float or array_like")
        
        # Number of segments
        n_seg = len(f_seg_arr)
        # Indices of the segment bounds
        i_seg = np.round(f_seg_arr / df).astype(int)
        # Sizes of all intervals
        segment_sizes = i_seg[1:] - i_seg[:-1]
        # Middle frequencies
        freqs_h = (f_seg_arr[:-1] + f_seg_arr[1:]) / 2.0

        # Compute the averages over each segment
        power_avg = jnp.array(
            [jnp.sum(power[i_seg[j]:i_seg[j+1]], axis=0) / segment_sizes[j]
            for j in range(n_seg-1)], dtype=power.dtype)
        
        return freqs_h, power_avg, segment_sizes