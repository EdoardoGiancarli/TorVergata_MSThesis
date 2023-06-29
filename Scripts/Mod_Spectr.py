# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 19:08:28 2023

@author: Edoardo Giancarli

Ref:
    [1] P. Virtanen et al, "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python" (2020)
    
        url (SciPy): https://scipy.org/
        url (scipy.signal): https://docs.scipy.org/doc/scipy/reference/signal.html
        url (Github doc): https://github.com/scipy/scipy/blob/v1.10.1/scipy/signal/_spectral_py.py#L617-L786
"""

#### ModuleMSThesis - Signals Spectrograms from SciPy library - version 1 ###############################################

####   libraries   #####

import numpy as np
from scipy import fft as sp_fft
# from . import _signaltools
from scipy.signal import get_window, signaltools
# from ._arraytools import const_ext, even_ext, odd_ext, zero_ext
import warnings

####    content    #####

# _spectral_helper
#
# _fft_helper
#
# triage_segment
#
# spectrogram

####    codes    #####

























#################################################################################################


def _triage_segments(window, nperseg, input_length):
    """
    Parses window and nperseg arguments for spectrogram and _spectral_helper.
    This is a helper function, not meant to be called externally.
    Parameters
    ----------
    window : string, tuple, or ndarray
        If window is specified by a string or tuple and nperseg is not
        specified, nperseg is set to the default of 256 and returns a window of
        that length.
        If instead the window is array_like and nperseg is not specified, then
        nperseg is set to the length of the window. A ValueError is raised if
        the user supplies both an array_like window and a value for nperseg but
        nperseg does not equal the length of the window.
    nperseg : int
        Length of each segment
    input_length: int
        Length of input signal, i.e. x.shape[-1]. Used to test for errors.
    Returns
    -------
    win : ndarray
        window. If function was called with string or tuple than this will hold
        the actual array used as a window.
    nperseg : int
        Length of each segment. If window is str or tuple, nperseg is set to
        256. If window is array_like, nperseg is set to the length of the
        window.
    """
    # parse window; if array like, then set nperseg = win.shape
    if isinstance(window, str) or isinstance(window, tuple):
        # if nperseg not specified
        if nperseg is None:
            nperseg = 256  # then change to default
        if nperseg > input_length:
            warnings.warn('nperseg = {0:d} is greater than input length '
                          ' = {1:d}, using nperseg = {1:d}'
                          .format(nperseg, input_length))
            nperseg = input_length
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if input_length < win.shape[-1]:
            raise ValueError('window is longer than input signal')
        if nperseg is None:
            nperseg = win.shape[0]
        elif nperseg is not None:
            if nperseg != win.shape[0]:
                raise ValueError("value specified for nperseg is different"
                                 " from length of window")
    return win, nperseg


#################################################################################################


def spectrogram(x, fs=1.0, window=('tukey', .25), nperseg=None, noverlap=None,
                nfft=None, detrend='constant', return_onesided=True,
                scaling='density', axis=-1, mode='psd'):
    """Compute a spectrogram with consecutive Fourier transforms.
    Spectrograms can be used as a way of visualizing the change of a
    nonstationary signal's frequency content over time.
    Parameters
    ----------
    x : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg.
        Defaults to a Tukey window with shape parameter of 0.25.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 8``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the power spectral density ('density')
        where `Sxx` has units of V**2/Hz and computing the power
        spectrum ('spectrum') where `Sxx` has units of V**2, if `x`
        is measured in V and `fs` is measured in Hz. Defaults to
        'density'.
    axis : int, optional
        Axis along which the spectrogram is computed; the default is over
        the last axis (i.e. ``axis=-1``).
    mode : str, optional
        Defines what kind of return values are expected. Options are
        ['psd', 'complex', 'magnitude', 'angle', 'phase']. 'complex' is
        equivalent to the output of `stft` with no padding or boundary
        extension. 'magnitude' returns the absolute magnitude of the
        STFT. 'angle' and 'phase' return the complex angle of the STFT,
        with and without unwrapping, respectively.
    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of segment times.
    Sxx : ndarray
        Spectrogram of x. By default, the last axis of Sxx corresponds
        to the segment times.
    
    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. In contrast to welch's method, where the
    entire data stream is averaged over, one may wish to use a smaller
    overlap (or perhaps none at all) when computing a spectrogram, to
    maintain some statistical independence between individual segments.
    It is for this reason that the default window is a Tukey window with
    1/8th of a window's length overlap at each end.
    .. versionadded:: 0.16.0
    
    References
    ----------
    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
           "Discrete-Time Signal Processing", Prentice Hall, 1999.
    """
    modelist = ['psd', 'complex', 'magnitude', 'angle', 'phase']
    if mode not in modelist:
        raise ValueError('unknown value for mode {}, must be one of {}'
                         .format(mode, modelist))

    # need to set default for nperseg before setting default for noverlap below
    window, nperseg = _triage_segments(window, nperseg,
                                       input_length=x.shape[axis])

    # Less overlap than welch, so samples are more statisically independent
    if noverlap is None:
        noverlap = nperseg // 8

    if mode == 'psd':
        freqs, time, Sxx = _spectral_helper(x, x, fs, window, nperseg,
                                            noverlap, nfft, detrend,
                                            return_onesided, scaling, axis,
                                            mode='psd')

    else:
        freqs, time, Sxx = _spectral_helper(x, x, fs, window, nperseg,
                                            noverlap, nfft, detrend,
                                            return_onesided, scaling, axis,
                                            mode='stft')

        if mode == 'magnitude':
            Sxx = np.abs(Sxx)
        elif mode in ['angle', 'phase']:
            Sxx = np.angle(Sxx)
            if mode == 'phase':
                # Sxx has one additional dimension for time strides
                if axis < 0:
                    axis -= 1
                Sxx = np.unwrap(Sxx, axis=axis)

        # mode =='complex' is same as `stft`, doesn't need modification

    return freqs, time, Sxx


# end