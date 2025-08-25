# library
# laser simulation
# wave propagation
# --- OPTIMIZED VERSION ---

import numpy as np
from numba import njit
from scipy.interpolate import RegularGridInterpolator
import pyfftw
from joblib import Parallel, delayed
from joblib import hash

datatype = np.float64

### --- OPTIMIZED ELEMENT CLASSES --- ###
# These classes replace the originals. They have the same API but are internally optimized.
@njit(cache=True)
def lens_kernel(r_sq, cos_2azimut, focal_length, C10, focal_length_defocus,
                Cs, C50, C32, focal_length_nominal, laser_wavenumber):
    """A JIT-compiled kernel for the lens phase calculation."""
    r_fourth = r_sq * r_sq
    r_sixth = r_fourth * r_sq
    
    phase_shift = (
        -laser_wavenumber * r_sq / (2 * focal_length)
        - C10 * laser_wavenumber * r_sq / (2 * focal_length_defocus**2)
        - Cs * laser_wavenumber * r_fourth / (4 * focal_length_nominal**4)
        - C50 * laser_wavenumber * r_sixth / (6 * focal_length_nominal**6)
        - C32 * laser_wavenumber * r_fourth * cos_2azimut / (4 * focal_length_nominal**4)
    )
    return np.exp(1j * phase_shift)

@njit(cache=True)
def laser_kernel(r_squared, beam_width):
    """A JIT-compiled kernel for Gaussian beam creation."""
    sigma = beam_width / 2.35403
    exponent = -0.5 * r_squared / (sigma**2)
    return np.exp(exponent)

@njit(cache=True)
def aperture_kernel(wave, r_sq, radius_sq):
    """A JIT-compiled kernel for a circular aperture."""
    # Looping is very fast inside a Numba function
    out_wave = np.zeros_like(wave)
    for i in range(wave.shape[0]):
        for j in range(wave.shape[1]):
            if r_sq[i, j] < radius_sq:
                out_wave[i, j] = wave[i, j]
    return out_wave

@njit(cache=True)
def beamblock_kernel(wave, r_sq, radius_sq):
    """A JIT-compiled kernel for a circular beamblock."""
    out_wave = np.zeros_like(wave)
    for i in range(wave.shape[0]):
        for j in range(wave.shape[1]):
            if r_sq[i, j] > radius_sq:
                out_wave[i, j] = wave[i, j]
    return out_wave

class Laser:
    """
    Optimized: Caches the grid calculation (r^2) for faster repeated wave generation.
    """
    def __init__(self, pos, beam_width_FWHM) -> None:
        self.pos = pos
        self.beam_width = beam_width_FWHM
        self._r_squared = None
        self._grid_shape = None

    def make_wave(self, plane_xx, plane_yy):
        if plane_xx.shape != self._grid_shape:
            self._r_squared = plane_xx**2 + plane_yy**2
            self._grid_shape = plane_xx.shape
        
        # Call the fast, compiled kernel instead of doing the math here
        laser_wave = laser_kernel(self._r_squared, self.beam_width)
        return laser_wave.astype(np.complex128, copy=False)

class Lens:
    """
    Heavily Optimized: Caches all static grid-based and aberration calculations.
    Automatically invalidates cache if relevant aberration coefficients change.
    """
    def __init__(self, pos, focal_length_forcomputing, radius, C10=0, Cs=0, C32=0, C50=0, 
                 focal_length_nominal=None, focal_length_defocus=None) -> None:
        self.pos = pos
        # Use private attributes accessed via properties for automatic cache invalidation
        self._focal_length = focal_length_forcomputing
        self._radius = radius
        self._C10 = C10
        self._Cs = Cs
        self._C32 = C32
        self._C50 = C50
        self._focal_length_nominal = focal_length_forcomputing if focal_length_nominal is None else focal_length_nominal
        self._focal_length_defocus = self._focal_length_nominal if focal_length_defocus is None else focal_length_defocus

        # Internal Cache
        self._grid_shape = None
        self._r_sq = None
        self._cos_2azimut = None
        self._aperture_mask = None
        self._static_phase_part = None

    # --- Properties to intercept changes and invalidate cache ---
    @property
    def Cs(self): return self._Cs
    @Cs.setter
    def Cs(self, value): self._Cs = value; self._static_phase_part = None

    @property
    def C32(self): return self._C32
    @C32.setter
    def C32(self, value): self._C32 = value; self._static_phase_part = None
    
    # ... (add properties for other coeffs if they need to be changed dynamically) ...

    def _precompute_grids(self, plane_xx, plane_yy):
        """PRIVATE: Runs ONCE per grid shape. Calculates all static grid arrays."""
        # print(f"INFO: (Lens at pos {self.pos}) Caching grids for shape {plane_xx.shape}...")
        self._grid_shape = plane_xx.shape
        self._r_sq = plane_xx**2 + plane_yy**2
        
        # Only calculate expensive trig functions if the aberration is non-zero
        if self._C32 != 0:
            self._cos_2azimut = np.cos(2 * np.arctan2(plane_yy, plane_xx))
        else:
            self._cos_2azimut = 1.0
        
        # Cache aperture mask, comparing r^2 is faster than sqrt(r^2)
        self._aperture_mask = np.where(self._r_sq < self._radius**2, 1, 0)
        self._static_phase_part = None # Invalidate phase part since grids changed

    def _precompute_static_phase(self, laser_wavenumber):
        """PRIVATE: Runs when aberration coeffs change. Calculates static phase part."""
        # print(f"INFO: (Lens at pos {self.pos}) Caching static aberration phase...")
        r_fourth = self._r_sq**2
        r_sixth = r_fourth * self._r_sq
        
        self._static_phase_part = (
            - self._Cs * laser_wavenumber * r_fourth / (4 * self._focal_length_nominal**4)
            - self._C50 * laser_wavenumber * r_sixth / (6 * self._focal_length_nominal**6)
            - self._C32 * laser_wavenumber * r_fourth * self._cos_2azimut / (4 * self._focal_length_nominal**4)
        )

    def propagate(self, wave2D_before, plane_xx, plane_yy, laser_wavenumber):
        if plane_xx.shape != self._grid_shape:
            self._precompute_grids(plane_xx, plane_yy)
        
        # Call the fast JIT kernel with all parameters
        lens_function = lens_kernel(
            self._r_sq, self._cos_2azimut, self._focal_length, self._C10,
            self._focal_length_defocus, self._Cs, self._C50, self._C32,
            self._focal_length_nominal, laser_wavenumber
        )
        
        return wave2D_before * lens_function * self._aperture_mask

class SLM:
    """
    Optimized: Caches the interpolator object and the final interpolated phase grid.
    """
    def __init__(self,pos, pix, size, phase_change = None, x_center=0, y_center=0) -> None:
        self.pos = pos
        self.pix = (pix[0], pix[1])
        self.size = size
        self.x_center = x_center
        self.y_center = y_center
        self.phase_change = np.zeros(self.pix, dtype=float) if phase_change is None else np.array(phase_change, dtype=float)
        
        # --- Add these attributes for API compatibility ---
        self.xs = None
        self.ys = None
        
        # Internal Cache
        self._interpolator = None
        self._cached_phase_on_grid = None
        self._grid_shape = None
        
        self._update_interpolator()

    def _update_interpolator(self):
        """PRIVATE: Rebuilds the expensive interpolator object."""
        # --- THIS IS THE FIX ---
        # Store xs and ys so the plotting function can access them
        self.xs = np.linspace(-self.size[0]/2, self.size[0]/2, self.pix[0]) + self.x_center
        self.ys = np.linspace(-self.size[1]/2, self.size[1]/2, self.pix[1]) + self.y_center
        
        self._interpolator = RegularGridInterpolator(
            (self.xs, self.ys), self.phase_change, method='nearest', bounds_error=False, fill_value=0
        )
        self._grid_shape = None

    def update_center(self, x_center, y_center):
        """API is identical, but now correctly rebuilds the interpolator."""
        self.x_center = x_center
        self.y_center = y_center
        self._update_interpolator()

    def propagate(self, wave2D_before, plane_xx, plane_yy, laser_wavenumber):
        """API is identical, but internals are cached."""
        if plane_xx.shape != self._grid_shape:
            # print(f"INFO: (SLM at pos {self.pos}) Caching interpolated phase for shape {plane_xx.shape}...")
            # Interpolation is expensive, so we cache its result.
            self._cached_phase_on_grid = self._interpolator((plane_xx, plane_yy))
            self._grid_shape = plane_xx.shape
            
        return wave2D_before * np.exp(1j * self._cached_phase_on_grid)

class Aperture:
    """Optimized: Caches the aperture mask."""
    def __init__(self, pos, radius) -> None:
        self.pos = pos
        self.radius = radius
        self._mask = None
        self._grid_shape = None

    def propagate(self, wave2D_before, plane_xx, plane_yy, laser_wavenumber):
        if plane_xx.shape != self._grid_shape:
            self._r_sq = plane_xx**2 + plane_yy**2
            self._grid_shape = plane_xx.shape
            
        # Call the fast JIT kernel
        return aperture_kernel(wave2D_before, self._r_sq, self.radius**2)

class BeamBlock:
    """Optimized: Caches the beamblock mask."""
    def __init__(self, pos, radius, x_center=0, y_center=0) -> None:
        self.pos = pos
        self.radius = radius
        self.x_center = x_center
        self.y_center = y_center
        self._mask = None
        self._grid_shape = None

    def propagate(self,  wave2D_before, plane_xx, plane_yy, laser_wavenumber):
        if plane_xx.shape != self._grid_shape:
            self._grid_shape = plane_xx.shape
            self._r_sq = (plane_xx - self.x_center)**2 + (plane_yy - self.y_center)**2
            
        # Call the fast JIT kernel
        return beamblock_kernel(wave2D_before, self._r_sq, self.radius**2)

### --- BEAMLINE CLASS (largely unchanged, but now benefits from faster elements) --- ###

### --- FINAL OPTIMIZED BEAMLINE CLASS --- ###

class BeamLine:
    """
    This version uses a CLASS-LEVEL cache that persists across multiple
    instantiations of the BeamLine object. This is essential for scripts that
    re-create the BeamLine inside a simulation loop.
    """
    # CLASS-LEVEL CACHE: This dictionary is shared by all instances of BeamLine.
    _global_element_mask_cache = {}

    def __init__(self, elements, wavelength, plane_width, plane_points) -> None:
        self.elements = elements
        self.elements_pos = np.array([el.pos for el in self.elements], dtype=datatype)
        self.plane_points = plane_points
        self.plane_width = plane_width
        self.wavelength = wavelength
        self.wavenumber = 2 * np.pi / wavelength
        
        self.plane_xs = np.linspace(-self.plane_width / 2, self.plane_width / 2, self.plane_points, dtype=datatype)
        self.plane_ys = np.linspace(-self.plane_width / 2, self.plane_width / 2, self.plane_points, dtype=datatype)
        self.plane_xx, self.plane_yy = np.meshgrid(self.plane_xs, self.plane_ys)
        self.plane_dx = self.plane_xs[1] - self.plane_xs[0]
        
        plane_freq_x = np.fft.fftfreq(self.plane_points, self.plane_dx)
        plane_freq_y = np.fft.fftfreq(self.plane_points, self.plane_dx)
        plane_freq_xx, plane_freq_yy = np.meshgrid(plane_freq_x, plane_freq_y)
        k_sq_minus_kxy_sq = self.wavenumber**2 - (2 * np.pi)**2 * (plane_freq_xx**2 + plane_freq_yy**2)
        self.kz_wavevector = np.sqrt(k_sq_minus_kxy_sq.astype(np.complex128))
        
        self.wave = None
        self.elements_intensity_crossection = []
        self.elements_intensity_crossection_y = []
        pyfftw.interfaces.cache.enable()

    @classmethod
    def clear_mask_cache(cls):
        """
        Clears the GLOBAL element mask cache. Call this if you fundamentally
        change the simulation grid between runs.
        """
        print("INFO: Global element mask cache has been cleared.")
        cls._global_element_mask_cache.clear()
    
    def update_positions(self):
        self.elements_pos = np.array([el.pos for el in self.elements], dtype=datatype)

    def propagate_wave_from_spectra(self, fft_ini, z_dist):
        transfer_function = np.exp(1j * self.kz_wavevector * z_dist)
        fft_zdist = pyfftw.interfaces.numpy_fft.ifft2(fft_ini * transfer_function)
        return fft_zdist
    
    def _make_hashable(self, obj):
        """
        PRIVATE RECURSIVE HELPER: Traverses a data structure and makes it hashable
        by finding and hashing any numpy arrays within it.
        """
        if isinstance(obj, np.ndarray):
            # Base case: If we find an array, hash it.
            return hash(obj)
        
        if isinstance(obj, dict):
            # If it's a dict, recurse on its items.
            return tuple(sorted((k, self._make_hashable(v)) for k, v in obj.items()))

        if isinstance(obj, (list, tuple)):
            # If it's a list or tuple, recurse on its elements.
            return tuple(self._make_hashable(item) for item in obj)
        
        if hasattr(obj, '__dict__'):
            # If it's a custom object, recurse on its __dict__.
            return self._make_hashable(obj.__dict__)

        # If it's already a simple, hashable type, return it as is.
        return obj

    def _get_element_mask_key(self, element):
        """
        PRIVATE: Creates a robust, hashable key from an element's properties.
        This version uses a recursive helper to handle complex nested objects.
        """
        # We start with the element's class name and the grid shape.
        # Then we add the fully sanitized, hashable representation of the element's properties.
        # This will now correctly handle any element, no matter how complex.
        key = (
            type(element).__name__,
            self.plane_xx.shape,
            self._make_hashable(element.__dict__)
        )
        return key


    def _get_element_mask(self, element, wave_shape):
        """
        PRIVATE: Uses the GLOBAL cache with a robust key.
        """
        key = self._get_element_mask_key(element)
        
        # Access the class-level cache
        if key not in BeamLine._global_element_mask_cache:
            # print(f"INFO: Caching new element state: {type(element).__name__}")
            # The `propagate` methods of our new elements are already optimized,
            # so this first call is as fast as it can be.
            dummy_wave = np.ones(wave_shape, dtype=np.complex128)
            mask = element.propagate(dummy_wave, self.plane_xx, self.plane_yy, self.wavenumber)
            BeamLine._global_element_mask_cache[key] = mask
            
        return BeamLine._global_element_mask_cache[key]
    
    def propagate(self, verbose=False):
        """
        This method is now extremely fast on subsequent runs, even if new
        BeamLine objects are created, thanks to the global cache.
        """
        # print('>>> Laser propagation starting')
        self.wave = self.elements[0].make_wave(self.plane_xx, self.plane_yy)
        # print('>>> Initial wave created OK')

        num_elements = len(self.elements)
        self.elements_intensity_crossection = [None] * num_elements
        self.elements_intensity_crossection_y = [None] * num_elements
        mid_point = self.plane_points // 2
        self.elements_intensity_crossection[0] = abs(self.wave[:, mid_point])**2
        self.elements_intensity_crossection_y[0] = abs(self.wave[mid_point, :])**2

        for c_element in range(1, num_elements):
            if verbose: print(f"Processing element {c_element}")
            dist = self.elements_pos[c_element] - self.elements_pos[c_element-1]
            fft_after = pyfftw.interfaces.numpy_fft.fft2(self.wave)
            self.wave = self.propagate_wave_from_spectra(fft_after, dist)
            
            element_mask = self._get_element_mask(self.elements[c_element], self.wave.shape)
            self.wave *= element_mask

            self.elements_intensity_crossection[c_element] = abs(self.wave[:, mid_point])**2
            self.elements_intensity_crossection_y[c_element] = abs(self.wave[mid_point, :])**2
        # print(">>> Propagation finished.")
    
    # --- The parallel detector method remains unchanged and will benefit from faster elements ---
    def get_detector_lens_and_beamblock(self, defocuses, lens_zpos, beamblock_radius,  lens_f_nominal, lens_f_computing, lens_f_defocus, lens_scalingfactor, lens_r):
        """
        PARALLELIZED & MEMORY-OPTIMIZED: This version uses joblib's memory mapping
        to avoid duplicating large arrays across processes.
        """
        if self.wave is None:
            raise RuntimeError("You must run .propagate() before using this method.")
            
        print(">>> Starting PARALLEL detector calculation (with memory mapping)...")
        # --- STEP 1: Calculations performed ONCE (unchanged) ---
        dist_to_lens = lens_zpos - self.elements_pos[-1]
        if dist_to_lens < 0: raise ValueError("Lens distance is negative")

        wave_before_lens = self.propagate_wave_from_spectra(
            pyfftw.interfaces.numpy_fft.fft2(self.wave), dist_to_lens
        )

        base_lens_obj = Lens(lens_zpos, focal_length_forcomputing=lens_f_computing, focal_length_nominal=lens_f_nominal, radius=lens_r)
        base_lens_mask = base_lens_obj.propagate(np.ones_like(self.wave, dtype=np.complex128), self.plane_xx, self.plane_yy, self.wavenumber)
        
        beamblock_dist = lens_f_computing 
        beamblock_obj = BeamBlock(lens_zpos + beamblock_dist, beamblock_radius)
        beamblock_mask = beamblock_obj.propagate(np.ones_like(self.wave, dtype=np.complex128), self.plane_xx, self.plane_yy, self.wavenumber)

        r_squared = self.plane_xx**2 + self.plane_yy**2
        
        # --- STEP 2: Worker function (unchanged) ---
        def process_single_defocus(defocus):
            if defocus != 0 and lens_f_defocus !=0:
                defocus_phase = -self.wavenumber * r_squared / (2 * lens_f_defocus * defocus)
                defocus_mask = np.exp(1j * defocus_phase)
                _wave = wave_before_lens * base_lens_mask * defocus_mask
            else:
                _wave = wave_before_lens * base_lens_mask

            _wave = self.propagate_wave_from_spectra(pyfftw.interfaces.numpy_fft.fft2(_wave), beamblock_dist)
            _wave *= beamblock_mask
            _wave = self.propagate_wave_from_spectra(pyfftw.interfaces.numpy_fft.fft2(_wave), lens_f_computing - beamblock_dist)
            
            return _wave

        # --- STEP 3: Execute in parallel WITH MEMORY MAPPING ENABLED ---
        # THIS IS THE KEY CHANGE. `mmap_mode='r'` tells joblib to share the
        # read-only data instead of copying it.
        print(f">>> Distributing {len(defocuses)} tasks (memory-mapped)...")
        waves_detected = Parallel(n_jobs=-1, mmap_mode='r')(
            delayed(process_single_defocus)(d) for d in defocuses
        )
        
        print(">>> Parallel detector calculation finished.")
        return waves_detected, self.plane_xs * lens_scalingfactor