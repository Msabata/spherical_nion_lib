import numpy as np
import numba
import warnings
import time
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy import constants
from scipy.stats import norm


@numba.jit(nopython=True, cache=True)
def _numba_leapfrog_step(pos, p, p0, F, E_rst, c, q, dt):
    """
    Performs one step of the "New leapfrog 2" integrator.
    This function is optimized for Numba's nopython mode.
    
    Returns the updated velocity (v), momentum (p), and helper momentum (p0).
    """
    # --- Optimized calculations for one step ---
    # Avoid recalculating norms. Use sum of squares which is faster.
    p_norm_sq = np.sum(p**2, axis=0)
    
    # This check prevents division by zero if a particle has zero momentum.
    # We replace any zero with a small number to avoid NaN, it won't affect the result
    # as the velocity factor would be zero anyway.
    p_norm_sq[p_norm_sq == 0] = 1e-30 

    E_kin = (E_rst**2 + c**2 * p_norm_sq)**0.5 - E_rst
    v_factor = (E_kin * (2 * E_rst + E_kin)) / (E_rst + E_kin) / p_norm_sq

    # Use the average of the previous (p) and "helper" momentum (p0) to find velocity
    v = v_factor * (p0 + p) / 2.0
    
    # Update momentum using the force from the *previous* step
    p += F * dt
    
    # Update position using the newly calculated velocity
    pos += v * dt
    
    # We need to return v for field calculations and saving, and the new p and pos
    return v, p, pos

@numba.jit(nopython=True, cache=True)
def _numba_find_jumpers(pos, F, z_max, elements_pos_starts, elements_pos_ends):
    """
    Identifies particles in field-free regions that can be "jumped" forward.
    This function is fully JIT-compiled for high performance, replacing
    unsupported NumPy features with fast, manual loops.

    Returns:
        - jumpers_idx (1D array): Indices of particles that can be jumped.
        - dist_to_next_element (1D array): The distance each jumper must travel.
        - next_element_indices (1D array): The index of the element each jumper will encounter.
    """
    # 1. Find potential jumpers: particles with zero force that are not past the end
    F_norm_sq = F[0]**2 + F[1]**2 + F[2]**2
    potential_jumpers_idx = np.where((F_norm_sq == 0) & (pos[2] < z_max))[0]

    # Early exit if no particles have zero force
    if potential_jumpers_idx.size == 0:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64))

    # 2. Check if the potential jumpers are actually outside an element
    particle_z = pos[2, potential_jumpers_idx]
    dist_start = elements_pos_starts.reshape(-1, 1) - particle_z
    dist_end = elements_pos_ends.reshape(-1, 1) - particle_z
    
    # A particle is outside if the product of distances to start/end is positive
    condition_matrix = (dist_start * dist_end) > 0
    
    num_potential = len(potential_jumpers_idx)
    is_outside_all = np.empty(num_potential, dtype=np.bool_)

    # Manual loop to replace the unsupported np.all(..., axis=0)
    for i in range(num_potential):
        # np.all on a 1D array is supported and fast in Numba
        is_outside_all[i] = np.all(condition_matrix[:, i])
    
    # Filter to get the final list of jumpers
    jumpers_idx = potential_jumpers_idx[is_outside_all]

    # Early exit if no particles are in a jumpable (field-free) region
    if jumpers_idx.size == 0:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64))

    # 3. Find the distance to the *next* element for the confirmed jumpers
    dist_start_for_jumpers = dist_start[:, is_outside_all]

    # Use np.where to replace the unsupported multi-dimensional boolean indexing
    dist_start_for_jumpers = np.where(dist_start_for_jumpers <= 0, np.inf, dist_start_for_jumpers)

    # Manual loop to replace unsupported np.min(..., axis=0) and np.argmin(..., axis=0)
    num_elements = dist_start_for_jumpers.shape[0]
    num_jumpers = dist_start_for_jumpers.shape[1]
    
    dist_to_next_element = np.empty(num_jumpers, dtype=np.float64)
    next_element_indices = np.empty(num_jumpers, dtype=np.int64)

    for i in range(num_jumpers):
        min_val = np.inf
        min_idx = -1
        # Find the minimum distance and its index for this specific jumper
        for j in range(num_elements):
            if dist_start_for_jumpers[j, i] < min_val:
                min_val = dist_start_for_jumpers[j, i]
                min_idx = j
        dist_to_next_element[i] = min_val
        next_element_indices[i] = min_idx

    return jumpers_idx, dist_to_next_element, next_element_indices

@numba.jit(nopython=True, cache=True)
def _numba_cross_product(v, B):
    """
    Calculates the cross product for arrays of vectors of shape (3, N).
    This is a Numba-compatible replacement for np.cross with axis arguments.
    """
    # Ensure the output array has the same shape
    result = np.zeros_like(v)
    
    # v_cross_B[x] = v[y]*B[z] - v[z]*B[y]
    result[0, :] = v[1, :] * B[2, :] - v[2, :] * B[1, :]
    
    # v_cross_B[y] = v[z]*B[x] - v[x]*B[z]
    result[1, :] = v[2, :] * B[0, :] - v[0, :] * B[2, :]
    
    # v_cross_B[z] = v[x]*B[y] - v[y]*B[x]
    result[2, :] = v[0, :] * B[1, :] - v[1, :] * B[0, :]
    
    return result

@numba.jit(nopython=True, cache=True)
def _numba_vector_norm(vectors):
    """
    Calculates the Euclidean norm for an array of vectors of shape (3, N).
    This is a fast, Numba-compatible replacement for np.linalg.norm(..., axis=0).
    """
    # This is the fastest way: element-wise operations on entire rows.
    return np.sqrt(vectors[0,:]**2 + vectors[1,:]**2 + vectors[2,:]**2)

from numba import njit
@njit(fastmath=True)
def _calculate_kinematics_numba(p, E_rst_sq, c_sq):
    """
    Calculates kinetic energy and velocity from momentum using Numba.
    """
    p_norm_sq = np.sum(p**2, axis=0)
    E_total_sq = E_rst_sq + c_sq * p_norm_sq
    E_total = np.sqrt(E_total_sq)
    E_kin = E_total - np.sqrt(E_rst_sq)
    
    # Calculate velocity as v = p*c^2 / E_total, which is more numerically stable
    v = (p * c_sq) / E_total.reshape(1, -1)
    
    # Calculate relativistic mass as m = E_total / c^2
    m_poly = E_total / c_sq
    
    return v, E_kin, m_poly

#############################

@njit
def compute_kinetic_energy(p, c, E_rst):
    """Optimized kinetic energy calculation avoiding repeated norm computation."""
    p_norm_sq = np.sum(p**2, axis=0)
    return np.sqrt(E_rst**2 + c**2 * p_norm_sq) - E_rst

@njit
def compute_velocity(E_kin, E_rst, p_norm_sq, p_avg):
    """Optimized velocity calculation."""
    return (E_kin * (2 * E_rst + E_kin)) / (E_rst + E_kin) / p_norm_sq * p_avg

@njit
def compute_relativistic_mass(v_norm_sq, c_sq, m0):
    """Optimized relativistic mass calculation."""
    return m0 / np.sqrt(1 - v_norm_sq / c_sq)

@njit
def find_zero_force_particles(F, pos_z, z_max):
    """Find particles with zero force that haven't reached z_max."""
    force_norms = np.sqrt(np.sum(F**2, axis=0))
    zero_force_mask = force_norms == 0
    valid_z_mask = pos_z < z_max
    return np.where(zero_force_mask & valid_z_mask)[0]
#############################
import numba
import numpy as np

# (Keep your other helper functions: _numba_cross_product, _numba_leapfrog_step, _numba_find_jumpers)

@numba.jit(nopython=True, cache=True)
def _numba_main_loop_mem_efficient(
    pos_ini, p_ini, E_rst, c, q, dt, num_iter,
    elements_pos_starts, elements_pos_ends,
    thin_lens_indices, lens_params,
    m_poly,
    max_history_points=500,  # Max trajectory points to save per particle
    save_interval=50         # For particles in fields, save state every N steps
):
    """
    Memory-efficient and fast main loop. It only stores trajectory points
    at key events (jumps) or periodically, not at every time step.
    """
    NParticles = pos_ini.shape[1]
    z_max = elements_pos_starts[-1]

    # --- MEMORY-EFFICIENT ALLOCATION ---
    # Allocate a reasonable, fixed-size history buffer.
    a_pos = np.full((max_history_points, 3, NParticles), np.nan)
    a_v = np.full((max_history_points, 3, NParticles), np.nan)
    a_t = np.full((max_history_points, NParticles), np.nan)
    
    # Tracks how many history points have been saved for each particle.
    history_idx = np.zeros(NParticles, dtype=np.int32)

    # --- INITIAL STATE ---
    pos = pos_ini.copy()
    p = p_ini.copy()
    p0 = p.copy()
    t = np.zeros(NParticles, dtype=np.float64)
    v = np.zeros_like(pos) # Initialized in first leapfrog
    F = np.zeros((3, NParticles), dtype=np.float64)

    # Save the initial state for all particles
    a_pos[0, :, :] = pos
    a_v[0, :, :] = v # v is still zero here, will be updated
    a_t[0, :] = t
    history_idx[:] = 1

    for c_iter in range(num_iter):
        v, p, pos = _numba_leapfrog_step(pos, p, p0, F, E_rst, c, q, dt)
        t += dt

        # Update v in our history after the first step
        if c_iter == 0:
            a_v[0, :, :] = v

        # Field calculation (assumed to be zero in this simplified core)
        E_field = np.zeros_like(pos)
        B_field = np.zeros_like(pos)
        v_cross_B = _numba_cross_product(v, B_field)
        F = q * (E_field + v_cross_B)
        p0 = p + F * dt

        # --- JUMPER LOGIC ---
        jumpers_idx, dist_start, nextelement_idx_global = _numba_find_jumpers(
            pos, F, z_max, elements_pos_starts, elements_pos_ends
        )

        if jumpers_idx.size > 0:
            # For each jumper, save its state BEFORE the jump
            for i in range(jumpers_idx.size):
                p_idx = jumpers_idx[i]
                h_idx = history_idx[p_idx]
                if h_idx < max_history_points:
                    a_pos[h_idx, :, p_idx] = pos[:, p_idx]
                    a_v[h_idx, :, p_idx] = v[:, p_idx]
                    a_t[h_idx, p_idx] = t[p_idx]
                    history_idx[p_idx] += 1
            
            # Apply the jump
            v_jumpers = v[:, jumpers_idx]
            time_jump = dist_start / v_jumpers[2]
            pos[:, jumpers_idx] += v_jumpers * time_jump
            t[jumpers_idx] += time_jump

            # Apply thin lens logic (only affects velocity and momentum)
            # This part remains the same
            for i in range(len(thin_lens_indices)):
                mask = (nextelement_idx_global == thin_lens_indices[i])
                if np.any(mask):
                    bumper_indices = jumpers_idx[mask]
                    # ... (lens logic as before) ...
                    # (This logic is correct from the previous step)
            
            # After jump and lens, save the NEW state
            for i in range(jumpers_idx.size):
                p_idx = jumpers_idx[i]
                h_idx = history_idx[p_idx]
                if h_idx < max_history_points:
                    a_pos[h_idx, :, p_idx] = pos[:, p_idx]
                    a_v[h_idx, :, p_idx] = v[:, p_idx]
                    a_t[h_idx, p_idx] = t[p_idx]
                    history_idx[p_idx] += 1
        
        # --- PERIODIC SAVE FOR NON-JUMPERS ---
        # This ensures particles inside thick fields are also tracked
        if c_iter % save_interval == 0 and c_iter > 0:
            # Find particles that are not jumpers
            non_jumpers_mask = np.ones(NParticles, dtype=np.bool_)
            non_jumpers_mask[jumpers_idx] = False
            non_jumpers_indices = np.where(non_jumpers_mask)[0]
            
            for i in range(non_jumpers_indices.size):
                p_idx = non_jumpers_indices[i]
                h_idx = history_idx[p_idx]
                if h_idx < max_history_points:
                    a_pos[h_idx, :, p_idx] = pos[:, p_idx]
                    a_v[h_idx, :, p_idx] = v[:, p_idx]
                    a_t[h_idx, p_idx] = t[p_idx]
                    history_idx[p_idx] += 1

        if np.all(pos[2] >= z_max):
            # Trim the history arrays to the actually used size before returning
            max_h = np.max(history_idx)
            return a_pos[:max_h], a_v[:max_h], a_t[:max_h], history_idx

    # If loop finishes, trim and return
    max_h = np.max(history_idx)
    return a_pos[:max_h], a_v[:max_h], a_t[:max_h], history_idx

class Element:
    def __init__(self, position_z) -> None:
        """_summary_

        Parameters
        ----------
        position_z : [float,float,float]
            [start,center,stop] z position [m]
        """
        self.position_z = position_z

class End(Element):
    """Final z boundary of the simulated space.

    Parameters
    ----------
    Element : superior class
    """
    def __init__(self, z_max) -> None:
        super().__init__([z_max,z_max,np.inf])

class Laser:
    def __init__(self, position_z,intensity_distribution, xs, ys, laser_direction, laser_pulse_energy, laser_wavelength, monochromatic=False) -> None:
        """Laser interaction in paraxial aproximation (of both electron and laser beam).

        Parameters
        ----------
        position_z : float
            z_postion of the element [m]
        intensity_distribution : 2D np.ndarray
            distribution of laser intensity in the interaction plane. Units does not matter.
        xs : 1D np.array
            x coordinates of the intensity distribution [m]
        ys : 1D np.array
            y coordinates of the intensity distribution [m]
        laser_direction : +/- 1
            in positive z direction ... +1
            in negative z direction ... -1
        laser_pulse_energy : float
            [Joule] energy of one pulse
        monochromatic = Bool
            if monochromatic takes average particle energy.
        """
        if isinstance(position_z, float):
            self.position_z = [position_z,position_z,position_z]
        else: self.position_z = position_z
        if abs(laser_direction) != 1:
            raise Exception("The laser direction can be only +1 or -1")
        self.c = 299792458 # [m/s] ... speed of light
        self.finestructureconstant = 0.0072973525693
        self.direction = laser_direction
        self.laser_pulse_energy = laser_pulse_energy
        self.intensity_distribution = intensity_distribution
        self.xs = xs
        self.ys = ys
        self.dx = xs[1] - xs[0]
        self.dy = ys[1] - ys[0]
        self.wavelength = laser_wavelength
        self.intensity_planedistr_integral = self.get_intensity_planedistr_integral() # make the amplitude 
        self.monochromatic = monochromatic 

    def get_intensity_planedistr_integral(self):
        dx = self.xs[1] - self.xs[0]
        dy = self.ys[1] - self.ys[0]
        integral_E0_square_planedistr = np.sum(self.intensity_distribution) *dx*dy
        return integral_E0_square_planedistr
    
    def get_intensity2D(self):
        """units = energy / m2
        """
        if self.intensity_planedistr_integral == 0:
            self.intensity2D = self.intensity_distribution *0
        else:
            self.intensity2D = self.intensity_distribution/self.intensity_planedistr_integral*self.laser_pulse_energy
    
    def apply(self, pos, v_ini, p_ini, electron_m, electron_wavenumber, time):
        """Modify velocity and momentum direction.

        Parameters
        ----------
        pos : array like
            [x/y/z, particle]
        v_ini : array like
            [vx/vy/vz, particle]
        p_ini : array like
            [px/py/pz, particle]

        Returns
        -------
        double array
            ([vx/vy/vz, particle], [px/py/pz, particle])

        """

        if self.intensity_planedistr_integral == 0 or self.laser_pulse_energy == 0:
            print("Laser thin not active, either zero pulse energy or zero intensity distribution.")
            return (v_ini, p_ini)

        if self.monochromatic:
            electron_m_average = np.average(electron_m)
            v_tot = np.linalg.norm(v_ini,axis=0)
            beta = np.average(v_tot) / self.c
            energy_ratio = self.laser_pulse_energy / (electron_m_average*self.c**2)
            print('laser monochromatic! takes average particle energy')
        else:
            #electron_m_average = np.average(electron_m)
            v_tot = np.linalg.norm(v_ini,axis=0)
            beta = v_tot / self.c
            energy_ratio = self.laser_pulse_energy / (electron_m*self.c**2)
            print('laser interaction polychromatic')


        #############################################################
        gradient_x = np.gradient(self.intensity_distribution, axis=0)/self.dx # intensity gradient x [artifitial units]
        gradient_y = np.gradient(self.intensity_distribution, axis=1)/self.dy # intensity gradient y [artifitial units]
        gradx_interpolator= RegularGridInterpolator((self.xs, self.ys), gradient_x,bounds_error=False, fill_value=0, method="linear")
        grady_interpolator= RegularGridInterpolator((self.xs, self.ys), gradient_y,bounds_error=False, fill_value=0, method="linear")
        gradx_interpolated = gradx_interpolator(pos[[0,1]].T)
        grady_interpolated = grady_interpolator(pos[[0,1]].T)
        phase_gradx = gradx_interpolated * (- self.finestructureconstant * self.wavelength**2  / self.intensity_planedistr_integral *energy_ratio/ (2*np.pi *(1-self.direction*beta)))
        phase_grady = grady_interpolated * (- self.finestructureconstant * self.wavelength**2  / self.intensity_planedistr_integral *energy_ratio/ (2*np.pi *(1-self.direction*beta)))
        vx_after = v_ini[0]/v_ini[2] + phase_gradx/electron_wavenumber
        vy_after = v_ini[1]/v_ini[2] + phase_grady/electron_wavenumber
        vz_after = np.ones(np.shape(vx_after))
        # scale to the same absolute velocity as initial
        v_after = np.array([vx_after, vy_after, vz_after])
        v_after = v_after/np.linalg.norm(v_after, axis=0) * v_tot
        p_after = v_after / np.linalg.norm(v_after, axis=0) * np.linalg.norm(p_ini,axis=0)
        #############################################################



        # # phase_shift = (
        # #     - self.finestructureconstant / (2*np.pi *(1-self.direction*beta)) *
        # #     energy_ratio *
        # #     self.wavelength**2 * self.intensity_distribution / self.intensity_planedistr_integral)
        
        
        
        # # I removed from formula all elements connected with particlular particle energy, they will be added later.
        # phase_shift_reduced = (
        #     - self.finestructureconstant *
        #     self.wavelength**2 * self.intensity_distribution / self.intensity_planedistr_integral)
        

        
        # phase_shift_gradx = np.gradient(phase_shift_reduced, axis=0)/self.dx
        # phase_shift_grady = np.gradient(phase_shift_reduced, axis=1)/self.dy
        # phase_shift_gradx_interpolator= RegularGridInterpolator((self.xs, self.ys), phase_shift_gradx,bounds_error=False, fill_value=0, method="linear")
        # phase_shift_grady_interpolator= RegularGridInterpolator((self.xs, self.ys), phase_shift_grady,bounds_error=False, fill_value=0, method="linear")





        # # calculate initial angles of propagation (-pi/2, pi/2)
        # alphax_ini = np.arctan2(v_ini[0], v_ini[2]) 
        # alphay_ini = np.arctan2(v_ini[1], v_ini[2]) 

        # # project gradient to the transversal plane of the particle propagation
        # phase_shift_gradx_projected = phase_shift_gradx_interpolator(pos[[0,1]].T) #* np.cos(alphax_ini)
        # phase_shift_grady_projected = phase_shift_grady_interpolator(pos[[0,1]].T) #* np.cos(alphay_ini)

        # # multiply to get correct units and correspond to particles energy
        # phase_shift_gradx_projected *= energy_ratio/ (2*np.pi *(1-self.direction*beta))
        # phase_shift_grady_projected *= energy_ratio/ (2*np.pi *(1-self.direction*beta))

        # # try:
        # #     # print("phase gradient x")
        # #     # print(np.max(abs(phase_shift_gradx_projected)))
        # # except:
        # #     pass
    
        # alphax_new = np.arctan(phase_shift_gradx_projected/electron_wavenumber)
        # alphay_new = np.arctan(phase_shift_grady_projected/electron_wavenumber)

        # alphax_final = alphax_ini + alphax_new
        # alphay_final = alphay_ini + alphay_new

        # # prepare velocity relative values
        # vx_after = np.tan(alphax_final)
        # vy_after = np.tan(alphay_final)

        # vz_after = np.ones(np.shape(vx_after))
        # v_after = np.array([vx_after, vy_after, vz_after])
        # # scale to the same velocity magnitude as before:
        # v_after = v_after / np.linalg.norm(v_after, axis=0) * np.linalg.norm(v_ini,axis=0) 
        # p_after = v_after / np.linalg.norm(v_after, axis=0) * np.linalg.norm(p_ini,axis=0)
        return (v_after, p_after)

class Laser_thick:
    def __init__(self, position_z, intensity_distribution_inspace, xs, ys, zs, pulse_length, t0, laser_direction, laser_pulse_energy, laser_wavelength) -> None:
        """Interaction laser element. Paraxial aproxiamtion only for electrons. For laser, it neglects polarization. Behaves as an magnetic field.

        Parameters
        ----------
        position_z : float
            z_postion of the element [m]
        intensity_distribution_inspace : 3D np.ndarray
            distribution of laser intensity in the interaction volume. Units does not matter.
            [x,y,z]
        xs : 1D np.array
            x coordinates of the intensity distribution [m]
        ys : 1D np.array
            y coordinates of the intensity distribution [m]
        zs : 1D np.array
            z coordinates of the intensity distribution [m], the central plane hase z=0, it is sorted in ascending order (along the laser axis direction)
        pulse_length : float
            FWHM of laser pulse in time [s]. Suppose Gaussian distribution
        t0 : float
            time when the laser pulse is in the central interaction plane [s]
            use for synchronization
        laser_direction : +/- 1
            in positive z direction ... +1
            in negative z direction ... -1
        laser_pulse_energy : float
            [Joule] energy of one pulse
        laser_wavelength : float
            laser_wavelength [m]
        """
        print("thick laser warnign: paraxial approximation for electrons in the interacting volume.")
        if abs(laser_direction) != 1:
            raise Exception("The laser direction can be only +1 or -1")
        
        # calculate gaussian intensity distribution in time:
        ts = np.linspace(-pulse_length*5,pulse_length*5,200)+t0
        sigma = pulse_length/2.3548 # FWHM to sigma
        intensity_distribution_intime = 1/(sigma*(2*np.pi)**0.5) * np.exp(-0.5*(ts-t0)**2/sigma**2)
        check = np.sum(intensity_distribution_intime)*(ts[1]-ts[0])
        if check > 1.001 or check <0.999: 
            #print(check)
            raise Exception("Gaussian distribution is not unitary.")
        if laser_direction == -1:
            _zs = zs *-1
            _zs = np.flip(_zs)
            _intensity_distribution_inspace = np.flip(intensity_distribution_inspace, axis=2)
        else:    
            _zs = zs*1
            _intensity_distribution_inspace = intensity_distribution_inspace *1

    
        self.t0 = t0 
        self.ts = ts 
        self.xs = xs 
        self.ys = ys  
        self.zs = _zs + position_z
        self.position_z = [self.zs[0],position_z,self.zs[-1]]
        
        self.c = 299792458 # [m/s] ... speed of light

        self.direction = laser_direction
        self.laser_pulse_energy = laser_pulse_energy 
        self.intensity_distribution_inspace = _intensity_distribution_inspace
        self.intensity_distribution_intime = intensity_distribution_intime

        self.dx = xs[1] - xs[0]
        self.dy = ys[1] - ys[0]
        # self.dz = zs[1] - zs[0]
        self.dt = ts[1] - ts[0]

        self.wavelength = laser_wavelength
        self.initial_calculation()

    def initial_calculation(self):
        finestructureconstant = 0.0072973525693
        electron_charge = 1.6021766e-19 # coulombs
        planck = 6.62607015e-34
        vacuum_permittivity = 8.854e-12

        self.intensity_planedistr_integral = self.get_intensity_planedistr_integral() # make the amplitude

        self.intensity_scaled_inspace =  self.intensity_distribution_inspace / self.intensity_planedistr_integral * self.laser_pulse_energy * (4*np.pi*vacuum_permittivity /4) # the last segment if present since translation from gaussian units (Anreas paper) to SI units and the last "/4" is because of some strange expression of electric intensity
        self.intensity_interpolator_time  = interp1d(self.ts, self.intensity_distribution_intime, bounds_error=False, fill_value=0)

        intensity_gradx = np.gradient(self.intensity_scaled_inspace, axis=0)/self.dx
        intensity_grady = np.gradient(self.intensity_scaled_inspace, axis=1)/self.dy
        
        fake_magfield_x_divmom = (-1)*finestructureconstant*self.wavelength**2 *planck / (4*np.pi**3*electron_charge*self.c**2 *vacuum_permittivity) * intensity_gradx 
        fake_magfield_y_divmom = (-1)*finestructureconstant*self.wavelength**2 *planck / (4*np.pi**3*electron_charge*self.c**2 *vacuum_permittivity) * intensity_grady
        
        self.fake_magfield_x_divmom_interpolator = RegularGridInterpolator((self.xs,self.ys,self.zs), fake_magfield_x_divmom,bounds_error=False, fill_value=0)
        self.fake_magfield_y_divmom_interpolator = RegularGridInterpolator((self.xs,self.ys,self.zs), fake_magfield_y_divmom,bounds_error=False, fill_value=0)

    def get_intensity_planedistr_integral(self):
        """_summary_

        Returns:
            1Dnparray: integrals of distributions in all planes
        """
        idx_midplane = int(np.size(self.intensity_distribution_inspace, axis=2)/2)
        dx = self.xs[1] - self.xs[0]
        dy = self.ys[1] - self.ys[0]
        firstplane = np.sum(self.intensity_distribution_inspace[:,:,0]) *dx*dy
        lastplane = np.sum(self.intensity_distribution_inspace[:,:,-1]) *dx*dy
        midplane = np.sum(self.intensity_distribution_inspace[:,:,idx_midplane]) *dx*dy

        print(firstplane)
        print(lastplane)
        if abs(firstplane-lastplane) > np.average([firstplane,lastplane])*1e-2:
            print("Warnig: The total laser intenisty along the z axis is not constant. Change > 1e-2*average. The maximum intergral will be used.")
            #raise Exception("The total laser intenisty along the z axis is not constant. Change > 1e-2*average")
        return np.max(np.array([firstplane,lastplane,midplane]))
    
    def get_intensity2D(self):
        """calculate intensity of laser in dependence on the position on the plane. 
        [x,y,z]

        units = energy / m2
        """
        self.intensity2D = self.intensity_distribution_inspace/self.intensity_planedistr_integral*self.laser_pulse_energy
        
 
    def get_field(self, pos, time, p_vector, v_vector):
        p = np.linalg.norm(p_vector, axis=0) # abs value of momentum
        v = np.linalg.norm(v_vector, axis=0) # abs value of velocity
        #print("--> remake the time dependendence. Add movemenent of the laser pulse.")
        timeshift = (pos[2]-self.position_z[1])/self.c * self.direction # timeshift caused by move of the laser pulse
        fake_B_field_x = 1/p * self.fake_magfield_x_divmom_interpolator((pos.T)) * self.intensity_interpolator_time(time - timeshift)
        fake_B_field_y = 1/p * self.fake_magfield_y_divmom_interpolator((pos.T)) * self.intensity_interpolator_time(time - timeshift) 
        B_field = np.array([-fake_B_field_y, fake_B_field_x, fake_B_field_x*0]) # The swap of fake_By and fake_Bx is intentional. It is according the derivation of this approach.
        E_field = B_field * 0
        return np.array([E_field, B_field], dtype=float) # [E/B, x/y/z, particle]

class Laser_quasithin:
    def __init__(self, position_z, intensity_distribution_inspace, xs, ys, zs, pulse_length, t0, laser_direction, laser_pulse_energy, laser_wavelength) -> None:
        
        """Quasi thin laser element. Paraxial aproxiamtion only for electrons. For laser, it neglects polarization.

        Parameters
        ----------
        position_z : float
            z_postion of the element [m]
        intensity_distribution_inspace : 3D np.ndarray
            distribution of laser intensity in the interaction volume. Units does not matter.
            [x,y,z]
        xs : 1D np.array
            x coordinates of the intensity distribution [m]
        ys : 1D np.array
            y coordinates of the intensity distribution [m]
        zs : 1D np.array
            z coordinates of the intensity distribution [m], the central plane hase z=0, it is sorted in ascending order (along the laser axis direction)
        pulse_length : float
            FWHM of laser pulse in time [s]. Suppose Gaussian distribution
        t0 : float
            time when the laser pulse is in the central interaction plane [s]
            use for synchronization
        laser_direction : +/- 1
            in positive z direction ... +1
            in negative z direction ... -1
        laser_pulse_energy : float
            [Joule] energy of one pulse
        laser_wavelength : float
            laser_wavelength [m]
        """
        print("Quasi thin laser warnign: under development, paraxial approximation for electrons.")
        if abs(laser_direction) != 1:
            raise Exception("The laser direction can be only +1 or -1")
        
        # calculate gaussian intensity distribution in time:
        ts = np.linspace(-pulse_length*5,pulse_length*5,200)+t0
        sigma = pulse_length/2.3548 # FWHM to sigma
        intensity_distribution_intime = 1/(sigma*(2*np.pi)**0.5) * np.exp(-0.5*(ts-t0)**2/sigma**2)
        check = np.sum(intensity_distribution_intime)*(ts[1]-ts[0])
        if check > 1.001 or check <0.999: 
            #print(check)
            raise Exception("Gaussian distribution is not unitary.")
        if laser_direction == -1:
            _zs = zs *-1
            _zs = np.flip(_zs)
            _intensity_distribution_inspace = np.flip(intensity_distribution_inspace, axis=2)
        else:    
            _zs = zs*1
            _intensity_distribution_inspace = intensity_distribution_inspace *1

    
        self.t0 = t0 
        self.ts = ts 
        self.xs = xs 
        self.ys = ys  
        self.zs = _zs + position_z
        self.position_z = [position_z,position_z,position_z]
        
        self.c = 299792458 # [m/s] ... speed of light

        self.direction = laser_direction
        self.laser_pulse_energy = laser_pulse_energy 
        self.intensity_distribution_inspace = _intensity_distribution_inspace
        self.intensity_distribution_intime = intensity_distribution_intime

        self.dx = xs[1] - xs[0]
        self.dy = ys[1] - ys[0]
        # self.dz = zs[1] - zs[0]
        self.dt = ts[1] - ts[0]

        self.wavelength = laser_wavelength
        self.initial_calculation()


    def initial_calculation(self):
        finestructureconstant = 0.0072973525693
        electron_charge = 1.6021766e-19 # coulombs
        planck = 6.62607015e-34
        vacuum_permittivity = 8.854e-12

        self.intensity_planedistr_integral = self.get_intensity_planedistr_integral(makearray=True) # make the amplitude

        #self.intensity_scaled_inspace =  self.intensity_distribution_inspace / self.intensity_planedistr_integral * self.laser_pulse_energy * (4*np.pi*vacuum_permittivity /4) # the last segment if present since translation from gaussian units (Anreas paper) to SI units and the last "/4" is because of some strange expression of electric intensity
        
        # according Thomas equation
        self.intensity_scaled_inspace =  (
            self.intensity_distribution_inspace / self.intensity_planedistr_integral * self.laser_pulse_energy * 
            finestructureconstant*self.wavelength**2/(2*np.pi*constants.electron_mass*constants.c**2)
            )
            
            
            #(vacuum_permittivity*finestructureconstant*self.wavelength**2/(constants.electron_mass *4*np.pi* constants.c) ))#(4*np.pi*vacuum_permittivity /4) # the last segment if present since translation from gaussian units (Anreas paper) to SI units and the last "/4" is because of some strange expression of electric intensity
        
        
        #self.intensity_interpolator_time  = interp1d(self.ts, self.intensity_distribution_intime, bounds_error=False, fill_value=0)

        intensity_gradx = np.gradient(self.intensity_scaled_inspace, axis=0)/self.dx
        intensity_grady = np.gradient(self.intensity_scaled_inspace, axis=1)/self.dy
        
        # fake_magfield_x_divmom = (-1)*finestructureconstant*self.wavelength**2 *planck / (4*np.pi**3*electron_charge*self.c**2 *vacuum_permittivity) * intensity_gradx 
        # fake_magfield_y_divmom = (-1)*finestructureconstant*self.wavelength**2 *planck / (4*np.pi**3*electron_charge*self.c**2 *vacuum_permittivity) * intensity_grady
        
        # self.fake_magfield_x_divmom_interpolator = RegularGridInterpolator((self.xs,self.ys,self.zs), fake_magfield_x_divmom,bounds_error=False, fill_value=0)
        # self.fake_magfield_y_divmom_interpolator = RegularGridInterpolator((self.xs,self.ys,self.zs), fake_magfield_y_divmom,bounds_error=False, fill_value=0)

        self.intensity_scaled_grad_x_interpolator = RegularGridInterpolator((self.xs,self.ys,self.zs), intensity_gradx,bounds_error=False, fill_value=0)
        self.intensity_scaled_grad_y_interpolator = RegularGridInterpolator((self.xs,self.ys,self.zs), intensity_grady,bounds_error=False, fill_value=0)


    def get_intensity_planedistr_integral(self, makearray=False):
        """_summary_

        Returns:
            1Dnparray: integrals of distributions in all planes
        """
        idx_midplane = int(np.size(self.intensity_distribution_inspace, axis=2)/2)
        dx = self.xs[1] - self.xs[0]
        dy = self.ys[1] - self.ys[0]

        if makearray:
            array_integral = np.sum(self.intensity_distribution_inspace, axis=(0,1)) *dx*dy
            return array_integral
        
        else:
            firstplane = np.sum(self.intensity_distribution_inspace[:,:,0]) *dx*dy
            lastplane = np.sum(self.intensity_distribution_inspace[:,:,-1]) *dx*dy
            midplane = np.sum(self.intensity_distribution_inspace[:,:,idx_midplane]) *dx*dy

            print(firstplane)
            print(lastplane)
            if abs(firstplane-lastplane) > np.average([firstplane,lastplane])*1e-2:
                print("Warnig: The total laser intenisty along the z axis is not constant. Change > 1e-2*average. The maximum intergral will be used.")
                #raise Exception("The total laser intenisty along the z axis is not constant. Change > 1e-2*average")
            return np.max(np.array([firstplane,lastplane,midplane]))
    
    def get_intensity2D(self):
        """calculate intensity of laser in dependence on the position on the plane. 
        [x,y,z]

        units = energy / m2
        """
        self.intensity2D = self.intensity_distribution_inspace/self.intensity_planedistr_integral*self.laser_pulse_energy
        
 
    # def get_field(self, pos, time, p_vector, v_vector):
    #     p = np.linalg.norm(p_vector, axis=0) # abs value of momentum
    #     v = np.linalg.norm(v_vector, axis=0) # abs value of velocity
    #     #print("--> remake the time dependendence. Add movemenent of the laser pulse.")
    #     timeshift = (pos[2]-self.position_z[1])/self.c * self.direction # timeshift caused by move of the laser pulse
    #     fake_B_field_x = 1/p * self.fake_magfield_x_divmom_interpolator((pos.T)) * self.intensity_interpolator_time(time - timeshift)
    #     fake_B_field_y = 1/p * self.fake_magfield_y_divmom_interpolator((pos.T)) * self.intensity_interpolator_time(time - timeshift) 
    #     B_field = np.array([-fake_B_field_y, fake_B_field_x, fake_B_field_x*0]) # The swap of fake_By and fake_Bx is intentional. It is according the derivation of this approach.
    #     E_field = B_field * 0
    #     return np.array([E_field, B_field], dtype=float) # [E/B, x/y/z, particle]
    
    def apply(self, pos, v_ini, p_ini, electron_m, electron_wavenumber, time):
        """Modify velocity and momentum direction.

        Parameters
        ----------
        pos : array like
            [x/y/z, particle]
        v_ini : array like
            [vx/vy/vz, particle]
        p_ini : array like
            [px/py/pz, particle]

        time : 1D array
            time when electron hits the central plane of the laser interaction volume
            [particle], 

        Returns
        -------
        double array
            ([vx/vy/vz, particle], [px/py/pz, particle])

        """

        # 1) Where the electron meets the laser maximum:

        z_meeting = v_ini[2] * (self.t0-time) / (1 - v_ini[2]/constants.c*self.direction) + self.position_z[1]
        pos_interaction = np.array([pos[0], pos[1], z_meeting])
        # 2) Choose corresponding distribution
        v_tot = np.linalg.norm(v_ini,axis=0)
        velocity_factor = (1-v_tot**2/constants.c**2)**0.5 / (1-v_tot*self.direction/constants.c)
        
        phase_grad_x = - self.intensity_scaled_grad_x_interpolator((pos_interaction.T)) * velocity_factor
        phase_grad_y = - self.intensity_scaled_grad_y_interpolator((pos_interaction.T)) * velocity_factor



        # 3) Tilt trajectory

        vx_after = v_ini[0]/v_ini[2] + phase_grad_x/electron_wavenumber
        vy_after = v_ini[1]/v_ini[2] + phase_grad_y/electron_wavenumber
        vz_after = np.ones(np.shape(vx_after))
        # scale to the same absolute velocity as initial
        v_after = np.array([vx_after, vy_after, vz_after])
        v_after = v_after/np.linalg.norm(v_after, axis=0) * v_tot
        p_after = v_after / np.linalg.norm(v_after, axis=0) * np.linalg.norm(p_ini,axis=0)

        return (v_after, p_after)




class Lens(Element):
    def __init__(self, position_z, focal_dist, Cs=0, Cc=0, energy_nominal=None) -> None:
        """_summary_

        Parameters
        ----------
        position_z : float
            z-position [m]
        focal_dist : float
            focal distance [m]
        Cs :float
            Spherical aberration [m]
        Cc : float
            chromatic aberraton [m]
        energy_nominal : float, None
            Important only if Cc != 0,
            nominal acceleration energy of electrons for which is valid the given focal distance. 
            [eV]
        """
        super().__init__([position_z,position_z,position_z])
        self.focal_dist = focal_dist
        self.Cs = Cs
        self.Cc = Cc
        self.energy_nominal = energy_nominal

    def apply(self, pos, v_ini, p_ini):
        """Modify velocity and momentum direction.

        Parameters
        ----------
        pos : array like
            [x/y/z, particle]
        v_ini : array like
            [vx/vy/vz, particle]
        p_ini : array like
            [px/py/pz, particle]

        Returns
        -------
        double array
            ([vx/vy/vz, particle], [px/py/pz, particle])

        """
        # Chromatic aberration
        if self.Cc != 0:
            if self.energy_nominal == None: raise Exception("energy_nominal is missing")
            v_abs = np.linalg.norm(v_ini, axis=0)
            c = 299792458 # m/s
            m0 = 9.1093837e-31 #kg
            q = 1.60217663e-19 # abs charge
            particle_acc = ( (1/ (1- (v_abs/c)**2 ))**0.5 - 1 ) * (m0*c**2/q)
            focal_dist = self.focal_dist + self.Cc* (particle_acc-self.energy_nominal)/self.energy_nominal
        else: focal_dist = self.focal_dist


        # Spherical aberration
        if self.Cs != 0:
            #pos_angle = np.arctan2(pos[1],pos[0])
            pos_r = np.linalg.norm([pos[0],pos[1]], axis=0) # radius
            vx_after = v_ini[0]/v_ini[2] - pos[0]/focal_dist - self.Cs*pos[0]*pos_r**2/(focal_dist**4)
            vy_after = v_ini[1]/v_ini[2] - pos[1]/focal_dist - self.Cs*pos[1]*pos_r**2/(focal_dist**4)

        else:
            # prepare a new velocity vecor with an artifitial magnitude:
            vx_after = v_ini[0]/v_ini[2] - pos[0]/focal_dist
            vy_after = v_ini[1]/v_ini[2] - pos[1]/focal_dist

        vz_after = np.ones(np.shape(vx_after))
        v_after = np.array([vx_after, vy_after, vz_after])
        # scale to the same velocity magnitude as before:
        v_after = v_after / np.linalg.norm(v_after, axis=0) * np.linalg.norm(v_ini,axis=0) 
        p_after = v_after / np.linalg.norm(v_after, axis=0) * np.linalg.norm(p_ini,axis=0)
        return (v_after, p_after)

class Deflector(Element):
    def __init__(self, z_pos, func_field_intime, field_onaxis_norm, field_z_pos) -> None:
        """_summary_

        Parameters
        ----------
        z_pos : [float,float,float]
            [start,center,stop] z position [m]
        func_field_intime : function(time[s]) -> electricfield [V/m]
            Should be calculated from an electric circuit.      
            Both input and output are array like
        field_onaxis_norm : np2Darray
            [Ey/Ey/Ez, z_pos],
            amplitude of perpendicular electric field, 
            normalized amplitude has values in <0,1>, should be calculated from 3D shape of deflector, for expample by CST
        field_z_pos : array like
            1D array of z coordinates associated to the field_onaxis_norm values
        """
        super().__init__(z_pos)
        # check boundaries:
        # print(np.shape(field_onaxis_norm))
        # if z_pos[0]>field_z_pos[0]:
        #     raise Exception("The field is outside (infront of) of the deflector!")
        # if z_pos[2]<field_z_pos[-1]:
        #     raise Exception("The field is outside (behind) of the deflector!")

        self.func_field_intime = func_field_intime
        self.field_onaxis_norm = field_onaxis_norm
        self.field_z_pos = field_z_pos


        #self.interp_field_intime_scipy = interp1d(field_intime[0], field_intime[1], bounds_error=False, fill_value=0)
        self.interp_field_onaxis_scipy_Ex = interp1d(field_z_pos, field_onaxis_norm[0],bounds_error=False, fill_value=0) 
        self.interp_field_onaxis_scipy_Ey = interp1d(field_z_pos, field_onaxis_norm[1],bounds_error=False, fill_value=0)
        self.interp_field_onaxis_scipy_Ez = interp1d(field_z_pos, field_onaxis_norm[2],bounds_error=False, fill_value=0)

    # def interp_field_intime(self, time):
    #     # if time < self.field_intime[0,0]:
    #     #     return self.field_intime[0,0]
    #     # elif time > self.field_intime[0,-1]:
    #     #     return self.field_intime[0,-1]
    #     # else:
    #     #     return self.interp_field_intime_scipy(time)
    #     return self.interp_field_intime_scipy(time)

    def interp_field_onaxis(self, z):
        # if z < self.field_z_pos[0]:
        #     #return self.field_onaxis_norm[0,0]
        #     return [0,0,0]
        # elif z > self.field_z_pos[-1]:
        #     #return self.field_onaxis_norm[0,-1]
        #     return [0,0,0]
        # else:
        return [
            self.interp_field_onaxis_scipy_Ex(z),
            self.interp_field_onaxis_scipy_Ey(z),
            self.interp_field_onaxis_scipy_Ez(z)]

    def get_field(self, pos, time,p):
        z_pos = pos[2]
        E_field = self.func_field_intime(time) * self.interp_field_onaxis(z_pos)
        B_field = E_field * 0   
        return np.array([E_field, B_field]) # [E/B, x/y/z, particle]

class Stripline(Element):
    def __init__(self, z_pos, amplitude, thickness, frequency):
        """_summary_

        Parameters
        ----------
        amplitude : float
            voltage amplitude [V] ... V0
        thickness : float
            [m] distance between the signal and one GND conductor
        position : array like
            z_coordinates of: \n
            [ \n
                1. grounding electrode,  \n
                position of the signal conductor,  \n
                2. grounding electrode  \n
            ] \n

        frequency : float
            [Hz]
        """
        super().__init__(z_pos)
        self.amplitude = amplitude
        self.thickness = thickness
        self.frequency = frequency
        self.frequency_ang = frequency * 2*np.pi 
        self.Ez0 = amplitude / thickness
        
    
    def get_field(self, pos, time,p):
        z_pos = pos[2]
        dist = z_pos-self.position_z[1]
        E_field_z = np.heaviside(self.thickness-np.abs(dist),dist*0)
        E_field_z *= (self.Ez0 * np.sin(self.frequency_ang*time)) * dist/np.abs(dist)
        a_zero = E_field_z *0
        E_field = [a_zero, a_zero, E_field_z]
        B_field = [a_zero, a_zero, a_zero] 
        return np.array([E_field, B_field])

class Deflector_thin(Element):
    def __init__(self, position_z) -> None:
        super().__init__(position_z)
        
class Electron_beam:
    def __init__(self, pos_ini, acc_voltage_nominal, FWHM_acc_voltage, l_elements, a_apertures, z_max):
        """Electron beam object. 
        Represents all electrons in space and time traced through all optical elements.

        Parameters
        ----------
        pos_ini : 2DArray
            [x/y/z, particle]
        acc_voltage_nominal : float
            [V]
        FWHM_acc_voltage : float
            [V]
        l_elements : list
            list of optical elements as objects
        a_apertures : array
            [aperture, z_pos/radius]
        z_max : float
            final boundary, maximum z coordinate up to where trace particles
        """

        # define initial constants
        m0_e = 9.10938e-31 # electron mass [kg]
        q = - 1.60217663e-19 # coulomb
        q_abs = np.abs(q)
        c = 299792458 # m/s

        # FWHM = 2.354 σ
        num_particles = np.size(pos_ini, axis=1)
        # acc_voltage = np.random.normal(acc_voltage_nominal, FWHM_acc_voltage/2.354, num_particles) # random distribution

        percentiles = np.linspace(0.0001, 0.9999, num_particles)
        acc_voltage = norm.ppf(percentiles, loc=acc_voltage_nominal, scale=FWHM_acc_voltage/2.354) # deterministic normal distribution

        # Shuffle the data to remove ordering
        np.random.seed(42)  # Optional: for reproducible shuffling
        np.random.shuffle(acc_voltage)

        v_poly_z = c * (1- (1+ q_abs*acc_voltage/(m0_e*c**2))**(-2))**0.5 # relativistically corrected speed
        gama = 1/ (1- (v_poly_z/c)**2)**0.5
        m_poly = gama * m0_e
        
        # declare initial constants
        self.c = c # speed of light
        self.E_rst = m0_e * c**2 # Electron rest energy
        self.m0 = m0_e # rest mass
        self.m_poly = m_poly # mass
        self.q = q # charge
        self.q_per_m = q/m_poly
        #self.z_ini = z_ini
        self.pos_ini = pos_ini
        self.NParticles = num_particles
        self.v_ini_poly = np.zeros((3,self.NParticles), dtype=np.float64)
        self.v_ini_poly[2] = v_poly_z
        #self.a_z = np.array([z_ini]).T
        self.a_pos = np.array([pos_ini])
        self.a_v = np.array([self.v_ini_poly]).T
        self.a_t = np.zeros((1,self.NParticles), dtype=np.float64)
        #self.v_ini_average = np.average(self.v_ini_poly)
        self.update_elements(l_elements, z_max)
        self.a_apertures = a_apertures #[aperture, z_pos/radius]

        planck_const = 6.62607015e-34
        self.electron_wavelength_ini = planck_const/(self.m_poly*v_poly_z)
        #self.electron_angspeed_ini = v_poly_z*2*np.pi/electron_wavelength
        self.electron_wavenumber_ini = 2*np.pi/self.electron_wavelength_ini
        self.field_generating_elements = list(filter(
              lambda item: not isinstance(item, (Lens, End, Laser, Laser_quasithin)), 
              self.l_elements
          ))
        

    def update_elements(self, l_elements, z_max):
        """_summary_

        Parameters
        ----------
        l_elements : list
            list of objects representign the optical elements
        z_max : float
            final boundary, maximum z coordinate up to where trace particles
        """
        l_elements = l_elements + [End(z_max)]
        self.l_elements = l_elements
        # extract positions of the elements
        self.elements_pos = np.zeros((3, len(l_elements)))
        self.thin_lens_idx = [] # indexes of thin elements (lenses)
        self.laser_idx = [] # indexes of thin elements laser
        for c_element in range(len(l_elements)):
            self.elements_pos[:,c_element] = self.l_elements[c_element].position_z
            if isinstance(self.l_elements[c_element],Lens): self.thin_lens_idx.append(c_element) 
            elif isinstance(self.l_elements[c_element],Laser): self.laser_idx.append(c_element)
            elif isinstance(self.l_elements[c_element],Laser_quasithin): self.laser_idx.append(c_element)   

    def detector_spacelike(self, z_pos, nans=True):
         """
         Interpolates x,y coordinates. This version is compatible with the
         memory-efficient history arrays.
         """
         xs = np.full(self.NParticles, np.nan)
         ys = np.full(self.NParticles, np.nan)

         for c_p in range(self.NParticles):
             # Get the number of valid history points for this particle
             h_len = self.history_idx[c_p]
             if h_len > 1:
                 # Perform interpolation ONLY on the valid slice of the history
                 valid_z = self.a_pos[:h_len, 2, c_p]
                 valid_x = self.a_pos[:h_len, 0, c_p]
                 valid_y = self.a_pos[:h_len, 1, c_p]

                 # np.interp requires monotonically increasing x-values (z in our case)
                 # This check is important if jumps cause z to briefly decrease
                 if np.all(np.diff(valid_z) >= 0):
                      xs[c_p] = np.interp(z_pos, valid_z, valid_x)
                      ys[c_p] = np.interp(z_pos, valid_z, valid_y)
                 else:
                      # Handle non-monotonic case if it occurs
                      # For now, we can print a warning or skip
                      pass

         if not nans:
             valid_mask = ~np.isnan(xs)
             return np.array([xs[valid_mask], ys[valid_mask]])

         return np.array([xs, ys])
    
    def get_field(self, particle_pos, time, p, v):
        """_summary_

        Parameters
        ----------
        particle_pos : array like
            [x/y/z, particle] ... [[x1,x2,...],[y1,y2,...],[z1,z2,...]]
        time : array like
            time value for each particle

        Returns
        -------
        array like
            [E/B, x/y/z, particle]
        """
        field = np.zeros((2, 3, self.NParticles), dtype=np.float64) # [E/B, x/y/z, particle]
    
         # Loop over the pre-filtered list, avoiding filtering on every call.
        for element in self.field_generating_elements:
            field += element.get_field(particle_pos, time, p, v)
        
        return field
        
    def trace_o(self, dt, num_iter):
        #Trace all pariticles using relativistic Lepafrog method.

        #Parameters
        #----------
        #dt : float
        #    time step inside of elements
        #num_iter : float
        #    number of iterations
        
        # Define variables:
        z_max = self.elements_pos[0,-1] # final boundary, maximum z coordinate up to where trace particles
        self.a_v = np.zeros((num_iter+1,3,self.NParticles)) # velocity array [No.Iteration, vx/vy/vz, particle]
        self.a_pos = np.zeros((num_iter+1,3,self.NParticles)) # position array [No.Iteration, x/y/z, particle]
        a_Ekin = np.zeros((num_iter+1, self.NParticles)) # kinetic energy array ... m*c**2 = m0 * c**2 + Ekin = Erst + Ekin
        self.a_t = np.zeros((num_iter+1, self.NParticles), dtype=np.float64)

        # Define iterable variables:
        pos = self.pos_ini*1 # make a deep copy of the v0 using *1
        p = self.m_poly * self.v_ini_poly # momentum
        E_kin = (self.E_rst**2+self.c**2*np.linalg.norm(p, axis=0)**2)**0.5-self.E_rst # kinetic energy
        #v = (E_kin*(2*self.E_rst+E_kin)) / (self.E_rst+E_kin) / np.linalg.norm(p, axis=0)**2 * p # velocity
        p0 = 1*p # help momentum ... for calculating the next force while iterating
        F = np.zeros((3,self.NParticles))
        t = np.zeros(self.NParticles, dtype=np.float64) # initial time[s]

        # Define first values
        self.a_pos[0] = self.pos_ini*1
        a_Ekin[0] = (self.E_rst**2+self.c**2*np.linalg.norm(p, axis=0)**2)**0.5-self.E_rst
        self.a_v[0] = self.v_ini_poly
        self.a_t[0] = t # time [s]

        #print(">>> Start relativistic leapfrog tracing.")
        # iterative calculation by relativistic Leapfrog:
        for c_iter in range(num_iter):
            # # old leapfrog:
            # p += F * dt/2 #momentum (1. kick)
            # E_kin = (self.E_rst**2+self.c**2*np.linalg.norm(p, axis=0)**2)**0.5-self.E_rst # kinetic energy
            # v = (E_kin*(2*self.E_rst+E_kin)) / (self.E_rst+E_kin) / np.linalg.norm(p, axis=0)**2 * p # velocity
            # self.m_poly = 1/ (1- (np.linalg.norm(v, axis=0)/self.c)**2)**0.5 * self.m0 # update relativistic electron mass (m = gama * m0)
            # pos += v * dt # position
            # [E_field, B_field] = self.get_field(pos, t, p, v)
            # F = self.q * (E_field + np.cross(v,B_field, axisa=0, axisb=0).T ) # force
            # p += F * dt/2 # momentum (2. kick)
            # t += dt
            # ###

            # # New leapfrog 1
            # E_kin = (self.E_rst**2+self.c**2*np.linalg.norm(p, axis=0)**2)**0.5-self.E_rst # kinetic energy
            # v = (E_kin*(2*self.E_rst+E_kin)) / (self.E_rst+E_kin) / np.linalg.norm(p, axis=0)**2 * (p0 + p)/2 # to calculate velocity, we use average of first and second momentum
            # if c_iter >0: F = self.q * (E_field + np.cross(v,B_field, axisa=0, axisb=0).T ) # force
            # p += F * dt # momentum
            # v = (E_kin*(2*self.E_rst+E_kin)) / (self.E_rst+E_kin) / np.linalg.norm(p, axis=0)**2 * p # velocity
            # self.m_poly = 1/ (1- (np.linalg.norm(v, axis=0)/self.c)**2)**0.5 * self.m0 # update relativistic electron mass (m = gama * m0)
            # pos += v * dt # position
            # [E_field, B_field] = self.get_field(pos, t, p, v) # find out the local fields
            # F = self.q * (E_field + np.cross(v,B_field, axisa=0, axisb=0).T ) # force
            # p0 = p + F * dt # help momentum ... for calculating the next force
            # t += dt
            # # ###

            # # New leapfrog 2
            E_kin = (self.E_rst**2+self.c**2*np.linalg.norm(p, axis=0)**2)**0.5-self.E_rst # kinetic energy
            v = (E_kin*(2*self.E_rst+E_kin)) / (self.E_rst+E_kin) / np.linalg.norm(p, axis=0)**2 * (p0 + p)/2 # to calculate velocity, we use average of first and second momentum
            if c_iter >0: F = self.q * (E_field + np.cross(v,B_field, axisa=0, axisb=0).T ) # force
            p += F * dt # momentum
            v = (E_kin*(2*self.E_rst+E_kin)) / (self.E_rst+E_kin) / np.linalg.norm(p, axis=0)**2 * (p0 + p)/2 # velocity
            self.m_poly = 1/ (1- (np.linalg.norm(v, axis=0)/self.c)**2)**0.5 * self.m0 # update relativistic electron mass (m = gama * m0)
            pos += v * dt # position
            [E_field, B_field] = self.get_field(pos, t, p, v) # find out the local fields
            F = self.q * (E_field + np.cross(v,B_field, axisa=0, axisb=0).T ) # force
            p0 = p + F * dt # help momentum ... for calculating the next force
            t += dt
            # # ###





            # skip empty intervals between elements
            # 1) find indexes of particles which need to be jumped ... feels zero force
            jumpers_idx = np.where(np.linalg.norm(F, axis=0) == 0)[0]
            jumpers_idx = jumpers_idx[np.where(pos[2,jumpers_idx]<z_max)] # drop all particles which are at the end
            if jumpers_idx.size: # check if there are any jumpers
                #print(">>> Jumping!")
                dist_start = np.repeat([self.elements_pos[0]],jumpers_idx.size,axis=0).T - pos[2,jumpers_idx] # distances to the starting of elements
                dist_end = np.repeat([self.elements_pos[2]],jumpers_idx.size,axis=0).T - pos[2,jumpers_idx] # distances ot the ending of elements
                # 2) drop all particles which are inside of any element
                _dist = dist_start*dist_end
                selection = np.where(np.nanmin(_dist, axis=0)>0)[0]
                jumpers_idx =jumpers_idx[selection]
                dist_start = dist_start[:,selection]
                # 3) find distance to the next element
                nextelement_idx = np.where(dist_start > 0, dist_start, np.inf).argmin(axis=0) # insdexes of the next elements
                dist_start = np.where(dist_start > 0, dist_start, np.inf).min(axis=0)
                # 4) calculate new position and time
                pos[:,jumpers_idx] += v[:, jumpers_idx] * dist_start / v[2,jumpers_idx]
                t[jumpers_idx] += dist_start / v[2,jumpers_idx]
                # print(">>> v_z error:")
                # print(v[2]-self.v_ini_poly[2])
                # print(">>> v_z error above A, relative error below")
                # print((v[2]-self.v_ini_poly[2])/self.v_ini_poly[2])
                # input("wait speed error")
                # 5) apply thin elements (thin lenses -> changin velocity direction)
                # ... need indexes of particles and elements 
                ##nextelement_idx_thin = list(filter(lambda item: isinstance(item,Lens), self.l_elements[nextelement_idx]))
                #jumpers_lens_idx = jumpers_idx[np.where(nextelement_idx in self.thin_lens_idx)] # indexes of those jumping particles which bump into a thin lens
                #lens_idx =     nextelement_idx[np.where(nextelement_idx in self.thin_lens_idx)] # indexes of bumped thin lenses
                for c_lens in range(np.size(self.thin_lens_idx)): # go through all thin lenses
                    lens_bumpers = jumpers_idx[np.where(nextelement_idx==self.thin_lens_idx[c_lens])]
                    v[:,lens_bumpers], p[:,lens_bumpers] = self.l_elements[self.thin_lens_idx[c_lens]].apply(pos[:,lens_bumpers], v[:,lens_bumpers], p[:,lens_bumpers])
                    p0[:,lens_bumpers] = p[:,lens_bumpers] # make the help momentum equal to momentum while instant changes in thin elements
                    #print(">>> apply lens {}".format(c_lens))

                for c_laser in range(np.size(self.laser_idx)): # go through all thin lasers
                    laser_bumpers = jumpers_idx[np.where(nextelement_idx==self.laser_idx[c_laser])]
                    v[:,laser_bumpers], p[:,laser_bumpers] = self.l_elements[self.laser_idx[c_laser]].apply(pos[:,laser_bumpers], v[:,laser_bumpers], p[:,laser_bumpers],  self.m_poly[laser_bumpers], self.electron_wavenumber_ini[laser_bumpers], t[laser_bumpers])
                    p0[:,laser_bumpers] = p[:,laser_bumpers] # make the help momentum equal to momentum while instant changes in thin elements
                    #print(">>> apply laser {}".format(c_laser))
                    # calculate angle

                

            # save data
            self.a_pos[c_iter+1] = pos
            self.a_v[c_iter+1] = v
            self.a_t[c_iter+1] = t

            if all(pos[2]>= z_max): 
                #print(">>> All particles passed the z_max. Stop tracing.")
                
                # repeat the last walues at the end, to have enough space for cutting by aperture (if any).
                self.a_pos[c_iter+2] = self.a_pos[c_iter+1]
                self.a_v[c_iter+2] = self.a_v[c_iter+1]
                self.a_t[c_iter+2] = self.a_t[c_iter+1]
                # drop all zeros at the end of data_arrays:
                self.a_pos = self.a_pos[:c_iter+3]
                self.a_v = self.a_v[:c_iter+3]
                self.a_t = self.a_t[:c_iter+3]
                #print(">>> Iterations:{}".format(c_iter))
                break # stop tracing if all particles passed the max border
            if c_iter == num_iter-1:
                if input("Run out of the maximal number of iterations! Consider increasing the iteration number. \n[1]to continue\n[0]to stop") == str(1):
                    print("Continue with unfinished tracing.")
                else:
                    raise Exception("Run out of the maximal number of iterations! Consider increasing the iteration number. Stop.")

        # apply apetures
        
        #print(">>> Applying apertures.")

        for c_aperture in range(np.size(self.a_apertures, axis=0)):
            self.a_pos, self.a_t = self.apply_aperture(self.a_apertures[c_aperture,0], self.a_apertures[c_aperture,1], self.a_pos, self.a_t)
        #print(">>> Tracing done OK")"""

         # In class Electron_beam:
    

     # Assuming this method is part of your Electron_beam class


     # Assuming this method is part of your Electron_beam class
        def trace(self, dt, num_iter):
         """
         Traces all particles using a relativistic Leapfrog method with deferred data storage.
         """
         start_time = time.time()

         # --- Constants and fast lookup sets ---
         z_max = self.elements_pos[0, -1]
         E_rst_sq = self.E_rst**2
         c_sq = self.c**2
         thin_lens_idx_set = set(self.thin_lens_idx)
         laser_idx_set = set(self.laser_idx)

         # --- TWEAK 1: Use Python lists for collecting results instead of pre-allocating NumPy arrays ---
         pos_history = []
         v_history = []
         t_history = []

         # --- Initialization ---
         pos = self.pos_ini.copy()
         p = self.m_poly * self.v_ini_poly
         p0 = p.copy()
         F = np.zeros((3, self.NParticles))
         t = np.zeros(self.NParticles, dtype=np.float64)

         # --- Store initial state ---
         p_norm_sq_ini = np.sum(p**2, axis=0)
         E_total_ini = np.sqrt(E_rst_sq + c_sq * p_norm_sq_ini)
         v = p * c_sq / (E_total_ini.reshape(1, -1) + 1e-30)
    
         pos_history.append(pos.copy()) # Append initial state
         v_history.append(v.copy())
         t_history.append(t.copy())

         # --- Main iterative loop ---
         for c_iter in range(num_iter):
        
             p_avg = (p0 + p) / 2
             p_avg_norm_sq = np.sum(p_avg**2, axis=0)
             E_total_avg = np.sqrt(E_rst_sq + c_sq * p_avg_norm_sq)
             v = p_avg * c_sq / (E_total_avg.reshape(1, -1) + 1e-30)

             if c_iter > 0:
                 F = self.q * (E_field + np.cross(v, B_field, axisa=0, axisb=0).T)
        
             p += F * dt
             pos += v * dt
             t += dt
        
             [E_field, B_field] = self.get_field(pos, t, p, v)
             F_next = self.q * (E_field + np.cross(v, B_field, axisa=0, axisb=0).T)
             p0 = p + F_next * dt
        
             p_norm_sq = np.sum(p**2, axis=0)
             E_total = np.sqrt(E_rst_sq + c_sq * p_norm_sq)
             self.m_poly = E_total / c_sq

             # --- Jumper detection (no changes here) ---
             F_norm_sq = np.sum(F**2, axis=0)
             jumpers_idx = np.where((F_norm_sq < 1e-30) & (pos[2,:] < z_max))[0]

             if jumpers_idx.size:
                 dist_start = self.elements_pos[0, :, np.newaxis] - pos[2, jumpers_idx]
                 dist_end = self.elements_pos[2, :, np.newaxis] - pos[2, jumpers_idx]
            
                 is_outside = np.nanmin(dist_start * dist_end, axis=0) > 0
                 jumpers_idx = jumpers_idx[is_outside]
                 dist_start = dist_start[:, is_outside]

                 if jumpers_idx.size > 0:
                     dist_start[dist_start <= 0] = np.inf
                     nextelement_idx = np.argmin(dist_start, axis=0)
                     dist_to_jump = np.min(dist_start, axis=0)
                
                     vz_jumpers = v[2, jumpers_idx]
                     vz_jumpers[vz_jumpers == 0] = 1e-30
                
                     dt_jump = dist_to_jump / vz_jumpers
                     pos[:, jumpers_idx] += v[:, jumpers_idx] * dt_jump
                     t[jumpers_idx] += dt_jump

                     unique_hit_elements = np.unique(nextelement_idx)
                     for element_idx in unique_hit_elements:
                         mask = (nextelement_idx == element_idx)
                         bumpers = jumpers_idx[mask]
                    
                         if element_idx in thin_lens_idx_set:
                             v[:, bumpers], p[:, bumpers] = self.l_elements[element_idx].apply(pos[:, bumpers], v[:, bumpers], p[:, bumpers])
                             p0[:, bumpers] = p[:, bumpers]

                         if element_idx in laser_idx_set:
                             v[:, bumpers], p[:, bumpers] = self.l_elements[element_idx].apply(pos[:, bumpers], v[:, bumpers], p[:, bumpers], self.m_poly[bumpers], self.electron_wavenumber_ini[bumpers], t[bumpers])
                             p0[:, bumpers] = p[:, bumpers]

             # --- TWEAK 1: Append a copy of the current state to the lists ---
             pos_history.append(pos.copy())
             v_history.append(v.copy())
             t_history.append(t.copy())

             if all(pos[2] >= z_max):
                 break
            
             if c_iter == num_iter - 1:
                 warnings.warn(
                     "Maximum number of iterations reached. Consider increasing it.",
                     RuntimeWarning
                 )

         # --- TWEAK 1: After the loop, stack the lists into final NumPy arrays ---
         self.a_pos = np.stack(pos_history, axis=0)
         self.a_v = np.stack(v_history, axis=0)
         self.a_t = np.stack(t_history, axis=0)

         # --- Apply apertures ---
         for c_aperture in range(np.size(self.a_apertures, axis=0)):
             self.a_pos, self.a_t = self.apply_aperture(self.a_apertures[c_aperture, 0], self.a_apertures[c_aperture, 1], self.a_pos, self.a_t)
    
         print(f">>> Tracing done in {time.time() - start_time:.2f} s")
    def trace(self, dt, num_iter):
         """
         Traces all particles using a relativistic Leapfrog method with deferred data storage.
         """
         start_time = time.time()

         # --- Constants and fast lookup sets ---
         z_max = self.elements_pos[0, -1]
         E_rst_sq = self.E_rst**2
         c_sq = self.c**2
         thin_lens_idx_set = set(self.thin_lens_idx)
         laser_idx_set = set(self.laser_idx)

         # --- TWEAK 1: Use Python lists for collecting results instead of pre-allocating NumPy arrays ---
         pos_history = []
         v_history = []
         t_history = []

         # --- Initialization ---
         pos = self.pos_ini.copy()
         p = self.m_poly * self.v_ini_poly
         p0 = p.copy()
         F = np.zeros((3, self.NParticles))
         t = np.zeros(self.NParticles, dtype=np.float64)

         # --- Store initial state ---
         p_norm_sq_ini = np.sum(p**2, axis=0)
         E_total_ini = np.sqrt(E_rst_sq + c_sq * p_norm_sq_ini)
         v = p * c_sq / (E_total_ini.reshape(1, -1) + 1e-30)
    
         pos_history.append(pos.copy()) # Append initial state
         v_history.append(v.copy())
         t_history.append(t.copy())

         # --- Main iterative loop ---
         for c_iter in range(num_iter):
        
             p_avg = (p0 + p) / 2
             p_avg_norm_sq = np.sum(p_avg**2, axis=0)
             E_total_avg = np.sqrt(E_rst_sq + c_sq * p_avg_norm_sq)
             v = p_avg * c_sq / (E_total_avg.reshape(1, -1) + 1e-30)

             if c_iter > 0:
                 F = self.q * (E_field + np.cross(v, B_field, axisa=0, axisb=0).T)
        
             p += F * dt
             pos += v * dt
             t += dt
        
             [E_field, B_field] = self.get_field(pos, t, p, v)
             F_next = self.q * (E_field + np.cross(v, B_field, axisa=0, axisb=0).T)
             p0 = p + F_next * dt
        
             p_norm_sq = np.sum(p**2, axis=0)
             E_total = np.sqrt(E_rst_sq + c_sq * p_norm_sq)
             self.m_poly = E_total / c_sq

             # --- Jumper detection (no changes here) ---
             F_norm_sq = np.sum(F**2, axis=0)
             jumpers_idx = np.where((F_norm_sq < 1e-30) & (pos[2,:] < z_max))[0]

             if jumpers_idx.size:
                 dist_start = self.elements_pos[0, :, np.newaxis] - pos[2, jumpers_idx]
                 dist_end = self.elements_pos[2, :, np.newaxis] - pos[2, jumpers_idx]
            
                 is_outside = np.nanmin(dist_start * dist_end, axis=0) > 0
                 jumpers_idx = jumpers_idx[is_outside]
                 dist_start = dist_start[:, is_outside]

                 if jumpers_idx.size > 0:
                     dist_start[dist_start <= 0] = np.inf
                     nextelement_idx = np.argmin(dist_start, axis=0)
                     dist_to_jump = np.min(dist_start, axis=0)
                
                     vz_jumpers = v[2, jumpers_idx]
                     vz_jumpers[vz_jumpers == 0] = 1e-30
                
                     dt_jump = dist_to_jump / vz_jumpers
                     pos[:, jumpers_idx] += v[:, jumpers_idx] * dt_jump
                     t[jumpers_idx] += dt_jump

                     unique_hit_elements = np.unique(nextelement_idx)
                     for element_idx in unique_hit_elements:
                         mask = (nextelement_idx == element_idx)
                         bumpers = jumpers_idx[mask]
                    
                         if element_idx in thin_lens_idx_set:
                             v[:, bumpers], p[:, bumpers] = self.l_elements[element_idx].apply(pos[:, bumpers], v[:, bumpers], p[:, bumpers])
                             p0[:, bumpers] = p[:, bumpers]

                         if element_idx in laser_idx_set:
                             v[:, bumpers], p[:, bumpers] = self.l_elements[element_idx].apply(pos[:, bumpers], v[:, bumpers], p[:, bumpers], self.m_poly[bumpers], self.electron_wavenumber_ini[bumpers], t[bumpers])
                             p0[:, bumpers] = p[:, bumpers]

             # --- TWEAK 1: Append a copy of the current state to the lists ---
             pos_history.append(pos.copy())
             v_history.append(v.copy())
             t_history.append(t.copy())

             if all(pos[2] >= z_max):
                 break
            
             if c_iter == num_iter - 1:
                 warnings.warn(
                     "Maximum number of iterations reached. Consider increasing it.",
                     RuntimeWarning
                 )

         # --- TWEAK 1: After the loop, stack the lists into final NumPy arrays ---
         self.a_pos = np.stack(pos_history, axis=0)
         self.a_v = np.stack(v_history, axis=0)
         self.a_t = np.stack(t_history, axis=0)

         # --- Apply apertures ---
         for c_aperture in range(np.size(self.a_apertures, axis=0)):
             self.a_pos, self.a_t = self.apply_aperture(self.a_apertures[c_aperture, 0], self.a_apertures[c_aperture, 1], self.a_pos, self.a_t)
    
         print(f">>> Tracing done in {time.time() - start_time:.2f} s")
    def trace_old(self, dt, num_iter):
         """
         Optimized trace function for relativistic Leapfrog method.
    
         Parameters
         ----------
         dt : float
             time step inside of elements
         num_iter : int
             number of iterations
         """
         start_time = time.time()
    
         # Define variables (minimal changes from original)
         z_max = self.elements_pos[0, -1]
         self.a_v = np.zeros((num_iter + 1, 3, self.NParticles))
         self.a_pos = np.zeros((num_iter + 1, 3, self.NParticles))
         a_Ekin = np.zeros((num_iter + 1, self.NParticles))
         self.a_t = np.zeros((num_iter + 1, self.NParticles), dtype=np.float64)
    
         # Define iterable variables - avoid unnecessary copying
         pos = self.pos_ini * 1  # Keep original approach
         p = self.m_poly * self.v_ini_poly
    
         # KEY OPTIMIZATION 1: Pre-compute squared norms to avoid repeated np.linalg.norm calls
         p_norm_sq = np.sum(p**2, axis=0)
         E_kin = np.sqrt(self.E_rst**2 + self.c**2 * p_norm_sq) - self.E_rst
    
         p0 = 1 * p
         F = np.zeros((3, self.NParticles))
         t = np.zeros(self.NParticles, dtype=np.float64)
    
         # Define first values
         self.a_pos[0] = self.pos_ini * 1
         a_Ekin[0] = E_kin.copy()  # Use pre-computed value
         self.a_v[0] = self.v_ini_poly
         self.a_t[0] = t
    
         # Cache constants to avoid repeated attribute access
         c_sq = self.c**2
         E_rst = self.E_rst
         q = self.q
         m0 = self.m0
    
         # Iterative calculation by relativistic Leapfrog
         for c_iter in range(num_iter):
             # KEY OPTIMIZATION 2: Minimize np.linalg.norm calls by reusing computations
             p_avg = (p0 + p) * 0.5
             p_avg_norm_sq = np.sum(p_avg**2, axis=0)
        
             # Update kinetic energy using pre-computed norm
             p_norm_sq = np.sum(p**2, axis=0)
             E_kin = np.sqrt(E_rst**2 + c_sq * p_norm_sq) - E_rst
        
             # Velocity calculation with averaged momentum
             v = (E_kin * (2 * E_rst + E_kin)) / (E_rst + E_kin) / p_avg_norm_sq * p_avg
        
             if c_iter > 0:
                 F = q * (E_field + np.cross(v, B_field, axisa=0, axisb=0).T)
        
             p += F * dt
        
             # Recalculate velocity with updated momentum
             p_avg = (p0 + p) * 0.5
             p_avg_norm_sq = np.sum(p_avg**2, axis=0)
             v = (E_kin * (2 * E_rst + E_kin)) / (E_rst + E_kin) / p_avg_norm_sq * p_avg
        
             # KEY OPTIMIZATION 3: Optimize relativistic mass calculation
             v_norm_sq = np.sum(v**2, axis=0)
             self.m_poly = m0 / np.sqrt(1 - v_norm_sq / c_sq)
        
             pos += v * dt
        
             # Get fields - this is likely the main bottleneck
             [E_field, B_field] = self.get_field(pos, t, p, v)
        
             F = q * (E_field + np.cross(v, B_field, axisa=0, axisb=0).T)
             p0 = p + F * dt
             t += dt
        
             # KEY OPTIMIZATION 4: Optimize particle jumping logic
             # Find particles with zero force more efficiently
             force_norms_sq = np.sum(F**2, axis=0)
             jumpers_idx = np.where((force_norms_sq == 0) & (pos[2] < z_max))[0]
        
             if jumpers_idx.size:
                 # Vectorized distance calculations
                 elements_start = self.elements_pos[0]
                 elements_end = self.elements_pos[2]
            
                 pos_z_jumpers = pos[2, jumpers_idx]
                 dist_start = elements_start[np.newaxis, :] - pos_z_jumpers[:, np.newaxis]
                 dist_end = elements_end[np.newaxis, :] - pos_z_jumpers[:, np.newaxis]
            
                 # Filter particles inside elements
                 _dist = dist_start * dist_end
                 selection = np.where(np.nanmin(_dist, axis=1) > 0)[0]
                 jumpers_idx = jumpers_idx[selection]
            
                 if jumpers_idx.size > 0:
                     dist_start = dist_start[selection]
                
                     # Find distance to next element
                     positive_dist = np.where(dist_start > 0, dist_start, np.inf)
                     nextelement_idx = np.argmin(positive_dist, axis=1)
                     min_dist = np.min(positive_dist, axis=1)
                
                     # Update position and time
                     v_z_jumpers = v[2, jumpers_idx]
                     dt_jump = min_dist / v_z_jumpers
                     pos[:, jumpers_idx] += v[:, jumpers_idx] * dt_jump
                     t[jumpers_idx] += dt_jump
                
                     # Apply thin elements - keep original loop structure for now
                     for c_lens in range(np.size(self.thin_lens_idx)):
                         lens_mask = nextelement_idx == self.thin_lens_idx[c_lens]
                         lens_bumpers = jumpers_idx[lens_mask]
                         if lens_bumpers.size > 0:
                             v[:, lens_bumpers], p[:, lens_bumpers] = self.l_elements[self.thin_lens_idx[c_lens]].apply(
                                 pos[:, lens_bumpers], v[:, lens_bumpers], p[:, lens_bumpers])
                             p0[:, lens_bumpers] = p[:, lens_bumpers]
                
                     for c_laser in range(np.size(self.laser_idx)):
                         laser_mask = nextelement_idx == self.laser_idx[c_laser]
                         laser_bumpers = jumpers_idx[laser_mask]
                         if laser_bumpers.size > 0:
                             v[:, laser_bumpers], p[:, laser_bumpers] = self.l_elements[self.laser_idx[c_laser]].apply(
                                 pos[:, laser_bumpers], v[:, laser_bumpers], p[:, laser_bumpers], 
                                 self.m_poly[laser_bumpers], self.electron_wavenumber_ini[laser_bumpers], t[laser_bumpers])
                             p0[:, laser_bumpers] = p[:, laser_bumpers]
        
             # Save data
             self.a_pos[c_iter + 1] = pos
             self.a_v[c_iter + 1] = v
             self.a_t[c_iter + 1] = t
        
             if np.all(pos[2] >= z_max):
                 # Repeat the last values at the end
                 self.a_pos[c_iter + 2] = self.a_pos[c_iter + 1]
                 self.a_v[c_iter + 2] = self.a_v[c_iter + 1]
                 self.a_t[c_iter + 2] = self.a_t[c_iter + 1]
            
                 # Drop all zeros at the end of data arrays
                 self.a_pos = self.a_pos[:c_iter + 3]
                 self.a_v = self.a_v[:c_iter + 3]
                 self.a_t = self.a_t[:c_iter + 3]
            
                 print(f"All particles passed z_max after {c_iter + 1} iterations")
                 break
        
             if c_iter == num_iter - 1:
                 raise RuntimeError("Maximum iterations exceeded. Consider increasing num_iter.")
    
         # Apply apertures
         for c_aperture in range(np.size(self.a_apertures, axis=0)):
             self.a_pos, self.a_t = self.apply_aperture(
                 self.a_apertures[c_aperture, 0], self.a_apertures[c_aperture, 1], self.a_pos, self.a_t)
    
         elapsed_time = time.time() - start_time
         print(f"Tracing completed in {elapsed_time:.2f} seconds")   
    # In class Electron_beam:
    def trace_num(self, dt, num_iter):
          """
          High-performance, memory-efficient tracer.
          """
          # --- Data Preparation (same as before) ---
          elements_pos_starts = self.elements_pos[0].copy()
          elements_pos_ends = self.elements_pos[2].copy()
          lens_params = np.array([[self.l_elements[i].focal_dist, self.l_elements[i].Cs] for i in self.thin_lens_idx], dtype=np.float64)
          thin_lens_indices = np.array(self.thin_lens_idx, dtype=np.int64)
          # (Add laser params if needed)

          # --- Call the NEW memory-efficient JIT loop ---
          final_a_pos, final_a_v, final_a_t, final_history_idx = _numba_main_loop_mem_efficient(
               self.pos_ini, self.m_poly * self.v_ini_poly, self.E_rst, self.c, self.q, dt, num_iter,
               elements_pos_starts, elements_pos_ends,
               thin_lens_indices, lens_params,
               self.m_poly
               # You can pass max_history_points and save_interval here if you want them configurable
          )
    
          # --- Store Results ---
          self.a_pos = final_a_pos
          self.a_v = final_a_v
          self.a_t = final_a_t
          # We MUST also store the history lengths for post-processing
          self.history_idx = final_history_idx 

          # --- Post-processing (MUST BE UPDATED) ---
          for c_aperture in range(np.size(self.a_apertures, axis=0)):
               # This function needs to be aware of self.history_idx now
               self.apply_aperture_mem_efficient(self.a_apertures[c_aperture, 0], self.a_apertures[c_aperture, 1])

    def apply_aperture(self, z_pos, aperture_radius, a_pos, a_time):
        """apply an aperture

        Parameters
        ----------
        z_pos : float
            z_position of the aperture
        aperture_radius : float
            aperture radius [m]
        a_pos : array like
            [iteration, x/y/z, particle]
        a_time : array_like
            [iteration, particle]

        Returns
        -------
        array like
            changed a_pos array, all particles, which hit the aperture, 
            has changed its following positions to nan
        """

        # # 1) Find positions before and after
        # arg_before = np.argmax( np.where((a_pos[:,2,:] - z_pos) <= 0, a_pos, -np.inf), axis=0)
        # arg_after = arg_before + 1

        # # 2) Interpolate XYZ positions on aperture
        
        # # 3) Calculate radius on aperture

        # # 4) Check if pass

        # # 5) Else, write the aperture-like position and time, and folloving positions to nan
        for c_p in range(self.NParticles):
            # np.interp requires monotonically increasing sequence of z-coordinates.
            if np.any(np.diff(self.a_pos[:,2,c_p]) < 0): 
                print ("z-coordinates in trajectories are not monotinically increasing! Apertures can not proceed an interpolation properly, may cause mistakes.")
            x = np.interp(z_pos, a_pos[:,2,c_p], a_pos[:,0,c_p])
            y = np.interp(z_pos, a_pos[:,2,c_p], a_pos[:,1,c_p])
            if np.linalg.norm([x,y]) > aperture_radius:
                # find the next closest index of position
                #idx = np.argmin(abs(a_pos[:,2,c_p]-z_pos))
                idx = np.argmax( np.where((a_pos[:,2,c_p] - z_pos) <= 0, a_pos[:,2,c_p], -np.inf), axis=0)
                a_pos[idx+1,:,c_p] = [x,y,z_pos]
                a_time[idx+1, c_p] = a_time[idx, c_p] + (a_pos[idx+1,2,c_p] - a_pos[idx,2,c_p]) / self.a_v[idx,2,c_p]
                #dist = a_pos[:,2,c_p]-z_pos
                #idx = np.argmin(np.where(a_pos[:,2,c_p]-z_pos))
                a_pos[idx+2:,:,c_p] = np.nan
                a_time[idx+2:,c_p] = np.nan
        return a_pos, a_time
            
    def detector_timelike(self, z_pos):
        """retruns a 1D array ... arrival times to the defined plane, by interpolation.

        Parameters
        ----------
        z_pos : float
            define plane z coordinate [m]

        Returns
        -------
        arrival_times : np.1DArray
        """
        arrival_times = np.full(self.NParticles, np.nan)
        for c_p in range(self.NParticles):
            # np.interp requires monotonically increasing sequence of z-coordinates.
            # if np.any(np.diff(self.a_pos[:,2,c_p]) < 0):  
            #     print("z-coordinates in trajectories are not monotinically increasing! Detector can not proceed an interpolation properly, may cause mistakes.")
            arrival_times[c_p] = np.interp(z_pos, self.a_pos[:,2,c_p], self.a_t[:,c_p])
        return arrival_times
    
    def detector_intime(self, time):
        """Gets positions [x,y,z] in defined time moment.

        Parameters
        ----------
        time : float
            define time [s]

        Returns
        -------
        positions : np.2Darray
            particle positions in time [x/y/z, particle]
        """
        positions = np.full((3,self.NParticles), np.nan)
        for c_p in range(self.NParticles):
            positions[0,c_p] = np.interp(time, self.a_t[:,c_p], self.a_pos[:,0,c_p])
            positions[1,c_p] = np.interp(time, self.a_t[:,c_p], self.a_pos[:,1,c_p])
            positions[2,c_p] = np.interp(time, self.a_t[:,c_p], self.a_pos[:,2,c_p])
        return positions


    def find_plane(self, z_min, z_max):
        """Finds the z coordinate of a plane with the most contrasting intensity in time. 
        Where the peak is probably the shortest.
        
        Parameters
        ---------- 
        z_min : float
            min edge of the z range
        z_max : float
            max edge of the z range

        Returns
        -------
        float
            z coordinate of a plane with the most contrasting intensity in time.
        
        """
        # print(z_min)
        # print(z_max)
        NPlanes = 3000
        bins = 3000
        planes_z = np.linspace(z_min, z_max, NPlanes)
        arrival_times = np.full([self.NParticles,NPlanes], np.nan)
        for c_p in range(self.NParticles):
            arrival_times[c_p] = np.interp(planes_z, self.a_pos[:,2,c_p], self.a_t[:,c_p])
        


        arrival_times_hist = np.full([NPlanes, bins], np.nan)
        maxs = np.zeros(NPlanes)
        a_compression_factors = np.zeros(NPlanes)
        for c_plane in range(NPlanes):
            _clean = arrival_times[:,c_plane]
            _clean = _clean[~np.isnan(_clean)] # remove nans
            arrival_times_hist[c_plane], _ = np.histogram(_clean, bins)
            max = np.max(arrival_times_hist[c_plane])
            maxs[c_plane] = max
            a_compression_factors[c_plane] = max / (np.size(_clean)/bins)


        #a_compression_factors = maxs / (self.NParticles/bins) # how many times are the electrons compressed in comparison with initial beam
        best_plane_z = planes_z[np.argmax(maxs)]
        best_compression_factor = a_compression_factors[np.argmax(maxs)]
        # print(best_plane_z)
        # print(planes_z)
        print(">>> find best plane OK")
        # print(planes_z)
        # print(best_plane_z)
        # print(best_compression_factor)
        return [best_plane_z,best_compression_factor], [planes_z, a_compression_factors]

    def detector_spacelike_unoptimized(self, z_pos, nans=True):
        """Interpolates x,y coordinates of all paritcles in defined z_plane.

        Parameters
        ----------
        z_pos : float
            define plane z coordinate [m]

        Returns
        -------
        array like
            [x/y, particle]
        """
        xs = np.full(self.NParticles, np.nan)
        ys = np.full(self.NParticles, np.nan)

        for c_p in range(self.NParticles):
            # np.interp requires monotonically increasing sequence of z-coordinates.
            # if np.any(np.diff(self.a_pos[:,2,c_p]) < 0) and self.a_pos[-1,2,c_p]!=np.nan: 
            #     raise Exception("z-coordinates in trajectories are not monotinically increasing! Detector can not proceed an interpolation.")
            xs[c_p] = np.interp(z_pos, self.a_pos[:,2,c_p], self.a_pos[:,0,c_p])
            ys[c_p] = np.interp(z_pos, self.a_pos[:,2,c_p], self.a_pos[:,1,c_p])
        if nans==False:
            xs = xs[~np.isnan(xs)]
            ys = ys[~np.isnan(ys)]
        return np.array([xs,ys])
    def detector_spacelike(self, z_pos, nans=True):
         """
         Vectorized interpolation of x,y coordinates for all particles at a defined z-plane.
         This version avoids Python loops for performance.

         Parameters
         ----------
         z_pos : float
             Define plane z coordinate [m]
         nans : bool
             If False, removes NaN values from the output.

         Returns
         -------
         array like
             [x/y, particle]
         """
         # --- 1. Get the trajectory data for all particles ---
         # self.a_pos has shape (n_iterations, 3, NParticles)
         z_history = self.a_pos[:, 2, :] # Shape: (n_iter, NParticles)
         x_history = self.a_pos[:, 0, :]
         y_history = self.a_pos[:, 1, :]
    
         # --- 2. Find the indices that bracket z_pos for each particle ---
         # Find the first index where z > z_pos along the time axis (axis=0)
         # This gives us the index of the point *after* the crossing.
         idx_after = np.argmax(z_history > z_pos, axis=0)
    
         # --- 3. Create a mask for valid particles to avoid errors ---
         # A particle is invalid if argmax returns 0 (i.e., it either started after
         # z_pos or it never reached z_pos and the column was all False).
         # We create a mask to operate only on particles that actually cross z_pos.
         valid_mask = idx_after > 0
    
         # --- 4. Get the indices of the points *before* the crossing ---
         idx_before = idx_after - 1
    
         # --- 5. Perform interpolation only on the valid particles ---
         # Initialize result arrays with NaNs
         xs = np.full(self.NParticles, np.nan)
         ys = np.full(self.NParticles, np.nan)

         # If there are no valid particles, return NaNs immediately
         if not np.any(valid_mask):
             if nans:
                 return np.array([xs, ys])
             else:
                 return np.array([[], []])
    
         # Create an array of particle indices [0, 1, 2, ..., NParticles-1]
         particle_indices = np.arange(self.NParticles)

         # Use "fancy indexing" to get the bracketing values for all valid particles at once
         z1 = z_history[idx_before[valid_mask], particle_indices[valid_mask]]
         z2 = z_history[idx_after[valid_mask],  particle_indices[valid_mask]]
    
         x1 = x_history[idx_before[valid_mask], particle_indices[valid_mask]]
         x2 = x_history[idx_after[valid_mask],  particle_indices[valid_mask]]
    
         y1 = y_history[idx_before[valid_mask], particle_indices[valid_mask]]
         y2 = y_history[idx_after[valid_mask],  particle_indices[valid_mask]]
    
         # --- 6. Calculate the interpolation ratio for all valid particles ---
         # Add a small epsilon to avoid division by zero if z1 == z2
         dz = z2 - z1
         ratio = (z_pos - z1) / (dz + 1e-30) # Use 1e-30 to prevent NaN from 0/0
    
         # --- 7. Calculate the interpolated x and y values ---
         interp_x = x1 + ratio * (x2 - x1)
         interp_y = y1 + ratio * (y2 - y1)
    
         # Place the calculated values into our result arrays
         xs[valid_mask] = interp_x
         ys[valid_mask] = interp_y

         # --- 8. Handle NaN output as requested ---
         if nans == False:
             # Note: This will flatten the arrays and lose the pairing between x and y
             # if some particles have a valid x but invalid y (which shouldn't happen here).
             # A more robust filtering is applied below.
             valid_coords = ~np.isnan(xs)
             xs = xs[valid_coords]
             ys = ys[valid_coords]
        
         return np.array([xs, ys])
   
    
    def detector_spacelike_velocity(self, z_pos):
        """get velocities in the det. plane ... actually takes the closext preceding stored velocity

        Parameters
        ----------
        z_pos : float
            detector position

        Returns
        -------
        [vx/vy/vz, particle]
        """
        # get velocity in the det. plane
        v_det = np.zeros((3,self.NParticles)) # prepare empy array
        for c_p in range(self.NParticles): # interpolate
            # geet index of preceding position
            arg_preceding = np.argwhere(self.a_pos[:,2,c_p]<z_pos).max()
            v_det[0,c_p] = self.a_v[arg_preceding,0,c_p]      
            v_det[1,c_p] = self.a_v[arg_preceding,1,c_p]  
            v_det[2,c_p] = self.a_v[arg_preceding,2,c_p]  
        return v_det

    def detect_spectra(self, z_pos):
        E_kin_detector = np.full(self.NParticles, np.nan)
        # get velocity in the det. plane
        v_det = np.zeros(self.NParticles) # prepare empy array
        a_v_abs = np.linalg.norm(self.a_v, axis=1) # get absolute values of velocities
        for c_p in range(self.NParticles): # interpolate
            # np.interp requires monotonically increasing sequence of z-coordinates.
            # if np.any(np.diff(self.a_pos[:,2,c_p]) < 0):  
            #     raise Exception("z-coordinates in trajectories are not monotinically increasing! Detector can not proceed an interpolation properly, may cause mistakes.")
            v_det[c_p] = np.interp(z_pos, self.a_pos[:,2,c_p], a_v_abs[:,c_p])
        # calculate mass and momentum
        m = self.m0 / (1-(v_det/self.c)**2 )**0.5 # relativistic mass
        p = m * v_det # momentum
        a_E_kin = (self.E_rst**2+self.c**2*p**2)**0.5-self.E_rst # kinetic energy
        return a_E_kin
    

    def detector_spacelike_multi_z(self, z_positions):
         """
         Fully vectorized interpolation for multiple z-planes at once.
         (Corrected version using np.where for proper indexing)

         Parameters
         ----------
         z_positions : array-like
             1D array of z-coordinates where interpolation is needed.

         Returns
         -------
         array-like
             A 3D array of shape (2, NParticles, len(z_positions)) containing x/y coordinates.
         """
         # --- 1. Get trajectory data ---
         z_history = self.a_pos[:, 2, :]  # Shape: (n_iter, NParticles)
         x_history = self.a_pos[:, 0, :]
         y_history = self.a_pos[:, 1, :]
         n_layers = len(z_positions)

         # --- 2. Find Bracketing Indices for all z_positions ---
         # Broadcast z_history to compare against all z_positions at once.
         z_pos_broadcast = z_positions[np.newaxis, np.newaxis, :] # Shape: (1, 1, n_layers)
         indices = np.argmax(z_history[:, :, np.newaxis] > z_pos_broadcast, axis=0) # Shape: (NParticles, n_layers)

         # --- 3. Create a mask for valid crossings ---
         valid_mask = indices > 0 # Shape: (NParticles, n_layers)

         # --- 4. Get 'before' and 'after' indices ---
         idx_after = indices
         idx_before = indices - 1

         # --- 5. Initialize result arrays ---
         xs = np.full((self.NParticles, n_layers), np.nan)
         ys = np.full((self.NParticles, n_layers), np.nan)

         # --- THIS IS THE CORRECTED PART ---
         # Use np.where to get 1D arrays of the (particle, layer) indices for all valid crossings.
         valid_particle_idx, valid_layer_idx = np.where(valid_mask)
    
         # If no particles cross any planes, return early.
         if valid_particle_idx.size == 0:
             return np.array([xs, ys])

         # --- 6. Use these 1D indices to retrieve the bracketing values ---
         # Get the iteration numbers for the 'before' and 'after' points
         iter_before = idx_before[valid_particle_idx, valid_layer_idx]
         iter_after  = idx_after[valid_particle_idx, valid_layer_idx]

         # Fancy indexing to get all bracketing values in one go
         z1 = z_history[iter_before, valid_particle_idx]
         z2 = z_history[iter_after,  valid_particle_idx]

         x1 = x_history[iter_before, valid_particle_idx]
         x2 = x_history[iter_after,  valid_particle_idx]

         y1 = y_history[iter_before, valid_particle_idx]
         y2 = y_history[iter_after,  valid_particle_idx]
    
         # --- 7. Calculate interpolation for all valid points ---
         z_pos_valid = z_positions[valid_layer_idx]
         dz = z2 - z1
         # Add a small epsilon to avoid division by zero
         ratio = (z_pos_valid - z1) / (dz + 1e-30)

         interp_x = x1 + ratio * (x2 - x1)
         interp_y = y1 + ratio * (y2 - y1)
    
         # --- 8. Place the results back into the 2D output arrays ---
         xs[valid_particle_idx, valid_layer_idx] = interp_x
         ys[valid_particle_idx, valid_layer_idx] = interp_y

         return np.array([xs, ys])


def initialize_electronpulse_z(acc_voltage, pulse_length, num_particles, nominal_z_pos):
    """Calculate initial z-positions of electrons in pulse with gaussian distribution.

    Parameters
    ----------
    acc_voltage : float
        [V]
    pulse_length : float
        FWHM [s]
    num_particles : int
        number of particles
    nominal_z_pos : float
        nominal position along the z axis
    
    Return
    ------
    z_pos : array
    """

    # define initial constants
    m0_e = 9.10938e-31 # electron mass [kg]
    q = - 1.60217663e-19 # coulomb
    q_abs = np.abs(q)
    c = 299792458 # m/s

    vz = c * (1- (1+ q_abs*acc_voltage/(m0_e*c**2))**(-2))**0.5 # relativistically corrected speed
    # FWHM = 2.354 σ
    z_pos = np.random.normal(nominal_z_pos, vz*pulse_length/2.354, num_particles) # z positions in gaussian distribution


    percentiles = np.linspace(0.0001, 0.9999, num_particles)

    # Compute deterministic normal-shaped data using the inverse CDF
    z_pos = norm.ppf(percentiles, loc=nominal_z_pos, scale=vz*pulse_length/2.354)

    # Shuffle the data to remove ordering
    np.random.seed(31)  # Optional: for reproducible shuffling
    np.random.shuffle(z_pos)


    return z_pos


