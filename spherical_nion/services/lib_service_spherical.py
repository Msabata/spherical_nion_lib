# load modules
import sys
import os
import time
from ..engine import lib_laser_wave_FFT
from ..engine import lib_laser_tracing
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy import constants as constants
from ..engine import lib_electron_tracing
from ..engine import lib_laser_tracing
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as mplcolors
from matplotlib.patches import Rectangle
from scipy.interpolate import  RegularGridInterpolator
import matplotlib.patches as mpatches
import scipy.constants as const
# from scipy.signal import fftconvolve
# from scipy.ndimage.filters import maximum_filter
# from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.signal import find_peaks




class Spherical_aberation_solver:
    def __init__(self, acc_voltage_nominal, acc_voltage_FWHM, beam_r, lens1_f, lens2_f, lens1_Cs, lens2_Cs, lens1_Cc, lens2_Cc, demagnification, laser_wavelength, n_electrons, laser_pulse_timelength, dt, tracing_maxiter, laser_direction, electron_initialcomposition="tophat", laserdist=200e-3) -> None:
        """Spherical aberration solver - laser and electrons.

        Args:
            acc_voltage_nominal (float): electron nominal acceleration voltage [ [m]]
            acc_voltage_FWHM (float): full width half maximum [V]
            beam_r (float): initial electron beam radius [m]
            lens1_f (float): fist lans focal distance [m]
            lens2_f (float): second lans focal distance [m]
            lens1_Cs (float): first lens spherical aberration [m]
            lens2_Cs (float): second lens spherical aberration [m]
            lens1_Cc (float): first lens chromatic aberration [m]
            lens2_Cc (float): second lens chromatic aberration [m]
            demagnification (float): demagnificaltion of the electron probe ... ratio of beam diameters in lenses
            laser_wavelength (float): laser wavelength [m]
            n_electrons (int): number of particles to be traced
            laser_pulse_timelength (float): FWHM of laser pulse length [s]
            dt (float): time step while tracing [s]
            tracing_maxiter (int): maximal number or iteration while particle tracing
            laser_direction (+/-1): laser_direction : +/- 1, in positive z direction ... +1,  in negative z direction ... -1
            laserdist (float): distance between laser and electron gun
        """

        self.acc_voltage_nominal = acc_voltage_nominal
        self.acc_voltage_FWHM = acc_voltage_FWHM
        self.beam_r = beam_r
        self.lens1_f = lens1_f
        self.lens2_f = lens2_f
        self.lens1_Cs = lens1_Cs
        self.lens2_Cs = lens2_Cs
        self.lens1_Cc = lens1_Cc
        self.lens2_Cc = lens2_Cc
        self.demagnification = demagnification
        self.laser_wavelength = laser_wavelength
        self.n_electrons = n_electrons
        self.laser_pulse_timelength = laser_pulse_timelength
        self.dt = dt
        self.tracing_maxiter = tracing_maxiter
        self.laser_direction = laser_direction
        self.laserdist = laserdist
        self.electron_initialcomposition = electron_initialcomposition

        # --- Initialize placeholder/default values ---
        self.laser_thick_pulse_energy = 1e-6 # Initial guess, corrected later
        self.defocus = 0.0 # Will be calculated later

    def build_elements(self):
        """
        Constructs the virtual optical bench: calculates lens positions and
        initializes the electron beam's starting positions and velocities.
        """
        func_start_time = time.time()
        print("  Running build_elements()...")

        # --- 1) Prepare lens and plane positions based on optical laws ---
        # These are simple arithmetic calculations and are extremely fast.
        self.WD = 1/(1/self.lens2_f - 1/(self.lens1_f*self.demagnification))
        lens2_z = -self.WD
        lens1_z = lens2_z - self.lens1_f*(self.demagnification+1)
        lens2back_z = -lens2_z
        lens1back_z = -lens1_z
        self.laser_plane_z = lens1_z - (1/(1/self.lens1_f - 1/(lens2_z-lens1_z)))
        self.laserback_plane_z = -self.laser_plane_z

        # --- 2) Initialize the starting positions of all electrons ---
        # PERFORMANCE NOTE: This is the most time-consuming part of this function.
        # Its execution time scales directly with `n_electrons`.
        init_electrons_start_time = time.time()

        if self.electron_initialcomposition == "tophat":
            # Create a square grid of electron starting points.
            points_per_axis = int(np.sqrt(self.n_electrons))
            xs = np.linspace(-self.beam_r, self.beam_r, points_per_axis)
            ys = np.linspace(-self.beam_r, self.beam_r, points_per_axis)
            xx, yy = np.meshgrid(xs, ys)
            n_actual_electrons = np.size(xx)
            z_ini_nominal = self.laser_plane_z - self.laserdist
            zs_ini = lib_electron_tracing.initialize_electronpulse_z(self.acc_voltage_nominal, self.laser_pulse_timelength, n_actual_electrons, z_ini_nominal)
            pos_ini = np.array([xx.flatten(), yy.flatten(), zs_ini])

        elif self.electron_initialcomposition == "oneline":
            # Create a single line of electrons along the x-axis.
            xs = np.linspace(-self.beam_r, self.beam_r, self.n_electrons)
            ys = np.zeros(self.n_electrons)
            z_ini_nominal= self.laser_plane_z -self.laserdist
            zs_ini = lib_electron_tracing.initialize_electronpulse_z(self.acc_voltage_nominal, self.laser_pulse_timelength, self.n_electrons, z_ini_nominal)
            pos_ini = np.array([xs, ys, zs_ini])

        self.z_max = -z_ini_nominal
        print(f"    Time to initialize {self.n_electrons} electrons: {time.time() - init_electrons_start_time:.4f} seconds.")

        # --- 3) Create the Electron Beam object ---
        # This is a fast object instantiation.
        l_elements = [] # Elements will be added later for tracing.
        apertures = np.array([[z_ini_nominal+10e-3, self.beam_r]])
        self.e_beam = lib_electron_tracing.Electron_beam(pos_ini, self.acc_voltage_nominal, self.acc_voltage_FWHM,l_elements, apertures, self.z_max)

        # --- 4) Create Lens objects ---
        # These are also fast object instantiations.
        self.lens1 = lib_electron_tracing.Lens(lens1_z, self.lens1_f)
        self.lens2 = lib_electron_tracing.Lens(lens2_z, self.lens2_f)
        self.lens2back = lib_electron_tracing.Lens(lens2back_z, self.lens2_f+self.defocus, self.lens2_Cs)
        self.lens1back = lib_electron_tracing.Lens(lens1back_z, self.lens1_f, self.lens1_Cs)
        self.lens2_aberated = lib_electron_tracing.Lens(lens2_z, self.lens2_f+self.defocus, self.lens2_Cs, self.lens2_Cc, energy_nominal=self.acc_voltage_nominal)
        self.lens1_aberated = lib_electron_tracing.Lens(lens1_z, self.lens1_f, self.lens1_Cs, self.lens1_Cc, energy_nominal=self.acc_voltage_nominal)

        # --- 5) Calculate final derived variables ---
        # These are fast calculations.
        self.conv_semiang = np.tan(self.beam_r*self.demagnification/self.WD)
        self.dd_jeol = 0.61*np.average(self.e_beam.electron_wavelength_ini)/np.sin(self.conv_semiang)
        self.dc_jeol = self.lens2_Cc * self.acc_voltage_FWHM/self.acc_voltage_nominal * self.conv_semiang
        self.ds_jeol = 0.5*self.lens2_Cs*self.conv_semiang**3
        self.d_jeol = (self.dd_jeol**2 + self.dc_jeol**2 + self.ds_jeol**2)**0.5
        print(f"  build_elements() finished in {time.time() - func_start_time:.4f} seconds.\n")

    def trace_backprop(self, evaluate=True):
        self.e_beam.update_elements([self.lens1, self.lens2, self.lens2back, self.lens1back], self.z_max) 
        self.e_beam.trace(self.dt, self.tracing_maxiter)

        # get positions and angles in positions for laser correction
        det_laserplane_pos = self.e_beam.detector_spacelike(self.laser_plane_z)
        det_laserplaneback_pos = self.e_beam.detector_spacelike(self.laserback_plane_z) * (-1) # flip coordinates in backpropagated pos to agree with forward data



        aligning_error = np.max(abs(det_laserplane_pos[~np.isnan(det_laserplane_pos)]-det_laserplaneback_pos[~np.isnan(det_laserplaneback_pos)]))
        if aligning_error>1e-12:
            raise Exception("alignig error high, check your setup. ELectrons are not at the same positions when comparing the laser plane and the backlaser plane.")

        det_laserplane_vel = self.e_beam.detector_spacelike_velocity(self.laser_plane_z)
        det_laserplaneback_vel = self.e_beam.detector_spacelike_velocity(self.laserback_plane_z)


        self._phase_gradx_1D_frompreviousiteration = None
        # def do_staff():
        #     ######################################################
        #     # compare data and calculate required laser intensity for correction
        #     ######################################################

        #     # compare angles ... angles wihich I want create by laser
        #     anglex_dif = np.arctan2(det_laserplaneback_vel[0],det_laserplaneback_vel[2]) - np.arctan2(det_laserplane_vel[0],det_laserplane_vel[2])
        #     angley_dif = np.arctan2(det_laserplaneback_vel[1],det_laserplaneback_vel[2]) - np.arctan2(det_laserplane_vel[1],det_laserplane_vel[2])

        #     # electron phase gradients
        #     phase_gradx = np.average(self.e_beam.electron_wavenumber_ini) * np.tan(anglex_dif)
        #     phase_grady = np.average(self.e_beam.electron_wavenumber_ini) * np.tan(angley_dif)


        #     ## version to avoid tangenting:
        #     # # compare angles ... angles wihich I want create by laser
        #     # anglex_dif = det_laserplaneback_vel[0]/det_laserplaneback_vel[2] - det_laserplane_vel[0]/det_laserplane_vel[2]
        #     # angley_dif = det_laserplaneback_vel[1]/det_laserplaneback_vel[2] - det_laserplane_vel[1]/det_laserplane_vel[2]
        #     # # electron phase gradients
        #     # phase_gradx = np.average(self.e_beam.electron_wavenumber_ini) * anglex_dif
        #     # phase_grady = np.average(self.e_beam.electron_wavenumber_ini) * angley_dif



            

        #     # reshape into regular mesh
        #     # laserplane_xmax = np.max(det_laserplane_pos)

        #     phase_gradx_mesh = np.reshape(phase_gradx,(int(np.sqrt(self.n_electrons)),int(np.sqrt(self.n_electrons))))
        #     phase_grady_mesh = np.reshape(phase_grady,(int(np.sqrt(self.n_electrons)),int(np.sqrt(self.n_electrons))))

        #     laserplane_xx_mesh = np.reshape(det_laserplane_pos[0], (int(np.sqrt(self.n_electrons)),int(np.sqrt(self.n_electrons))))

        #     # continue in >>> 1D <<< integrate (suppose axial symetry)
        #     phase_gradx_1D = phase_gradx_mesh[int(np.sqrt(self.n_electrons)/2),:]
        #     laser_plane_xs = laserplane_xx_mesh[int(np.sqrt(self.n_electrons)/2),:]
        #     # remove nans
        #     phase_gradx_1D = phase_gradx_1D[~np.isnan(laser_plane_xs)] # do it according xs in both cases, sinc only xs has nans
        #     laser_plane_xs = laser_plane_xs[~np.isnan(laser_plane_xs)]
        #     # print(laser_plane_xs)
        #     # print(phase_gradx_1D)
        #     if laser_plane_xs.size != phase_gradx_1D.size:
        #         raise Exception("1D arrays have different number of points! {} and {}".format(laser_plane_xs.size,phase_gradx_1D.size))


        #     # print("phase_gradx_1D : " +str(phase_gradx_1D))
        #     if self._phase_gradx_1D_frompreviousiteration is not None:
        #         phase_gradx_1D += self._phase_gradx_1D_frompreviousiteration
            

        #     phase_shift_x = np.cumsum(phase_gradx_1D)*np.gradient(laser_plane_xs)

        #     # shift phaseshift to be only negative (... and the laser intensity will be only positive)
        #     phase_shift_x = phase_shift_x - np.max(phase_shift_x)

        #     # calculate laser El. field distribution (accodrding the equation (D11) from the letter TRANSVERSE ELECTRON BEAM SHAPING WITH LIGHT (Mihaila 2022))
        #     laserideal_Intensity_distr_artifitial_units = - phase_shift_x # ... ElField**2 [a.u.]
        #     if any(laserideal_Intensity_distr_artifitial_units<0):
        #         raise Exception("Idela laser has negative values!")

        #     ##############################################################
        #     # 5) Calculate laser pulse energy
        #     ##############################################################
        #     # specify laser properties
        #     finestructureconstant = 0.0072973525693
        #     beta = np.average(det_laserplane_vel) / self.e_beam.c # electron_speed / light_speed
        #     electron_energy = (np.average(self.e_beam.m_poly)*self.e_beam.c**2)
        #     laser_direction = -1 # laser_direction : +/- 1, in positive z direction ... +1,  in negative z direction ... -1

        #     def integral_phase_planedistr(xs, phase):
        #         dx = np.gradient(xs)
        #         integral_phase_planedistr = np.sum(phase * np.pi *abs(xs) * dx)
        #         return integral_phase_planedistr

        #     laser_pulse_energy = (
        #         2*np.pi * (1-laser_direction*beta) * electron_energy * (-1)*integral_phase_planedistr(laser_plane_xs,phase_shift_x) 
        #         / (finestructureconstant*self.laser_wavelength**2))
        #     #print('laser_pulse_energy= {} J'.format(laser_pulse_energy))

        #     # store variables
        #     self.laserideal_xs = laser_plane_xs
        #     self.laserideal_Intensity_distr_artifitial_units = laserideal_Intensity_distr_artifitial_units
        #     self.laserideal_pulse_energy = laser_pulse_energy #[J]
        #     self._phase_gradx_1D_frompreviousiteration = phase_gradx_1D

        


        if evaluate:
            self._evaluate_backpropagation_precise(det_laserplaneback_vel, det_laserplane_vel, det_laserplane_pos)
            # trace ideal correction
            # compare with backpropagation
            # add gradients

            # pass
            # plt.plot(laserideal_Intensity_distr_artifitial_units)
            # plt.title("laserideal_Intensity_distr_artifitial_units")
            # plt.show()

            for c_iter in range(10):
                self.trace_idealcorrected()
                det_laserplane_vel = self.e_beam.detector_spacelike_velocity(self.laser_plane_z+1e-9)
                print("D50 ideal = {}".format(self.detector_idealcorrection_D50))
                self._evaluate_backpropagation_precise(det_laserplaneback_vel, det_laserplane_vel, det_laserplane_pos)
            self._phase_gradx_1D_frompreviousiteration = None

    def trace_backprop_oneline(self):
        """Do backpropagation with electrons only at x axis. In one line. Expecting axialy symetric abberations only.

        Args:
            evaluate (bool, optional): _description_. Defaults to True.

        Raises:
            Exception: _description_
        """
        ######################################################
        # 1) prepare electron beam with electrons in one line on the x axis
        ######################################################
        z_ini_nominal= self.laser_plane_z -30e-3 # initial z positions
        l_elements = [] # ad and update elements later
        apertures = np.array([[z_ini_nominal+10e-3, self.beam_r]]) #[aperture, z_pos/radius]
        xs = np.linspace(-self.beam_r, self.beam_r, int(self.n_electrons/100))
        pos_ini = np.array([xs, xs*0, xs*0+ z_ini_nominal])
        self.e_beam = lib_electron_tracing.Electron_beam(pos_ini, self.acc_voltage_nominal, self.acc_voltage_FWHM,l_elements, apertures, self.z_max)
        self.e_beam.update_elements([self.lens1, self.lens2, self.lens2back, self.lens1back], self.z_max) 
        self.e_beam.trace(self.dt, self.tracing_maxiter)
        ######################################################
        # 2) get positions and angles in positions for laser correction
        ######################################################
        det_laserplane_pos = self.e_beam.detector_spacelike(self.laser_plane_z)
        det_laserplaneback_pos = self.e_beam.detector_spacelike(self.laserback_plane_z) * (-1) # flip coordinates in backpropagated pos to agree with forward data

        aligning_error = np.max(abs(det_laserplane_pos[~np.isnan(det_laserplane_pos)]-det_laserplaneback_pos[~np.isnan(det_laserplaneback_pos)]))
        if aligning_error>1e-12:
            raise Exception("alignig error high, check your setup. ELectrons are not at the same positions when comparing the laser plane and the backlaser plane.")

        det_laserplane_vel = self.e_beam.detector_spacelike_velocity(self.laser_plane_z)
        det_laserplaneback_vel = self.e_beam.detector_spacelike_velocity(self.laserback_plane_z)



        ######################################################
        # 3 compare data and calculate required laser intensity for correction - all on the x axis
        ######################################################

        # compare angles ... angles wihich I want create by laser
        anglex_dif = np.arctan2(det_laserplaneback_vel[0],det_laserplaneback_vel[2]) - np.arctan2(det_laserplane_vel[0],det_laserplane_vel[2])

        # electron phase gradients
        phase_gradx = np.average(self.e_beam.electron_wavenumber_ini) * np.tan(anglex_dif)


        laser_plane_xs = det_laserplane_pos[0]
        phase_gradx_1D = phase_gradx[~np.isnan(laser_plane_xs)] # do it according xs in both cases, sinc only xs has nans
        laser_plane_xs = laser_plane_xs[~np.isnan(laser_plane_xs)]


        if laser_plane_xs.size != phase_gradx_1D.size:
            raise Exception("1D arrays have different number of points! {} and {}".format(laser_plane_xs.size,phase_gradx_1D.size))


        phase_shift_x = np.cumsum(phase_gradx_1D)*np.gradient(laser_plane_xs)
        phase_shift_x = (phase_shift_x + np.flip(phase_shift_x))/2 # correction of the cumsum - remove the directional artefact

        # shift phaseshift to be only negative (... and the laser intensity will be only positive)
        phase_shift_x = phase_shift_x - np.max(phase_shift_x)

        # calculate laser El. field distribution (accodrding the equation (D11) from the letter TRANSVERSE ELECTRON BEAM SHAPING WITH LIGHT (Mihaila 2022))
        laserideal_Intensity_distr_artifitial_units = - phase_shift_x # ... ElField**2 [a.u.]
        if any(laserideal_Intensity_distr_artifitial_units<0):
            raise Exception("Idela laser has negative values!")

        ##############################################################
        # 5) Calculate laser pulse energy
        ##############################################################
        # specify laser properties
        finestructureconstant = 0.0072973525693
        beta = np.average(np.linalg.norm(det_laserplane_vel, axis=0)) / self.e_beam.c # electron_speed / light_speed
        electron_energy = (np.average(self.e_beam.m_poly)*self.e_beam.c**2)
        def integral_phase_planedistr(xs, phase):
            dx = np.gradient(xs)
            integral_phase_planedistr = np.sum(phase * np.pi *abs(xs) * dx)
            return integral_phase_planedistr

        laser_pulse_energy = (
            2*np.pi * (1-self.laser_direction*beta) * electron_energy * (-1)*integral_phase_planedistr(laser_plane_xs,phase_shift_x) 
            / (finestructureconstant*self.laser_wavelength**2))

        # store variables
        self.laserideal_xs = laser_plane_xs
        self.laserideal_Intensity_distr_artifitial_units = laserideal_Intensity_distr_artifitial_units
        self.laserideal_pulse_energy = laser_pulse_energy #[J]
        self._phase_gradx_1D_frompreviousiteration = phase_gradx_1D

    def calculate_laser_analycialy(self, save=True, plot=False, aberration='s', zs0=None, npix=1000):
        """ Analyticaly calculate ideal intensity distribution of laser in the interaction region which is in the conjugated plane to the objective lens.
        Suppose that the lens1 is without aberrations

                                    |  |
                                    |  |
        0 laser    -----------------|--x0-------------------- 
                                    |   \
                    d0              |    \
                                    |     \
        1 lens1    -----------------|-----x1-----------------
                                    |     / 
                                    |    /
                                    |   /
                                    |  /
                                    | /
                                    |/
                                    /
                                   /|
                    d1            / |
                                 /  |
                                /   |
                               /    |
                              /     |                                  
                             /      |
                            /       |
                           /        |
                          /         |
        2 lens2    ------x2---------|------------------------
                           \        |
                    d2        \     |
                                 \  |
        3 sample   -----------------0------------------------

        zs0 (array): distances along the z axis from the center of the the interaction volume.
        """

        # 1) calculate electron trajectories in corrected beam
        d0 = self.lens1.position_z[0] - self.laser_plane_z
        d1 = self.lens2.position_z[0] - self.lens1.position_z[0]
        d2 = 0 - self.lens2.position_z[0]

        #f2_ideal = self.lens2_f
        f2 = self.lens2_aberated.focal_dist
        Cs2 = self.lens2_Cs

        _beam_r_extended = self.beam_r+2e-8 # extend the laser beam radius to be sure, that the laser is wider than electron beam.
        xs0 = np.linspace(-_beam_r_extended, _beam_r_extended, npix) # electron positions in laser plane
        xs2 = - xs0 * d1/d0 # image is upsidedown and magnified # electron positions in lens2
        xs1 = -(-xs2/d2 + xs2/f2 + Cs2*xs2**3/f2**4) * d1 + xs2 # electron positions in lens1
        xs3 = xs0*0 # in the sample plane all focused in one spot
        dx0 = xs0[1]-xs0[0]


        #############################################
        if plot:
            self.trace_backprop_oneline()




            legend_list = []
            fig, ax = plt.subplots(1,1)
            ray_x =  self.e_beam.a_pos[:,0,:]
            ray_z =  self.e_beam.a_pos[:,2,:]
            ax.plot(ray_x, ray_z, 'g')
            legend_list.append(mpatches.Patch(color='green', label="backpropagation"))
            legend_list.append(mpatches.Patch(color='blue', label="analytical"))
            ax.plot(ray_x, ray_z*(-1), 'orange')
            legend_list.append(mpatches.Patch(color='orange', label="backpropagation-inverted"))
            #ax.plot(self.e_beam.a_pos[:,0,int(self.n_electrons/2-np.sqrt(self.n_electrons)/2) : int(self.n_electrons/2+np.sqrt(self.n_electrons)/2)], self.e_beam.a_pos[:,2,int(self.n_electrons/2-np.sqrt(self.n_electrons)/2) : int(self.n_electrons/2+np.sqrt(self.n_electrons)/2)], 'g')
            # plot aperture
            _ap = self.e_beam.a_apertures
            for c_ap in range (np.size(_ap, axis=0)):
                ax.plot([-10*_ap[c_ap,1],-_ap[c_ap,1]],[_ap[c_ap,0],_ap[c_ap,0]], 'k')
                ax.plot([ 10*_ap[c_ap,1], _ap[c_ap,1]],[_ap[c_ap,0],_ap[c_ap,0]], 'k')
            for c_element in range(np.size(self.e_beam.l_elements)):
                _z = self.e_beam.l_elements[c_element].position_z[1]
                ax.plot([-0.2e-3,0.2e-3],[_z,_z], 'r')
            # plot conjugated planes
            ax.plot([-0.2e-3,0.2e-3],[self.laser_plane_z,self.laser_plane_z], 'r')
            ax.plot([-0.2e-3,0.2e-3],[self.laserback_plane_z,self.laserback_plane_z], 'r')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('z [m]')
            ax.legend(handles=legend_list)
            ax.invert_yaxis()
            ax.set_title('Backtracing\nacceleration voltage nominal {} kV\nNo.Particles{}\n convsemiang={}mrad'.format(self.acc_voltage_nominal/1000, self.n_electrons, self.conv_semiang*1000))


            ax.plot(np.array([xs0,xs1,xs2,xs3]),np.array([self.laser_plane_z,self.lens1.position_z[0] ,self.lens2.position_z[0],0]), 'b', linewidth=3, alpha=0.5, label="analytical")



            plt.show()
        #############################################

        # 2) calculate laser induced phasechange
        k = np.average(self.e_beam.electron_wavenumber_ini)
        if aberration == 's': # correct only spherical aberration
            if self.defocus ==0:
                phasechange = xs0**4 * (k*d1**4*Cs2/(f2**4*d0**4*4)) # no defocus
            else:
                phasechange = xs0**4 * (k*d1**4*Cs2/(f2**4*d0**4*4)) + xs0**2*k/d0 * (-d1**2/(2*d0*d2) + d1**2/(2*d0*f2) - d1/(2*d0) - 0.5) # consider defocus
                
                # print("d0 {}".format(d0))
                # print("d1 {}".format(d1))
                # print("d2 {}".format(d2))
                # print("Cs2 {}".format(Cs2))
                # print("f2 {}".format(f2))

                # plt.close()
                # plt.plot(xs0, phasechange)
                # plt.show()


        elif aberration == 'sc': # correct both spherical and chromatic
            
            # calculate local acc voltages
            acc_poly = (-1+(1/(1-(((zs0+self.laserdist)/self.laserdist)**2*(1-(1+(const.elementary_charge*self.acc_voltage_nominal)/(const.m_e*const.c**2))**(-2)))))**0.5)*const.m_e*const.c**2/const.elementary_charge
            f2_polychromatic = f2+self.lens2_Cc*(acc_poly-self.acc_voltage_nominal)/self.acc_voltage_nominal
            k_poly = 2*np.pi*(2*const.m_e*const.elementary_charge*acc_poly*(1+const.elementary_charge/(2*const.m_e*const.c**2)*acc_poly))**0.5/const.Planck
            phasechange = np.full((np.size(xs0), np.size(zs0)), np.nan)
            for c, z0 in enumerate(zs0):
                phasechange[:,c] = xs0**4 * (k_poly[c]*d1**4*Cs2/(f2_polychromatic[c]**4*d0**4*4)) + xs0**2*k_poly[c]/d0 * (-d1**2/(2*d0*d2) + d1**2/(2*d0*f2_polychromatic[c]) - d1/(2*d0) - 0.5) # consider defocus
            


            # print("d0 {}".format(d0))
            # print("d1 {}".format(d1))
            # print("d2 {}".format(d2))
            # print("Cs2 {}".format(Cs2))
            # print("f2 {}".format(f2))

            # plt.close()
            # plt.plot(xs0, phasechange)
            # plt.show()


        elif aberration == 'c': # correct only chromatic
            # calculate local acc voltages
            acc_poly = (-1+(1/(1-(((zs0+self.laserdist)/self.laserdist)**2*(1-(1+(const.elementary_charge*self.acc_voltage_nominal)/(const.m_e*const.c**2))**(-2)))))**0.5)*const.m_e*const.c**2/const.elementary_charge
            f2_polychromatic = f2+self.lens2_Cc*(acc_poly-self.acc_voltage_nominal)/self.acc_voltage_nominal
            k_poly = 2*np.pi*(2*const.m_e*const.elementary_charge*acc_poly*(1+const.elementary_charge/(2*const.m_e*const.c**2)*acc_poly))
            phasechange = np.full((np.size(xs0), np.size(zs0)), np.nan)
            for c, z0 in enumerate(zs0):
                phasechange[:,c] = xs0**2*k_poly[c]/d0 * (-d1**2/(2*d0*d2) + d1**2/(2*d0*f2_polychromatic[c]) - d1/(2*d0) - 0.5) # consider defocus

        else: raise Exception('aberration kind not recognized set either s(spherical) or sc(spherical and chromatic)')

        phasechange -= np.max(phasechange) # phasechange can be only negative
        

        #############################################
        if plot:

            # numericaly calculate the phasechange
            phasechange_numerical = k/d0 * np.cumsum(xs1-xs0)*dx0
            phasechange_numerical = (phasechange_numerical+np.flip(phasechange_numerical))/2
            phasechange_numerical -=np.max(phasechange_numerical)
            

            fig, ax = plt.subplots(1,1)
            #ax.plot(self.laserideal_xs, self.laserideal_Intensity_distr_artifitial_units / np.max(abs(self.laserideal_Intensity_distr_artifitial_units)) * np.max(abs(phasechange[1:-1]-phasechange[1]))*(-1)+phasechange[1], label="-1*laserideal_Intensity_distr_artifitial_units [a.u.]")
            ax.plot(self.laserideal_xs, self.laserideal_Intensity_distr_artifitial_units / np.max(abs(self.laserideal_Intensity_distr_artifitial_units)) * np.max(abs(phasechange))*(-1), label="-1*laserideal_Intensity_distr_artifitial_units [a.u.]")

            ax.plot(xs0, phasechange, label="phasechange [rad]")
            ax.plot(xs0, phasechange_numerical, '--',label="phasechange_numerical [rad]")
            ax.legend()
            plt.show()
        #############################################

        # 3) calculate laser intensity distribution
            
        laser_intensity_artifitial_units = -phasechange 
        laser_intensity_artifitial_units = laser_intensity_artifitial_units - np.min(laser_intensity_artifitial_units)
        # plt.plot(xs0,laser_intensity_artifitial_units)
        # plt.show()
        #############################################
        if plot:
            fig, ax = plt.subplots(1,1)
            ax.plot(xs0,laser_intensity_artifitial_units)
            plt.show()
        #############################################


        # 4) calculate laser pulse energy
        finestructureconstant = 0.0072973525693
        beta = np.average(np.linalg.norm(self.e_beam.v_ini_poly, axis=0)) / self.e_beam.c # electron_speed / light_speed
        electron_energy = (np.average(self.e_beam.m_poly)*self.e_beam.c**2)
        
        if self.defocus ==0 and aberration=='s': 
            print("energy calculated analyticaly")
            laser_pulse_energy = (np.pi**2 * electron_energy * (1-self.laser_direction*beta))/(finestructureconstant*self.laser_wavelength**2) * (_beam_r_extended**6 * k * d1**4 * Cs2)/(f2**4 * d0**4) * 1/3
        else: # evaluate numerically
            print("laser pulse energy calculated numerically")
            if aberration=='sc' or aberration == 'c':
                electron_energy = const.m_e*const.c**2 + const.elementary_charge*acc_poly
                laser_pulse_energy = np.full(np.size(zs0), np.nan)
                v_poly = const.c*(1-(1+const.elementary_charge*acc_poly/(const.m_e*const.c**2))**(-2))**(0.5)
                beta_poly =  v_poly/ self.e_beam.c # electron_speed / light_speed
                for c,z0 in enumerate(zs0):
                    phasechange_integral = np.sum(phasechange[:,c] *np.pi*np.abs(xs0)*dx0)
                    laser_pulse_energy[c] = (np.pi*2 * electron_energy[c] * (1-self.laser_direction*beta_poly[c]))/(finestructureconstant*self.laser_wavelength**2) * (- phasechange_integral)
            else:
                phasechange_integral = np.sum(phasechange *np.pi*np.abs(xs0)*dx0)
                laser_pulse_energy = (np.pi*2 * electron_energy * (1-self.laser_direction*beta))/(finestructureconstant*self.laser_wavelength**2) * (- phasechange_integral)
            print("Note: Electron energy calculated numerically, because the defocus is not zero, or aberrations are not spherical only.")
        #############################################
        if plot:
            print("analytical laser pulse energy = {} J".format(laser_pulse_energy))
            print("numerical  laser pulse energy = {} J".format(self.laserideal_pulse_energy))
            #print("defocus = {}".format(self.defocus))
        #############################################

        # 5) check if the laser makes the correct phasechange and trajectory
        #############################################
        if plot:

            self.laserideal_Intensity_distr_artifitial_units = laser_intensity_artifitial_units
            self.laserideal_pulse_energy = laser_pulse_energy
            self.build_elements()
            self.trace_idealcorrected()
            det_above = self.e_beam.detector_spacelike(-5e-9, nans=False)
            det_below = self.e_beam.detector_spacelike(5e-9, nans=False)
            d50_above = np.median(np.linalg.norm(det_above, axis=0))
            d50_below = np.median(np.linalg.norm(det_below, axis=0))

            print("self.detector_idealcorrection_D50 = {}".format(d50_above))
            print("self.detector_idealcorrection_D50 = {}".format(self.detector_idealcorrection_D50))
            print("self.detector_idealcorrection_D50 = {}".format(d50_below))
            self.plot_electron_tracing_fine()

            plt.scatter(det_above[0], det_above[1], marker='.', s=1, color="green")
            plt.scatter(det_below[0], det_below[1], marker='.', s=1, color="red")
            plt.scatter(self.detector_idealcorrection[0], self.detector_idealcorrection[1], marker='.', s=1, color="black")
            plt.show()
        #############################################

        # 6) save data
        if save:
            self.laserideal_pulse_energy = laser_pulse_energy
            self.laserideal_Intensity_distr_artifitial_units = laser_intensity_artifitial_units
            self.laserideal_xs = xs0

    def analysis_aberated(self, beam_rs=None, trace=True, plot=False):
        """_summary_

        Parameters
        ----------
        beam_rs : _type_, optional
            _description_, by default None
        trace : bool, optional
            _description_, by default True
        plot : bool, optional
            _description_, by default False

        ## return
        return(r3_50,r3_90,r3_max) [m]
        """
        if beam_rs is None:
            beam_rs = np.array([self.beam_r])
            

            if trace:
                self.trace_aberrated()
                traced_rs3 = np.linalg.norm(self.detector_aberrated, axis=0)
                traced_rs3_hist, traced_hist_bins = np.histogram(traced_rs3,bins=100, density=True)

                plt.scatter(self.detector_aberrated[0],self.detector_aberrated[1])
                plt.show()

        conv_semiangs = np.tan(beam_rs*self.demagnification/self.WD)
        # JEOL
        Ds_jeol = 0.5*self.lens2_Cs*conv_semiangs**3
        Dd_jeol = 0.61*np.average(self.e_beam.electron_wavelength_ini)/np.sin(conv_semiangs)
        Dc_jeol = self.lens2_Cc*self.acc_voltage_FWHM/self.acc_voltage_nominal*conv_semiangs

        D_jeol = (Dd_jeol**2+Ds_jeol**2+Dc_jeol**2)**0.5

        # My calculation of marginal radiuses
        
        f1 = self.lens1_aberated.focal_dist
        f2 = self.lens2_aberated.focal_dist
        d0 = self.lens1.position_z[1] - self.laser_plane_z # distance laser to lens1
        d1 = self.lens2.position_z[1] - self.lens1.position_z[1] # distance lens1 to lens2
        d2 = 0 - self.lens2.position_z[1] # distnace lens2 to sample
        R0 = beam_rs
        R1 = R0*1
        
        R2 = -R1 * (d1-f1)/f1
        R3 = R2 - d2*R1/f1 - d2*R2/f2 - d2*self.lens2_Cs *R2**3/f2**4 # marginal radius in sample plane

        # electron intensity distribution
        if np.size(beam_rs) == 1:
            rs1= np.linspace(0,R1[0],1000)
            rs2 = -rs1*(d1-f1)/f1
            rs3 = rs2 - d2*rs1/f1 - d2*rs2/f2 - d2*self.lens2_Cs*rs2**3/f2**4
            Ir3 = -(-(d1-f1)/f1 - d1/f1 + (d1-f1)*d2/(f1*f2) + self.lens2_Cs*((d1-f1)/f1)**3*rs1**2*3*d2/f2**4)**(-1)*2*np.pi*rs1 # [electrons/m]
            r3_max = np.max(np.abs(rs3))
            

            Ir3_histogram,Ir3_hist_bins = np.histogram(np.abs(rs3),100,weights=Ir3, density=True)
            integral = np.cumsum(Ir3_histogram)
            r3_50 = Ir3_hist_bins[np.argmin(np.abs(integral/np.max(integral)-0.5))]
            r3_90 = Ir3_hist_bins[np.argmin(np.abs(integral/np.max(integral)-0.9))]

            if plot:
                fig,ax = plt.subplots(1,1)
                Ir3dx3 = Ir3/np.gradient(rs3)
                # ax.plot(rs3,Ir3dx3 /np.max(Ir3dx3)*np.max(traced_rs3_hist) * 100, label="Ir3/dx3", marker='.')
                ax.plot(rs3,Ir3dx3 /Ir3dx3[-1]*traced_rs3_hist[-1] , label="Ir3/dx3", marker='.')
                ax.set_xlabel("r3 [m]")
                ax.set_ylabel("[electrons/m]")
                ax.plot(R3,0,'o', label="R3")
                ax.plot(Ds_jeol/2,0,'o', label="Ds_jeol")
                ax.plot(r3_max,0,'o', label="r3_max")
                ax.plot(r3_50,0,'o', label="r3_50")
                ax.plot(r3_90,0,'o', label="r3_90")
            
                ax.plot(traced_hist_bins[1:],traced_rs3_hist, label="histogram from tracing")
                ax.plot(Ir3_hist_bins[1:],Ir3_histogram, label="histogram from analytical")
                #ax.hist(traced_rs3, bins=100,density=True)
                ax.legend()
                ax.set_title("defocus= {} m".format(self.defocus))
                
                plt.show()

        if plot:
            fig, ax = plt.subplots(1,1)
            ax.plot(conv_semiangs, Ds_jeol, label="Ds_jeol")
            ax.plot(conv_semiangs, Dd_jeol, label="Dd_jeol")
            ax.plot(conv_semiangs, Dc_jeol, label="Dc_jeol")
            ax.plot(conv_semiangs, D_jeol, label="D_jeol")
            ax.plot(conv_semiangs, R3*2, label="R3*2")
            ax.legend()
            plt.show()

        return(r3_50,r3_90,r3_max)

    def find_best_defocus_aberated(self, iterations=3, steps=21, defocus_min = -10e-6, defocus_max = 10e-6):
        best_defocus = np.nan
        sequence_defocus = np.linspace(defocus_min,defocus_max,steps)
        array_r90 = np.full(steps,np.nan)
        for c_iter in range(iterations):
            print(c_iter)
            for c_defocus in range(steps):
                self.defocus = sequence_defocus[c_defocus]
                self.build_elements()
                array_r90[c_defocus],_,_ = self.analysis_aberated(beam_rs = np.array([self.beam_r]), trace=False, plot=False)
            
            best_defocus_idx = np.argmin(array_r90)
            best_defocus = sequence_defocus[best_defocus_idx]
            print(best_defocus)
            print(array_r90[best_defocus_idx])
            print()
            sequence_defocus = np.linspace(sequence_defocus[best_defocus_idx-1],sequence_defocus[best_defocus_idx+1],steps)
        self.defocus = best_defocus
        return(best_defocus)
    
    def find_best_defocus_lasercorrected(self, iterations=3, steps=21, defocus_min = -10e-6, defocus_max = 10e-6):
        best_defocus = np.nan
        sequence_defocus = np.linspace(defocus_min,defocus_max,steps)
        array_r90 = np.full(steps,np.nan)
        for c_iter in range(iterations):
            print(c_iter)
            for c_defocus in range(steps):
                self.defocus = sequence_defocus[c_defocus]
                self.build_elements()
                # trace thin laser
                self.trace_corrected_real(laser="thin")
                array_r90[c_defocus] = self.detector_realcorrection_D90
            best_defocus_idx = np.argmin(array_r90)
            best_defocus = sequence_defocus[best_defocus_idx]
            print(best_defocus)
            print(array_r90[best_defocus_idx])
            print()
            sequence_defocus = np.linspace(sequence_defocus[best_defocus_idx-1],sequence_defocus[best_defocus_idx+1],steps)
        self.defocus = best_defocus
        return(best_defocus)

    def _evaluate_backpropagation_precise(self, det_laserplaneback_vel, det_laserplane_vel, det_laserplane_pos):
        print("precise backprop evaluation")
        ######################################################
        # compare data and calculate required laser intensity for correction
        ######################################################

        # # compare angles ... angles wihich I want create by laser
        # anglex_dif = np.arctan2(det_laserplaneback_vel[0],det_laserplaneback_vel[2]) - np.arctan2(det_laserplane_vel[0],det_laserplane_vel[2])
        # angley_dif = np.arctan2(det_laserplaneback_vel[1],det_laserplaneback_vel[2]) - np.arctan2(det_laserplane_vel[1],det_laserplane_vel[2])

        # # electron phase gradients
        # phase_gradx = np.average(self.e_beam.electron_wavenumber_ini) * np.tan(anglex_dif)
        # phase_grady = np.average(self.e_beam.electron_wavenumber_ini) * np.tan(angley_dif)


        # version to avoid tangenting:
        # compare angles ... angles wihich I want create by laser
        phase_gradx = det_laserplaneback_vel[0]/det_laserplaneback_vel[2] - det_laserplane_vel[0]/det_laserplane_vel[2]


        #angley_dif = det_laserplaneback_vel[1]/det_laserplaneback_vel[2] - det_laserplane_vel[1]/det_laserplane_vel[2]

        n = 940
        print("\n")
        print("det_laserplaneback_vel[0]")
        print(det_laserplaneback_vel[0,n])
        print("det_laserplaneback_vel[2]")
        print(det_laserplaneback_vel[2,n])
        print("det_laserplaneback_vel[0]/det_laserplaneback_vel[2]")
        print(det_laserplaneback_vel[0][n]/det_laserplaneback_vel[2][n])
        print("")

        

        print("det_laserplane_vel[0]")
        print(det_laserplane_vel[0,n])
        print("det_laserplane_vel[2]")
        print(det_laserplane_vel[2,n])
        print("det_laserplane_vel[0]/det_laserplane_vel[2]")
        print(det_laserplane_vel[0][n]/det_laserplane_vel[2][n])

        print("")
        print("phase_gradx")
        print(phase_gradx[n])

        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # ax.plot_surface(det_laserplane_pos[0], det_laserplane_pos[1], det_laserplane_vel[0], cmap='viridis', edgecolor='green')
        # plt.show()
        plt.plot(det_laserplane_vel[0][~np.isnan(det_laserplane_pos[0])], marker='.', label='det_laserplane_vel[0]')
        plt.plot(det_laserplaneback_vel[0][~np.isnan(det_laserplane_pos[0])], marker='.', label='det_laserplaneback_vel[0]')
        plt.plot(phase_gradx[~np.isnan(det_laserplane_pos[0])]/np.max(phase_gradx[~np.isnan(det_laserplane_pos[0])])*np.max(det_laserplaneback_vel[0][~np.isnan(det_laserplane_pos[0])]), marker='.', label='phase_gradx scaled')
        plt.title("det_laserplane_vel x")
        plt.legend()
        plt.show()

        plt.plot((det_laserplane_vel[0][~np.isnan(det_laserplane_pos[0])])/np.max(det_laserplane_vel[0][~np.isnan(det_laserplane_pos[0])] )* np.max(det_laserplaneback_vel[0][~np.isnan(det_laserplane_pos[0])]), marker='.', label='det_laserplane_vel[0] scaled')
        plt.plot(det_laserplaneback_vel[0][~np.isnan(det_laserplane_pos[0])], marker='.', label='det_laserplaneback_vel[0]')
        plt.title("det_laserplane_vel x scaled")
        plt.legend()
        plt.show()


        plt.plot(phase_gradx, marker='.', label='phase_gradx')
        plt.title("phase_gradx")
        plt.show()

        plt.plot(det_laserplane_vel[2][~np.isnan(det_laserplane_pos[0])], marker='.', label='det_laserplane_vel[2]')
        plt.plot(det_laserplaneback_vel[2][~np.isnan(det_laserplane_pos[0])], marker='.', label='det_laserplaneback_vel[2]')
        plt.title("det_laserplane_vel z")
        plt.show()
        plt.plot(det_laserplane_vel[0][~np.isnan(det_laserplane_pos[0])]/det_laserplane_vel[2][~np.isnan(det_laserplane_pos[0])], marker='.', label='det_laserplane_vel')
        plt.plot(det_laserplaneback_vel[0][~np.isnan(det_laserplane_pos[0])]/det_laserplaneback_vel[2][~np.isnan(det_laserplane_pos[0])], marker='.', label='det_laserplaneback_vel')
        plt.title("det_laserplane_vel x/z")
        plt.show()
        plt.plot(det_laserplaneback_vel[0][~np.isnan(det_laserplane_pos[0])]/det_laserplaneback_vel[2][~np.isnan(det_laserplane_pos[0])] - det_laserplane_vel[0][~np.isnan(det_laserplane_pos[0])]/det_laserplane_vel[2][~np.isnan(det_laserplane_pos[0])])
        plt.title("det_laserplane_vel dif")
        plt.show()

      



        # electron phase gradients
        #phase_gradx = anglex_dif
        #phase_grady = np.average(self.e_beam.electron_wavenumber_ini) * angley_dif



        

        # reshape into regular mesh
        # laserplane_xmax = np.max(det_laserplane_pos)

        phase_gradx_mesh = np.reshape(phase_gradx,(int(np.sqrt(self.n_electrons)),int(np.sqrt(self.n_electrons))))
        #phase_grady_mesh = np.reshape(phase_grady,(int(np.sqrt(self.n_electrons)),int(np.sqrt(self.n_electrons))))

        laserplane_xx_mesh = np.reshape(det_laserplane_pos[0], (int(np.sqrt(self.n_electrons)),int(np.sqrt(self.n_electrons))))

        # continue in >>> 1D <<< integrate (suppose axial symetry)
        phase_gradx_1D = phase_gradx_mesh[int(np.sqrt(self.n_electrons)/2),:]
        laser_plane_xs = laserplane_xx_mesh[int(np.sqrt(self.n_electrons)/2),:]





        # remove nans
        phase_gradx_1D = phase_gradx_1D[~np.isnan(laser_plane_xs)] # do it according xs in both cases, sinc only xs has nans
        laser_plane_xs = laser_plane_xs[~np.isnan(laser_plane_xs)]

        plt.plot(laser_plane_xs,phase_gradx_1D)
        plt.title('Check selection')
        plt.show()

        # print(laser_plane_xs)
        # print(phase_gradx_1D)
        if laser_plane_xs.size != phase_gradx_1D.size:
            raise Exception("1D arrays have different number of points! {} and {}".format(laser_plane_xs.size,phase_gradx_1D.size))


        if self._phase_gradx_1D_frompreviousiteration is not None:
            phase_gradx_1D += self._phase_gradx_1D_frompreviousiteration
        

        phase_shift_x = np.cumsum(phase_gradx_1D)*np.gradient(laser_plane_xs)
        #phase_shift_x = np.cumsum(phase_gradx_1D)* (laser_plane_xs[1]-laser_plane_xs[0])#np.gradient(laser_plane_xs)

        # shift phaseshift to be only negative (... and the laser intensity will be only positive)
        phase_shift_x = phase_shift_x - np.max(phase_shift_x)

        # calculate laser El. field distribution (accodrding the equation (D11) from the letter TRANSVERSE ELECTRON BEAM SHAPING WITH LIGHT (Mihaila 2022))
        laserideal_Intensity_distr_artifitial_units = - phase_shift_x # ... ElField**2 [a.u.]
        if any(laserideal_Intensity_distr_artifitial_units<0):
            raise Exception("Idela laser has negative values!")

        ##############################################################
        # 5) Calculate laser pulse energy
        ##############################################################
        # specify laser properties
        finestructureconstant = 0.0072973525693
        beta = np.average(det_laserplane_vel) / self.e_beam.c # electron_speed / light_speed
        electron_energy = (np.average(self.e_beam.m_poly)*self.e_beam.c**2)
        #laser_direction =  #-1 # laser_direction : +/- 1, in positive z direction ... +1,  in negative z direction ... -1

        def integral_phase_planedistr(xs, phase):
            dx = np.gradient(xs)
            integral_phase_planedistr = np.sum(phase * np.pi *abs(xs) * dx)
            return integral_phase_planedistr

        laser_pulse_energy = (
            2*np.pi * (1-self.laser_direction*beta) * electron_energy * (-1)*integral_phase_planedistr(laser_plane_xs,phase_shift_x) 
            / (finestructureconstant*self.laser_wavelength**2) * np.average(self.e_beam.electron_wavenumber_ini))
        #print('laser_pulse_energy= {} J'.format(laser_pulse_energy))

        # store variables
        self.laserideal_xs = laser_plane_xs
        self.laserideal_Intensity_distr_artifitial_units = laserideal_Intensity_distr_artifitial_units
        self.laserideal_pulse_energy = laser_pulse_energy #[J]
        self._phase_gradx_1D_frompreviousiteration = phase_gradx_1D








        ######################################################
        # self check 
        ######################################################
        # make 2D array
        laser_xx, laser_yy = np.meshgrid(self.laserideal_xs, self.laserideal_xs)
        laser_rr = (laser_xx**2+laser_yy**2)**0.5
        interpolator = interp1d(self.laserideal_xs, self.laserideal_Intensity_distr_artifitial_units, bounds_error=False, fill_value=0)

        laser_intensity_distr_2D_ideal = interpolator(laser_rr)


        #if self.monochromatic:
        electron_m_average = np.average(self.e_beam.m_poly)
        v_tot = np.linalg.norm(self.e_beam.v_ini_poly,axis=0)
        beta = np.average(v_tot) / self.e_beam.c
        energy_ratio = self.laserideal_pulse_energy / (electron_m_average*self.e_beam.c**2)
        

        # else:
        #     #electron_m_average = np.average(electron_m)
        #     v_tot = np.linalg.norm(v_ini,axis=0)
        #     beta = v_tot / self.c
        #     energy_ratio = self.laser_pulse_energy / (electron_m*self.c**2)
        #     print('laser interaction polychromatic')



        # phase_shift = (
        #     - self.finestructureconstant / (2*np.pi *(1-self.direction*beta)) *
        #     energy_ratio *
        #     self.wavelength**2 * self.intensity_distribution / self.intensity_planedistr_integral)
        
        dx = self.laserideal_xs[1] - self.laserideal_xs[0]
        dy = self.laserideal_xs[1] - self.laserideal_xs[0]
        _integral_E0_square_planedistr = np.sum(laser_intensity_distr_2D_ideal) *dx*dy
        
        
        # I removed from formula all elements connected with particlular particle energy, they will be added later.
        _scaling_constant = -finestructureconstant * self.laser_wavelength**2 / _integral_E0_square_planedistr

        # phase_shift_reduced = (
        #     -finestructureconstant *
        #     self.laser_wavelength**2 * laser_intensity_distr_2D_ideal/ _integral_E0_square_planedistr)
        
        phase_shift_reduced = _scaling_constant * laser_intensity_distr_2D_ideal
        

        

        
        phase_shift_gradx = np.gradient(phase_shift_reduced, axis=0)/dx
        phase_shift_grady = np.gradient(phase_shift_reduced, axis=1)/dy
        phase_shift_gradx_interpolator= RegularGridInterpolator((self.laserideal_xs, self.laserideal_xs), phase_shift_gradx,bounds_error=False, fill_value=0, method="linear")
        phase_shift_grady_interpolator= RegularGridInterpolator((self.laserideal_xs, self.laserideal_xs), phase_shift_grady,bounds_error=False, fill_value=0, method="linear")

        # calculate initial angles of propagation (-pi/2, pi/2)
        alphax_ini = np.arctan2(det_laserplane_vel[0], det_laserplane_vel[2]) 
        alphay_ini = np.arctan2(det_laserplane_vel[1], det_laserplane_vel[2]) 

        # project gradient to the transversal plane of the particle propagation
        phase_shift_gradx_projected = phase_shift_gradx_interpolator(det_laserplane_pos[[0,1]].T) #* np.cos(alphax_ini)
        phase_shift_grady_projected = phase_shift_grady_interpolator(det_laserplane_pos[[0,1]].T) #* np.cos(alphay_ini)

        # multiply to get correct units and correspond to particles energy
        phase_shift_gradx_projected *= energy_ratio/ (2*np.pi *(1-self.laser_direction*beta))
        phase_shift_grady_projected *= energy_ratio/ (2*np.pi *(1-self.laser_direction*beta))

        # check gradient
        plt.plot(phase_shift_gradx_projected[~np.isnan(det_laserplane_pos[0])], marker='.', label='receive: phase_shift_gradx_projected nonan')
        plt.plot(phase_gradx[~np.isnan(det_laserplane_pos[0])]*np.max(phase_shift_gradx_projected[~np.isnan(det_laserplane_pos[0])])/np.max(phase_gradx[~np.isnan(det_laserplane_pos[0])]), marker='.', label='prepare: phase_gradx nonan')
        plt.title("selfcheck phase_gradx_1D")
        plt.legend()
        plt.show()

    


        alphax_new = np.arctan(phase_shift_gradx_projected/self.e_beam.electron_wavenumber_ini)
        alphay_new = np.arctan(phase_shift_grady_projected/self.e_beam.electron_wavenumber_ini)

        alphax_final = alphax_ini + alphax_new
        alphay_final = alphay_ini + alphay_new

        # prepare velocity relative values
        vx_after = np.tan(alphax_final)
        vy_after = np.tan(alphay_final)

        vz_after = np.ones(np.shape(vx_after))
        v_after = np.array([vx_after, vy_after, vz_after])
        # scale to the same velocity magnitude as before:
        v_after = v_after / np.linalg.norm(v_after, axis=0) * np.linalg.norm(det_laserplane_vel,axis=0) 
        p_after = v_after / np.linalg.norm(v_after, axis=0) * np.linalg.norm(det_laserplane_pos,axis=0)
        #return (v_after, p_after)


        plt.plot(det_laserplane_vel[0][~np.isnan(det_laserplane_pos[0])], marker='.', label='det_laserplane_vel[0]')
        plt.plot(det_laserplaneback_vel[0][~np.isnan(det_laserplane_pos[0])], marker='.', label='det_laserplaneback_vel[0]')
        plt.plot(v_after[0][~np.isnan(det_laserplane_pos[0])], marker='.', label='v_after[0]')
        plt.title("det_laserplane_vel x after selfcheck")
        plt.legend()
        plt.show()

    def _evaluate_backpropagation(self, det_laserplaneback_vel, det_laserplane_vel, det_laserplane_pos):
        ######################################################
        # compare data and calculate required laser intensity for correction
        ######################################################

        # compare angles ... angles wihich I want create by laser
        anglex_dif = np.arctan2(det_laserplaneback_vel[0],det_laserplaneback_vel[2]) - np.arctan2(det_laserplane_vel[0],det_laserplane_vel[2])
        angley_dif = np.arctan2(det_laserplaneback_vel[1],det_laserplaneback_vel[2]) - np.arctan2(det_laserplane_vel[1],det_laserplane_vel[2])

        # electron phase gradients
        phase_gradx = np.average(self.e_beam.electron_wavenumber_ini) * np.tan(anglex_dif)
        phase_grady = np.average(self.e_beam.electron_wavenumber_ini) * np.tan(angley_dif)


        ## version to avoid tangenting:
        # # compare angles ... angles wihich I want create by laser
        # anglex_dif = det_laserplaneback_vel[0]/det_laserplaneback_vel[2] - det_laserplane_vel[0]/det_laserplane_vel[2]
        # angley_dif = det_laserplaneback_vel[1]/det_laserplaneback_vel[2] - det_laserplane_vel[1]/det_laserplane_vel[2]
        # # electron phase gradients
        # phase_gradx = np.average(self.e_beam.electron_wavenumber_ini) * anglex_dif
        # phase_grady = np.average(self.e_beam.electron_wavenumber_ini) * angley_dif



        

        # reshape into regular mesh
        # laserplane_xmax = np.max(det_laserplane_pos)

        phase_gradx_mesh = np.reshape(phase_gradx,(int(np.sqrt(self.n_electrons)),int(np.sqrt(self.n_electrons))))
        phase_grady_mesh = np.reshape(phase_grady,(int(np.sqrt(self.n_electrons)),int(np.sqrt(self.n_electrons))))

        laserplane_xx_mesh = np.reshape(det_laserplane_pos[0], (int(np.sqrt(self.n_electrons)),int(np.sqrt(self.n_electrons))))

        # continue in >>> 1D <<< integrate (suppose axial symetry)
        phase_gradx_1D = phase_gradx_mesh[int(np.sqrt(self.n_electrons)/2),:]
        laser_plane_xs = laserplane_xx_mesh[int(np.sqrt(self.n_electrons)/2),:]
        # remove nans
        phase_gradx_1D = phase_gradx_1D[~np.isnan(laser_plane_xs)] # do it according xs in both cases, sinc only xs has nans
        laser_plane_xs = laser_plane_xs[~np.isnan(laser_plane_xs)]
        # print(laser_plane_xs)
        # print(phase_gradx_1D)
        if laser_plane_xs.size != phase_gradx_1D.size:
            raise Exception("1D arrays have different number of points! {} and {}".format(laser_plane_xs.size,phase_gradx_1D.size))


        # print("phase_gradx_1D : " +str(phase_gradx_1D))
        if self._phase_gradx_1D_frompreviousiteration is not None:
            phase_gradx_1D += self._phase_gradx_1D_frompreviousiteration
        

        phase_shift_x = np.cumsum(phase_gradx_1D)*np.gradient(laser_plane_xs)

        # shift phaseshift to be only negative (... and the laser intensity will be only positive)
        phase_shift_x = phase_shift_x - np.max(phase_shift_x)

        # calculate laser El. field distribution (accodrding the equation (D11) from the letter TRANSVERSE ELECTRON BEAM SHAPING WITH LIGHT (Mihaila 2022))
        laserideal_Intensity_distr_artifitial_units = - phase_shift_x # ... ElField**2 [a.u.]
        if any(laserideal_Intensity_distr_artifitial_units<0):
            raise Exception("Idela laser has negative values!")

        ##############################################################
        # 5) Calculate laser pulse energy
        ##############################################################
        # specify laser properties
        finestructureconstant = 0.0072973525693
        beta = np.average(det_laserplane_vel) / self.e_beam.c # electron_speed / light_speed
        electron_energy = (np.average(self.e_beam.m_poly)*self.e_beam.c**2)
        laser_direction = -1 # laser_direction : +/- 1, in positive z direction ... +1,  in negative z direction ... -1

        def integral_phase_planedistr(xs, phase):
            dx = np.gradient(xs)
            integral_phase_planedistr = np.sum(phase * np.pi *abs(xs) * dx)
            return integral_phase_planedistr

        laser_pulse_energy = (
            2*np.pi * (1-laser_direction*beta) * electron_energy * (-1)*integral_phase_planedistr(laser_plane_xs,phase_shift_x) 
            / (finestructureconstant*self.laser_wavelength**2))
        #print('laser_pulse_energy= {} J'.format(laser_pulse_energy))

        # store variables
        self.laserideal_xs = laser_plane_xs
        self.laserideal_Intensity_distr_artifitial_units = laserideal_Intensity_distr_artifitial_units
        self.laserideal_pulse_energy = laser_pulse_energy #[J]
        self._phase_gradx_1D_frompreviousiteration = phase_gradx_1D

    def trace_aberrated(self):
        """
        Traces the electron beam through the defined aberrated lens setup.
        """
        func_start_time = time.time()
        print("  Running trace_aberrated()...")

        # Update the beam object with the lenses it needs to pass through.
        self.e_beam.update_elements([self.lens1_aberated, self.lens2_aberated], z_max=1e-3)

        # --- CORE SIMULATION ---
        # PERFORMANCE NOTE: This is the main computational bottleneck.
        # It iterates through time steps for every single electron.
        # Its execution time is proportional to: (n_electrons * tracing_maxiter).
        trace_start_time = time.time()
        self.e_beam.trace(self.dt, self.tracing_maxiter)
        print(f"    Time for core e_beam.trace(): {time.time() - trace_start_time:.4f} seconds.")

        # --- Post-processing: Analyze the beam at the detector ---
        # These calculations (norm, median, quantile) are very fast, as they
        # are vectorized numpy operations.
        post_proc_start_time = time.time()
        detector_aberrated = self.e_beam.detector_spacelike(0, nans=False)
        detector_aberrated_r = np.linalg.norm(detector_aberrated, axis=0)

        # Store key metrics about the final beam spot.
        self.detector_aberrated = detector_aberrated
        self.detector_aberrated_D50 = 2* np.median(detector_aberrated_r)
        self.detector_aberrated_D90 = 2* np.quantile(detector_aberrated_r,0.9)
        print(f"    Time for post-trace analysis: {time.time() - post_proc_start_time:.4f} seconds.")
        print(f"  trace_aberrated() finished in {time.time() - func_start_time:.4f} seconds.\n")

    def calculate_defocus(self, defocus_min = -10e-6, defocus_max = 0, num_planes_to_chack = 300, update_defocus = True):
        """refocus the aberated objective to move the smallest beam diameter to the detector plane
        """
        self.trace_aberrated()
        planesz = np.linspace(defocus_min,defocus_max,num_planes_to_chack)
        D50s = []
        for c_plane in range(num_planes_to_chack):
            detector_aberrated = self.e_beam.detector_spacelike(planesz[c_plane], nans=False)
            detector_aberrated_r = np.linalg.norm(detector_aberrated, axis=0)
            D50s.append(np.median(detector_aberrated_r))
        D50s = np.array(D50s)
        defocus = planesz[np.argmin(D50s)]
        if update_defocus:
            self.defocus += defocus

        return defocus

    def plot_electron_tracing(self):

        plot_selection = 1

        fig, ax = plt.subplots(1,1)
        ax.plot(self.e_beam.a_pos[:,0,::plot_selection], self.e_beam.a_pos[:,2,::plot_selection], '.-')
        # plot aperture
        _ap = self.e_beam.a_apertures
        for c_ap in range (np.size(_ap, axis=0)):
            ax.plot([-10*_ap[c_ap,1],-_ap[c_ap,1]],[_ap[c_ap,0],_ap[c_ap,0]], 'k')
            ax.plot([ 10*_ap[c_ap,1], _ap[c_ap,1]],[_ap[c_ap,0],_ap[c_ap,0]], 'k')
        for c_element in range(np.size(self.e_beam.l_elements)):
            _z = self.e_beam.l_elements[c_element].position_z[1]
            ax.plot([-0.2e-3,0.2e-3],[_z,_z], 'r')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        ax.invert_yaxis()
        ax.set_title('acceleration voltage nominal {} kV\nNo.Particles{}\n convsemiang={}mrad'.format(self.acc_voltage_nominal/1000, self.n_electrons, self.conv_semiang*1000))
        plt.show()

    def plot_electron_tracing_fine(self):
        fig, ax = plt.subplots(1,1)
        ax.plot(self.e_beam.a_pos[:,0,int(self.n_electrons/2-np.sqrt(self.n_electrons)/2) : int(self.n_electrons/2+np.sqrt(self.n_electrons)/2)], self.e_beam.a_pos[:,2,int(self.n_electrons/2-np.sqrt(self.n_electrons)/2) : int(self.n_electrons/2+np.sqrt(self.n_electrons)/2)], 'g')
        # plot aperture
        _ap = self.e_beam.a_apertures
        for c_ap in range (np.size(_ap, axis=0)):
            ax.plot([-10*_ap[c_ap,1],-_ap[c_ap,1]],[_ap[c_ap,0],_ap[c_ap,0]], 'k')
            ax.plot([ 10*_ap[c_ap,1], _ap[c_ap,1]],[_ap[c_ap,0],_ap[c_ap,0]], 'k')
        for c_element in range(np.size(self.e_beam.l_elements)):
            _z = self.e_beam.l_elements[c_element].position_z[1]
            ax.plot([-0.2e-3,0.2e-3],[_z,_z], 'r')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        ax.invert_yaxis()
        ax.set_title('acceleration voltage nominal {} kV\nNo.Particles{}\n convsemiang={}mrad'.format(self.acc_voltage_nominal/1000, self.n_electrons, self.conv_semiang*1000))
        plt.show()

    def plot_electron_backtracing(self,plot_ideal=True, oneline=True):
        legend_list = []
        
        fig, ax = plt.subplots(1,1)
        if oneline: 
            ray_x =  self.e_beam.a_pos[:,0,:]
            ray_z =  self.e_beam.a_pos[:,2,:]
        else: 
            ray_x = self.e_beam.a_pos[:,0,int(self.n_electrons/2-np.sqrt(self.n_electrons)/2) : int(self.n_electrons/2+np.sqrt(self.n_electrons)/2)]
            ray_z = self.e_beam.a_pos[:,2,int(self.n_electrons/2-np.sqrt(self.n_electrons)/2) : int(self.n_electrons/2+np.sqrt(self.n_electrons)/2)]
        ax.plot(ray_x, ray_z, 'g')
        legend_list.append(mpatches.Patch(color='green', label="backpropagation"))
        ax.plot(ray_x, ray_z*(-1), 'orange')
        legend_list.append(mpatches.Patch(color='orange', label="backpropagation-inverted"))
        #ax.plot(self.e_beam.a_pos[:,0,int(self.n_electrons/2-np.sqrt(self.n_electrons)/2) : int(self.n_electrons/2+np.sqrt(self.n_electrons)/2)], self.e_beam.a_pos[:,2,int(self.n_electrons/2-np.sqrt(self.n_electrons)/2) : int(self.n_electrons/2+np.sqrt(self.n_electrons)/2)], 'g')
        # plot aperture
        _ap = self.e_beam.a_apertures
        for c_ap in range (np.size(_ap, axis=0)):
            ax.plot([-10*_ap[c_ap,1],-_ap[c_ap,1]],[_ap[c_ap,0],_ap[c_ap,0]], 'k')
            ax.plot([ 10*_ap[c_ap,1], _ap[c_ap,1]],[_ap[c_ap,0],_ap[c_ap,0]], 'k')
        for c_element in range(np.size(self.e_beam.l_elements)):
            _z = self.e_beam.l_elements[c_element].position_z[1]
            ax.plot([-0.2e-3,0.2e-3],[_z,_z], 'r')

        # plot conjugated planes
        ax.plot([-0.2e-3,0.2e-3],[self.laser_plane_z,self.laser_plane_z], 'r')
        ax.plot([-0.2e-3,0.2e-3],[self.laserback_plane_z,self.laserback_plane_z], 'r')

        if plot_ideal:
            #self.laserideal_pulse_energy *=1.10
            self.build_elements()
            self.trace_idealcorrected()
            ray_x = self.e_beam.a_pos[:,0,int(self.n_electrons/2-np.sqrt(self.n_electrons)/2) : int(self.n_electrons/2+np.sqrt(self.n_electrons)/2)]
            ray_z = self.e_beam.a_pos[:,2,int(self.n_electrons/2-np.sqrt(self.n_electrons)/2) : int(self.n_electrons/2+np.sqrt(self.n_electrons)/2)]
            ax.plot(ray_x, ray_z, '--r')
            legend_list.append(mpatches.Patch(color='red', label='Ideal correction D50={}'.format(self.detector_idealcorrection_D50)))

        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')

        ax.legend(handles=legend_list)
        ax.invert_yaxis()
        ax.set_title('Backtracing\nacceleration voltage nominal {} kV\nNo.Particles{}\n convsemiang={}mrad'.format(self.acc_voltage_nominal/1000, self.n_electrons, self.conv_semiang*1000))
        plt.show()

    def plot_laser_profile(self, thicklaser=True):
        print("print laser start")
        fig, ax = plt.subplots(1,1)
        ax.plot(
            self.laserideal_xs,
            self.laser_ideal.intensity2D[int(np.shape(self.laser_ideal.intensity2D)[1]/2)], 
            label='desired {}μJ'.format(self.laserideal_pulse_energy*1000000))

        if thicklaser:
            for c_plane in range(len(self.laser_setup.detectplanesz)):
                laser_intensity_2D = self.laser_thick.intensity2D[:,:,c_plane]
                laser_intensity_2D_cross = laser_intensity_2D[int(np.shape(laser_intensity_2D)[1]/2)]
                ax.plot(
                    self.laser_setup.detector_xs,
                    laser_intensity_2D_cross, 'orange',
                    label='LightPipes {}μJ'.format(self.laser_thick_pulse_energy*1000000))

        ax.set_xlabel('x [m]')
        ax.legend()
        ax.set_ylabel('laser intensity')
        plt.show()
        print("print laser done")
            
    def trace_idealcorrected(self):
        """
        Trace ideally corrected setup
        """
        laser_xx, laser_yy = np.meshgrid(self.laserideal_xs, self.laserideal_xs)
        laser_rr = (laser_xx**2+laser_yy**2)**0.5
        interpolator = interp1d(self.laserideal_xs, self.laserideal_Intensity_distr_artifitial_units, bounds_error=False, fill_value=0)

        laser_intensity_distr_2D_ideal = interpolator(laser_rr)
        self.laser_ideal = lib_electron_tracing.Laser(self.laser_plane_z, laser_intensity_distr_2D_ideal, self.laserideal_xs, self.laserideal_xs, self.laser_direction, self.laserideal_pulse_energy, self.laser_wavelength)
        self.e_beam.update_elements([self.laser_ideal,self.lens1_aberated, self.lens2_aberated], z_max=1e-3) 
        self.e_beam.trace(self.dt, self.tracing_maxiter)
        detector_idealcorrection = self.e_beam.detector_spacelike(0, nans=False)
        detector_idealcorrection_r = np.linalg.norm(detector_idealcorrection, axis=0)

        # store variables
        self.detector_idealcorrection = detector_idealcorrection
        self.detector_idealcorrection_D50 = 2*np.median(detector_idealcorrection_r)
        self.detector_idealcorrection_D90 = 2*np.quantile(detector_idealcorrection_r,0.9)

    def build_laser_setup(self, laser_wavelength, plane0_width, plane_points, laser_beam_width, SLM_npix, SLM_size, detectplanesz, laser_zpos, laser_xcenter, SLM_zpos, SLM_xcenter, SLM_dist, lens1_f, lens2_f, lens3_f, beamblock_r, lens3_zdist, lens1_r, lens2_r, lens3_r, rescaling_f=0.1):
        """_summary_

        Parameters
        ----------
        laser_thick : Laser_setup object
        """

        if detectplanesz is "auto":
            # estimate where the pulses interacts
            electron_speed = np.average(self.e_beam.v_ini_poly[2])
            laser_pulse_length = self.laser_pulse_timelength * const.c
            electron_pulse_length = self.laser_pulse_timelength * electron_speed
            interaction_length = (laser_pulse_length + electron_pulse_length) * const.c / (2* (const.c - self.laser_direction * electron_speed))
            interaction_length_longer = interaction_length * 10
            detectplanesz = np.linspace(-1,1,21)*interaction_length_longer
            #raise Exception("not defined so far")


        self.laser_setup = Laser_setup(laser_wavelength, plane0_width, plane_points, laser_beam_width, SLM_npix, SLM_size, detectplanesz, laser_zpos, laser_xcenter, SLM_zpos, SLM_xcenter, SLM_dist, lens1_f, lens2_f, lens3_f, beamblock_r, lens3_zdist, lens1_r, lens2_r, lens3_r, rescaling_f=rescaling_f)
        self.laser_setup.build_setup()
        # self.laser_setup.propagate()
        # self.laser_setup.detect()

    def trace_corrected_real(self, laser="thick", synchronization_error=0):
        """_summary_

        Args:
            laser (str, optional): _description_. Defaults to "thick".
            synchronization_error (float, optional): time synchronization error [s]. Defaults to 0.

        Raises:
            Exception: _description_
        """

        ##############################################################
        # Trace corrected setup from LightPipes
        ##############################################################
        
        # do synchronization
        # electron arrival time
        electron_arrivaltime = self.e_beam.detector_timelike(self.laser_plane_z)
        electron_arrivaltime = np.average(electron_arrivaltime[~np.isnan(electron_arrivaltime)])
        laser_t0 = electron_arrivaltime + synchronization_error#[s]
        
        if laser == "thick":
            self.laser_thick = lib_electron_tracing.Laser_thick(
                self.laser_plane_z, 
                self.laser_setup.detector_2D_array,
                self.laser_setup.detector_xs,
                self.laser_setup.detector_xs,
                self.laser_setup.detectplanesz,
                self.laser_pulse_timelength,
                laser_t0,
                self.laser_direction,
                self.laser_thick_pulse_energy,
                self.laser_wavelength)
            self.e_beam.update_elements([self.laser_thick,self.lens1_aberated, self.lens2_aberated], z_max=1e-3) 
        elif laser == "thin":
            self.laser_thin = lib_electron_tracing.Laser(
                self.laser_plane_z, 
                self.laser_setup.detector_2D_array[:,:,np.argmin(abs(self.laser_setup.detectplanesz))], # pick up the laser distribution in the central plane
                self.laser_setup.detector_xs,
                self.laser_setup.detector_xs, 
                self.laser_direction, 
                self.laser_thick_pulse_energy,
                self.laser_wavelength)
            self.e_beam.update_elements([self.laser_thin,self.lens1_aberated, self.lens2_aberated], z_max=1e-3) 

        elif laser == "quasithin":
            self.laser_thick = lib_electron_tracing.Laser_quasithin(
                self.laser_plane_z, 
                self.laser_setup.detector_2D_array,
                self.laser_setup.detector_xs,
                self.laser_setup.detector_xs,
                self.laser_setup.detectplanesz,
                self.laser_pulse_timelength,
                laser_t0,
                self.laser_direction,
                self.laser_thick_pulse_energy,
                self.laser_wavelength)
            self.e_beam.update_elements([self.laser_thick,self.lens1_aberated, self.lens2_aberated], z_max=1e-3)

        else: raise Exception("Laser kind not specified.")
    
        # trace   
        self.e_beam.trace(self.dt, self.tracing_maxiter)
        self.detector_realcorrection = self.e_beam.detector_spacelike(0, nans=False)
        detector_realcorrection_r = np.linalg.norm(self.detector_realcorrection, axis=0)
        self.detector_realcorrection_D50 = 2* np.median(detector_realcorrection_r)
        self.detector_realcorrection_D90 = 2* np.quantile(detector_realcorrection_r,0.9)

    def evaluate_laser_error(self):
        # correct pulse energy
        self.laser_ideal.get_intensity2D()
        self.laser_thick.get_intensity2D()

        laserthick_intensity_2D_midplane = self.laser_thick.intensity2D[:,:,int(len(self.laser_setup.detectplanesz)/2)]

        laser_thick_intensity_2D_cross = laserthick_intensity_2D_midplane[int(np.shape(laserthick_intensity_2D_midplane)[1]/2)]
        laser_ideal_intensity_2D_cross = self.laser_ideal.intensity2D[int(np.shape(self.laser_ideal.intensity2D)[1]/2)]

        laser_thick_max = np.max((laser_thick_intensity_2D_cross))
        laser_ideal_max = np.max((laser_ideal_intensity_2D_cross))
        
        print("real  laser max = {}".format(laser_thick_max))
        print("ideal laser max = {}".format(laser_ideal_max))
        laser_thick_pulse_energy_new = self.laser_thick_pulse_energy*laser_ideal_max/laser_thick_max
        self.laser_thick_pulse_energy = laser_thick_pulse_energy_new
        self.laser_thick.laser_pulse_energy = self.laser_thick_pulse_energy
        print("setting a new thick laser pulse energy = {}".format(laser_thick_pulse_energy_new))

        self.laser_thick.initial_calculation()
        # evaluate laser error 
        self.laser_ideal.get_intensity2D()
        self.laser_thick.get_intensity2D()

        laser_thick_interpolator_forcomparison = RegularGridInterpolator((self.laser_thick.xs, self.laser_thick.ys), self.laser_thick.intensity2D[:,:,int(len(self.laser_setup.detectplanesz)/2)],bounds_error=False, fill_value=0)
        laser_ideal_xx,laser_ideal_yy = np.meshgrid(self.laser_ideal.xs, self.laser_ideal.ys)
        laser_ideal_rr = (laser_ideal_xx**2 + laser_ideal_yy**2)**0.5
        laser_thick_intensity2D_interpolated = laser_thick_interpolator_forcomparison(np.array([laser_ideal_xx,laser_ideal_yy]).T)

        self.laser_error = np.sqrt(np.sum((np.array(np.gradient(self.laser_ideal.intensity2D)) - np.array(np.gradient(laser_thick_intensity2D_interpolated)))**2))
        return self.laser_error
        

    def update_all(self):
        self.laser_setup.build_setup()
        self.laser_setup.propagate()
        self.laser_setup.detect()
        self.evaluate_laser_error()
        self.trace_corrected_real()
        self.evaluate_laser_error()
        self.trace_corrected_real()



    def make_errormap(self, SLM_lens_f_inverse_array,SLM_phasefactor1_array):
        """_summary_

        Parameters
        ----------
        SLM_lens_f_inverse : np1D array 
            _description_ 
        SLM_phasefactor1 : np1D array 
            _description_ 
        """
        error_map = np.zeros((np.size(SLM_lens_f_inverse_array), np.size(SLM_phasefactor1_array)))
        D50_map = np.zeros((np.size(SLM_lens_f_inverse_array), np.size(SLM_phasefactor1_array)))
        pulse_energy_map = np.zeros((np.size(SLM_lens_f_inverse_array), np.size(SLM_phasefactor1_array)))
        for c_f in range(np.size(SLM_lens_f_inverse_array)):
            SLM_lens_f_inverse = SLM_lens_f_inverse_array[c_f]
            for c_ph1 in range(np.size(SLM_phasefactor1_array)):
                SLM_phasefactor1 = SLM_phasefactor1_array[c_ph1]

                self.laser_setup.SLM_lens_f_inverse = SLM_lens_f_inverse
                self.laser_setup.SLM_phasefactor1 = SLM_phasefactor1

                # update all and optimize pulse energy
                self.laser_setup.build_setup()
                self.laser_setup.propagate()
                self.laser_setup.detect()
                self.evaluate_laser_error()
                self.trace_corrected_real()

                # collect data
                error_map[c_f,c_ph1] = self.evaluate_laser_error()
                self.trace_corrected_real()
                D50_map[c_f,c_ph1] = self.detector_realcorrection_D50
                pulse_energy_map[c_f,c_ph1] = self.laser_thick.laser_pulse_energy

        return error_map, D50_map, pulse_energy_map

    def optimize(self,SLM_phasefactor1_iniguess=None, SLM_lens_f_inverse_iniguess=None, correct_defocus=False, plot=True):
        # 0) correct defocus
        if correct_defocus:
            self.trace_aberrated()
            self.calculate_defocus(defocus_min = -10e-6, defocus_max = 0, num_planes_to_chack = 100, update_defocus = True)
            self.build_elements()
            self.trace_aberrated()

        # update all
        self.build_elements()

        self.trace_backprop()
        self.trace_aberrated()
        self.trace_idealcorrected()
        self.trace_corrected_real()
        self.evaluate_laser_error()

        if plot: self.plot_laser_profile()


        # 1) make map to find global maximum (roughly)
        if SLM_lens_f_inverse_iniguess == None or SLM_phasefactor1_iniguess == None:
            SLM_lens_f_inverse_array = np.linspace(-1,0,10)
            SLM_lens_f_inverse_array = np.linspace(-0.7,-0.5,3)
            SLM_lens_f_inverse_array = np.linspace(-10,-0,31)
            #SLM_phasefactor1_array = np.linspace(0,3000,10)
            SLM_phasefactor1_array = np.linspace(0,0,1)
            
            error_map, D50_map, pulse_energy_map = self.make_errormap(SLM_lens_f_inverse_array,SLM_phasefactor1_array)
            ind = np.unravel_index(np.argmin(D50_map, axis=None), D50_map.shape)
            SLM_lens_f_inverse_iniguess = SLM_lens_f_inverse_array[ind[0]]
            SLM_phasefactor1_iniguess = SLM_phasefactor1_array[ind[1]]

            if plot:
                if all(np.shape(D50_map))>1:
                    xx,yy = np.meshgrid(SLM_phasefactor1_array,SLM_lens_f_inverse_array)
                    fig,ax = plt.subplots()
                    cs = ax.contourf(xx,yy,D50_map,levels=100)
                    cs = ax.imshow()
                    cbar = fig.colorbar(cs) 
                    ax.plot(SLM_phasefactor1_iniguess, SLM_lens_f_inverse_iniguess, 'o')
                    ax.set_ylabel("SLM_lens_f_inverse [m]")
                    ax.set_xlabel("SLM phasefactor1 [m]")
                    ax.set_title("D50_map")
                    plt.show()

                    xx,yy = np.meshgrid(SLM_phasefactor1_array,SLM_lens_f_inverse_array)
                    fig,ax = plt.subplots()
                    cs = ax.contourf(xx,yy,error_map,levels=100)
                    cbar = fig.colorbar(cs) 
                    ax.plot(SLM_phasefactor1_iniguess, SLM_lens_f_inverse_iniguess, 'o')
                    ax.set_ylabel("SLM_lens_f_inverse [m]")
                    ax.set_xlabel("SLM phasefactor1 [m]")
                    ax.set_title("error_map")
                    plt.show()
                else:
                    fig,ax = plt.subplots()
                    ax.plot(SLM_lens_f_inverse_array,D50_map[:,0])
                    ax.plot(SLM_lens_f_inverse_iniguess,0, 'o')
                    ax.set_xlabel("SLM_lens_f_inverse [m]")
                    ax.set_ylabel("D50_map [m]")
                    plt.show()

                    fig,ax = plt.subplots()
                    ax.plot(SLM_lens_f_inverse_array,error_map[:,0])
                    ax.plot(SLM_lens_f_inverse_iniguess,0, 'o')
                    ax.set_xlabel("SLM_lens_f_inverse [m]")
                    ax.set_ylabel("error_map")
                    plt.show()

    

        if False:

            # 2) optimize by puleni intervalu
            if D50_map[ind[0]-1,ind[1]] < D50_map[ind[0]+1,ind[1]]:
                param_left = SLM_lens_f_inverse_array[ind[0]-1]
                param_right = SLM_lens_f_inverse_array[ind[0]]
                val_left = D50_map[ind[0]-1,ind[1]]
                val_right = D50_map[ind[0],ind[1]]
            elif D50_map[ind[0]-1,ind[1]] > D50_map[ind[0]+1,ind[1]]:
                param_left = SLM_lens_f_inverse_array[ind[0]]
                param_right = SLM_lens_f_inverse_array[ind[0]+1]
                val_left = D50_map[ind[0],ind[1]]
                val_right = D50_map[ind[0]+1,ind[1]]

            for c_iter in range(10):
                print("iter {}".format(c_iter))
                # calculate middle
                param_middle = (param_left + param_right)/2
                self.laser_setup.SLM_lens_f_inverse = param_middle
                self.laser_setup.SLM_phasefactor1 = SLM_phasefactor1_iniguess

                # update all and optimize pulse energy
                self.laser_setup.build_setup()
                self.laser_setup.propagate()
                self.laser_setup.detect()
                self.evaluate_laser_error()
                self.trace_corrected_real()

                # collect data
                self.evaluate_laser_error()
                self.trace_corrected_real()
                val_middle = self.detector_realcorrection_D50

                # evaluate
                if val_left > val_right:
                    val_left = val_middle
                    param_left = param_middle
                elif val_left < val_right:
                    val_right = val_middle
                    param_right = param_middle
                print("val_middle = {}".format(val_middle))

                print(param_left)
                print(param_middle)
                print(param_right)

                print(val_left)
                print(val_middle)
                print(val_right)
                #input("wait")
            SLM_lens_f_inverse_iniguess = param_middle


        # 3) trace properly 
        # 4) return data and parameters 
        return (SLM_lens_f_inverse_iniguess,SLM_phasefactor1_iniguess)


    # def _detect_peaks(self, image):
    #     """
    #     Takes an image and detect the peaks usingthe local maximum filter.
    #     Returns a boolean mask of the peaks (i.e. 1 when
    #     the pixel's value is the neighborhood maximum, 0 otherwise)
    #     """

    #     # define an 8-connected neighborhood
    #     neighborhood = generate_binary_structure(2,5)

    #     #apply the local maximum filter; all pixel of maximal value 
    #     #in their neighborhood are set to 1
    #     local_max = maximum_filter(image, footprint=neighborhood)==image
    #     #local_max is a mask that contains the peaks we are 
    #     #looking for, but also the background.
    #     #In order to isolate the peaks we must remove the background from the mask.

    #     #we create the mask of the background
    #     background = (image==0)

    #     #a little technicality: we must erode the background in order to 
    #     #successfully subtract it form local_max, otherwise a line will 
    #     #appear along the background border (artifact of the local maximum filter)
    #     eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #     #we obtain the final mask, containing only peaks, 
    #     #by removing the background from the local_max mask (xor operation)
    #     detected_peaks = local_max ^ eroded_background

    #     return detected_peaks


    # def calculate_resolution(self, psf, psf_framewidth):
    #     """calculate resolution limit from the point spred function of a focused beam. Simulates high contrast sample and checks what is the smallest distance between points which is still resolvable.

    #     Args:
    #         psf (2D np array): centered, symetric 2D point spred function, amplitude units does not matter
    #         psf_framewidth (float): total width of psf array [m]

    #     Raises:
    #         Exception: _description_
    #         Exception: _description_

    #     Returns:
    #         resolution: [m] closest distance of points which are still resolvable as separated spots
    #     """
    #     if  np.shape(psf)[0] % 2 == 0 or np.shape(psf)[1] % 2 == 0: raise Exception("psf has to have even shape")
    #     if np.shape(psf)[0] != np.shape(psf)[1]: raise Exception("psf has to have square shape")
    #     npix_psf = np.shape(psf)[0]
    #     pix_size = psf_framewidth / npix_psf
    #     npix_sample = int(10*npix_psf)
    #     best_resolution = np.nan
    #     print("calculating resolution")

    #     dist_factor = int(npix_psf/2) # minimal distance between points [pixels]
    #     while True:
    #         dist_factor = int(dist_factor*0.9)
    #         if dist_factor <= 2: raise Exception("It is not possible to find lowest resolution, provide psf with finer sampling")
    #         resolution = dist_factor * pix_size

    #         # prepare sample
    #         sample = np.zeros((npix_sample,npix_sample))
    #         npoints = npix_sample *1
    #         points = np.random.random((npoints,npoints))*npix_sample/dist_factor
    #         points = np.floor(points)*dist_factor
    #         points = np.array(points, dtype=int)
    #         sample[points[0], points[1]] = 1

    #         # plt.imshow(sample)
    #         # plt.title("sample")
    #         # plt.show()
    #         #plt.close('all')


    #         # convolution
    #         convolution = fftconvolve(sample, psf)
    #         plt.imshow(convolution)
    #         plt.title("convolution {}".format(resolution))
    #         plt.pause(0.2)
    #         convolution = np.where(convolution<np.max(convolution)/3,0,convolution) # remove noise - mostly form numerical artefacts, rounding errors

            
    #         # plt.imshow(convolution)
    #         # plt.title("convolution")
    #         # plt.show()

    #         detected_peaks = self._detect_peaks(convolution)
    #         detected_peaks = detected_peaks[int(npix_psf/2):-int(npix_psf/2),int(npix_psf/2):-int(npix_psf/2)]

    #         # plt.imshow(detected_peaks)
    #         # plt.title("detected_peaks")
    #         # plt.show()

    #         # plt.imshow(sample-detected_peaks)
    #         # plt.title("sample - detected_peaks")
    #         # plt.show()

    #         detection_errors = np.sum(abs(sample-detected_peaks)) # each error means that there was an arror in localizing a peak ... under resolution limit
    #         detection_errors = np.sum((sample-detected_peaks)) # each error means that there was an arror in localizing a peak ... under resolution limit
    #         if detection_errors == 0:
    #             best_resolution = resolution*1
    #             print(best_resolution)
    #         else: 
    #             print("break")
    #             print(detection_errors)
    #             plt.imshow(sample-detected_peaks)
    #             plt.title("sample - detected_peaks")
    #             plt.show()
    #             break

    #     return best_resolution


    # def create_psf(self, histogram, rs):
    #     npix = int(np.size(histogram)*2-1)
    #     r_max = np.max(rs)
    #     psf_xs = np.linspace(-r_max,r_max,npix)
    #     XX,YY = np.meshgrid(psf_xs, psf_xs)
    #     rr = np.linalg.norm(np.array([XX,YY]), axis = 0)
    #     psf = np.interp(rr, rs, histogram/(2*np.pi*rs))
    #     #psf = np.reshape(psf, (npix, npix))

    #     plt.close('all')
    #     plt.imshow(psf)
    #     plt.title('psf')
    #     plt.show()
    #     return (psf, psf_xs)
    
    # def get_resolution(self, detector_rs):
    #     hist, bins = np.histogram(detector_rs, 200)
    #     psf, psfxs = self.create_psf(hist,bins[1:])
    #     resolution = self.calculate_resolution(psf, 2*np.max(psfxs))
    #     return resolution






    def get_resolution(self, detector_rs):
        hist, bins = np.histogram(detector_rs, 2000)
        dx = bins[1] - bins[0]
        rs = bins[1:] - dx/2
        psf_cross = hist/(2*np.pi*rs)
        psf_cross = np.append(np.flip(psf_cross), psf_cross)

        plt.close("all")
        fig, ax = plt.subplots(1,3)
        scale_factor = 1
        best_resolution = np.nan
        while True:
            if int(np.size(psf_cross)*scale_factor) <= 6: 
                plt.show()
                raise Exception("not able to find best resolution, try to increase sampling of psf")

            
            resolution = np.max(rs)*2*scale_factor
            sample = np.zeros(int(np.size(psf_cross)*scale_factor))
            sample[0] = 1
            sample[-1] = 1

            convolution = np.convolve(sample,psf_cross)

            peaks, _ = find_peaks(convolution, height=np.max(convolution))


            
            
            ax[0].plot(psf_cross)
            ax[1].plot(sample)
            ax[2].plot(convolution)
            ax[2].plot(peaks[:2], convolution[peaks[:2]], 'o')
            ax[2].set_title(resolution)
            print(resolution)
            plt.pause(0.4)
            
            scale_factor *= 0.9
            

        resolution = self.calculate_resolution(psf, 2*np.max(psfxs))
        return resolution

    def plot_chromaticaberration_overview(self, export=False):
        """ This function exist only to keep compatibility with older scripts."""
        self.plot_overview(aberration='chromatic', export=export)



    def plot_overview(self,aberration, export=False):
        detector_timelike_laserplane = self.e_beam.detector_timelike(self.laser_plane_z)
        detector_timelike_laserplane = detector_timelike_laserplane[~np.isnan(detector_timelike_laserplane)]
        detector_intime_laservolume_z = self.e_beam.detector_intime(np.average(detector_timelike_laserplane))[2]
        

        detector_spacelike_laserplane_velocity = self.e_beam.detector_spacelike_velocity(self.laser_plane_z)
        detector_spacelike_laserplane_velocity = np.linalg.norm(detector_spacelike_laserplane_velocity, axis=0)
        detector_spacelike_laserplane_velocity = detector_spacelike_laserplane_velocity[~np.isnan(detector_intime_laservolume_z)]
        detector_intime_laservolume_z = detector_intime_laservolume_z[~np.isnan(detector_intime_laservolume_z)]

        idx_sorting = np.argsort(detector_spacelike_laserplane_velocity)
        idx_sorting_accpoly = np.argsort(self.e_beam.v_ini_poly[2])
        detector_r_initial = np.linalg.norm(self.e_beam.detector_spacelike(self.laser_plane_z, nans=False), axis=0)
        idx_sorting_r = np.argsort(detector_r_initial)
        detector_r_initial_nans = np.linalg.norm(self.e_beam.pos_ini[:2], axis=0)
        idx_sorting_r_nans = np.argsort(detector_r_initial_nans)

        fig,ax = plt.subplots(2,3)
        figl, axl =  plt.subplots(2,3)


        plot_selection = 10
        if aberration == 'spherical':
            a_pos_sorted = np.take(self.e_beam.a_pos,idx_sorting_r_nans, axis=2)
        else:
            a_pos_sorted = np.take(self.e_beam.a_pos,idx_sorting_accpoly, axis=2)
        #ax[0,0].plot(self.e_beam.a_pos[:,0,::plot_selection], self.e_beam.a_pos[:,2,::plot_selection], '.-')
        ax[0,0].plot(a_pos_sorted[:,0,:int(self.n_electrons/3):plot_selection],                             a_pos_sorted[:,2,:int(self.n_electrons/3):plot_selection], '.-', color='blue', alpha=0.5)
        ax[0,0].plot(a_pos_sorted[:,0,int(self.n_electrons/3):int(self.n_electrons*2/3):plot_selection],    a_pos_sorted[:,2,int(self.n_electrons/3):int(self.n_electrons*2/3):plot_selection], '.-', color='orange', alpha=0.5)
        ax[0,0].plot(a_pos_sorted[:,0,int(self.n_electrons*2/3)::plot_selection],                           a_pos_sorted[:,2,int(self.n_electrons*2/3)::plot_selection], '.-', color='green', alpha=0.5)
        _ap = self.e_beam.a_apertures
        for c_ap in range (np.size(_ap, axis=0)):
            ax[0,0].plot([-10*_ap[c_ap,1],-_ap[c_ap,1]],[_ap[c_ap,0],_ap[c_ap,0]], 'k')
            ax[0,0].plot([ 10*_ap[c_ap,1], _ap[c_ap,1]],[_ap[c_ap,0],_ap[c_ap,0]], 'k')
        for c_element in range(np.size(self.e_beam.l_elements)):
            _z = self.e_beam.l_elements[c_element].position_z[1]
            _z0 = self.e_beam.l_elements[c_element].position_z[0]
            _z2 = self.e_beam.l_elements[c_element].position_z[2]
            ax[0,0].plot([-0.1e-3,0.1e-3],[_z,_z], 'r')
            rectangle = Rectangle((-0.1e-3, _z0), 0.2e-3, _z2-_z0,color="red",  alpha=0.30)
            ax[0,0].add_patch(rectangle)
        ax[0,0].set_xlabel('x [m]')
        ax[0,0].set_ylabel('z [m]')
        ax[0,0].invert_yaxis()
        ax[0,0].set_title('Thick Laser\nacceleration voltage nominal {} kV\nNo.Particles{}\nconvergencesemiangle={}mrad'.format(self.acc_voltage_nominal/1000, self.e_beam.NParticles, self.conv_semiang*1000))

        
        # pulses on time
        _hist, _bins = np.histogram(detector_timelike_laserplane,100)
        #ax[0,1].hist( _hist,_bins, color='navy')
        axl[0,1].hist( detector_timelike_laserplane,100, color='navy')
        axl[0,1].plot(self.laser_thick.ts, self.laser_thick.intensity_distribution_intime/np.max(self.laser_thick.intensity_distribution_intime), color='orange')
        axl[0,1].set_xlabel('t [s]')
        axl[0,1].set_title('Laser and electron pulses in interaction plane')
        patch_orange = mpatches.Patch(color='orange', label='laser pulse')
        patch_blue = mpatches.Patch(color='black', label='electron pulse')
        axl[0,1].legend(handles=[patch_orange, patch_blue])


        # pulses in time
        _hist, _bins = np.histogram(detector_intime_laservolume_z,100)
        axl[0,2].hist( detector_intime_laservolume_z,100, color='navy')
        axl[0,2].plot((self.laser_thick.ts-self.laser_thick.t0)* constants.c * self.laser_direction +self.laser_plane_z , self.laser_thick.intensity_distribution_intime/np.max(self.laser_thick.intensity_distribution_intime), color='orange')
        axl[0,2].set_xlabel('z [m]')
        axl[0,2].set_title('Laser and electron pulses in interaction plane')
        patch_orange = mpatches.Patch(color='orange', label='laser pulse')
        patch_blue = mpatches.Patch(color='black', label='electron pulse')
        axl[0,2].legend(handles=[patch_orange, patch_blue])



        # # get_resolution
        # resolution_aberrated = self.get_resolution(np.linalg.norm(self.detector_aberrated, axis = 0))
        # resolution_realcorrection= self.get_resolution(np.linalg.norm(self.detector_realcorrection, axis = 0))
        # if aberration == 'spherical':
        #     resolution_ideal = self.get_resolution(np.linalg.norm(self.detector_idealcorrection, axis = 0))


        detector_corrected_r = np.linalg.norm(self.detector_realcorrection, axis = 0)
        detector_aberrated_r = np.linalg.norm(self.detector_aberrated, axis = 0)
        n_electrons_passed = np.size(detector_aberrated_r)
        _, bins = np.histogram(detector_aberrated_r, 200)
        dr = bins[1]-bins[0]
        rs = bins[1:] -dr/2
        _factor = 2*dr*n_electrons_passed # dividing factor to put histogram units to be [electrons/m2] normalized, valid only for oneline mode
        if aberration == 'spherical':
            detector_idealcorrection_r = np.linalg.norm(self.detector_idealcorrection, axis = 0)
            #sort electrons arrording their initial radius
            # detector_r_initial = np.linalg.norm(self.e_beam.pos_ini[:2], axis=0)
            # detector_r_initial = np.linalg.norm(self.e_beam.detector_spacelike(self.e_beam.pos_ini[2,0]+1e-2, nans=False), axis=0)
            # detector_r_initial = np.linalg.norm(self.e_beam.detector_spacelike(self.laser_plane_z, nans=False), axis=0)
            # idx_sorting_r = np.argsort(detector_r_initial)

            detector_aberrated_r_sorted = np.take(detector_aberrated_r, idx_sorting_r)
            detector_corrected_r_sorted = np.take(detector_corrected_r, idx_sorting_r)
            detector_idealcorrection_r_sorted = np.take(detector_idealcorrection_r, idx_sorting_r)


            
            
            
            hist1, _ = np.histogram(detector_aberrated_r_sorted[:int(n_electrons_passed/3)], bins)
            hist2, _ = np.histogram(detector_aberrated_r_sorted[int(n_electrons_passed/3):int(n_electrons_passed*2/3)], bins)
            hist3, _ = np.histogram(detector_aberrated_r_sorted[int(n_electrons_passed*2/3):], bins)
            axl[1,0].stackplot(rs, hist1/_factor, hist2/_factor, hist3/_factor,
                               colors =['black', 'black', 'black'], alpha = 0.4)
            
            hist1, _ = np.histogram(detector_idealcorrection_r_sorted[:int(n_electrons_passed/3)], bins)
            hist2, _ = np.histogram(detector_idealcorrection_r_sorted[int(n_electrons_passed/3):int(n_electrons_passed*2/3)], bins)
            hist3, _ = np.histogram(detector_idealcorrection_r_sorted[int(n_electrons_passed*2/3):], bins)
            axl[1,0].stackplot(rs, hist1/_factor, hist2/_factor, hist3/_factor,
                               colors =['red', 'red', 'red'], alpha = 0.4)
            
            hist1, _ = np.histogram(detector_corrected_r_sorted[:int(n_electrons_passed/3)], bins)
            hist2, _ = np.histogram(detector_corrected_r_sorted[int(n_electrons_passed/3):int(n_electrons_passed*2/3)], bins)
            hist3, _ = np.histogram(detector_corrected_r_sorted[int(n_electrons_passed*2/3):], bins)
            axl[1,0].stackplot(rs, hist1/_factor, hist2/_factor, hist3/_factor,
                               colors =['blue', 'cyan', 'green'], alpha = 0.4)
            
            # calculate diffraction effect
            difraction = 0.61*np.average(self.e_beam.electron_wavelength_ini)/np.sin(self.conv_semiang)
            axl[1,0].plot([],[], color='k', label='aberrated D50={}\n D50_diff={}'.format(self.detector_aberrated_D50, (self.detector_aberrated_D50**2+difraction**2)**0.5))
            axl[1,0].plot([],[], color='blue', label='real correction D50={}\n D50_diff={}'.format(self.detector_realcorrection_D50, (self.detector_realcorrection_D50**2+difraction**2)**0.5))
            axl[1,0].plot([],[], color='red', label='ideal correction D50={}\n D50_diff={}'.format(self.detector_idealcorrection_D50, (self.detector_idealcorrection_D50**2+difraction**2)**0.5))

            if self.electron_initialcomposition == 'oneline':
                axl[1,0].set_ylabel('electron density crossection normalized [electrons/m2/electronstotal]')
            else: axl[1,0].set_ylabel('electron density crossection normalized [electrons/m/electronstotal]')


        else:
            axl[1,0].hist(detector_aberrated_r, bins, density=True, alpha=0.5, label='aberrated D50={}'.format(self.detector_aberrated_D50), color='black')
            detector_corrected_r_sorted = np.take(detector_corrected_r, idx_sorting)
            hist1, _ = np.histogram(detector_corrected_r_sorted[:int(n_electrons_passed/3)], bins, density=True)
            hist2, _ = np.histogram(detector_corrected_r_sorted[int(n_electrons_passed/3):int(n_electrons_passed*2/3)], bins, density=True)
            hist3, _ = np.histogram(detector_corrected_r_sorted[int(n_electrons_passed*2/3):], bins, density=True)
            axl[1,0].stackplot(rs, hist1/3, hist2/3, hist3/3,
                               colors =['blue', 'cyan', 'green'], alpha = 0.4)
            axl[1,0].plot([],[], color='blue', label='real correction D50={}'.format(self.detector_realcorrection_D50))






            # axl[1,0].hist(detector_corrected_r, 100, density=True, alpha=0.4, label='real correction D50={}'.format(self.detector_realcorrection_D50), color='red')
        axl[1,0].legend()
        axl[1,0].set_xlabel('r [m]')

        





        _laserplaneidx = int(np.size(self.laser_thick.zs)/2)
        im = ax[1,1].contourf(
            self.laser_setup.detector_xs, self.laser_setup.detector_xs , 
            self.laser_setup.detector_2D_array[:,:,_laserplaneidx],
            levels=100, cmap='gray')
        ax[1,1].set_title("thin-laser and electrons in interaction plane 0")
        ax[1,1].set_xlabel("x [m]")
        ax[1,1].set_ylabel("y [m]")
        detector_interacitionplane = self.e_beam.detector_spacelike(self.laser_plane_z, False)
        ax[1,1].scatter(detector_interacitionplane[0],detector_interacitionplane[1], s=1)

        from matplotlib.colors import to_rgb, to_hex
        def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
            c1=np.array(to_rgb(c1))
            c2=np.array(to_rgb(c2))
            return to_hex((1-mix)*c1 + mix*c2)

        # define curve colors:
        _colormap = mplcm.copper
        colorparams = np.arange(np.size(self.laser_thick.zs))
        normalize = mplcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))


        dict_profiles = {}
        dict_profiles["xs"] = self.laser_thick.xs
        
        _laserthick_nplanes = np.shape(self.laser_thick.intensity_distribution_inspace)[2]
        center_y_idx = int(np.shape(self.laser_thick.intensity_distribution_inspace)[1]/2)
        _dx = self.laser_thick.xs[1]-self.laser_thick.xs[0]
        _volume = np.sum(self.laser_thick.intensity_distribution_inspace[:,center_y_idx,int(_laserthick_nplanes/2)] * abs(self.laser_thick.xs))*np.pi*_dx
        _factor = self.laser_thick.laser_pulse_energy/_volume # scale laser profile to intensity in SI units [J/m2]
        try: # in case of chromatic aberration correction _factor is an array which needs to be reduced to single value float
            _factor = _factor[0]
        except:
            pass
        for c_plane in range(_laserthick_nplanes):
            _color = _colormap(normalize(c_plane))
            ax[1,2].plot(self.laser_thick.xs, self.laser_thick.intensity_distribution_inspace[:,center_y_idx,c_plane]*_factor, label= 'z={}m'.format((self.laser_thick.zs[c_plane])), color=_color)
            dict_profiles[str(self.laser_thick.zs[c_plane] - self.laser_plane_z)] = self.laser_thick.intensity_distribution_inspace[:,center_y_idx,c_plane]*_factor

        if aberration == 'spherical':
            _dx = self.laserideal_xs[1]-self.laserideal_xs[0]
            _volume =  np.sum(self.laserideal_Intensity_distr_artifitial_units * abs(self.laserideal_xs))*np.pi*_dx
            _factor = self.laserideal_pulse_energy / _volume # scale laser profile to intensity in SI units [J/m2]
            ax[1,2].plot(self.laserideal_xs, self.laserideal_Intensity_distr_artifitial_units*_factor, label='ideal {}J'.format(self.laserideal_pulse_energy))
            dict_profile_ideal = {}
            dict_profile_ideal['laser ideal'] = self.laserideal_Intensity_distr_artifitial_units*_factor
            dict_profile_ideal["xs_ideal"] = self.laserideal_xs
            
        #ax[1,2].legend()
        ax[1,2].set_ylabel('[ J/m2 ]')
        ax[1,2].set_xlabel('x[m]')
        ax[1,2].set_title('laser real {}J'.format(self.laser_thick_pulse_energy))


        

        # laser crossection at elements
        try:
            axl[1,1].plot(self.laser_setup.beamline.wave.xvalues,np.array(self.laser_setup.beamline.elements_intensity_crossection).T, label=["laser", "SLM", "lens1", "lens2", "lens3","beamblock"])
            axl[1,1].legend()
        except:
            print("can not find complex wave data")


        
        detector_intime_laservolume_z_sorted = np.take(detector_intime_laservolume_z,idx_sorting)
        num_detected = np.size(detector_spacelike_laserplane_velocity)
        detector_intime_laservolume_z_sorted_slow = detector_intime_laservolume_z_sorted[:int(num_detected/3)]
        detector_intime_laservolume_z_sorted_medi = detector_intime_laservolume_z_sorted[int(num_detected/3):int(num_detected*2/3)]
        detector_intime_laservolume_z_sorted_fast = detector_intime_laservolume_z_sorted[int(num_detected*2/3):]


        kwargs = dict(histtype='stepfilled', alpha=0.5, density=True, bins=100)
        ax[0,2].hist(detector_intime_laservolume_z_sorted_slow, **kwargs)
        ax[0,2].hist(detector_intime_laservolume_z_sorted_medi, **kwargs)
        ax[0,2].hist(detector_intime_laservolume_z_sorted_fast, **kwargs)
        _hist, _bins = np.histogram(detector_intime_laservolume_z_sorted_medi,100, density=True)
        ax[0,2].plot((self.laser_thick.ts-self.laser_thick.t0)* constants.c * self.laser_direction +self.laser_plane_z , self.laser_thick.intensity_distribution_intime/np.max(self.laser_thick.intensity_distribution_intime)*np.max(_hist), color='red', linewidth = 2)
        ax[0,2].set_xlabel('z [m]')
        ax[0,2].set_title('Laser and electron pulses in interaction plane')
        patch_orange = mpatches.Patch(color='red', label='laser pulse')
        # patch_blue = mpatches.Patch(color='black', label='electron pulse')
        ax[0,2].legend(handles=[patch_orange])
        for c_plane in range(np.size(self.laser_thick.zs)):
            _color = _colormap(normalize(c_plane))
            _z = self.laser_thick.zs[c_plane]
            ax[0,2].plot([_z,_z], [0,np.max(_hist)], color=_color)


        # 
        if aberration == 'spherical':
            detector_realcorrection_sorted = np.take(self.detector_realcorrection,idx_sorting_r, axis=1)
        else:
            detector_realcorrection_sorted = np.take(self.detector_realcorrection,idx_sorting, axis=1)
        ax[1,0].scatter(detector_realcorrection_sorted[0,:int(num_detected/3)],                        detector_realcorrection_sorted[1,:int(num_detected/3)], s=1, label='slow', alpha=0.5)
        ax[1,0].scatter(detector_realcorrection_sorted[0,int(num_detected/3):int(num_detected*2/3)],   detector_realcorrection_sorted[1,int(num_detected/3):int(num_detected*2/3)], s=1, label='medi', alpha=0.5)
        ax[1,0].scatter(detector_realcorrection_sorted[0,int(num_detected*2/3):],                      detector_realcorrection_sorted[1,int(num_detected*2/3):], s=1, label='fast', alpha=0.5)
        ax[1,0].set_xlabel('x [m]')
        ax[1,0].set_ylabel('y [m]')
        ax[1,0].set_title('real correction D90 = {} m\nno correction D90 = {} m'.format(self.detector_realcorrection_D90, self.detector_aberrated_D90))
        ax[1,0].legend()

        self.laser_setup.setup_raytracing(ax=ax[0,1])


        # SLM
        _SLM = self.laser_setup.SLM_phasechange
        _SLM_x = self.laser_setup.SLM.xs
        _SLM_y = self.laser_setup.SLM.ys
        
        _cs = axl[0,0].contourf(_SLM_x,_SLM_y,_SLM,100)
        _cbar = figl.colorbar(_cs)
        
        axl[1,2].plot(_SLM_x, _SLM[int(np.size(_SLM_x)/2)])







        plt.show()


        if export:
            # export spot distributions
            hist_nocorrection, bins_nocorrection = np.histogram(detector_aberrated_r, 200, density=True)
            hist_corrected, _ = np.histogram(detector_corrected_r,bins_nocorrection, density=True) # use teh same bins for better ploting in excel
            if aberration == 'spherical':
                hist_ideal_correction, _ = np.histogram(detector_idealcorrection_r,bins_nocorrection, density=True) # use teh same bins for better ploting in excel
                hist = {
                    'hist_corrected': hist_corrected ,
                    'hist_corrected_ideal': hist_ideal_correction ,
                    'hist_nocorrection': hist_nocorrection ,
                    'bins': bins_nocorrection[1:]}
            else:
                hist = {
                    'hist_corrected': hist_corrected,
                    'hist_nocorrection': hist_nocorrection ,
                    'bins': bins_nocorrection[1:]}
            df = pd.DataFrame(hist)        
            df.to_csv('data\\datanew\\spot_histograms__electrons_per_m_dividedbyelectronstotal.csv', sep='\t', index=False,header=True)



            

            # export pulses
            hist_all, _bins = np.histogram(detector_intime_laservolume_z_sorted, 100, density=True)
            _binsize = _bins[1] - _bins[0]
            hist_slow, _ = np.histogram(detector_intime_laservolume_z_sorted_slow, _bins, density=True)
            hist_medi, _ = np.histogram(detector_intime_laservolume_z_sorted_medi, _bins, density=True)
            hist_fast, _ = np.histogram(detector_intime_laservolume_z_sorted_fast, _bins, density=True)

            laser_interpolated = np.interp(
                _bins[1:]-_binsize/2, 
                (self.laser_thick.ts-self.laser_thick.t0)* constants.c + self.laser_plane_z, 
                self.laser_thick.intensity_distribution_intime/np.max(self.laser_thick.intensity_distribution_intime))

            electron_pulses = {'hist_slow' : hist_slow,
                               'hist_medi' : hist_medi,
                               'hist_fast' : hist_fast,
                               'hist_all' : hist_all,
                               'bins_z' : _bins[1:]-_binsize/2 - self.laser_plane_z,
                               'laser_interpolated' : laser_interpolated}
            
            df = pd.DataFrame(electron_pulses)        
            df.to_csv('data\\datanew\\electronpulses__electrons_permetr_dividedbyelectronstotal.csv', sep='\t', index=False,header=True)




            # # export laser profiles
            df = pd.DataFrame(dict_profiles)        
            df.to_csv('data\\datanew\\laser_profiles_J_per_m2.csv', sep='\t', index=False,header=True)
            if aberration == 'spherical':
                df = pd.DataFrame(dict_profile_ideal)        
                df.to_csv('data\\datanew\\dict_profile_ideal_J_per_m2.csv', sep='\t', index=False,header=True)
            


            # export laser profile on z
            laser_profile_z =  {'zs' : (self.laser_thick.ts-self.laser_thick.t0)* constants.c,
                                'laser profile' : self.laser_thick.intensity_distribution_intime/np.max(self.laser_thick.intensity_distribution_intime)}
            
            df = pd.DataFrame(laser_profile_z)        
            df.to_csv('data\\datanew\\laser_profile_z.csv', sep='\t', index=False,header=True)


    def export_laserprofiles(self, laser_xs, laser_zs, laser_profiles, laser_energies):
        dict_profiles = {}
        dict_profiles["xs"] = laser_xs
        
        _laserthick_nplanes = np.shape(laser_profiles)[1]
        _dx = laser_xs[1]-laser_xs[0]
        _volume = np.sum(laser_profiles[:,0]*abs(laser_xs)*np.pi*_dx)#np.sum(self.laser_thick.intensity_distribution_inspace[:,center_y_idx,int(_laserthick_nplanes/2)] * abs(self.laser_thick.xs))*np.pi*_dx
        _factor = laser_energies[0]/_volume # scale laser profile to intensity in SI units [J/m2]
        for c_plane in range(_laserthick_nplanes):
            dict_profiles[str(laser_zs[c_plane])] = laser_profiles[:,c_plane]*_factor

        # # export laser profiles
        df = pd.DataFrame(dict_profiles)        
        df.to_csv('data\\datanew\\laser_profiles_manual_J_per_m2.csv', sep='\t', index=False,header=True)


            

            





          




class Laser_setup:
    def __init__(self, laser_wavelength, plane0_width, plane_points, laser_beam_width, SLM_npix, SLM_size, detectplanesz, laser_zpos, laser_xcenter, SLM_zpos, SLM_xcenter, SLM_dist, lens1_f, lens2_f, lens3_f, beamblock_r, lens3_zdist, lens1_r, lens2_r, lens3_r, rescaling_f = 0.1) -> None:
        """Laser setup with the 3. lens as parabolic mirror

        Args:
            laser_wavelength (float): laser_wavelength [m]
            plane0_width (float): width of lightpipes plane [m]
            plane_points (int): number of points in one dimension in lightpipes plane (pixels)
            laser_beam_width (float): initial diameter of gaussian laser waist
            SLM_npix ([int,int]): number of SLM pixels [xpixels, ypixels]
            SLM_size ([float,float]): SLM screen size [xsize, ysize] [m]
            detectplanesz (NP1DArray): planes in which the laser profile will be detected in respect to the central interaction plane, ie. [-1,-0.5,0,0.5.1]e-3  [m]
            laser_zpos (float): z position of laser [m]
            laser_xcenter (float): [m]
            SLM_zpos (float): SLM z position [m]
            SLM_xcenter (float): [m]
            SLM_dist (float): distance between SLM and Lens1 [m]
            lens1_f (float): lens 1 focal dist [m]
            lens2_f (float): lens 2 focal dist [m]
            lens3_f (float): lens 3 focal dist [m]
            beamblock_r (float): mirror hole radius [m]
            lens3_zdist (float): distance between lens2 and lens3 [m]
            lens1_r(float): lens1 radius, if None then not applied
            lens2_r(float): lens2 radius, if None then not applied
            lens3_r(float): lens3 radius, if None then not applied
            rescaling_f(float): rescaling factor in range <0.1,1>. 1 means no rescaling.
        """
        self.laser_wavelength = laser_wavelength
        self.plane0_width = plane0_width
        self.plane_points = plane_points
        self.laser_beam_width = laser_beam_width
        self.SLM_npix = SLM_npix
        self.SLM_size = SLM_size
        self.detectplanesz = detectplanesz
        self.laser_zpos = laser_zpos
        self.laser_xcenter = laser_xcenter
        self.SLM_zpos = SLM_zpos
        self.SLM_xcenter = SLM_xcenter
        self.SLM_dist = SLM_dist
        self.lens1_f = lens1_f
        self.lens2_f = lens2_f
        self.lens3_f_nominal = lens3_f
        self.beamblock_r = beamblock_r
        self.lens3_zdist = lens3_zdist

        self.lens1_r = lens1_r
        self.lens2_r = lens2_r
        self.lens3_r = lens3_r

        self.rescaling_f = rescaling_f
        self.lens3_f_forcomputing = self.lens3_f_nominal / self.rescaling_f

        self.lens1_zpos = SLM_zpos + SLM_dist
        self.conjplane1_zpos = self.lens1_zpos + self.lens1_f
        self.lens2_zpos = self.conjplane1_zpos + self.lens2_f
        self.lens3_zpos = self.lens2_zpos + self.lens3_zdist
        self.beamblock_zpos_info = self.lens3_zpos + self.lens3_r/ self.rescaling_f # planar mirror 45° has its center just lens3_r from the lens3
        self.conjplane2_zpos = self.lens3_zpos + self.lens3_f_forcomputing

        self.SLM_phasefactor1 = 0
        self.SLM_phasefactor2 = 0
        self.SLM_phasefactor3 = 0
        self.SLM_lens_f_inverse = 0 #  = 1/f
        self.SLM_doughnut_turns = 0
        self.SLM_sphericalaberrationcoef = 0 # artifitial coefficient determinig spherical aberration of the SLM lensing
        self.SLM_lens_f_inverse_2 = 0


    def build_setup(self):
        SLM_xs = np.linspace(-self.SLM_size[0]/2, self.SLM_size[0]/2, self.SLM_npix[0])
        SLM_ys = np.linspace(-self.SLM_size[1]/2, self.SLM_size[1]/2, self.SLM_npix[1])
        xx,yy = np.meshgrid(SLM_xs, SLM_ys)
        rr = (xx**2+yy**2)**0.5
        SLM_angle = np.arctan2(xx,yy)


        self.SLM_phasechange = rr * self.SLM_phasefactor1 + rr**2 *self.SLM_phasefactor2 + rr**3* self.SLM_phasefactor3
        # SLM Doughnout
        self.SLM_phasechange += SLM_angle * self.SLM_doughnut_turns
        # SLM lens
        _k = 2*np.pi/self.laser_wavelength
        self.SLM_phasechange += -_k*(rr**2)/2*self.SLM_lens_f_inverse
        # SLM spherical aberration
        self.SLM_phasechange += -_k*(rr**4)/2*self.SLM_lens_f_inverse*self.SLM_sphericalaberrationcoef
        # SLM second lens
        self.SLM_phasechange += -_k*(rr**2)/2*self.SLM_lens_f_inverse_2

        # modulate SLM phasechange to be in interval (0,2pi)
        self.SLM_phasechange = np.remainder(self.SLM_phasechange, np.pi*2)



        

        self.laser = lib_laser_wave_FFT.Laser(self.laser_zpos, self.laser_beam_width)
        self.SLM = lib_laser_wave_FFT.SLM(self.SLM_zpos,self.SLM_npix, self.SLM_size, self.SLM_phasechange, x_center=self.SLM_xcenter)
        self.lens1 = lib_laser_wave_FFT.Lens(self.lens1_zpos, self.lens1_f,self.lens1_r)
        self.lens2 = lib_laser_wave_FFT.Lens(self.lens2_zpos, self.lens2_f,self.lens2_r)
        #self.lens3 = lib_laser_wave_FFT.Lens(self.lens3_zpos, self.lens3_f_forcomputing,self.lens3_r)
        #self.beamblock = lib_laser_wave_FFT.BeamBlock(self.beamblock_zpos, self.beamblock_r, 0,0)
        self.elements = [self.laser, self.SLM, self.lens1, self.lens2]
        self.beamline = lib_laser_wave_FFT.BeamLine(self.elements, self.laser_wavelength, self.plane0_width, self.plane_points)

    def propagate(self):
        self.beamline.propagate()

    def detect(self):
        # detector_2D_array = []
        # detector_intensity_crossections = []
        # detector_intensity_integral = []
        # x_coord_center = None



        detected_planes_complexwave, detector_xs = self.beamline.get_detector_lens_and_beamblock(
            defocuses= self.detectplanesz,
            lens_zpos= self.lens3_zpos,
            beamblock_radius= self.beamblock_r,
            lens_f_nominal = self.lens3_f_nominal,
            lens_f_computing = self.lens3_f_forcomputing,
            lens_f_defocus = self.lens3_f_nominal,
            lens_scalingfactor = self.rescaling_f,
            lens_r = self.lens3_r,
        )

        detected_planes_complexwave



        # for c_plane in range(np.size(self.detectplanesz)):
        #     xs, detector2_intensity_crossection, x_coord_center, y_coord_center, detector_center_2D = self.beamline.get_detector_lens_and_beamblock(self.conjplane2_zpos, self.lens3, self.beamblock, kind='intensity_crossection_x_center2D')
        #     detector_2D_array.append(detector_center_2D)
        #     # df_2D = pd.DataFrame(detector_center_2D)
        #     # df_2D.to_csv('data\\series3_spehricalaberration\\plane{}.csv'.format(detectplanesz[c_plane]))
        #     print("plane {}".format(c_plane))
        #     detector_intensity_crossections.append(detector2_intensity_crossection)
        #     dx = x_coord_center[1]-x_coord_center[0]
        #     detector_intensity_integral.append(np.sum(detector_center_2D)*dx**2)
        

        self.detector_xs = detector_xs
        self.detector_2D_array = np.array(detected_planes_complexwave) # [z,x,y]
        self.detector_2D_array = abs(self.detector_2D_array)**2 # convert to intensity
        self.detector_2D_array = np.rollaxis(self.detector_2D_array, 0, 3) # [x,y,z]
        # self.detector_intensity_crossections = np.array(detector_intensity_crossections)
        # self.detector_intensity_integrals = np.array(detector_intensity_integral)


    def check_losses(self, plot=False):
            if plot:
                plt.plot(self.beamline.wave.xvalues,np.array(self.beamline.elements_intensity_crossection).T, label=["laser", "SLM", "lens1", "lens2", "lens3","beamblock"])
                # plt.plot(beamline.wave.xvalues,np.array(detector_intensity_crossections).T, "red", label="detector")
                plt.legend()
                plt.show()


    def setup_raytracing(self, stop=True, ax=None):
        #raise Exception("not finished so far")
        
        beam_r = self.laser_beam_width/2
        num_rays = 100

        ini_pos_x = np.linspace(-beam_r,beam_r,num_rays)
        x = ini_pos_x * abs(ini_pos_x)
        ini_pos_x = x/ x.max() * beam_r
        #ini_pos_x = np.random.normal(0,beam_r, num_rays)

        ini_pos_y = np.zeros(num_rays)
        ini_pos_z = np.full(num_rays,self.laser_zpos)

        ini_dir_x = np.zeros(num_rays)
        ini_dir_y = np.zeros(num_rays)
        ini_dir_z = np.ones(num_rays)

        ini_pos = np.array([ini_pos_x, ini_pos_y, ini_pos_z])
        ini_dir = np.array([ini_dir_x, ini_dir_y, ini_dir_z])


        print("Warnig: Ray tracing without SLM doughnut turns, since it would confuse tracing result.")
        _SLM_doughnut_turns = self.SLM_doughnut_turns +0
        self.SLM_doughnut_turns = 0
        self.build_setup()


        laser = lib_laser_tracing.Laser(0, ini_pos, ini_dir)
        SLM = lib_laser_tracing.SLM(self.SLM.pos, self.SLM_size, self.SLM_npix, self.SLM_phasechange, self.laser_wavelength, xcenter=self.SLM_xcenter)
        lens1 = lib_laser_tracing.Lens(self.lens1_zpos, self.lens1_f, self.lens1_r)
        lens2 = lib_laser_tracing.Lens(self.lens2_zpos, self.lens2_f, self.lens2_r)
        lens3 = lib_laser_tracing.Lens(self.lens3_zpos, self.lens3_f_nominal, self.lens3_r)
        beam_block = lib_laser_tracing.BeamBlock(self.beamblock_zpos_info, self.beamblock_r)

        self.SLM_doughnut_turns = _SLM_doughnut_turns
        self.build_setup()
        

        elements = [laser,SLM, lens1, lens2, lens3, beam_block]
        #elements = [laser,lens1, lens2]
        setup_raytracing = lib_laser_tracing.Optical_assembly(elements)

        setup_raytracing.trace()

        cros_zs = np.linspace(self.laser_zpos+1e-9, self.lens3_zpos+ self.lens3_f_nominal*1.1 ,500)
        crossection = setup_raytracing.crossection(cros_zs)

        if ax == None:
            fig, ax = plt.subplots(1,1)
        
        ax.plot(crossection[:,0], crossection[:,2], 'firebrick')
        for c_element in range(1, np.size(elements, axis=0)):
            element = elements[c_element]
            if isinstance(element, lib_laser_tracing.Lens):
                ax.plot([-element.radius,element.radius], [element.pos,element.pos], 's-', color='k')
            elif isinstance(element, lib_laser_tracing.BeamBlock):
                ax.plot([-element.radius,element.radius], [element.pos,element.pos], '-', color='k', linewidth=3)
            elif isinstance(element, lib_laser_tracing.SLM):
                ax.plot([element.xs[0],element.xs[-1]], [element.pos,element.pos], '-', color='navy', linewidth=3)

        
        for c_plane in range(np.size(self.detectplanesz)):
            ax.plot([-self.lens3_r/10,self.lens3_r/10], [self.lens3_zpos + self.lens3_f_nominal+self.detectplanesz[c_plane],self.lens3_zpos + self.lens3_f_nominal+self.detectplanesz[c_plane]], 'gray')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        ax.invert_yaxis()
        ax.set_title("doughnut turns ignored in raytracing")


   



