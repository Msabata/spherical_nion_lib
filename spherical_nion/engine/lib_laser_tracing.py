
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from scipy.special import jn, jn_zeros

class Laser:
    def __init__(self, pos, ray_pos_ini, ray_dir_ini) -> None:
        self.pos = pos
        self.ray_pos_ini = ray_pos_ini
        self.ray_dir_ini = ray_dir_ini

    def propagate(self):
        return (self.ray_pos_ini, self.ray_dir_ini)

class Lens:
    def __init__(self, pos, focal_dist, radius) -> None:
        self.pos = pos
        self.focal_dist = focal_dist
        self.radius = radius

    def propagate(self, ray_pos_before, ray_dir_before):
        ray_pos_after = ray_pos_before
        # prepare a new velocity vecor with an artifitial magnitude:
        vx_after = ray_dir_before[0]/ray_dir_before[2] - ray_pos_before[0]/self.focal_dist
        vy_after = ray_dir_before[1]/ray_dir_before[2] - ray_pos_before[1]/self.focal_dist
        vz_after = np.ones(np.shape(vx_after))
        ray_dir_after = np.array([vx_after, vy_after, vz_after])
        # scale to the same velocity magnitude as before:
        ray_dir_after = ray_dir_after / np.linalg.norm(ray_dir_after, axis=0) * np.linalg.norm(ray_dir_before,axis=0) 
        # aperture
        pos_rr = (ray_pos_before[0]**2 + ray_pos_before[1]**2)**0.5
        ray_dir_after = np.where(pos_rr<self.radius, ray_dir_after, np.nan)
        ray_pos_after = np.where(pos_rr<self.radius, ray_pos_after, np.nan)

        return ray_pos_after, ray_dir_after
    


class Lens_radial:
    def __init__(self, pos, focal_dist, radius) -> None:
        self.pos = pos
        self.focal_dist = focal_dist
        self.radius = radius

    def propagate(self, ray_pos_before, ray_dir_before):
        ray_pos_after = ray_pos_before
        pos_rr = (ray_pos_before[0]**2 + ray_pos_before[1]**2)**0.5
        # prepare a new velocity vecor with an artifitial magnitude:
        ray_dir_before_radial = (ray_dir_before[0]*ray_pos_before[0] + ray_dir_before[1]*ray_pos_before[1]) / pos_rr
        ray_dir_before_azimut = (ray_dir_before[0]*ray_pos_before[0]*(-1) + ray_dir_before[1]*ray_pos_before[1]) / pos_rr

        vr_after = ray_dir_before_radial/ray_dir_before[2] - pos_rr/self.focal_dist

        vx_after = ray_pos_after[0]/pos_rr * vr_after
        vy_after = ray_pos_after[1]/pos_rr * vr_after


        # vx_after = ray_dir_before[0]/ray_dir_before[2] - ray_pos_before[0]/self.focal_dist
        # vy_after = ray_dir_before[1]/ray_dir_before[2] - ray_pos_before[1]/self.focal_dist
        vz_after = np.ones(np.shape(vx_after))
        ray_dir_after = np.array([vx_after, vy_after, vz_after])
        # scale to the same velocity magnitude as before:
        ray_dir_after = ray_dir_after / np.linalg.norm(ray_dir_after, axis=0) * np.linalg.norm(ray_dir_before,axis=0) 
        # aperture
        
        ray_dir_after = np.where(pos_rr<self.radius, ray_dir_after, np.nan)
        ray_pos_after = np.where(pos_rr<self.radius, ray_pos_after, np.nan)

        return ray_pos_after, ray_dir_after



# class Lens_thick_spherical:
#     def __init__(self, pos, radius, rs, zs, ns) -> None:
#         """Thick spehrical lens

#         Parameters
#         ----------
#         pos : [float,float,float]
#             z posistion along the optical axis [m]
#         radius : float
#             lens radius (half of diameter)
#         rs : np1DArray
#             radiuses of surfaces. C-shape has negative radius, D-shape has positive.
#         zs : np1DArray
#             centers of spheres of surfaces
#         ns : np1DArray
#             refractrion indes in the following volume
#         """
#         self.pos = pos
#         self.radius = radius
#         self.rs = rs
#         self.zs = zs
#         self.ns = ns

#     def propagate(self, ray_pos_before, ray_dir_before):
        
#         ray_pos_new = np.full((np.size(self.rs)+1,np.shape(ray_pos_before)[0],np.shape(ray_pos_before)[1]), np.nan)
#         # propagate from beginnig (pos0) to the interface



#         ray_pos_after = ray_pos_before
#         # prepare a new velocity vecor with an artifitial magnitude:
#         vx_after = ray_dir_before[0]/ray_dir_before[2] - ray_pos_before[0]/self.focal_dist
#         vy_after = ray_dir_before[1]/ray_dir_before[2] - ray_pos_before[1]/self.focal_dist
#         vz_after = np.ones(np.shape(vx_after))
#         ray_dir_after = np.array([vx_after, vy_after, vz_after])
#         # scale to the same velocity magnitude as before:
#         ray_dir_after = ray_dir_after / np.linalg.norm(ray_dir_after, axis=0) * np.linalg.norm(ray_dir_before,axis=0) 
#         # aperture
#         pos_rr = (ray_pos_before[0]**2 + ray_pos_before[1]**2)**0.5
#         ray_dir_after = np.where(pos_rr<self.radius, ray_dir_after, np.nan)

#         return ray_pos_after, ray_dir_after

class SLM:
    def __init__(self, pos, size, num_pixels, phase_shift, wavelength, xcenter=0, ycenter=0) -> None:
        """_summary_

        Parameters
        ----------
        pos : float
            z-position [m]
        size : double (float, float)
            x,y size of the slm screnn
        num_pixels : double (float,float)
            number of pixels in x and y direction
        phase_shift : np.ndarry 2D
            phase shift in each pixel of slm
        wavelength : float
            _description_
        """
        self.pos = pos
        self.size = size
        self.num_pixels = num_pixels
        xs = np.linspace(-size[0]/2, size[0]/2, num_pixels[0]) + xcenter
        ys = np.linspace(-size[1]/2, size[1]/2, num_pixels[1]) + ycenter
        self.xs = xs
        self.ys = ys
        self.dx = xs[1] - xs[0]
        self.dy = ys[1] - ys[0]
        self.phase_shift = phase_shift
        phase_shift_grad = np.gradient(phase_shift) # [phase change difference / meter]

        self.phase_shift_grad_x_interpolator = RegularGridInterpolator((xs,ys), phase_shift_grad[0]/self.dx)
        self.phase_shift_grad_y_interpolator = RegularGridInterpolator((xs,ys), phase_shift_grad[1]/self.dy)
        self.wave_number = 2*np.pi/wavelength

    def update(self, phase_shift, xcenter=0, ycenter=0):
        self.xs = np.linspace(-self.size[0]/2, self.size[0]/2, self.num_pixels[0]) + xcenter
        self.ys = np.linspace(-self.size[1]/2, self.size[1]/2, self.num_pixels[1]) + ycenter
        self.phase_shift = phase_shift
        phase_shift_grad = np.gradient(phase_shift) # [phase change difference / meter]
        self.phase_shift_grad_x_interpolator = RegularGridInterpolator((self.xs,self.ys), phase_shift_grad[0]/self.dx)
        self.phase_shift_grad_y_interpolator = RegularGridInterpolator((self.xs,self.ys), phase_shift_grad[1]/self.dy)


    def propagate(self, ray_pos_before, ray_dir_before):
        # so far only for straight initial beam
        if any(ray_dir_before[0]) or any(ray_dir_before[1]) != 0:
            raise Exception("SLM: so far only for straight initial beam")
        tan_x = self.phase_shift_grad_x_interpolator(ray_pos_before[[0,1]].T)/self.wave_number
        dir_x = tan_x * ray_dir_before[2]
        tan_y = self.phase_shift_grad_y_interpolator(ray_pos_before[[0,1]].T)/self.wave_number
        dir_y = tan_y * ray_dir_before[2]
        ray_dir_after = np.array([dir_x, dir_y, np.ones(np.shape(dir_x))])
        return ray_pos_before, ray_dir_after

class BeamBlock:
    def __init__(self, pos, radius, x_center=0, y_center=0) -> None:
        self.pos = pos
        self.radius = radius
        self.x_center = x_center
        self.y_center = y_center

    def propagate(self, ray_pos_before, ray_dir_before):
        rr = ((ray_pos_before[0]-self.x_center)**2 + (ray_pos_before[1]-self.y_center)**2)**0.5
        ray_dir_after_z = np.where(rr>self.radius, ray_dir_before[2], np.nan)
        ray_dir_after = np.array([ray_dir_before[0], ray_dir_before[1], ray_dir_after_z])
        return ray_pos_before, ray_dir_after

class Optical_assembly:
    def __init__(self, elements, ) -> None:
        self.elements = elements
        self.elements_zpos = []
        for c_element in range(np.size(elements)):
            self.elements_zpos.append(self.elements[c_element].pos)
        self.elements_zpos = np.array(self.elements_zpos)
        print('elements_zpos: {}'.format(self.elements_zpos))
        self.ray_pos = None
        self.ray_dir = None


    # def trace(self, ray_pos_ini, ray_dir_ini):
    #     ray_pos = [ray_pos_ini]
    #     ray_dir = [ray_dir_ini]

    #     for c_element in range(np.size(self.elements)):
    #         element = self.elements[c_element]
    #         dist = element.pos - ray_pos[-1][2,0]
    #         ray_pos_before = self.propagate(ray_pos[-1], ray_dir[-1],dist)
    #         ray_pos_after, ray_dir_after = element.propagate(ray_pos_before, ray_dir_ini)
    #         ray_pos.append(ray_pos_after)
    #         ray_dir.append(ray_dir_after)

    #     self.ray_pos = np.array(ray_pos)
    #     self.ray_dir = np.array(ray_dir)

    def trace(self):

        ray_pos, ray_dir = self.elements[0].propagate()
        ray_pos = [ray_pos]
        ray_dir = [ray_dir]


        for c_element in range(1,np.size(self.elements)):
            element = self.elements[c_element]
            dist = element.pos -  self.elements[c_element-1].pos
            ray_pos_before = self.propagate(ray_pos[-1], ray_dir[-1],dist)
            ray_pos_after, ray_dir_after = element.propagate(ray_pos_before, ray_dir[-1])
            ray_pos.append(ray_pos_after)
            ray_dir.append(ray_dir_after)

        self.ray_pos = np.array(ray_pos)
        self.ray_dir = np.array(ray_dir)
        

    def propagate(self, ray_pos_ini, ray_dir_ini, dist):
        """propagate rays over the distance

        Parameters
        ----------
        ray_pos_ini : np.ndarray
            [x/y/z, ray]
        ray_dir_ini : np.ndarray
            [x/y/z, ray]
        dist : float
            distance to propagate [m]
        """
        ray_dir_ini_norm = ray_dir_ini/ray_dir_ini[2] # be suere, that the z-component is one 
        ray_pos_new = ray_pos_ini + ray_dir_ini_norm*dist
        return ray_pos_new

    def get_detector(self, z_pos, rs=False):
        # 1) find closest preceding element ########
        z_plane_preceding   = self.elements_zpos[self.elements_zpos<z_pos].max() # z-coordinate of preceding plane
        idx_plane_preceding = self.elements_zpos[self.elements_zpos<z_pos].argmax() # number of preceding plane
        # 2) propagate wave ########################
        z_dist = z_pos - z_plane_preceding
        detector_ray_pos = self.propagate(self.ray_pos[idx_plane_preceding], self.ray_dir[idx_plane_preceding], z_dist)
        if rs == True:
            xs = detector_ray_pos[0]
            ys = detector_ray_pos[1]
            rs = (xs**2 + ys**2)**0.5
            return rs
        else:
            return detector_ray_pos
        
    def get_airy(self):
        x = np.linspace(-10,10,100)
        # The jinc, or "sombrero" function, J0(x)/x
        jinc = lambda x: jn(1, x) / x
        airy = (2 * jinc(x))**2

        # Aperture radius (mm), light wavelength (nm)
        a, lam = 1.5, 500
        # wavenumber (mm-1)
        k = 2 * np.pi / (lam / 1.e6)
        # First zero in J1(x)
        x1 = jn_zeros(1, 1)[0]
        theta1 = np.arcsin(x1 / k / a)
        # Convert from radians to arcsec
        theta1 = np.degrees(theta1) * 60 * 60

        print('Maximum resolving power for pupil diameter {} mm at {} nm is {:.1f}'
            ' arcsec'.format(2*a, lam, theta1))
        





        
    def get_detector_phase(self, z_pos, wavelength):
        detector_ray_pos = self.get_detector(z_pos)
        positions_all = np.append(self.ray_pos,[detector_ray_pos], axis=0)
        difs = positions_all[1:]-positions_all[:-1]
        traj_lengths = np.linalg.norm(difs,axis=1)
        traj_lengths = np.sum(traj_lengths, axis=0)
        phase = traj_lengths/wavelength*2*np.pi 
        print(np.shape(phase))
        phase = phase[~np.isnan(phase)]  
        
        print(np.shape(phase))
        complex = np.exp(1j*phase)

        # get kx, ky
        ray_dir_last = self.ray_dir[-1]
        ray_dir_lastx = ray_dir_last[0,~np.isnan(ray_dir_last[0])] 
        ray_dir_lasty = ray_dir_last[1,~np.isnan(ray_dir_last[1])] 
        ray_dir_lastz = ray_dir_last[2,~np.isnan(ray_dir_last[2])] 
        ray_dir_last = np.array([ray_dir_lastx, ray_dir_lasty, ray_dir_lastz])

        velocity = np.linalg.norm(ray_dir_last, axis=0)
        k = np.pi*2/wavelength
        kx = ray_dir_last[0] / velocity * k
        ky = ray_dir_last[1] / velocity * k

        npix = 100

        plt.plot(kx, ky, '.')
        plt.show()

        kk_real, binsx, binsy = np.histogram2d(kx, ky, bins=[npix,npix], weights=np.real(np.exp(1j*phase)))
        kk_imag, binsx, binsy = np.histogram2d(kx, ky, bins=[npix,npix], weights=np.imag(np.exp(1j*phase)))

        plt.imshow(abs(kk_real))
        plt.show()

        kk_complex = kk_real + 1j*kk_imag

        psf = np.fft.ifftshift(kk_complex)
        psf = np.fft.ifft2(psf)
        psf = np.fft.fftshift(psf)
        psf = abs(psf)

        pixsizex = 2*np.pi/np.max(kx)/2
        pixsizey = 2*np.pi/np.max(ky)/2
        psf_xs = np.linspace(-pixsizey*npix/2, pixsizey*npix/2, npix)
        psf_ys = np.linspace(-pixsizey*npix/2, pixsizey*npix/2, npix)
        plt.contourf(psf_xs,psf_ys,psf)
        plt.title('psf')
        plt.show()



        

    
    def crossection(self, zs):
        crossection_ray_pos = []
        for c_z in range(np.size(zs)):
            det_ray_pos = self.get_detector(zs[c_z])
            crossection_ray_pos.append(det_ray_pos)
        print("crossection shape {}".format(np.shape(crossection_ray_pos)))
        return np.array(crossection_ray_pos)
    
    #def optimize(self, variables, variables_ini, parameters_wanted)

    # def get_image_size(self,img_pos):
    #     detector_ray_pos = self.get_detector(img_pos)
    #     xs = detector_ray_pos[0]
    #     ys = detector_ray_pos[1]
    #     rs = (xs**2 + ys**2)**0.5
    #     r_average = np.average(rs)
    #     return r_average
    
    # def get_detec
