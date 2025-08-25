
from LightPipes.zernikemath import zernike
import numpy as np





def Zernike_polar(rs, wavenumber, n, m, R, A = 1.0, norm=True, units='opd'):
    """
    Caluculates Zernike aberration phase distribution in the wave for cylindrical coordinates.
    This is a remake of LightPipes.Zernike  

    Input
    -----
    rs : np.array 1D float
        radial coordinates where to calculate Zernike phase shift
        [m]
    wavenumber : float
        wave number [rad/m]
    n : int, float
        radial ordern
    m: int, float 
        azimuthal order, n-\\|m\\| must be even, \\|m\\|<=n
    R: int, float
        radius of the aberrated aperture
    A: int, float
        size of the aberration
    norm: bool
        if True, normalize integral(Z over unit circle)=1, if False
                Z(rho=1)=1 on edge (-> True=rms const, False=PtV const) (default = True)
    units: string
        units: 'opd': A given in meters as optical path difference (default = 'opd')
                'lam': A given in multiples of lambda
                'rad': A given in multiples of 2pi

    Returns
    -------
    phase_shift : np.array 1D float
        phase shift in the rs coordinates, [rad]
        >>> wave *= np.exp(1j*phase_shift)


    Description
    -----------
    :math:`F_{out}(x,y)=e^{\\phi^m_n (x,y)}F_{in}(x,y)`
    
     with:
    
    :math:`\\phi^m_n(x,y)=-j \\frac{2 \\pi }{ \\lambda } Z^m_n {(\\rho (x,y) ,\\theta (x,y)) }`
    
    :math:`\\rho(x,y)=  \\sqrt{ \\frac{x^2+y^2}{R^2} }`
    
    :math:`\\theta (x,y)=atan \\big( \\frac{y}{x} \\big)`
    
    :math:`Z^m_n(\\rho , \\theta)=A \\sqrt{ \\frac{2(n+1)}{1+\\delta_{m0}} } V^m_n(\\rho)cos(m\\theta)`
    
    :math:`Z^{-m}_n(\\rho , \\theta)=A \\sqrt{ \\frac{2(n+1)}{1+\\delta_{m0}} }V^m_n(\\rho)sin(m\\theta)`
    
    :math:`V^m_n(\\rho)= \\sum_{s=0}^{ \\frac{n-m}{2} }  \\frac{(-1)^s(n-s)!}{s!( \\frac{n+m}{2}-s)!( \\frac{n-m}{2}-s )! } \\rho^{n-2s}`
    
    :math:`\\delta_{m0}= \\begin{cases}1 & m = 0\\\\0 & m  \\neq  0\\end{cases}`
    
    :Example:
        if norm=True and Aunit='lambda' and A=1.0, the wavefront
        will have an rms error of 1lambda, but PtV depending on n/m.
        If norm=False and Aunit='lambda' and A=1.0, the wavefront will
        have a PtV value of 2lambda (+ and -1 lambda!), but rms error
        depending on n/m.

    .. seealso::
    
        * :ref:`Manual: Zernike polynomials.<Zernike polynomials.>`
        * :ref:`Examples: Zernike aberration.<Zernike aberration.>`
        * `https://en.wikipedia.org/wiki/Zernike_polynomials <https://en.wikipedia.org/wiki/Zernike_polynomials>`_
    """
    mcorrect = False
    ncheck = n
    while ncheck >= -n:
        if ncheck == m:
            mcorrect = True
        ncheck -= 2
    if not mcorrect:
        raise ValueError('Zernike: n,m must fulfill: n>0, |m|<=n and n-|m|=even')
        
    #Fout = Field.copy(Fin)
    
    if units=='opd':
        A = wavenumber*A #if A=1e-6 [==1um~1lambda], this will yield 2pi/lam*1um, e.g. 1lambda OPD
    elif units=='lam':
        A = 2*np.pi*A #if A=1, this will yield 1 lambda OPD
    elif units=='rad':
        A = A #if A=2pi, this will yield 1lambda OPD
    else:
        raise ValueError('Unknown value for option units={}'.format(units))
    
    if norm:
        if m==0:
            # wikipedia has modulo Pi? -> ignore for now
            # keep backward compatible and since not dep. on n/m irrelevant
            Nnm = np.sqrt(n+1)
        else:
            Nnm = np.sqrt(2*(n+1))
    else:
        Nnm = 1
    
    #r, phi = Fout.mgrid_polar
    phi = 0
    rho = rs/R
    phase_shift = -A*Nnm*zernike(n,m, rho, phi)
    # Fout.field *= _np.exp(1j*phase_shift)
    # return Fout
    return phase_shift