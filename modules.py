
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import fitsio as ft
from astropy.table import Table, vstack, hstack

from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy import stats

arcsec = u.arcsec


mag2r_old = lambda mag: 1630. * 1.396**(-mag)  # the DR9 radius-mag relation

class LRGRad:
    
    def __init__(self):
        from scipy.interpolate import interp1d
        
        mags = np.array([4.0, 9.0, 10.0, 10.5, 11.5, 12.0, 12.5, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 17.0, 18.0])
        radii = np.array([429.18637985, 80.95037032, 57.98737129, 36.80882682,
                26.36735446, 25.29190318, 21.40616169, 15.33392671,
                13.74150366, 13.56870306, 12.03092488, 11.10823009,
                 9.79334208, 7.01528803, 5.02527796])
        log_radii = np.log10(radii)
        f_radius_log_south = interp1d(mags, log_radii, bounds_error=False, fill_value='extrapolate')
        self.f_radius_south = lambda mags: 10**f_radius_log_south(mags)


        mags = np.array([4.0, 9.0, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 17.0, 18.0])
        radii = np.array([429.18637985, 80.95037032, 60., 60.,
                60., 47.46123803, 38.68173428, 32.73883553,
                27.70897871, 23.45188791, 19.84883862, 16.79934664,
                13.67150555, 11.57107301, 7.83467367, 5.61223042,
                 4.02022236])
        log_radii = np.log10(radii)
        f_radius_log_north = interp1d(mags, log_radii, bounds_error=False, fill_value='extrapolate')
        self.f_radius_north = lambda mags: 10**f_radius_log_north(mags)              


class ELGRad:
    def __init__(self):
        from scipy.interpolate import interp1d
        
        # observed ELG
        mags = np.array([4., 5., 6., 7.] + np.arange(7.5, 18.5, 0.5).tolist())

        radii_north = [690., 590., 550., 510., 310., 
                       270., 260., 250., 150., 120., 95.,
                       85.,  70.,  65.,  40.,  35., 
                       30.,  25.,  20.,  18., 15.,
                       12.,  10.,  10., 8.0, 7.0]
        
        log_radii = np.log10(radii_north)
        f_radius_log_north = interp1d(mags, log_radii, bounds_error=False, fill_value='extrapolate')
        self.f_radius_north = lambda mags: 10**f_radius_log_north(mags)
        
                
        radii_south = [700., 500., 350., 260., 200., 
                       180., 170., 120., 100., 75., 65., 
                       55., 50., 40., 30., 25., 22.,
                       19., 18., 14., 11., 10.0, 7.5, 
                       6.0, 5.5, 4.2]        
        log_radii = np.log10(radii_south)
        f_radius_log_south = interp1d(mags, log_radii, bounds_error=False, fill_value='extrapolate')
        self.f_radius_south = lambda mags: 10**f_radius_log_south(mags) 


def mag2rad(mag, field):
    rad = ELGRad()
    # Radius-mag relationship
    if field=='north':
        return rad.f_radius_north(mag)
    elif field=='south':
        return rad.f_radius_south(mag)
    else:
        raise ValueError(f"{field} not implemented.")

class DataLoader:
    """
        Class to facilitate input parameters to the code
    """
    def __init__(self, **kw):   
        # initiate the paths
        self.data_dir = kw.get('data_dir', '/fs/ess/PHS0336/data/tanveer/elgmask')
        self.gaia_path = kw.get('gaia_path', '/fs/ess/PHS0336/data/tanveer/elgmask/gaia_lrg_mask_v1.fits')
        self.gaia_suppl_path = kw.get('gaia_suppl_path', '/fs/ess/PHS0336/data/tanveer/elgmask/gaia_reference_suppl_dr9.fits')
        
        self.gaia_columns = kw.get('gaia_columns', ['RA', 'DEC', 'mask_mag'])
        self.target_class = kw.get('target_class', 'ELG')

        self.maskbits = kw.get('maskbits', [1, 12, 13])
        self.min_nobs = kw.get('min_nobs', 1)

        # return the variables
        msg = ''
        for k,v in self.__dict__.items():
            msg += f'{k:20s}: {v}\n'
        print(msg)
            
    def read_cat(self, field):
        """ Read the target catalog """    

        self.photsys = 'S' if field=='south' else 'N'
        cat_path = os.path.join(self.data_dir, 'dr9_{}_{}_1.0.0_basic.fits'.format(self.target_class.lower(), field))
        cat = Table(ft.read(cat_path))
        return self.__clean_cat(cat)
        
    def read_randoms(self, field):
        
        self.photsys = 'S' if field=='south' else 'N'
        
        cat_path = os.path.join(self.data_dir, 'randoms_for_elgs.fits')
        randoms = Table(ft.read(cat_path))
        is_good = (randoms['PHOTSYS']==self.photsys)
        randoms = randoms[is_good]
        
        return self.__clean_cat(randoms)
        
        
    def __clean_cat(self, cat):
        
        print(f'# of targets: {len(cat)}')

        # Apply MASKBITS
        mask_clean = np.ones(len(cat), dtype=bool)
        for bit in self.maskbits:
            mask_clean &= (cat['MASKBITS'] & 2**bit)==0
        cat = cat[mask_clean]
        print(f'# of targets (after maskbits={self.maskbits}): {len(cat)}')

        # Remove pixels near the LMC
        ramin, ramax, decmin, decmax = 58, 110, -90, -56
        mask_remove = (cat['RA']>ramin) & (cat['RA']<ramax) & (cat['DEC']>decmin) & (cat['DEC']<decmax)
        cat = cat[~mask_remove]
        print(f'# of targets (after LMC cut): {len(cat)}')

        mask = (cat['NOBS_G']>=self.min_nobs) & (cat['NOBS_R']>=self.min_nobs) & (cat['NOBS_Z']>=self.min_nobs)
        cat = cat[mask]
        print(f'# of targets (after min_nobs >= {self.min_nobs}): {len(cat)}')
        
        return cat
    
    def read_siena(self, field):
        d = ft.FITS('/fs/ess/PHS0336/data/templates/SGA-2020.fits')
        dd = Table(d[1].read(columns=['RA', 'DEC', 'D26', 'MAG_LEDA']))
        
        
        print(f'# of Siena objects: {len(dd)}') 
        if field=='south':
            mask = (dd['DEC']<36.)
        else:
            mask = (dd['DEC']>30.)
            mask &= (dd['RA']<310.) & (dd['RA']>75.)
            
        dd = dd[mask]
        print(f'# of gaia objects ({field}): {len(dd)}')
        dd['radius'] = dd['D26']*60 # arcsec
        
        return dd
        
        
    
    def read_gaia(self, field):
        """ Read Gaia star catalog """
        gaia = Table(ft.read(self.gaia_path, columns=self.gaia_columns))
        gaia_suppl = Table(ft.read(self.gaia_suppl_path, columns=['ra', 'dec', 'mask_mag']))
        gaia_suppl.rename_columns(gaia_suppl.colnames, self.gaia_columns)
        gaia = vstack([gaia, gaia_suppl], join_type='exact')
        
        print(f'# of gaia objects: {len(gaia)}')        
        
        if field=='south':
            mask = (gaia['DEC']<36.)
        else:
            mask = (gaia['DEC']>30.)
            mask &= (gaia['RA']<310.) & (gaia['RA']>75.)
            
        gaia = gaia[mask]
        print(f'# of gaia objects ({field}): {len(gaia)}')
        
        #gaia['radius'] = 1630. * 1.396**(-gaia['mask_mag'])  # the DR9 radius-mag relation
        gaia['radius'] = mag2rad(gaia['mask_mag'], field)
        
        return gaia
    
    def read_ranvar(self, field):
        randoms_result_path = './elgmask_dev/density_rand_gaia_{}_minobs_{}_maskbits_{}.npy'\
                                .format(field, self.min_nobs, ''.join([str(tmp) for tmp in self.maskbits]))
        randoms_results = np.load(randoms_result_path, allow_pickle=True)[()]
        return randoms_results

    
def cat2sky(cat):
    ra2 = cat['RA'].data
    dec2 = cat['DEC'].data
    sky2 = SkyCoord(ra2*u.degree, dec2*u.degree, frame='icrs')
    return sky2

def select_gaia(gaia, gaia_min, gaia_max):   
    mask = (gaia['mask_mag']>gaia_min) & (gaia['mask_mag']<=gaia_max)    
    gaia1 = gaia[mask]
    return gaia1

def get_residuals(gaia, cat):
    ra1, dec1 = gaia['RA'], gaia['DEC']
    ra2, dec2 = cat['RA'], cat['DEC']
    
    d_ra = (ra2-ra1)*3600.    # in arcsec
    d_dec = (dec2-dec1)*3600. # in arcsec
    
    # Convert d_ra to actual arcsecs
    mask = d_ra > 180*3600
    d_ra[mask] = d_ra[mask] - 360.*3600
    mask = d_ra < -180*3600
    d_ra[mask] = d_ra[mask] + 360.*3600
    d_ra = d_ra * np.cos(dec1/180*np.pi)
    
    return (d_ra, d_dec)

def get_refden(d2d, search_radius):
    
    # Paramater for estimating the overdensities
    annulus_min, annulus_max = search_radius/3.*2, search_radius

    # without randoms
    ntot_annulus = np.sum((d2d>annulus_min) & (d2d<annulus_max))
    density_annulus = ntot_annulus/(np.pi*(annulus_max**2 - annulus_min**2)) 
    
    return density_annulus


def get_refden_randoms(randoms_results, bins, gaia_min, gaia_max):
    # with randoms
    key_str = '{:g}_{:g}'.format(gaia_min, gaia_max)
    if not np.allclose(randoms_results[key_str+'_bins'], bins):
        raise ValueError()
    density_rand = randoms_results[key_str+'_density_rand']
    count = randoms_results[key_str+'_count']
    density_rand[count<30] = np.nan
    
    return density_rand
    
def check_maskbits(cat):    
    # Check mask bits
    mask_clean = np.ones(len(cat), dtype=bool)
    for bit in [1, 12, 13]:
            mask_clean &= ((cat['MASKBITS'] & 2**bit) == 0)
    print('{:} ({:.1f}%) objects flagged by maskbits'.format(np.sum(~mask_clean), np.sum(~mask_clean)/len(mask_clean)*100))

    mask_bright = ((cat['MASKBITS'] & 2**1) == 0)
    mask_medium = ((cat['MASKBITS'] & 2**11) == 0)
    print('{:} ({:.1f}%) objects flagged by BRIGHT mask'.format(np.sum(~mask_bright), np.sum(~mask_bright)/len(mask_bright)*100))
    print('{:} ({:.1f}%) objects flagged by MEDIUM mask'.format(np.sum(~mask_medium), np.sum(~mask_medium)/len(mask_medium)*100))


    
def shiftra(ra):
    return ra-360*(ra>290)
    
    
def plot_gaia_specs(gaia):
    
    fg, ax = plt.subplots(ncols=2, figsize=(12, 5))
    
    ax[0].hist(gaia['mask_mag'], 100, log=True);
    print(gaia['mask_mag'].min(), gaia['mask_mag'].max())

    ax[1].hist(gaia['radius'], 100, log=True)
    print(gaia['radius'].min(), gaia['radius'].max())   
    
    ax[0].set_xlabel('MAG')
    ax[1].set_xlabel('Radius')
    fg.show()
    
def plot_cat_on_gaia(cat, gaia, field):
    
    fsize = (16, 8) if field=='south' else (12, 8)
    
    plt.figure(figsize=fsize)
    
    kw = dict(marker='.', alpha=0.5)
    plt.scatter(shiftra(gaia['RA'][::100]), gaia['DEC'][::100], 0.1, **kw)    
    plt.scatter(shiftra(cat['RA'][::40]), cat['DEC'][::40], 0.1, **kw)
    
    plt.xlabel('RA')
    plt.ylabel('DEC')
    
    plt.gca().invert_xaxis()
    plt.show()
    
    
def binned_mean(x, y, vmin=None, vmax=None, nbins=25):
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(x, [0.5, 99.5])
    bins = np.linspace(vmin, vmax, nbins)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bin_mean, bin_edges, binnumber = stats.binned_statistic(x, y, statistic=np.nanmean, bins=bins)
    bin_center = (bin_edges[1:] + bin_edges[:-1])/2
    return bin_center, bin_edges, bin_mean

def get_density(d_ra, d_dec, d2d, plot_radius, nbins=101, min_count=None):

    bins = np.linspace(-plot_radius, plot_radius, nbins)
    bin_spacing = bins[1] - bins[0]
    bincenter = (bins[1:]+bins[:-1])/2
    mesh_ra, mesh_dec = np.meshgrid(bincenter, bincenter)
    mesh_d2d = np.sqrt(mesh_ra**2 + mesh_dec**2)
    mask = (d2d>0.01)
    count = np.histogram2d(d_ra[mask], d_dec[mask], bins=bins)[0]
    if min_count is not None:
        count[count<min_count] = np.nan
    density = count/(bin_spacing**2)
    mask = mesh_d2d >= bins.max()-bin_spacing
    density[mask] = np.nan

    return bins, density, count

def get_relative_density(bins, density, ref_density, nbins):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        density_ratio = density/ref_density
        
    bincenter = (bins[1:]+bins[:-1])/2
    mesh_ra, mesh_dec = np.meshgrid(bincenter, bincenter)
    mesh_d2d = np.sqrt(mesh_ra**2 + mesh_dec**2)

    bin_center, bin_edges, bin_mean = binned_mean(mesh_d2d.flatten(), density_ratio.flatten()-1, 
                                                  vmin=0, vmax=bins.max(), nbins=25)
    return (bin_center, bin_mean), (mesh_d2d, density_ratio)


def relative_density_subplots(bins, density, ref_density, nbins=101, vmin=-2, vmax=2,
                              xlabel1='$\Delta$RA (arcsec)', ylabel1='$\Delta$DEC (arcsec)',
                              xlabel2='distance (arcsec)', ylabel2='fractional overdensity'):

    (bin_center, bin_mean), (mesh_d2d, density_ratio) = get_relative_density(bins, density, ref_density, nbins)
    extent = bins.max()*np.array([-1, 1, -1, 1])
    plot_radius = bins.max()

    fig, ax = plt.subplots(1, 2, figsize=(17, 6.5))
    dens = ax[0].imshow(density_ratio.transpose()-1, origin='lower', aspect='equal', 
               cmap='seismic', extent=extent, vmin=vmin, vmax=vmax)
    fig.colorbar(dens, ax=ax[0], fraction=0.046, pad=0.04)
    ax[0].axis([-plot_radius*1.03, plot_radius*1.03, -plot_radius*1.03, plot_radius*1.03])
    ax[0].set_xlabel(xlabel1)
    ax[0].set_ylabel(ylabel1)
    # plt.show()
    
    ax[1].plot(mesh_d2d.flatten(), density_ratio.flatten()-1, '.', markersize=1.5)
    ax[1].plot(bin_center, bin_mean)
    # plt.axvline(mask_length1/2, lw=1, color='k')
    # plt.axvline(mask_length2/2, lw=1, color='k')
    ax[1].set_xlabel(xlabel2)
    ax[1].set_ylabel(ylabel2)
    ax[1].grid(alpha=0.5)
    ax[1].axis([0, plot_radius, -1.1, 1.6])
    
    return ax, density_ratio    



def relative_density_subplots_fast(bins, rdens, vmin=-2, vmax=2,
                              xlabel1='$\Delta$RA (arcsec)', ylabel1='$\Delta$DEC (arcsec)',
                              xlabel2='distance (arcsec)', ylabel2='fractional overdensity'):
    
    (bin_center, bin_mean), (mesh_d2d, density_ratio) = rdens
    extent = bins.max()*np.array([-1, 1, -1, 1])
    plot_radius = bins.max()

    fig, ax = plt.subplots(1, 2, figsize=(17, 6.5))
    dens = ax[0].imshow(density_ratio.transpose()-1, origin='lower', aspect='equal', 
               cmap='seismic', extent=extent, vmin=vmin, vmax=vmax)
    fig.colorbar(dens, ax=ax[0], fraction=0.046, pad=0.04)
    ax[0].axis([-plot_radius*1.03, plot_radius*1.03, -plot_radius*1.03, plot_radius*1.03])
    ax[0].set_xlabel(xlabel1)
    ax[0].set_ylabel(ylabel1)
    # plt.show()
    
    ax[1].plot(mesh_d2d.flatten(), density_ratio.flatten()-1, '.', markersize=1.5)
    ax[1].plot(bin_center, bin_mean)
    #ax[1].axhline(0.2, lw=1, color='k', ls='-')
    #ax[1].axhline(-0.2, lw=1, color='k', ls='-')    
    # plt.axvline(mask_length2/2, lw=1, color='k')
    ax[1].set_xlabel(xlabel2)
    ax[1].set_ylabel(ylabel2)
    ax[1].grid(alpha=0.5)
    ax[1].axis([0, plot_radius, -1.1, 1.6])
    
    return ax, density_ratio    