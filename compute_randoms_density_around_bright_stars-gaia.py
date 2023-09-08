# Pre-compute the density of randoms around GAIA stars
# srun -N 1 -C haswell -c 64 -t 04:00:00 -L cfs -q interactive python compute_randoms_density_around_bright_stars-gaia.py south

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
# import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, vstack, hstack
import fitsio

from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy import stats


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


time_start = time.time()

randoms_dir = '/global/cfs/cdirs/desi/target/catalogs/dr9/0.49.0/randoms/resolve'
n_randoms_catalogs = 4

gaia_path = '/global/cfs/cdirs/desi/users/rongpu/desi_mask/gaia_reference_dr9.fits'
gaia_suppl_path = '/global/cfs/cdirs/desi/users/rongpu/desi_mask/gaia_reference_suppl_dr9.fits'
gaia_columns = ['RA', 'DEC', 'mask_mag']

randoms_columns = ['RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'PHOTSYS', 'MASKBITS']

randoms_paths = sorted(glob.glob('/global/cfs/cdirs/desi/target/catalogs/dr9/0.49.0/randoms/resolve/randoms-[0-9]*.fits'))
randoms_paths = randoms_paths[:n_randoms_catalogs]

randoms_density = fitsio.read_header(randoms_paths[0], ext=1)['DENSITY']  # randoms per sq. deg.

# field = 'north'
field = str(sys.argv[1])
field = field.lower()

maskbits = [1, 12, 13]
min_nobs = 1

if field=='south':
    photsys = 'S'
else:
    photsys = 'N'

gaia = Table(fitsio.read(gaia_path, columns=gaia_columns))
gaia_suppl = Table(fitsio.read(gaia_suppl_path, columns=['ra', 'dec', 'mask_mag']))
gaia_suppl.rename_columns(gaia_suppl.colnames, gaia_columns)
gaia = vstack([gaia, gaia_suppl], join_type='exact')
print(len(gaia))

gaia['radius'] = 1630 * 1.396**(-gaia['mask_mag'])  # the DR9 radius-mag relation

if field=='south':
    mask = (gaia['DEC']<36)
else:
    mask = (gaia['DEC']>30)
    mask &= (gaia['RA']<310) & (gaia['RA']>75)
gaia = gaia[mask]
print(len(gaia))

##################################################################################################################################

# randoms_stack = []
# for randoms_path in randoms_paths:
#     print(randoms_path)
#     randoms = Table(fitsio.read(randoms_path, columns=randoms_columns))
#     randoms_stack.append(randoms)
# randoms = vstack(randoms_stack)
# del randoms_stack
# randoms.write('/global/cfs/cdirs/desi/users/rongpu/tmp/randoms_for_elgs.fits')

randoms = Table(fitsio.read('/global/cfs/cdirs/desi/users/rongpu/tmp/randoms_for_elgs.fits'))

mask = (randoms['PHOTSYS']==photsys)
randoms = randoms[mask]

mask = (randoms['NOBS_G']>=min_nobs) & (randoms['NOBS_R']>=min_nobs) & (randoms['NOBS_Z']>=min_nobs)
randoms = randoms[mask]

# Apply masks
randoms_clean = np.ones(len(randoms), dtype=bool)
for bit in maskbits:
    randoms_clean &= (randoms['MASKBITS'] & 2**bit)==0
randoms = randoms[randoms_clean]

# Remove pixels near the LMC
ramin, ramax, decmin, decmax = 58, 110, -90, -56
mask_remove = (randoms['RA']>ramin) & (randoms['RA']<ramax) & (randoms['DEC']>decmin) & (randoms['DEC']<decmax)
randoms = randoms[~mask_remove]
print(len(randoms))

print('randoms:', len(randoms))

##################################################################################################################################

ra2_rand = np.array(randoms['RA'])
dec2_rand = np.array(randoms['DEC'])
sky2_rand = SkyCoord(ra2_rand*u.degree, dec2_rand*u.degree, frame='icrs')

gaia_min_list = [-np.inf, 4, 5, 6, 7, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5]
gaia_max_list = [4, 5, 6, 7, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0]

data = {}

for index in range(len(gaia_min_list)):

    gaia_min, gaia_max = gaia_min_list[index], gaia_max_list[index]

    nbins = 101

    mask = (gaia['mask_mag']>gaia_min) & (gaia['mask_mag']<=gaia_max)
    ra1 = gaia['RA'][mask]
    dec1 = gaia['DEC'][mask]
    gaia1 = gaia[mask]

    search_radius = np.minimum(gaia1['radius'].max() * 4.1, 3600)

    if gaia_min==-np.inf:
        title = 'GAIA_G <= {:.1f}'.format(gaia_max, np.sum(mask))
    else:
        title = '{:.1f} < GAIA_G <= {:.1f}'.format(gaia_min, gaia_max, np.sum(mask))

    print(title, '{} stars'.format(np.sum(mask)))

    sky1 = SkyCoord(ra1*u.degree, dec1*u.degree, frame='icrs')

    # randoms
    idx1_rand, idx2_rand, d2d_rand, _ = sky2_rand.search_around_sky(sky1, seplimit=search_radius*u.arcsec)
    if len(idx1_rand)==0:
        continue
    print('{} nearby objects around {} stars'.format(len(np.unique(idx2_rand)), len(np.unique(idx1_rand))))
    d2d_rand = np.array(d2d_rand.to(u.arcsec))  # convert distances to numpy array in arcsec
    d_ra_rand = (ra2_rand[idx2_rand]-ra1[idx1_rand])*3600.     # in arcsec
    d_dec_rand = (dec2_rand[idx2_rand]-dec1[idx1_rand])*3600.  # in arcsec
    # Convert d_ra_rand to actual arcsecs
    mask = d_ra_rand > 180*3600
    d_ra_rand[mask] = d_ra_rand[mask] - 360.*3600
    mask = d_ra_rand < -180*3600
    d_ra_rand[mask] = d_ra_rand[mask] + 360.*3600
    d_ra_rand = d_ra_rand * np.cos(dec1[idx1_rand]/180*np.pi)

    bins, density_rand, count_rand = get_density(d_ra_rand, d_dec_rand, d2d_rand, search_radius, nbins=nbins, min_count=None)
    key_str = '{:g}_{:g}'.format(gaia_min, gaia_max)
    data[key_str+'_bins'] = bins
    data[key_str+'_density_rand'] = density_rand
    data[key_str+'_count'] = count_rand

data['n_randoms'] = len(randoms)

save_path = '/global/u2/r/rongpu/notebooks/desi_mask/data/elgmask_dev/density_rand_gaia_{}_minobs_{}_maskbits_{}.npy'.format(field, min_nobs, ''.join([str(tmp) for tmp in maskbits]))
np.save(save_path, data)

print('Done!', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))
