from __future__ import print_function
from __future__ import division
from builtins import map
from past.utils import old_div
import universe
reload(universe)
from universe import *

import halo_fit
reload(halo_fit)
from halo_fit import *

import weight
reload(weight)
from weight import *

import pn_2d
reload(pn_2d)
from pn_2d import *

import cmb
reload(cmb)
from cmb import *

import flat_map
reload(flat_map)
from flat_map import *


##################################################################################
print("Map properties")

# number of pixels for the flat map
nX = 400 #1200
nY = 400 #1200

# map dimensions in degrees
sizeX = 10.
sizeY = 10.

# basic map object
baseMap = FlatMap(nX=nX, nY=nY, sizeX=sizeX*np.pi/180., sizeY=sizeY*np.pi/180.)

# multipoles to include in the lensing reconstruction
lMin = 30.; lMax = 3.5e3

# ell bins for power spectra
nBins = 21  # number of bins
lRange = (1., 2.*lMax)  # range for power spectra


##################################################################################
print("CMB experiment properties")

# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S3 specs
cmb = StageIVCMB(beam=1., noise=1., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False)



##################################################################################
print("CMB lensing power spectrum")

u = UnivPlanck15()
halofit = Halofit(u, save=False)
w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)


##################################################################################
print("Generate GRF kappa map")

kCmbFourier = baseMap.genGRF(p2d_cmblens.fPinterp, test=False)
kCmb = baseMap.inverseFourier(kCmbFourier)


##################################################################################
def makeMap():
   ''' Makes unlensed CMB map, lenses, then adds sources. Upates the
   total CMB power spectrum. Returns the map (in Fourier space).
   '''
   cmb0Fourier = baseMap.genGRF(cmb.funlensedTT, test=False)
   cmb0 = baseMap.inverseFourier(cmb0Fourier)

   lensedCmb = baseMap.doLensing(cmb0, kappaFourier=kCmbFourier)
   lensedCmbFourier = baseMap.fourier(lensedCmb)

   singleA = old_div((baseMap.fSky * 4.*np.pi), (baseMap.nX*baseMap.nY))
   poisson = baseMap.genPoissonWhiteNoise(nbar=5.e4, norm=False, test=False)
   scaledPoisson = old_div(poisson * sbar, singleA)
   scaledPoissonFourier = baseMap.fourier(scaledPoisson)

   totalLensedFourier = lensedCmbFourier + scaledPoissonFourier
   totalLensed = baseMap.inverseFourier(totalLensedFourier)

   const = old_div(sbar**2., (4. * np.pi * baseMap.fSky))
   poissonTheory = lambda l: const * np.sum(poisson) + l*0

   # Total power spectrum, for the lens reconstruction
   forCtotal = lambda l: cmb.flensedTT(l) + poissonTheory(l)
   # reinterpolate: gain factor 10 in speed
   L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
   F = np.array(list(map(forCtotal, L)))
   cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

   return totalLensedFourier



def reconstructKappa(totalLensedFourier, pathkBH="./output/kBH.txt"):
   ''' Takes the lensed CMB with point sources, recontructs kappa
   using the point source hardened estimator.
   '''
   baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=totalLensedFourier, test=False, path=pathkBH)
   kBHFourier = baseMap.loadDataFourier(pathkBH) 

   lCen, Cl, sCl = baseMap.crossPowerSpectrum(kBHFourier, kCmbFourier, theory=[p2d_cmblens.fPinterp, fNkPSHCmb_fft], plot=True, save=False)   

   return lCen, Cl



##################################################################################
##################################################################################
print("Doing things")

totalLensedFourier = makeMap()
lCen, Cl = reconstructKappa(totalLensedFourier=totalLensedFourier)
print(lCen)
print(Cl)

