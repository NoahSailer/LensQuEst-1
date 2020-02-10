from __future__ import division
from builtins import map
from past.utils import old_div
import importlib
import sys

import universe
importlib.reload(universe)
from universe import *

import halo_fit
importlib.reload(halo_fit)
from halo_fit import *

import weight
importlib.reload(weight)
from weight import *

import pn_2d
importlib.reload(pn_2d)
from pn_2d import *

import cmb
importlib.reload(cmb)
from cmb import *

import flat_map
importlib.reload(flat_map)
from flat_map import *

##################################################################################
##################################################################################
#print("Map properties")

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
##################################################################################
#print("Generate Poisson point source map")

sbar = 1.5e-4
singleA = old_div((baseMap.fSky * 4.*np.pi), (baseMap.nX*baseMap.nY))
poisson = baseMap.genPoissonWhiteNoise(nbar=5.e4, norm=False, test=False)
scaledPoisson = old_div(poisson * sbar, singleA)
scaledPoissonFourier = baseMap.fourier(scaledPoisson)

const = old_div(sbar**2., (4. * np.pi * baseMap.fSky))
poissonTheory = lambda l: const * np.sum(poisson) + l*0

#print "plot scaled poisson map"
#baseMap.plot(scaledPoisson, save=False)
#print "check the power spectrum"
#lCen, Cl, sCl = baseMap.powerSpectrum(scaledPoissonFourier, theory=[poissonTheory], plot=False, save=False)


##################################################################################
#print("Genenerate GRF map with point source power spectrum, calculate noise")

#Theoretical noise for point source min var QE
normalizationS = baseMap.forecastN0S(poissonTheory, lMin=lMin, lMax=lMax, test=False)
#
pathSN = "./output/psn.txt"
poissonNoiseMap = baseMap.genGRF(poissonTheory, test=False)
baseMap.computeQuadEstSNorm(poissonTheory, lMin=lMin, lMax=lMax, dataFourier=poissonNoiseMap, test=False, path=pathSN)
pSNFourier = baseMap.loadDataFourier(pathSN)
lCen, Cl, sCl = baseMap.powerSpectrum(pSNFourier,theory=[normalizationS], plot=False, save=False)


##################################################################################
#print("Reconstructing S before adding CMB: standard QE")

pathS0 = "./output/ps0.txt"
baseMap.computeQuadEstSNorm(poissonTheory, lMin=lMin, lMax=lMax, dataFourier=scaledPoissonFourier, test=False, path=pathS0)
pS0Fourier = baseMap.loadDataFourier(pathS0)
#Estimator is really for \Tilde{S}, need to scale by sbar to compare to poissonTheory
pS0Fourier /= sbar

##print "Auto-power: S_rec"
#lCen, Cl, sCl = baseMap.powerSpectrum(pS0Fourier,theory=[poissonTheory], plot=False, save=False)

##print "Cross-power: S_rec x S_true"
#lCen, Cl, sCl = baseMap.crossPowerSpectrum(pS0Fourier, scaledPoissonFourier, theory=[poissonTheory], plot=False, save=False)

##print "Plot S_rec"
#pS0 = baseMap.inverseFourier(pS0Fourier)
#baseMap.plot(np.abs(pS0))

##print "Plot difference"
#baseMap.plot(np.abs(pS0 - scaledPoisson))


##################################################################################
##################################################################################
#print("CMB experiment properties")

# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S3 specs
cmb = StageIVCMB(beam=1., noise=1., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False)
# Total power spectrum, for the lens reconstruction
forCtotal = lambda l: cmb.flensedTT(l) #+ cmb.fdetectorNoise(l)
# reinterpolate: gain factor 10 in speed
L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
F = np.array(list(map(forCtotal, L)))
cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)


##################################################################################
#print("CMB lensing power spectrum")

u = UnivPlanck15()
halofit = Halofit(u, save=False)
w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)


##################################################################################
#print("Generate GRF unlensed CMB map (debeamed)")

cmb0Fourier = baseMap.genGRF(cmb.funlensedTT, test=False)
cmb0 = baseMap.inverseFourier(cmb0Fourier)
##print "plot unlensed CMB map"
#baseMap.plot(cmb0)
##print "check the power spectrum"
lCen, Cl, sCl = baseMap.powerSpectrum(cmb0Fourier, theory=[cmb.funlensedTT], plot=False, save=False)


#################################################################################
#################################################################################
#print("Adding sources to unlensed CMB map")

totalUnlensedFourier = cmb0Fourier + scaledPoissonFourier
totalUnlensed = baseMap.inverseFourier(totalUnlensedFourier)
##print "plot unlensed CMB + point sources"
#baseMap.plot(totalUnlensed)
##print "check the power spectrum"
unlensedWithSources = lambda l: cmb.funlensedTT(l) + poissonTheory(l)
lCen, Cl, sCl = baseMap.powerSpectrum(totalUnlensedFourier, theory=[unlensedWithSources], plot=False, save=False)


##################################################################################
#print("Reconstructing S (unlensed with sources): standard QE")

fCtotal = lambda l: poissonTheory(l) + cmb.funlensedTT(l)

pathS1 = "./output/ps1.txt"
baseMap.computeQuadEstSNorm(fCtotal, lMin=lMin, lMax=lMax, dataFourier=totalUnlensedFourier, test=False, path=pathS1)
pS1Fourier = baseMap.loadDataFourier(pathS1)
#Estimator is really for \Tilde{S}, need to scale by sbar to compare to poissonTheory
pS1Fourier /= sbar

##print "Auto-power: S_rec"
lCen, Cl, sCl = baseMap.powerSpectrum(pS1Fourier,theory=[poissonTheory], plot=False, save=False)

##print "Cross-power: S_rec x S_true"
lCen, Cl, sCl = baseMap.crossPowerSpectrum(pS1Fourier, scaledPoissonFourier, theory=[poissonTheory], plot=False, save=False)

##print "Plot S_rec"
#pS1 = baseMap.inverseFourier(pS1Fourier)
#baseMap.plot(np.abs(pS1))

##print "Plot difference"
#baseMap.plot(np.abs(pS1 - scaledPoisson))


#################################################################################
#print "Sanity check: cross-correlation (unlensed with sources)"

pathQ9Cmb = "./output/q9Cmb.txt"
baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, fCtotal, lMin=lMin, lMax=lMax, dataFourier=totalUnlensedFourier, test=False, path=pathQ9Cmb)
q9CmbFourier = baseMap.loadDataFourier(pathQ9Cmb)

response = baseMap.response(cmb.funlensedTT, fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None)
fNqCmb_fft = baseMap.forecastN0Kappa(cmb.funlensedTT, fCtotal, lMin=lMin, lMax=lMax, test=False)

cross_theory = lambda l: response(l) * fNqCmb_fft(l) * poissonTheory(l) * sbar**2.

lCen, Cl, sCl = baseMap.crossPowerSpectrum(q9CmbFourier, pS1Fourier*sbar, theory=[cross_theory], plot=False, save=False)


##################################################################################
##################################################################################
#print "Generate GRF kappa map"

kCmbFourier = baseMap.genGRF(p2d_cmblens.fPinterp, test=False)
kCmb = baseMap.inverseFourier(kCmbFourier)
#print "plot kappa map"
#baseMap.plot(kCmb)
#print "check the power spectrum"
lCen, Cl, sCl = baseMap.powerSpectrum(kCmbFourier, theory=[p2d_cmblens.fPinterp], plot=False, save=False)


##################################################################################
#print "Lens the CMB map"

lensedCmb = baseMap.doLensing(cmb0, kappaFourier=kCmbFourier)
lensedCmbFourier = baseMap.fourier(lensedCmb)
#print "plot lensed CMB map"
#baseMap.plot(lensedCmb, save=False)
#print "check the power spectrum"
lCen, Cl, sCl = baseMap.powerSpectrum(lensedCmbFourier, theory=[cmb.funlensedTT, cmb.flensedTT], plot=False, save=False)


##################################################################################
#print "Reconstructing kappa (lensed without sources): standard QE"

pathQCmb = "./output/qCmb.txt"
baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=lensedCmbFourier, test=False, path=pathQCmb)
qCmbFourier = baseMap.loadDataFourier(pathQCmb)

#print "Compute the statistical uncertainty on the reconstructed lensing convergence"
fNqCmb_fft = baseMap.forecastN0Kappa(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)

#print "Auto-power: kappa_rec"
lCen, Cl, sCl = baseMap.powerSpectrum(qCmbFourier,theory=[p2d_cmblens.fPinterp, fNqCmb_fft], plot=False, save=False)

#print "Cross-power: kappa_rec x kappa_true"
lCen, Cl, sCl = baseMap.crossPowerSpectrum(qCmbFourier, kCmbFourier, theory=[p2d_cmblens.fPinterp, fNqCmb_fft], plot=False, save=False)


#################################################################################
#print "Sanity check: cross-correlation (lensed without sources)"

pathS9 = "./output/ps9.txt"
baseMap.computeQuadEstSNorm(cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=lensedCmbFourier, test=False, path=pathS9)
pS9Fourier = baseMap.loadDataFourier(pathS9)
#Estimator is really for \Tilde{S}, need to scale by sbar to compare to poissonTheory
pS9Fourier /= sbar

response = baseMap.response(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None)

cross_theory = lambda l: response(l) * fNqCmb_fft(l) * poissonTheory(l) * sbar**2.

lCen, Cl, sCl = baseMap.crossPowerSpectrum(qCmbFourier, pS9Fourier, theory=[cross_theory], plot=False, save=False)


#################################################################################
#################################################################################
#print "Add sources to lensed CMB map"

totalLensedFourier = lensedCmbFourier + scaledPoissonFourier
totalLensed = baseMap.inverseFourier(totalLensedFourier)
#print "plot lensed CMB + point sources"
#baseMap.plot(totalLensed)
#print "check the power spectrum"
lensedWithSources = lambda l: cmb.flensedTT(l) + poissonTheory(l)
lCen, Cl, sCl = baseMap.powerSpectrum(totalLensedFourier, theory=[lensedWithSources], plot=False, save=False)


##################################################################################
#print "Reconstructing S (lensed with sources): standard QE"

fCtotal = lambda l: poissonTheory(l) + cmb.flensedTT(l)

pathS2 = "./output/ps2.txt"
baseMap.computeQuadEstSNorm(fCtotal, lMin=lMin, lMax=lMax, dataFourier=totalLensedFourier, test=False, path=pathS2)
pS2Fourier = baseMap.loadDataFourier(pathS2)

#print "Auto-power: S_rec"
lCen, Cl, sCl = baseMap.powerSpectrum(old_div(pS2Fourier,sbar),theory=[poissonTheory], plot=False, save=False)

#print "Cross-power: S_rec x S_true"
lCen, Cl, sCl = baseMap.crossPowerSpectrum(old_div(pS2Fourier,sbar), scaledPoissonFourier, theory=[poissonTheory], plot=False, save=False)

##print "Plot S_rec"
#pS2 = baseMap.inverseFourier(pS2Fourier)
#baseMap.plot(np.abs(pS2))

##print "Plot difference"
#baseMap.plot(np.abs(pS2 - scaledPoisson))


##################################################################################
#print "Reconstructing kappa (lensed with sources): standard QE"

pathQ1Cmb = "./output/q1Cmb.txt"
baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, fCtotal, lMin=lMin, lMax=lMax, dataFourier=totalLensedFourier, test=False, path=pathQ1Cmb)
q1CmbFourier = baseMap.loadDataFourier(pathQ1Cmb)


#print "Compute the statistical uncertainty on the reconstructed lensing convergence"
fNqCmb_fft = baseMap.forecastN0Kappa(cmb.funlensedTT, fCtotal, lMin=lMin, lMax=lMax, test=False)

#print "Auto-power: kappa_rec"
lCen, Cl, sCl = baseMap.powerSpectrum(q1CmbFourier,theory=[p2d_cmblens.fPinterp, fNqCmb_fft], plot=False, save=False)

#print "Cross-power: kappa_rec x kappa_true"
lCen, Cl, sCl = baseMap.crossPowerSpectrum(q1CmbFourier, kCmbFourier, theory=[p2d_cmblens.fPinterp, fNqCmb_fft], plot=False, save=False)


#################################################################################
#print "Calculate noise for BH QE"

fNkPSHCmb_fft = baseMap.forecastN0KappaPSH(cmb.funlensedTT, fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None)


#################################################################################
#print "Bias hardened kappa"

pathkBH = "./output/kBH.txt"
baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.funlensedTT, fCtotal, lMin=lMin, lMax=lMax, dataFourier=totalLensedFourier, test=False, path=pathkBH)
kBHFourier = baseMap.loadDataFourier(pathkBH) 

#print "Auto-power: kappa_rec"
lCen, Cl, sCl = baseMap.powerSpectrum(kBHFourier,theory=[p2d_cmblens.fPinterp, fNkPSHCmb_fft], plot=False, save=False)
#print "Cross-power: kappa_rec x kappa_true"
lCen, Cl, sCl = baseMap.crossPowerSpectrum(kBHFourier, kCmbFourier, theory=[p2d_cmblens.fPinterp, fNkPSHCmb_fft], plot=False, save=False)
