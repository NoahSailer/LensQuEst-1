from __future__ import division
from builtins import map
from past.utils import old_div
#from importlib import reload
import sys

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
print("CMB experiment properties")

# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S3 specs
cmb = CMB(beam=1.4, noise=7., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False, fg=True)
# Total power spectrum, for the lens reconstruction
forCtotal = lambda l: cmb.ftotalTT(l) #cmb.flensedTT(l) #+ cmb.fdetectorNoise(l)
# reinterpolate: gain factor 10 in speed
L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
F = np.array(list(map(forCtotal, L)))
cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)


#################################################################################
print("Calculate noises and response")

fNqCmb_fft = baseMap.forecastN0Kappa(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
fNs_fft = baseMap.forecastN0S(cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
fNqBHCmb_fft = baseMap.forecastN0KappaBH(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
response = baseMap.response(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None)


#################################################################################
print("Checking the noises")

tFourier = baseMap.genGRF(cmb.fCtotal, test=False)
t = baseMap.inverseFourier(tFourier)
print("plot the GRF")
baseMap.plot(t, save=False)
print "check the power spectrum"
lCen, Cl, sCl = baseMap.powerSpectrum(tFourier, theory=[cmb.fCtotal], plot=True, save=False, dataLabel=r'$t\times t$', theoryLabel=r'$C^\text{tot}_\ell$')

pathQ = "./output/pQ.txt"
baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=tFourier, test=False, path=pathQ)
pQFourier = baseMap.loadDataFourier(pathQ)

pathS = "./output/ps.txt"
baseMap.computeQuadEstSNorm(cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=tFourier, test=False, path=pathS)
pSFourier = baseMap.loadDataFourier(pathS)

pathQBH = "./output/pQBH.txt"
baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=tFourier, test=False, path=pathQBH)
pQBHFourier = baseMap.loadDataFourier(pathQBH)


print("Auto-power: kappa_rec")
lCen, Cl, sCl = baseMap.powerSpectrum(pQFourier,theory=[fNqCmb_fft], plot=True, save=False, dataLabel=r'$\hat{\kappa}[t]\times\hat{\kappa}[t]$', theoryLabel=r'$N^{\kappa}_\ell$')

print("Auto-power: S_rec")
lCen, Cl, sCl = baseMap.powerSpectrum(pSFourier,theory=[fNs_fft], plot=True, save=False, dataLabel=r'$\hat{S}[t]\times\hat{S}[t]$',theoryLabel=r'$N^{S}_\ell$')

print("Auto-power: kappa^BH_rec")
lCen, Cl, sCl = baseMap.powerSpectrum(pQBHFourier,theory=[fNqBHCmb_fft], plot=True, save=False, dataLabel=r'$\hat{\kappa}^\text{BH}[t]\times\hat{\kappa}^\text{BH}[t]$', theoryLabel=r'$N^{\kappa^\text{BH}}_\ell$')


##################################################################################
print("CMB lensing power spectrum")

u = UnivPlanck15()
halofit = Halofit(u, save=False)
w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)


##################################################################################
print("Generate GRF unlensed CMB map (debeamed)")

cmb0Fourier = baseMap.genGRF(cmb.funlensedTT, test=False)
cmb0 = baseMap.inverseFourier(cmb0Fourier)
#print("plot unlensed CMB map")
#baseMap.plot(cmb0)
#print("check the power spectrum")
#lCen, Cl, sCl = baseMap.powerSpectrum(cmb0Fourier, theory=[cmb.funlensedTT], plot=True, save=False)


##################################################################################
print("Generate GRF kappa map")

kCmbFourier = baseMap.genGRF(p2d_cmblens.fPinterp, test=False)
kCmb = baseMap.inverseFourier(kCmbFourier)
#print("plot kappa map")
#baseMap.plot(kCmb)
#print("check the power spectrum")
#lCen, Cl, sCl = baseMap.powerSpectrum(kCmbFourier, theory=[p2d_cmblens.fPinterp], plot=True, save=False)


##################################################################################
print("Lens the CMB map")

lensedCmb = baseMap.doLensing(cmb0, kappaFourier=kCmbFourier)
lensedCmbFourier = baseMap.fourier(lensedCmb)
print("plot lensed CMB map")
baseMap.plot(lensedCmb, save=False)
print "check the power spectrum"
lCen, Cl, sCl = baseMap.powerSpectrum(lensedCmbFourier, theory=[cmb.flensedTT], plot=True, save=False, dataLabel=r'$T\times T$', theoryLabel=r'$C_\ell$')


##################################################################################
print("Checking cross correlation on lensed CMB map")

pathQ = "./output/pQ.txt"
baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=lensedCmbFourier, test=False, path=pathQ)
pQFourier = baseMap.loadDataFourier(pathQ)

pathS = "./output/ps.txt"
baseMap.computeQuadEstSNorm(cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=lensedCmbFourier, test=False, path=pathS)
pSFourier = baseMap.loadDataFourier(pathS)

pathQBH = "./output/pQBH.txt"
baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=lensedCmbFourier, test=False, path=pathQBH)
pQBHFourier = baseMap.loadDataFourier(pathQBH)


print("Cross-power: kappa_rec")
lCen, Cl, sCl = baseMap.crossPowerSpectrum(pQFourier, kCmbFourier, theory=[p2d_cmblens.fPinterp], plot=True, save=False, dataLabel=r'$\hat{\kappa}[T] \times \kappa$', theoryLabel=r'$C^\kappa_\ell$')


print("Cross-power: S_rec")
lCen, Cl, sCl = baseMap.crossPowerSpectrumFiltered(pSFourier, kCmbFourier, fNs_fft, p2d_cmblens.fPinterp, theory=[response], plot=True, save=False, dataLabel=r'$(\hat{S}[T]/N^S) \times (\kappa/C^\kappa)$', theoryLabel=r'$\mathcal{R}_\ell$')

print("Cross-power: kappa^BH_rec")
lCen, Cl, sCl = baseMap.crossPowerSpectrum(pQBHFourier, kCmbFourier, theory=[p2d_cmblens.fPinterp], plot=True, save=False, dataLabel=r'$\hat{\kappa}^\text{BH}[T] \times \kappa$', theoryLabel=r'$C^\kappa_\ell$')


##################################################################################
print("Generate Poisson point source map")

nbar = 5.e3 
scaledPoisson = baseMap.genPoissonWhiteNoise(nbar=nbar, fradioPoisson=cmb.fradioPoisson, norm=False, test=False)
scaledPoissonFourier = baseMap.fourier(scaledPoisson)

poissonTheory = lambda l: cmb.fradioPoisson(l)

print "plot scaled poisson map"
baseMap.plot(scaledPoisson, save=False)
print "check the power spectrum"
lCen, Cl, sCl = baseMap.powerSpectrum(scaledPoissonFourier, theory=[poissonTheory], plot=True, save=False, dataLabel=r'$\tilde{S}\times \tilde{S}$', theoryLabel=r"$C^{\Tilde{S}}_\ell$")


##################################################################################
print("Checking cross correlation on tilde{S} map")

pathQ = "./output/pQ.txt"
baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=scaledPoissonFourier, test=False, path=pathQ)
pQFourier = baseMap.loadDataFourier(pathQ)

pathS = "./output/ps.txt"
baseMap.computeQuadEstSNorm(cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=scaledPoissonFourier, test=False, path=pathS)
pSFourier = baseMap.loadDataFourier(pathS)

pathQBH = "./output/pQBH.txt"
baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=scaledPoissonFourier, test=False, path=pathQBH)
pQBHFourier = baseMap.loadDataFourier(pathQBH)


# number of objects per pixel
Ngal_perpix = nbar * baseMap.dX * baseMap.dY
# number of objects per map
Ngal = Ngal_perpix * baseMap.nX * baseMap.nY
# area of single pixel
singleA = (baseMap.fSky * 4.*np.pi) / (baseMap.nX*baseMap.nY)
# rms flux of sources
sRMS = np.sqrt(np.sum(scaledPoisson**2.)/Ngal) * singleA

sTrueFourier = scaledPoissonFourier * sRMS
sTrueTheory = lambda l: poissonTheory(l) * sRMS**2.

print("Cross-power: kappa_rec")
lCen, Cl, sCl = baseMap.crossPowerSpectrumFiltered(pQFourier, sTrueFourier, fNqCmb_fft, sTrueTheory, theory=[response], plot=True, save=False, dataLabel=r'$(\hat{\kappa}[\tilde{S}]/N^\kappa) \times (S/C^S)$', theoryLabel=r'$\mathcal{R}_\ell$')

print("Cross-power: S_rec")
lCen, Cl, sCl = baseMap.crossPowerSpectrum(pSFourier, sTrueFourier, theory=[sTrueTheory], plot=True, save=False, dataLabel=r'$\hat{S}[\tilde{S}] \times S$', theoryLabel=r'$C^S_\ell$')

print("Cross-power: kappa^BH_rec")
lCen, Cl, sCl = baseMap.crossPowerSpectrumFiltered(pQBHFourier, sTrueFourier, fNqBHCmb_fft, sTrueTheory, theory=[response], plot=True, save=False, dataLabel=r'$(\hat{\kappa}^\text{BH}[\tilde{S}]/N^{\kappa^\text{BH}}) \times (S/C^S)$', theoryLabel=r'$\mathcal{R}_\ell$')


##################################################################################
print("Plotting the deteminant")

determinant = lambda l: np.abs(1. - fNqCmb_fft(l) * fNqBHCmb_fft(l) * response(l)**2.)

plt.loglog(L,1./determinant(L),label=r'$(1-N^\kappa_\ell N^S_\ell \mathcal{R}^2_\ell)^{-1}$',c='k',lw=2)
plt.xlim(min(L), max(L))
plt.xlabel(r'$\ell$')
plt.legend(loc=0)
plt.show()

import sys
sys.exit()

##################################################################################
print("Response calculation check: s_rec x k_true on the lensed CMB map")

print("Reconstructing S: standard QE")
pathS = "./output/ps.txt"
baseMap.computeQuadEstSNorm(cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=lensedCmbFourier, test=False, path=pathS)
pSFourier = baseMap.loadDataFourier(pathS)

Ns = baseMap.forecastN0S(cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)

print("Calculating response")
response = baseMap.response(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None)


print("Cross-power: s_rec x kappa_true")
lCen, Cl, sCl = baseMap.crossPowerSpectrumFiltered(pSFourier, kCmbFourier, Ns, p2d_cmblens.fPinterp, theory=[response], plot=True, save=False, ylabel = r"$\mathcal{R}=C^{\hat{S}[T] \times\kappa} / N^S C^\kappa $")


##################################################################################
##################################################################################


import sys
sys.exit()


##################################################################################
print("Response calculation check: k_rec x s_true on point source map")

print("Reconstructing kappa: standard QE")
pathQ = "./output/pQ.txt"
baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=scaledPoissonFourier, test=False, path=pathQ)
pQFourier = baseMap.loadDataFourier(pathQ)

Nq = baseMap.forecastN0Kappa(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)

print("Calculating response")
response = baseMap.response(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None)

print("Cross-power: kappa_rec x s_true")
lCen, Cl, sCl = baseMap.crossPowerSpectrumFiltered(pQFourier, scaledPoissonFourier * sbar, Nq, tmptheory, theory=[response], plot=True, save=False, ylabel = r"$\mathcal{R}=C^{\hat{\kappa}[\Tilde{S}]\times S}/N^\kappa C^S $")


##################################################################################
print("Sanity check: kBH x s_true on point source map")

print("Reconstructing kappa: bias hardened estimator")
pathQBH = "./output/pQBH.txt"
baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=scaledPoissonFourier, test=False, path=pathQBH)
pQBHFourier = baseMap.loadDataFourier(pathQBH)

null = lambda l: l*0.

print("Cross-power: kappa_bh x s_true vs kappa_rec x s_true")
lCen, Cl, sCl = baseMap.crossPowerSpectrumComp(pQBHFourier, pQFourier, scaledPoissonFourier * sbar, theory=[null], plot=True, save=False, label1 = r"$C^{\hat{\kappa}^\text{BH}[\Tilde{S}]\times S} $",label2 = r"$C^{\hat{\kappa}[\Tilde{S}]\times S} $")




sys.exit()
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
