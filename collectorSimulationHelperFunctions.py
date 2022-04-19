import numpy as np

from elementPT import HalbachLensSim,Drift,LensIdeal
from ParticleClass import Swarm,Particle
from ParticleTracerLatticeClass import ParticleTracerLattice
from helperTools import *
from scipy.special import voigt_profile
from scipy.optimize import curve_fit


@np.vectorize
def voigt(r,a,b,sigma,gamma):
    #be very cautious about FWHM vs HWHM here for gamma. gamma for scipy is HWHM per the docs
    assert r>=0
    gamma=gamma/2.0 #convert to HWHM per scipy docs
    v0=voigt_profile(0,sigma,gamma)
    v=voigt_profile(r,sigma,gamma)/v0
    v=a*v+b
    return v

def make_Density_And_Pos_Arr(yArr: np.ndarray,zArr: np.ndarray, vxArr: np.ndarray,numBins: int) \
        -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    assert len(yArr)==len(zArr)
    image,binx,biny=np.histogram2d(yArr,zArr,bins=numBins,weights=1/np.abs(vxArr))
    binx=np.linspace(yArr.min(),yArr.max(),numBins+1)
    binx=binx[:-1]+(binx[1]-binx[0])/2.0
    biny=biny[:-1]+(biny[1]-biny[0])/2.0
    binArea=(binx[1]-binx[0])*(biny[1]-biny[0])
    yHistArr,zHistArr=np.meshgrid(binx,biny)
    yHistArr=np.ravel(yHistArr)
    zHistArr=np.ravel(zHistArr)
    countsArr=np.ravel(image)
    rArr=np.sqrt(yHistArr**2+zHistArr**2)
    rArr=rArr[countsArr!=0]
    countsArr=countsArr[countsArr!=0]
    sortIndices=np.argsort(rArr)
    rArr=rArr[sortIndices]
    countsArr=countsArr[sortIndices]
    densityArr=countsArr/binArea
    densityErr=np.sqrt(countsArr)/binArea
    assert len(rArr)==len(densityArr)
    return rArr,densityArr,densityErr

def make_Radial_Signal_Arr(yArr: np.ndarray,zArr: np.ndarray,vxArr: np.ndarray,numBins: int,numSamples: int=3) \
        -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    assert np.all(np.abs(vxArr)>=1.0)
    rArr,signalArr,signalErrArr=make_Density_And_Pos_Arr(yArr,zArr,vxArr,numBins)
    for i in range(1,numSamples):
        rArrTemp,signalArrTemp,signalErrArrTemp=make_Density_And_Pos_Arr(yArr,zArr,vxArr,numBins+i)
        rArr=np.append(rArr,rArrTemp)
        signalArr=np.append(signalArr,signalArrTemp)
        signalErrArr=np.append(signalErrArr,signalErrArrTemp)
    return rArr,signalArr,signalErrArr

class Interpolater:
    def __init__(self,swarm,PTL):
        self.swarm=swarm
        self.PTL=PTL
        self.endDriftLength=abs(self.PTL.elList[-1].r2[0]-self.PTL.elList[-1].r1[0])
        self.lens=self._get_Lens()
        self.xMin,self.xMax=self._get_x_Min_Max()
    def _get_Lens(self):
        lens=None
        numDrift=0
        for el in self.PTL.elList:
            if type(el)==HalbachLensSim or type(el)==LensIdeal:
                assert lens is None #should only be one lens
                lens=el
            if type(el)==Drift:
                numDrift+=1
        assert lens is not None
        assert numDrift+1==len(self.PTL.elList)
        return lens
    def _get_x_Min_Max(self):
        return -self.PTL.elList[-1].r1[0],-self.PTL.elList[-1].r2[0]
    def _get_Sweep_Range_Trans_Vel_Max(self,voltageRange: float)-> float:
        freqRange=voltageRange*565e6
        vTMax=freqRange*671e-9
        vTMax/=2 #need to half because validity is assesed with radial velocity
        return vTMax
    def __call__(self,xOrbit,maxRadius=np.inf,vTMax=np.inf,laserScanRangeVolts=np.inf,returnP=False,useAssert=True,useInitial=False):
        #xOrbit: Distance in orbit frame, POSITIVE to ease with analyze. Know that the tracing is done with x being negative
        #returns in units of mm
        #vTMax: maximum transverse velocity for interpolation
        #useAssert: I can use this interplater elsewhere if I turn this off
        if not self.xMin<xOrbit<self.xMax or (vTMax!=np.inf and laserScanRangeVolts!=np.inf):
            raise ValueError
        yList=[]
        zList=[]
        pList=[]
        vTMax=self._get_Sweep_Range_Trans_Vel_Max(laserScanRangeVolts)

        for particle in self.swarm.particles:
            if useInitial==True:
                p,q=particle.pi,particle.qi
            else:
                p,q=particle.pf,particle.qf
            vT=np.sqrt(p[1]**2+p[2]**2)
            if (q[0]<-xOrbit and vT<vTMax) or useInitial==True:
                stepFrac=(abs(q[0])-xOrbit)/self.endDriftLength
                ySlope=p[1]/p[0]
                y=q[1]+stepFrac*self.endDriftLength*ySlope
                zSlope=p[2]/p[0]
                z=q[2]+stepFrac*self.endDriftLength*zSlope
                yList.append(y)
                zList.append(z)
                pList.append(p)
        yArr=np.asarray(yList)*1e3
        zArr=np.asarray(zList)*1e3
        rArr=np.sqrt(yArr**2+zArr**2)
        yArr=yArr[rArr<maxRadius]
        zArr=zArr[rArr<maxRadius]
        pArr=np.asarray(pList)[rArr<maxRadius]
        returnArgs=[yArr,zArr]
        assert len(pArr)==len(yArr) and len(pArr)==len(yArr)
        if returnP==True:
            returnArgs.append(pArr)
        return returnArgs


def get_FWHM(x: float,interpFunction: Callable,Plot: bool=False,rMax: float=10.0,w: float=.3,
             laserScanRangeVolts: float=np.inf)-> float:
    """

    :param x: location to get FWHM, cm
    :param interpFunction: Function for interpolating particle positions
    :param Plot: wether to plot the focus
    :param rMax: Maximum radius to consider, mm
    :param w: Interrogation size, mm. This will be edge size of bins used to construct the histogram for signal
    :param laserScanRangeVolts: Full voltage scan range. Used to limit transverse velocities contributing as per our
        actual method
    :return: the FWHM of the voigt
    """
    numBins=int(2*rMax/w)
    yArr,zArr,pArr=interpFunction(x,maxRadius=rMax,laserScanRangeVolts=laserScanRangeVolts,returnP=True)
    vxArr=pArr[:,0]
    rArr,signalArr,signalErrArr=make_Radial_Signal_Arr(yArr,zArr,vxArr,numBins,numSamples=3)
    sigmaArr=signalErrArr/signalArr

    guess=[signalArr.max(),signalArr.min(),1.0,1.0]
    bounds=[0.0,np.inf]
    params=curve_fit(voigt,rArr,signalArr,p0=guess,bounds=bounds,sigma=sigmaArr)[0]
    if Plot==True:
        plt.scatter(rArr,signalArr)
        rPlot=np.linspace(0.0,rArr.max(),1_000)
        plt.plot(rPlot,voigt(rPlot,*params),c='r')
        plt.xlabel('radial position, mm')
        plt.ylabel('signal, au')
        plt.show()
    sigma=params[2]
    gamma=params[3]
    fL=gamma #already FWHM
    fG=2*sigma*np.sqrt(2*np.log(2)) #np.log is natural logarithm
    FWHM=.5346*fL+np.sqrt(.2166*fL**2 + fG**2)
    return FWHM

def make_Fake_Flat_Data():
    numGridEdge=500
    rMax0=1.0
    xGridArr=np.linspace(-rMax0,rMax0,numGridEdge)
    yGridArr=xGridArr.copy()
    coords=np.asarray(np.meshgrid(xGridArr,yGridArr)).T.reshape(-1,2)
    coords=coords[np.linalg.norm(coords,axis=1)<rMax0]
    xArr,yArr=coords.T
    density0=1.0/(xGridArr[1]-xGridArr[0])**2
    pxArr=np.ones((len(xArr),3))
    return xArr,yArr,pxArr,density0
def make_Fake_Image(rMax,numSamplesGoal,sigma,gamma):
    np.random.seed(42)
    numSamples=0
    data=[]
    while numSamples<numSamplesGoal:
        x,y=rMax*(2*(np.random.random_sample(2)-.5))
        r=np.sqrt(x**2+y**2)
        prob=voigt(r,1.0,0.0,sigma,gamma)
        accept=np.random.random_sample()<prob
        if accept==True:
            data.append([x,y])
            numSamples+=1
    yArr,zArr=np.asarray(data).T
    return yArr,zArr






def test__make_Density_And_Pos_Arr():
    numBins=100
    xArr,yArr,pArr,density0=make_Fake_Flat_Data()
    vxArr=pArr[:,0]
    rArr,densityArr,densityErrArr=make_Density_And_Pos_Arr(xArr,yArr,vxArr,numBins)
    assert np.all(densityErrArr<=densityArr)
    assert abs(len(densityArr)/(numBins**2*(np.pi/4.0))-1.0)<.1 #geometry works as expected
    assert abs(np.mean(densityArr)/density0-1.0)<.1 #assert density mostly works
def test__Radial_Signal_Arr():
    numBins=100
    xArr,yArr,pArr,density0=make_Fake_Flat_Data()
    vxArr=pArr[:,0]
    rArr1,signalArr1,signalErrArr1=make_Radial_Signal_Arr(xArr,yArr,vxArr,numBins,numSamples=1)
    rArr2,signalArr2,signalErrArr2=make_Radial_Signal_Arr(xArr,yArr,vxArr,numBins,numSamples=4)
    assert abs(signalArr1.mean()/signalArr2.mean()-1.0)<.05
    assert abs(signalArr1.mean()/density0-1.0)<.05
    assert abs(signalArr2.mean()/density0-1.0)<.05
    assert np.all(signalErrArr1<=signalArr1) and np.all(signalErrArr2<=signalArr2)
def test__get_FWHM():
    sigma0,gamma0=1.0,2.0
    rMax0=10.0
    w0=.3
    numBins0=int(2*rMax0/w0)
    numSamplesGoal0=100_000
    yArr,zArr=make_Fake_Image(rMax0,numSamplesGoal0,sigma0,gamma0)
    vxArr=np.ones(len(yArr))
    rArr,signalArr,signalErrArr=make_Density_And_Pos_Arr(yArr,zArr,vxArr,numBins0)
    sigmaArr=signalErrArr/signalArr


    guess=[signalArr.max(),signalArr.min(),1.0,1.0]
    bounds=[0.0,np.inf]
    params=curve_fit(voigt,rArr,signalArr,p0=guess,bounds=bounds,sigma=sigmaArr)[0]
    # plt.scatter(rArr,signalArr)
    # rPlot=np.linspace(0.0,rArr.max(),1_000)
    # plt.plot(rPlot,voigt(rPlot,*params),c='r')
    # plt.plot(rPlot,voigt(rPlot,*[signalArr.max(),0.0,sigma0,gamma0]),c='g')
    # plt.show()
    sigma=params[2]
    gamma=params[3]
    fL=gamma
    fG=2*sigma*np.sqrt(2*np.log(2)) #np.log is natural logarithm
    FWHM=.5346*fL+np.sqrt(.2166*fL**2 + fG**2)
    pFakeArr=np.ones((len(yArr),3))
    fakeInterpFunc=lambda x,**kwargs: (yArr,zArr,pFakeArr)
    
    FWHM_FromMyFunc=get_FWHM(np.nan,fakeInterpFunc,Plot=False,w=w0,rMax=rMax0)
    assert abs(gamma/gamma0-1)<.1 and abs(sigma/sigma0-1)<.1
    assert abs(FWHM_FromMyFunc/FWHM -1.0)<1e-3

def test__Interpolator():
    """dirty tester"""
    LObject = 72.0E-2
    LImage = 85E-2
    LLensHardEdge = 15.24e-2
    rpLens = (5e-2, 5e-2 + 2.54e-2)
    magnetWidth = (.0254, .0254 * 1.5)

    voltScanRange=.25
    freqRange=voltScanRange*565e6
    vTMax=freqRange*671e-9
    vTMax/=2 #need to half because validity is assesed with radial velocity


    fringeFrac = 1.5
    LFringe = fringeFrac * max(rpLens)
    LLens = LLensHardEdge + 2 * LFringe
    LObject -= LFringe
    LImage -= LFringe

    PTL = ParticleTracerLattice(v0Nominal=210.0, latticeType='injector', fieldDensityMultiplier=1.0,
                                standardMagnetErrors=False)
    PTL.add_Drift(LObject, ap=.07)
    PTL.add_Halbach_Lens_Sim(rpLens, LLens, apFrac=None, magnetWidth=magnetWidth)
    PTL.add_Drift(LImage * 2, ap=.07)
    assert PTL.elList[1].fringeFracOuter==fringeFrac and abs(PTL.elList[1].Lm-LLensHardEdge)<1e-9
    PTL.end_Lattice()
    lensIndex = 1
    assert type(PTL.elList[lensIndex]) == HalbachLensSim
    assert abs((PTL.elList[lensIndex].L - PTL.elList[lensIndex].Lm) / 2 - LFringe) < 1e-9

    v0 = 210.0
    xEnd = -PTL.elList[-1].r2[0] + 1e-3
    swarmTest = Swarm()
    thetaArr = np.linspace(.01, 1.5*vTMax/v0, 30)
    thetaPhiList = []
    for theta in thetaArr:
        numSamples = int(np.random.random_sample() * 15 + 3)
        phiArr = np.linspace(0.0, 2 * np.pi, numSamples)[:-1]
        for phi in phiArr:
            thetaPhiList.append([theta, phi])
            particle = Particle()
            # tan is used because of the slope
            p = np.array([-1.0, np.tan(theta) * np.sin(phi), np.tan(theta) * np.cos(phi)]) * v0
            q = np.array([-xEnd, 0.0, 0.0])
            particle.qf = q
            particle.pf = p
            swarmTest.particles.append(particle)
    thetaPhiArr = np.array(thetaPhiList)
    interpFunc = Interpolater(swarmTest, PTL)


    xTest = 1.75
    deltaX = abs(xEnd) - xTest
    yArr, zArr, pArr = interpFunc(xTest, returnP=True)
    rArr = np.sqrt(yArr ** 2 + zArr ** 2)
    assert abs(rArr.max() - deltaX * np.tan(thetaArr.max()) * 1e3) < 1e-12
    assert len(thetaPhiArr) == len(yArr) and len(thetaPhiArr) == len(zArr) and len(thetaPhiArr) == len(pArr)
    #test that interpolation results agree with geoemtric model
    yGeomArr = np.sin(thetaPhiArr[:, 1]) * deltaX * np.tan(-thetaPhiArr[:, 0]) * 1e3
    zGeomArr = np.cos(thetaPhiArr[:, 1]) * deltaX * np.tan(-thetaPhiArr[:, 0]) * 1e3
    assert np.all(np.abs(np.sort(zArr) - np.sort(zGeomArr)) < 1e-6)
    assert np.all(np.abs(np.sort(yArr) - np.sort(yGeomArr)) < 1e-6)
    #test that momentum is reduced by using laser scan range as expcted
    yArr, zArr, pArr = interpFunc(xTest, returnP=True, laserScanRangeVolts=voltScanRange)
    vTArr = np.sqrt(pArr[:, 1] ** 2 + pArr[:, 2] ** 2)
    assert vTArr.max() <= interpFunc._get_Sweep_Range_Trans_Vel_Max(voltScanRange)
    assert len(yArr)<len(yGeomArr)

    import pytest
    testVals = [0.0, -PTL.elList[-1].r2[0], -1.0, np.inf]
    for val in testVals:
        with pytest.raises(ValueError):
            interpFunc(val)