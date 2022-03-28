
from helperTools import *
from scipy.special import voigt_profile
from scipy.optimize import curve_fit
import scipy.interpolate as spi
@np.vectorize
def voigt(r,a,b,sigma,gamma):
    #be very cautious about FWHM vs HWHM here for gamma. gamma for scipy is HWHM per the docs
    assert r>=0
    gamma=gamma/2.0 #convert to HWHM per scipy docs
    v0=voigt_profile(0,sigma,gamma)
    v=voigt_profile(r,sigma,gamma)/v0
    v=a*v+b
    return v

def make_Density_And_Pos_Arr(yArr: np.ndarray,zArr: np.ndarray,numBins: int) \
        -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    assert len(yArr)==len(zArr)
    image,binx,biny=np.histogram2d(yArr,zArr,bins=numBins)
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

def make_Radial_Signal_Arr(yArr: np.ndarray,zArr: np.ndarray,numBins: int,numSamples: int=3) \
        -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    rArr,signalArr,signalErrArr=make_Density_And_Pos_Arr(yArr,zArr,numBins)
    for i in range(1,numSamples):
        rArrTemp,signalArrTemp,signalErrArrTemp=make_Density_And_Pos_Arr(yArr,zArr,numBins+i)
        rArr=np.append(rArr,rArrTemp)
        signalArr=np.append(signalArr,signalArrTemp)
        signalErrArr=np.append(signalErrArr,signalErrArrTemp)
    return rArr,signalArr,signalErrArr


def get_FWHM(x: float,interpFunction: Callable,Plot: bool=False,rMax: float=10.0,w: float=.3,useSweepRange=True,
             numSamples: int=3)-> float:
    """

    :param x: location to get FWHM, cm
    :param interpFunction: Function for interpolating particle positions
    :param Plot: wether to plot the focus
    :param rMax: Maximum radius to consider, mm
    :param w: Interrogation size, ie laser beam waist, mm
    :return: the FWHM of the voigt
    """
    numBins=int(2*rMax/w)
    yArr,zArr=interpFunction(x,maxRadius=rMax,useSweepRange=useSweepRange)
    rArr,signalArr,signalErrArr=make_Radial_Signal_Arr(yArr,zArr,numBins,numSamples=numSamples)
    sigmaArr=signalErrArr/signalArr

    guess=[signalArr.max(),signalArr.min(),1.0,1.0]
    bounds=[0.0,np.inf]
    params=curve_fit(voigt,rArr,signalArr,p0=guess,bounds=bounds,sigma=sigmaArr)[0]
    if Plot==True:
        plt.scatter(rArr,signalArr)
        rPlot=np.linspace(0.0,rArr.max(),1_000)
        plt.plot(rPlot,voigt(rPlot,*params),c='r')
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
    return xArr,yArr,density0
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
# def test_make_Density_And_Pos_Arr():
#     numBins=100
#     xArr,yArr,density0=make_Fake_Flat_Data()
#     rArr,densityArr,densityErrArr=make_Density_And_Pos_Arr(xArr,yArr,numBins)
#     assert np.all(densityErrArr<=densityArr)
#     assert abs(len(densityArr)/(numBins**2*(np.pi/4.0))-1.0)<.1 #geometry works as expected
#     assert abs(np.mean(densityArr)/density0-1.0)<.1 #assert density mostly works
# def test_Radial_Signal_Arr():
#     numBins=100
#     xArr,yArr,density0=make_Fake_Flat_Data()
#     rArr1,signalArr1,signalErrArr1=make_Radial_Signal_Arr(xArr,yArr,numBins,numSamples=1)
#     rArr2,signalArr2,signalErrArr2=make_Radial_Signal_Arr(xArr,yArr,numBins,numSamples=4)
#     assert abs(signalArr1.mean()/signalArr2.mean()-1.0)<.05
#     assert abs(signalArr1.mean()/density0-1.0)<.05
#     assert abs(signalArr2.mean()/density0-1.0)<.05
#     assert np.all(signalErrArr1<=signalArr1) and np.all(signalErrArr2<=signalArr2)
# def test_get_FWHM():
#     sigma0,gamma0=1.0,2.0
#     rMax0=10.0
#     w0=.3
#     numBins0=int(2*rMax0/w0)
#     numSamplesGoal0=100_000
#     yArr,zArr=make_Fake_Image(rMax0,numSamplesGoal0,sigma0,gamma0)
#
#     rArr,signalArr,signalErrArr=make_Density_And_Pos_Arr(yArr,zArr,numBins0)
#     sigmaArr=signalErrArr/signalArr
#
#
#     guess=[signalArr.max(),signalArr.min(),1.0,1.0]
#     bounds=[0.0,np.inf]
#     params=curve_fit(voigt,rArr,signalArr,p0=guess,bounds=bounds,sigma=sigmaArr)[0]
#     # plt.scatter(rArr,signalArr)
#     # rPlot=np.linspace(0.0,rArr.max(),1_000)
#     # plt.plot(rPlot,voigt(rPlot,*params),c='r')
#     # plt.plot(rPlot,voigt(rPlot,*[signalArr.max(),0.0,sigma0,gamma0]),c='g')
#     # plt.show()
#     sigma=params[2]
#     gamma=params[3]
#     fL=gamma
#     fG=2*sigma*np.sqrt(2*np.log(2)) #np.log is natural logarithm
#     FWHM=.5346*fL+np.sqrt(.2166*fL**2 + fG**2)
#     fakeInterpFunc=lambda x,**kwargs: (yArr,zArr)
#     FWHM_FromMyFunc=get_FWHM(np.nan,fakeInterpFunc,Plot=False,numSamples=1,w=w0,rMax=rMax0)
#     # print(FWHM_FromMyFunc,FWHM)
#     # print(gamma,gamma0,sigma,sigma0)
#     assert abs(gamma/gamma0-1)<.1 and abs(sigma/sigma0-1)<.1
#     assert abs(FWHM_FromMyFunc/FWHM -1.0)<1e-3
