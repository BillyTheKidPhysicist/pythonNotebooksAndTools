import numpy as np

def simple_Square_Itegrate(func,bounds):
    #func: a N dimensional function that take N arguments. will look like func(x1,x2,x3,..xn).
    #bounds: a list of the start,end,numberpoints of the integral. will look like [(x1a,x1b,x1Num),...,(xna,xnb,xnNum)].
    #make sure to pick a reasonable number of points! double check that the result is invariant to the required degree
    #to changes in the number of points
    numPoints=1
    for bound in bounds:
      assert bound[1]>bound[0] and len(bound)==3
      assert isinstance(bound[2],int) and bound[2]>1
      numPoints=numPoints*bound[2]
    if numPoints>50_000:
      print('Warning: You are evaluating alot of points!! A total of: ',numPoints)
    axisPointsList=[np.linspace(b[0],b[1],b[2]) for b in bounds]
    integralElement=np.prod([axePoints[1]-axePoints[0] for axePoints in axisPointsList]) #dA or dV etc.
    coords=np.asarray(np.meshgrid(*axisPointsList)).T.reshape(-1,len(bounds))
    funcSum=sum([func(*coord) for coord in coords])
    integral=funcSum*integralElement #now multiply by area/volume/line element
    return integral

func1=lambda x: (1/np.sqrt(2*np.pi))*np.exp(-.5*x**2)
bounds1=[(-100.0,100.0,10_000)]
print(simple_Square_Itegrate(func1,bounds1)) #integral should be one!
assert abs(simple_Square_Itegrate(func1,bounds1)-1.0)<1e-6

import sympy as sym
x,y,z=sym.symbols('x y z',real=True)
func2Sym=sym.sin(x)**4*sym.cos(y)**2*sym.exp(-z**2)
actualValue=float(sym.N(sym.integrate(func2Sym,(x,-1,2.0),(y,-2,5),(z,-3.14,5))))
func2=sym.lambdify([x,y,z],func2Sym)
bounds2=[(-1.0,2.0,100),(-2.0,5.0,100),(-3.14,5,100)]
print(actualValue)
print(simple_Square_Itegrate(func2,bounds2))
