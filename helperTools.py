import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import numba
import time
import multiprocess as mp
from typing import Optional,Union,Callable,Any
list_tuple_arr=Union[list,tuple,np.ndarray]

def tool_Parallel_Process(func: Callable, args: Any,resultsAsArray: bool=False,processes: int=-1,reRandomize: bool=True,
                          reapplyArgs: Optional[int]=None)->Union[list,np.ndarray]:

    def randomized_Wrapper(arg,seed):
        if reRandomize==True: np.random.seed(seed)
        return func(arg)

    assert type(processes)==int and processes>=-1
    assert (type(reapplyArgs)==int and reapplyArgs>=1) or reapplyArgs is None

    argsIter= (args,)*reapplyArgs if reapplyArgs is not None else args
    seedArr=int(time.time())+np.arange(len(argsIter))
    poolIter=zip(argsIter,seedArr)
    processes = mp.cpu_count() if processes == -1 else processes

    with mp.Pool(processes) as pool:
        results = pool.starmap(randomized_Wrapper, poolIter)

    results=np.array(results) if resultsAsArray==True else results
    return results

def tool_Make_Image_Cartesian(func: Callable,xGridEdges: list_tuple_arr,yGridEdges: list_tuple_arr,
                              extraArgs: list_tuple_arr =(), arrInput=False)-> tuple[np.ndarray,list]:
    coords = np.array(np.meshgrid(xGridEdges, yGridEdges)).T.reshape(-1, 2)
    if arrInput==True:
        print("using single array input to make image data")
        vals=func(coords,*extraArgs)
    else:
        print("looping over function inputs to make image data")
        vals=np.asarray([func(*coord, *extraArgs) for coord in coords])
    image = vals.reshape(len(xGridEdges), len(yGridEdges))
    image = np.rot90(image)
    extent=[xGridEdges.min(),xGridEdges.max(),yGridEdges.min(),yGridEdges.max()]
    return image,extent

def _test_tool_Make_Image_Cartesian():
    def fake_Func(x,y,z):
        assert z is None
        if np.abs(x + 1.0) < 1e-9 and np.abs(y + 1.0) < 1e-9:  # x=-1.0,y=-1.0   BL
            return 1
        if np.abs(x + 1.0) < 1e-9 and np.abs(y - 1.0) < 1e-9:  # x=-1.0, y=1.0 TL
            return 2
        if np.abs(x - 1.0) < 1e-9 and np.abs(y - 1.0) < 1e-9:  # x=1.0,y=1.0   TR
            return 3
        return 0.0
    xArr = np.linspace(-1, 1, 15)
    yArr = np.linspace(-1, 1, 5)
    image=tool_Make_Image_Cartesian(fake_Func, xArr, yArr,extraArgs=[None])
    assert image[len(yArr)-1,0]==1 and image[0,0]==2 and image[0,len(xArr)-1]==3

# def main():
#
#     _test_tool_Make_Image_Cartesian()
# if __name__=="__main__":
#     main()