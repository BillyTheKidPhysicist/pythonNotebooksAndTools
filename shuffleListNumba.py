import random
import numba
from numba.typed import List
import time
@numba.njit()
def shuffle_List_In_Place(aList):
    #take a list and shuffle it in place. Nothing is returned. Fisher-Yates shuffle
    #aList: a list to be shuffled in place. Must be numba list: numba.typed.List
    indexCurrent=len(aList)-1
    while indexCurrent>0:
        index=random.randint(0,indexCurrent)
        if index!=indexCurrent:
            aList[index],aList[indexCurrent]=aList[indexCurrent],aList[index]
        indexCurrent-=1

@numba.njit()
def work():
    temp=List(range(10))
    for _ in range(1_000_000):
        shuffle_List_In_Place(temp)
work()
t=time.time()
work()
print(time.time()-t)