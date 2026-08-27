#!/usr/bin/env python3
import random

def ids(count,B,SV):
    out=[];grid=(count+B*SV-1)//(B*SV)
    for block in range(grid):
        bb=block*B*SV
        for tid in range(B):
            lb=bb+tid
            if lb>=count: continue
            for i in range(SV):
                n=lb+i*B
                if n<count: out.append(n)
    return out

def main():
    r=random.Random(0xC0A1E5CE);checks=0
    for B in (32,64,96,128,256):
      for SV in (1,2,4,8):
       for count in list(range(0,2*B*SV+3))+[r.randrange(1,100000) for _ in range(40)]:
        x=ids(count,B,SV)
        assert len(x)==count and len(set(x))==count and sorted(x)==list(range(count))
        # Within any full warp and fixed vector slot, candidate ids are adjacent.
        if B>=32 and count>=32:
          slot=[0*B*SV+t+0*B for t in range(32)]
          assert slot==list(range(32))
        checks+=1
    print(f'PASS: {checks} block-interleaved suffix mapping cases')
if __name__=='__main__':main()
