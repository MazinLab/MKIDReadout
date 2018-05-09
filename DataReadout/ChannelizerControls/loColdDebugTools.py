import numpy as np

def getDefaultRegList():
    return [0x80000028, 0x69500009, 0x8800000A, 0x0000000B, 0xca9ce00c, 0x108e968d, 0x5441100E]

def changeRegBits(regVal, bitRange, newVal):
    if bitRange.__class__==int:
        bitmask = 2**bitRange
        if newVal>1:
            raise Exception('Not enough bits for newVal')
    else:
        bitmask = 2**(bitRange[1] + 1) - 1 - (2**(bitRange[0]) - 1)
        if newVal > (2**(bitRange[1] - bitRange[0] + 1) - 1):
            raise Exception('Not enough bits for newVal')

    regVal -= (regVal & bitmask)
    regVal += (newVal << bitRange[0])
    print bin(regVal)
    return regVal
