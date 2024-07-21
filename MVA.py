def minvar(VectorArray):
  import numpy as np
  from numpy import linalg as npla

  # Discard any entries where some values are invalid (ie. NaN)
  workingVectors = VectorArray[np.where(np.isfinite(VectorArray).any(axis=1))]
  steps = workingVectors.shape[0]
  spacialDims = workingVectors.shape[1]

  # Pre-init mva matrix
  MVA_matrix = np.zeros((spacialDims,spacialDims))

  # Component averages
  vecAvg = np.mean(workingVectors, axis=0)
  for dim1 in range(spacialDims):
    for dim2 in range(spacialDims):
      MVA_matrix[dim1,dim2] = np.sum(workingVectors[:,dim1]*workingVectors[:,dim2])/steps - (vecAvg[dim1]*vecAvg[dim2])
  # My attempt to do this directly via numpy:
  ####Note:  not this function!
  ##  EigVals, EigVecs = npla.eigh(MVA_matrix)
  EigVals, EigVecs = npla.eig(MVA_matrix)

  #performs lines 65 through 69 from minvar.pro
  indices = np.fabs(EigVals).argsort()[::-1]
  sortedEVals = EigVals[indices]
  sortedEVecs = EigVecs[:,indices]
  
  YcrosZdotX = sortedEVecs[0,0]*(sortedEVecs[1,1]*sortedEVecs[2,2] - sortedEVecs[2,1]*sortedEVecs[1,2])
  if YcrosZdotX < 0:
      sortedEVecs[:,1] = -sortedEVecs[:,1]
  
  return (sortedEVals, sortedEVecs)
