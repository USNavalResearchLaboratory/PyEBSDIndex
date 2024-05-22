# This software was developed by employees of the US Naval Research Laboratory (NRL), an
# agency of the Federal Government. Pursuant to title 17 section 105 of the United States
# Code, works of NRL employees are not subject to copyright protection, and this software
# is in the public domain. PyEBSDIndex is an experimental system. NRL assumes no
# responsibility whatsoever for its use by other parties, and makes no guarantees,
# expressed or implied, about its quality, reliability, or any other characteristic. We
# would appreciate acknowledgment if the software is used. To the extent that NRL may hold
# copyright in countries other than the United States, you are hereby granted the
# non-exclusive irrevocable and unconditional right to print, publish, prepare derivative
# works and distribute this software, in any medium, or authorize others to do so on your
# behalf, on a royalty-free basis throughout the world. You may improve, modify, and
# create derivative works of the software or any portion of the software, and you may copy
# and distribute such modifications or works. Modified works should carry a notice stating
# that you changed the software and should note the date and nature of any such change.
# Please explicitly acknowledge the US Naval Research Laboratory as the original source.
# This software can be redistributed and/or modified freely provided that any derivative
# works bear some notice that they are derived from it, and any modified versions bear
# some notice that they have been modified.
#
# This borrows heavily from Marc De Graef's EMsoft package. Many functions here
# are Python translations of functions from it.
#
#
# Author: David Rowenhorst, Patrick Callahan;
# The US Naval Research Laboratory Date: 22 May 2024


import numpy as np
import math as math
from functools import reduce

## This borrows heavily from Marc De Graef's EMsoft package. Many functions here
## are Python translations of functions from it.



class CrystalPlane:
    def __init__(self,hkl = [0,0,0],centering = 'F'):
        self.hkl = np.array(hkl)
        self.allowed = True
        self.lattice_centering = centering
        self.d = 0.0
        self.g = 0.0

    def is_g_allowed(self):
        self.allowed = True
        if self.lattice_centering == 'P':
            self.allowed = True
        if self.lattice_centering == 'F':
            selection = np.sum(np.mod(self.hkl,2))
            if ((selection == 1) or (selection == 2)):
                self.allowed = False


    # def get_planar_distance(self):
    #     g = np.array(np.sqrt(gg*(np.matrix(self.reciprocalMetricTensor))*gg.transpose()))
    #     self.g = g[0]
    #     d = 1/self.g
    #     self.d = d

    def get_plane_normal(self):
        if self.lattice_centering == 'F':
            self.gvec = self.hkl

class Crystal:
    def __init__(self,name,a,b,c,alpha,beta,gamma):
        self.name = name 
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.elambda = 0.0
        self.spaceGroup = ''
        smallThreshold = 1e-6

        ca = np.cos(alpha*np.pi/180)
        cb = np.cos( beta*np.pi/180)
        cg = np.cos(gamma*np.pi/180)
        sa = np.sin(alpha*np.pi/180)
        sb = np.sin( beta*np.pi/180)
        sg = np.sin(gamma*np.pi/180)
        tg = np.tan(gamma*np.pi/180)
        fabg = ca*cb-cg


        #compute the real space metric tensor
        self.metricTensor = np.zeros([3,3])
        self.metricTensor[0,0] = a**2.0
        self.metricTensor[1,1] = b**2.0
        self.metricTensor[2,2] = c**2.0
        self.metricTensor[1,0] = a*b*cg
        self.metricTensor[0,1] = self.metricTensor[1,0]
        self.metricTensor[2,0] = a*c*cb
        self.metricTensor[0,2] = self.metricTensor[2,0]
        self.metricTensor[2,1] = b*c*ca
        self.metricTensor[1,2] = self.metricTensor[2,1]
        self.mtDet = ((a*b*c)**2)*(1.0-ca**2 - cb**2 - cg**2 + 2.0*ca*cb*cg)
        self.volume = np.sqrt(self.mtDet)
        self.metricTensor = self.metricTensor.transpose()
        self.metricTensor[np.abs(self.metricTensor) < smallThreshold] = 0
        
        #Compute the reciprocal metric tensor
        self.reciprocalMetricTensor = np.zeros([3,3])
        self.reciprocalMetricTensor[0,0] = (b*c*sa)**2
        self.reciprocalMetricTensor[1,1] = (a*c*sb)**2
        self.reciprocalMetricTensor[2,2] = (a*b*sg)**2
        self.reciprocalMetricTensor[1,0] = a*b*c**2*(ca*cb-cg)
        self.reciprocalMetricTensor[0,1] = self.reciprocalMetricTensor[1,0]
        self.reciprocalMetricTensor[2,0] = a*b**2*c*(cg*ca-cb)
        self.reciprocalMetricTensor[0,2] = self.reciprocalMetricTensor[2,0]
        self.reciprocalMetricTensor[2,1] = a**2*b*c*(cb*cg-ca)
        self.reciprocalMetricTensor[1,2] = self.reciprocalMetricTensor[2,1]
        self.reciprocalMetricTensor = self.reciprocalMetricTensor/self.mtDet
        self.reciprocalMetricTensor = self.reciprocalMetricTensor.transpose()
        self.reciprocalMetricTensor[np.abs(self.reciprocalMetricTensor) < smallThreshold] = 0
        
        #Compute direct stucture matrix
        self.directStructureMatrix = np.zeros([3,3])
        self.directStructureMatrix[0,0] = a
        self.directStructureMatrix[1,0] = b*cg
        self.directStructureMatrix[2,0] = c*cb
        self.directStructureMatrix[0,1] = 0.0
        self.directStructureMatrix[1,1] = b*sg
        self.directStructureMatrix[2,1] = -c*(cb*cg-ca)/sg
        self.directStructureMatrix[0,2] = 0.0
        self.directStructureMatrix[1,2] = 0.0
        self.directStructureMatrix[2,2] = self.volume/(a*b*sg)
        self.directStructureMatrix = self.directStructureMatrix.transpose()
        self.directStructureMatrix[np.abs(self.directStructureMatrix) < smallThreshold] = 0

        # Compute inverse direct stucture matrix
        self.invDirectStructureMatrix = np.zeros([3, 3])
        self.invDirectStructureMatrix[0, 0] = 1.0/a
        self.invDirectStructureMatrix[1, 0] = -1.0/(a * tg)
        self.invDirectStructureMatrix[2, 0] = b*c * (cg*ca-cb)/(self.volume * sg)
        self.invDirectStructureMatrix[0, 1] = 0.0
        self.invDirectStructureMatrix[1, 1] = 1.0/(b * sg)
        self.invDirectStructureMatrix[2, 1] = a*c*(cb*cg-ca)/(self.volume*sg)
        self.invDirectStructureMatrix[0, 2] = 0.0
        self.invDirectStructureMatrix[1, 2] = 0.0
        self.invDirectStructureMatrix[2, 2] = a*b*sg/(self.volume)
        self.invDirectStructureMatrix = self.invDirectStructureMatrix.transpose()
        self.invDirectStructureMatrix[np.abs(self.invDirectStructureMatrix) < smallThreshold] = 0
        
        #compute reciprocal structure matrix
        self.reciprocalStructureMatrix = np.zeros([3,3])
        self.reciprocalStructureMatrix[0,0] = 1.0/a
        self.reciprocalStructureMatrix[1,0] = 0.0
        self.reciprocalStructureMatrix[2,0] = 0.0
        self.reciprocalStructureMatrix[0,1] = -1.0/(a*tg)
        self.reciprocalStructureMatrix[1,1] = 1.0/(b*sg)
        self.reciprocalStructureMatrix[2,1] = 0.0
        self.reciprocalStructureMatrix[0,2] = b*c*(cg*ca-cb)/(self.volume*sg)
        self.reciprocalStructureMatrix[1,2] = a*c*(cb*cg-ca)/(self.volume*sg)
        self.reciprocalStructureMatrix[2,2] = (a*b*sg)/self.volume
        self.reciprocalStructureMatrix = self.reciprocalStructureMatrix.transpose()
        self.reciprocalStructureMatrix[np.abs(self.reciprocalStructureMatrix) < smallThreshold] = 0

        # Compute inverse reciprocal stucture matrix
        self.invReciprocalStructureMatrix = np.zeros([3, 3])
        self.invReciprocalStructureMatrix = np.transpose(self.directStructureMatrix)

    # def get_lattice_centering(self):
    #     self.latticeCentering = self.spaceGroup[0]

    # def is_g_allowed(self,g):
    #     get_lattice_centering()
    #     allowed = True
    #     if lattice_centering == 'F':
    #         seo = np.sum(np.mod(g + 100,2))
    #         if ((seo == 1) or (seo == 2)):
    #             allowed = False
    #     # if allowed == True:
    #     #     self.gVectors = self.gVectors.append(g.tolist)
    #     return allowed


    def generate_list_of_planes(self,maxhkl = 3,dmin = 0.0):
        gVectorList = []
        try:
            print('generating list')
            g = [0,0,0]
            g0 = CrystalPlane(g,'F')
            g0.d = np.Infinity
            g0.g = 0.0
            # g0
            gVectorList.append(g0)
            for h in range(-maxhkl,maxhkl+1):
                for k in range(-maxhkl,maxhkl+1):
                    for l in range(-maxhkl,maxhkl+1):
                        if np.abs(h)+np.abs(k)+np.abs(l) != 0:
                            g = np.array([h,k,l]) # the current gvector
                            g1 = CrystalPlane(g,'F')
                            g1.is_g_allowed()
                            if g1.allowed == True:
                                gg = np.matrix(g)
                                dval = np.array(1.0/np.sqrt(gg*(np.matrix(self.reciprocalMetricTensor))*gg.transpose()))[0]
                                g1.d = dval
                                g1.g = 1/dval
                                if g1.d > dmin:
                                    gVectorList.append(g1)
            self.planeList = gVectorList
        except:
            print('error')



def uvtw2uvw(uvtw):
    uvw = np.zeros(3)
    uvw[0] = uvtw[0] - uvtw[2]
    uvw[1] = uvtw[1] - uvtw[2]
    uvw[2] = uvtw[3]
    #find greatest common denominator and divide
    gcd = reduce(math.gcd,(uvw.astype(int)))
    uvw = uvw/gcd
    return uvw

# def rotVecQuat(v,qu):
#     # Rotate a 3-vector v by quaternion qu.
#     qv = quaternion.Quaternion([0,v[0],v[1],v[2]])
#     qvr = qu.conjugate() * qv * qu
#     vecRot = np.array(qvr[1:])
#     return vecRot

	# def wavelength(voltage):
	# 	## enter voltage in Volts
	# 	h = 6.62606957e-34 # Js planck's constant
	# 	me = 9.10938291e-31 # kg
	# 	e = 1.60217657e-19 # C electron charge
	# 	c = 299792458.0 # m/s  speed light
	# 	tmp1 = 1.0e9*h/sqrt(2.0*me*e)
	# 	tmp2 = 0.5*e*voltage/(me*c**2.0)

	# 	relcor = 1.0 + 2.0*tmp2 # relativistic correction factor
	# 	psi = voltage*(1.0+tmp2)
	# 	elambda = tmp1/sqrt(psi)


	# def a(self):
	# 	return self
	# def __str__(self):
	# 	return str(self.name)
	# def __repr__(self):
	# 	return str(self)

# if __name__ == "__main__":
# xt = Crystal('Ni', 0.352, 0.352, 0.352, 90.0, 90.0, 90.0)