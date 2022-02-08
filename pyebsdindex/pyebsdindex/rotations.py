####################################################################################################
# Copyright (c) 2017-2020, Martin Diehl/Max-Planck-Institut für Eisenforschung GmbH
# Copyright (c) 2013-2014, Marc De Graef/Carnegie Mellon University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
#     - Redistributions of source code must retain the above copyright notice, this list
#        of conditions and the following disclaimer.
#     - Redistributions in binary form must reproduce the above copyright notice, this
#        list of conditions and the following disclaimer in the documentation and/or
#        other materials provided with the distribution.
#     - Neither the names of Marc De Graef, Carnegie Mellon University nor the names
#        of its contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
####################################################################################################

# Note: An object oriented approach to use this conversions is available in DAMASK, see
# https://damask.mpie.de and https://github.com/eisenforschung/DAMASK

import numpy as np

P = 1

# parameters for conversion from/to cubochoric
sc   = np.pi**(1./6.)/6.**(1./6.)
beta = np.pi**(5./6.)/6.**(1./6.)/2.
R1   = (3.*np.pi/4.)**(1./3.)
#---------- Quaternion ----------

def qu2om(qu):
    qq = qu[...,0:1]**2-(qu[...,1:2]**2 + qu[...,2:3]**2 + qu[...,3:4]**2)
    om = np.block([qq + 2.0*qu[...,1:2]**2,
                   2.0*(qu[...,2:3]*qu[...,1:2]-P*qu[...,0:1]*qu[...,3:4]),
                   2.0*(qu[...,3:4]*qu[...,1:2]+P*qu[...,0:1]*qu[...,2:3]),
                   2.0*(qu[...,1:2]*qu[...,2:3]+P*qu[...,0:1]*qu[...,3:4]),
                   qq + 2.0*qu[...,2:3]**2,
                   2.0*(qu[...,3:4]*qu[...,2:3]-P*qu[...,0:1]*qu[...,1:2]),
                   2.0*(qu[...,1:2]*qu[...,3:4]-P*qu[...,0:1]*qu[...,2:3]),
                   2.0*(qu[...,2:3]*qu[...,3:4]+P*qu[...,0:1]*qu[...,1:2]),
                   qq + 2.0*qu[...,3:4]**2,
                  ]).reshape(qu.shape[:-1]+(3,3))
    return om


def qu2eu(qu):
    """Quaternion to Bunge-Euler angles."""
    q02   = qu[...,0:1]*qu[...,2:3]
    q13   = qu[...,1:2]*qu[...,3:4]
    q01   = qu[...,0:1]*qu[...,1:2]
    q23   = qu[...,2:3]*qu[...,3:4]
    q03_s = qu[...,0:1]**2+qu[...,3:4]**2
    q12_s = qu[...,1:2]**2+qu[...,2:3]**2
    chi = np.sqrt(q03_s*q12_s)

    eu = np.where(np.abs(q12_s) < 1.0e-8,
            np.block([np.arctan2(-P*2.0*qu[...,0:1]*qu[...,3:4],qu[...,0:1]**2-qu[...,3:4]**2),
                      np.zeros(qu.shape[:-1]+(2,))]),
                  np.where(np.abs(q03_s) < 1.0e-8,
                      np.block([np.arctan2(   2.0*qu[...,1:2]*qu[...,2:3],qu[...,1:2]**2-qu[...,2:3]**2),
                                np.broadcast_to(np.pi,qu.shape[:-1]+(1,)),
                                np.zeros(qu.shape[:-1]+(1,))]),
                      np.block([np.arctan2((-P*q02+q13)*chi, (-P*q01-q23)*chi),
                                np.arctan2( 2.0*chi,          q03_s-q12_s    ),
                                np.arctan2(( P*q02+q13)*chi, (-P*q01+q23)*chi)])
                          )
                 )
    # reduce Euler angles to definition range
    eu[np.abs(eu)<1.e-6] = 0.0
    eu = np.where(eu<0, (eu+2.0*np.pi)%np.array([2.0*np.pi,np.pi,2.0*np.pi]),eu)                # needed?
    return eu


def qu2ax(qu):
    """
    Quaternion to axis angle pair.

    Modified version of the original formulation, should be numerically more stable
    """
    with np.errstate(invalid='ignore',divide='ignore'):
        s = np.sign(qu[...,0:1])/np.sqrt(qu[...,1:2]**2+qu[...,2:3]**2+qu[...,3:4]**2)
        omega = 2.0 * np.arccos(np.clip(qu[...,0:1],-1.0,1.0))
        ax = np.where(np.broadcast_to(qu[...,0:1] < 1.0e-8,qu.shape),
                      np.block([qu[...,1:4],np.broadcast_to(np.pi,qu.shape[:-1]+(1,))]),
                      np.block([qu[...,1:4]*s,omega]))
    ax[np.isclose(qu[...,0],1.,rtol=0.0)] = [0.0, 0.0, 1.0, 0.0]
    return ax


def qu2ro(qu):
    """Quaternion to Rodrigues-Frank vector."""
    with np.errstate(invalid='ignore',divide='ignore'):
        s  = np.linalg.norm(qu[...,1:4],axis=-1,keepdims=True)
        ro = np.where(np.broadcast_to(np.abs(qu[...,0:1]) < 1.0e-12,qu.shape),
                      np.block([qu[...,1:2], qu[...,2:3], qu[...,3:4], np.broadcast_to(np.inf,qu.shape[:-1]+(1,))]),
                      np.block([qu[...,1:2]/s,qu[...,2:3]/s,qu[...,3:4]/s,
                                np.tan(np.arccos(np.clip(qu[...,0:1],-1.0,1.0)))
                               ])
                   )
    ro[np.abs(s).squeeze(-1) < 1.0e-12] = [0.0,0.0,P,0.0]
    return ro


def qu2ho(qu):
    """Quaternion to homochoric vector."""
    with np.errstate(invalid='ignore'):
        omega = 2.0 * np.arccos(np.clip(qu[...,0:1],-1.0,1.0))
        ho = np.where(np.abs(omega) < 1.0e-12,
                      np.zeros(3),
                      qu[...,1:4]/np.linalg.norm(qu[...,1:4],axis=-1,keepdims=True) \
                      * (0.75*(omega - np.sin(omega)))**(1./3.))
    return ho


def qu2cu(qu):
    """Quaternion to cubochoric vector."""
    return ho2cu(qu2ho(qu))


#---------- Rotation matrix ----------

def om2qu(om):
    """
    Rotation matrix to quaternion.

    This formulation is from  www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion.
    The original formulation had issues.
    """
    trace = om[...,0,0:1]+om[...,1,1:2]+om[...,2,2:3]

    with np.errstate(invalid='ignore',divide='ignore'):
        s = [
             0.5 / np.sqrt( 1.0 + trace),
             2.0 * np.sqrt( 1.0 + om[...,0,0:1] - om[...,1,1:2] - om[...,2,2:3]),
             2.0 * np.sqrt( 1.0 + om[...,1,1:2] - om[...,2,2:3] - om[...,0,0:1]),
             2.0 * np.sqrt( 1.0 + om[...,2,2:3] - om[...,0,0:1] - om[...,1,1:2] )
             ]
        qu= np.where(trace>0,
                     np.block([0.25 / s[0],
                              (om[...,2,1:2] - om[...,1,2:3] ) * s[0],
                              (om[...,0,2:3] - om[...,2,0:1] ) * s[0],
                              (om[...,1,0:1] - om[...,0,1:2] ) * s[0]]),
                     np.where(om[...,0,0:1] > np.maximum(om[...,1,1:2],om[...,2,2:3]),
                              np.block([(om[...,2,1:2] - om[...,1,2:3]) / s[1],
                                        0.25 * s[1],
                                        (om[...,0,1:2] + om[...,1,0:1]) / s[1],
                                        (om[...,0,2:3] + om[...,2,0:1]) / s[1]]),
                              np.where(om[...,1,1:2] > om[...,2,2:3],
                                       np.block([(om[...,0,2:3] - om[...,2,0:1]) / s[2],
                                                 (om[...,0,1:2] + om[...,1,0:1]) / s[2],
                                                 0.25 * s[2],
                                                 (om[...,1,2:3] + om[...,2,1:2]) / s[2]]),
                                       np.block([(om[...,1,0:1] - om[...,0,1:2]) / s[3],
                                                 (om[...,0,2:3] + om[...,2,0:1]) / s[3],
                                                 (om[...,1,2:3] + om[...,2,1:2]) / s[3],
                                                 0.25 * s[3]]),
                                      )
                             )
                    )*np.array([1,P,P,P])
        qu[qu[...,0]<0] *=-1
    return qu


def om2eu(om):
    """Rotation matrix to Bunge-Euler angles."""
    with np.errstate(invalid='ignore',divide='ignore'):
        zeta = 1.0/np.sqrt(1.0-om[...,2,2:3]**2)
        eu = np.where(np.isclose(np.abs(om[...,2,2:3]),1.0,1e-9),
                      np.block([np.arctan2(om[...,0,1:2],om[...,0,0:1]),
                                np.pi*0.5*(1-om[...,2,2:3]),
                                np.zeros(om.shape[:-2]+(1,)),
                               ]),
                      np.block([np.arctan2(om[...,2,0:1]*zeta,-om[...,2,1:2]*zeta),
                                np.arccos( om[...,2,2:3]),
                                np.arctan2(om[...,0,2:3]*zeta,+om[...,1,2:3]*zeta)
                               ])
                      )
    eu[np.abs(eu)<1.e-8] = 0.0
    eu = np.where(eu<0, (eu+2.0*np.pi)%np.array([2.0*np.pi,np.pi,2.0*np.pi]),eu)
    return eu


def om2ax(om):
    """Rotation matrix to axis angle pair."""
    #return qu2ax(om2qu(om)) # HOTFIX
    diag_delta = -P*np.block([om[...,1,2:3]-om[...,2,1:2],
                               om[...,2,0:1]-om[...,0,2:3],
                               om[...,0,1:2]-om[...,1,0:1]
                             ])
    t = 0.5*(om.trace(axis2=-2,axis1=-1) -1.0).reshape(om.shape[:-2]+(1,))
    w,vr = np.linalg.eig(om)
    # mask duplicated real eigenvalues
    w[np.isclose(w[...,0],1.0+0.0j),1:] = 0.
    w[np.isclose(w[...,1],1.0+0.0j),2:] = 0.
    vr = np.swapaxes(vr,-1,-2)
    ax = np.where(np.abs(diag_delta)<1e-12,
                         np.real(vr[np.isclose(w,1.0+0.0j)]).reshape(om.shape[:-2]+(3,)),
                  np.abs(np.real(vr[np.isclose(w,1.0+0.0j)]).reshape(om.shape[:-2]+(3,))) \
                  *np.sign(diag_delta))
    ax = np.block([ax,np.arccos(np.clip(t,-1.0,1.0))])
    ax[np.abs(ax[...,3])<1.e-8] = [ 0.0, 0.0, 1.0, 0.0]
    return ax


def om2ro(om):
    """Rotation matrix to Rodrigues-Frank vector."""
    return eu2ro(om2eu(om))


def om2ho(om):
    """Rotation matrix to homochoric vector."""
    return ax2ho(om2ax(om))


def om2cu(om):
    """Rotation matrix to cubochoric vector."""
    return ho2cu(om2ho(om))


#---------- Bunge-Euler angles ----------

def eu2qu(eu):
    """Bunge-Euler angles to quaternion."""
    ee = 0.5*eu
    cPhi = np.cos(ee[...,1:2])
    sPhi = np.sin(ee[...,1:2])
    qu = np.block([    cPhi*np.cos(ee[...,0:1]+ee[...,2:3]),
                   -P*sPhi*np.cos(ee[...,0:1]-ee[...,2:3]),
                   -P*sPhi*np.sin(ee[...,0:1]-ee[...,2:3]),
                   -P*cPhi*np.sin(ee[...,0:1]+ee[...,2:3])])
    qu[qu[...,0]<0.0]*=-1
    return qu


def eu2om(eu):
    """Bunge-Euler angles to rotation matrix."""
    c = np.cos(eu)
    s = np.sin(eu)
    om = np.block([+c[...,0:1]*c[...,2:3]-s[...,0:1]*s[...,2:3]*c[...,1:2],
                   +s[...,0:1]*c[...,2:3]+c[...,0:1]*s[...,2:3]*c[...,1:2],
                   +s[...,2:3]*s[...,1:2],
                   -c[...,0:1]*s[...,2:3]-s[...,0:1]*c[...,2:3]*c[...,1:2],
                   -s[...,0:1]*s[...,2:3]+c[...,0:1]*c[...,2:3]*c[...,1:2],
                   +c[...,2:3]*s[...,1:2],
                   +s[...,0:1]*s[...,1:2],
                   -c[...,0:1]*s[...,1:2],
                   +c[...,1:2]
                   ]).reshape(eu.shape[:-1]+(3,3))
    om[np.abs(om)<1.e-12] = 0.0
    return om


def eu2ax(eu):
    """Bunge-Euler angles to axis angle pair."""
    t = np.tan(eu[...,1:2]*0.5)
    sigma = 0.5*(eu[...,0:1]+eu[...,2:3])
    delta = 0.5*(eu[...,0:1]-eu[...,2:3])
    tau   = np.linalg.norm(np.block([t,np.sin(sigma)]),axis=-1,keepdims=True)
    alpha = np.where(np.abs(np.cos(sigma))<1.e-12,np.pi,2.0*np.arctan(tau/np.cos(sigma)))
    with np.errstate(invalid='ignore',divide='ignore'):
        ax = np.where(np.broadcast_to(np.abs(alpha)<1.0e-12,eu.shape[:-1]+(4,)),
                      [0.0,0.0,1.0,0.0],
                      np.block([-P/tau*t*np.cos(delta),
                                -P/tau*t*np.sin(delta),
                                -P/tau*  np.sin(sigma),
                                 alpha
                                ]))
    ax[(alpha<0.0).squeeze()] *=-1
    return ax


def eu2ro(eu):
    """Bunge-Euler angles to Rodrigues-Frank vector."""
    ax = eu2ax(eu)
    ro = np.block([ax[...,:3],np.tan(ax[...,3:4]*.5)])
    ro[ax[...,3]>=np.pi,3] = np.inf
    ro[np.abs(ax[...,3])<1.e-16] = [ 0.0, 0.0, P, 0.0 ]
    return ro


def eu2ho(eu):
    """Bunge-Euler angles to homochoric vector."""
    return ax2ho(eu2ax(eu))


def eu2cu(eu):
    """Bunge-Euler angles to cubochoric vector."""
    return ho2cu(eu2ho(eu))


#---------- Axis angle pair ----------

def ax2qu(ax):
    """Axis angle pair to quaternion."""
    c = np.cos(ax[...,3:4]*.5)
    s = np.sin(ax[...,3:4]*.5)
    qu = np.where(np.abs(ax[...,3:4])<1.e-6,[1.0, 0.0, 0.0, 0.0],np.block([c, ax[...,:3]*s]))
    return qu


def ax2om(ax):
    """Axis angle pair to rotation matrix."""
    c = np.cos(ax[...,3:4])
    s = np.sin(ax[...,3:4])
    omc = 1. -c
    om = np.block([c+omc*ax[...,0:1]**2,
                   omc*ax[...,0:1]*ax[...,1:2] + s*ax[...,2:3],
                   omc*ax[...,0:1]*ax[...,2:3] - s*ax[...,1:2],
                   omc*ax[...,0:1]*ax[...,1:2] - s*ax[...,2:3],
                   c+omc*ax[...,1:2]**2,
                   omc*ax[...,1:2]*ax[...,2:3] + s*ax[...,0:1],
                   omc*ax[...,0:1]*ax[...,2:3] + s*ax[...,1:2],
                   omc*ax[...,1:2]*ax[...,2:3] - s*ax[...,0:1],
                   c+omc*ax[...,2:3]**2]).reshape(ax.shape[:-1]+(3,3))
    return om if P < 0.0 else np.swapaxes(om,-1,-2)


def ax2eu(ax):
    """Rotation matrix to Bunge Euler angles."""
    return om2eu(ax2om(ax))


def ax2ro(ax):
    """Axis angle pair to Rodrigues-Frank vector."""
    ro = np.block([ax[...,:3],
                   np.where(np.isclose(ax[...,3:4],np.pi,atol=1.e-15,rtol=.0),
                            np.inf,
                            np.tan(ax[...,3:4]*0.5))
                  ])
    ro[np.abs(ax[...,3])<1.e-6] = [.0,.0,P,.0]
    return ro


def ax2ho(ax):
    """Axis angle pair to homochoric vector."""
    f = (0.75 * ( ax[...,3:4] - np.sin(ax[...,3:4]) ))**(1.0/3.0)
    ho = ax[...,:3] * f
    return ho


def ax2cu(ax):
    """Axis angle pair to cubochoric vector."""
    return ho2cu(ax2ho(ax))


#---------- Rodrigues-Frank vector ----------

def ro2qu(ro):
    """Rodrigues-Frank vector to quaternion."""
    return ax2qu(ro2ax(ro))


def ro2om(ro):
    """Rodgrigues-Frank vector to rotation matrix."""
    return ax2om(ro2ax(ro))


def ro2eu(ro):
    """Rodrigues-Frank vector to Bunge-Euler angles."""
    return om2eu(ro2om(ro))


def ro2ax(ro):
    """Rodrigues-Frank vector to axis angle pair."""
    with np.errstate(invalid='ignore',divide='ignore'):
        ax = np.where(np.isfinite(ro[...,3:4]),
             np.block([ro[...,0:3]*np.linalg.norm(ro[...,0:3],axis=-1,keepdims=True),2.*np.arctan(ro[...,3:4])]),
             np.block([ro[...,0:3],np.broadcast_to(np.pi,ro[...,3:4].shape)]))
    ax[np.abs(ro[...,3]) < 1.e-8]  = np.array([ 0.0, 0.0, 1.0, 0.0 ])
    return ax


def ro2ho(ro):
    """Rodrigues-Frank vector to homochoric vector."""
    f = np.where(np.isfinite(ro[...,3:4]),2.0*np.arctan(ro[...,3:4]) -np.sin(2.0*np.arctan(ro[...,3:4])),np.pi)
    ho = np.where(np.broadcast_to(np.sum(ro[...,0:3]**2.0,axis=-1,keepdims=True) < 1.e-8,ro[...,0:3].shape),
                  np.zeros(3), ro[...,0:3]* (0.75*f)**(1.0/3.0))
    return ho


def ro2cu(ro):
    """Rodrigues-Frank vector to cubochoric vector."""
    return ho2cu(ro2ho(ro))


#---------- Homochoric vector----------

def ho2qu(ho):
    """Homochoric vector to quaternion."""
    return ax2qu(ho2ax(ho))


def ho2om(ho):
    """Homochoric vector to rotation matrix."""
    return ax2om(ho2ax(ho))


def ho2eu(ho):
    """Homochoric vector to Bunge-Euler angles."""
    return ax2eu(ho2ax(ho))


def ho2ax(ho):
    """Homochoric vector to axis angle pair."""
    tfit = np.array([+1.0000000000018852,      -0.5000000002194847,
                     -0.024999992127593126,    -0.003928701544781374,
                     -0.0008152701535450438,   -0.0002009500426119712,
                     -0.00002397986776071756,  -0.00008202868926605841,
                     +0.00012448715042090092,  -0.0001749114214822577,
                     +0.0001703481934140054,   -0.00012062065004116828,
                     +0.000059719705868660826, -0.00001980756723965647,
                     +0.000003953714684212874, -0.00000036555001439719544])
    hmag_squared = np.sum(ho**2.,axis=-1,keepdims=True)
    hm = hmag_squared.copy()
    s = tfit[0] + tfit[1] * hmag_squared
    for i in range(2,16):
        hm *= hmag_squared
        s  += tfit[i] * hm
    with np.errstate(invalid='ignore'):
        ax = np.where(np.broadcast_to(np.abs(hmag_squared)<1.e-8,ho.shape[:-1]+(4,)),
                      [ 0.0, 0.0, 1.0, 0.0 ],
                      np.block([ho/np.sqrt(hmag_squared),2.0*np.arccos(np.clip(s,-1.0,1.0))]))
    return ax


def ho2ro(ho):
    """Axis angle pair to Rodrigues-Frank vector."""
    return ax2ro(ho2ax(ho))


def ho2cu(ho):
    """
    Homochoric vector to cubochoric vector.

    References
    ----------
    D. Roşca et al., Modelling and Simulation in Materials Science and Engineering 22:075013, 2014
    https://doi.org/10.1088/0965-0393/22/7/075013

    """
    rs = np.linalg.norm(ho,axis=-1,keepdims=True)

    xyz3 = np.take_along_axis(ho,_get_pyramid_order(ho,'forward'),-1)

    with np.errstate(invalid='ignore',divide='ignore'):
        # inverse M_3
        xyz2 = xyz3[...,0:2] * np.sqrt( 2.0*rs/(rs+np.abs(xyz3[...,2:3])) )
        qxy = np.sum(xyz2**2,axis=-1,keepdims=True)

        q2 = qxy + np.max(np.abs(xyz2),axis=-1,keepdims=True)**2
        sq2 = np.sqrt(q2)
        q = (beta/np.sqrt(2.0)/R1) * np.sqrt(q2*qxy/(q2-np.max(np.abs(xyz2),axis=-1,keepdims=True)*sq2))
        tt = np.clip((np.min(np.abs(xyz2),axis=-1,keepdims=True)**2\
            +np.max(np.abs(xyz2),axis=-1,keepdims=True)*sq2)/np.sqrt(2.0)/qxy,-1.0,1.0)
        T_inv = np.where(np.abs(xyz2[...,1:2]) <= np.abs(xyz2[...,0:1]),
                            np.block([np.ones_like(tt),np.arccos(tt)/np.pi*12.0]),
                            np.block([np.arccos(tt)/np.pi*12.0,np.ones_like(tt)]))*q
        T_inv[xyz2<0.0] *= -1.0
        T_inv[np.broadcast_to(np.isclose(qxy,0.0,rtol=0.0,atol=1.0e-12),T_inv.shape)] = 0.0
        cu = np.block([T_inv, np.where(xyz3[...,2:3]<0.0,-np.ones_like(xyz3[...,2:3]),np.ones_like(xyz3[...,2:3])) \
                              * rs/np.sqrt(6.0/np.pi),
                      ])/ sc

    cu[np.isclose(np.sum(np.abs(ho),axis=-1),0.0,rtol=0.0,atol=1.0e-16)] = 0.0
    cu = np.take_along_axis(cu,_get_pyramid_order(ho,'backward'),-1)

    return cu

#---------- Cubochoric ----------

def cu2qu(cu):
    """Cubochoric vector to quaternion."""
    return ho2qu(cu2ho(cu))


def cu2om(cu):
    """Cubochoric vector to rotation matrix."""
    return ho2om(cu2ho(cu))


def cu2eu(cu):
    """Cubochoric vector to Bunge-Euler angles."""
    return ho2eu(cu2ho(cu))


def cu2ax(cu):
    """Cubochoric vector to axis angle pair."""
    return ho2ax(cu2ho(cu))


def cu2ro(cu):
    """Cubochoric vector to Rodrigues-Frank vector."""
    return ho2ro(cu2ho(cu))


def cu2ho(cu):
    """
    Cubochoric vector to homochoric vector.

    References
    ----------
    D. Roşca et al., Modelling and Simulation in Materials Science and Engineering 22:075013, 2014
    https://doi.org/10.1088/0965-0393/22/7/075013

    """
    with np.errstate(invalid='ignore',divide='ignore'):
        # get pyramide and scale by grid parameter ratio
        XYZ = np.take_along_axis(cu,_get_pyramid_order(cu,'forward'),-1) * sc
        order = np.abs(XYZ[...,1:2]) <= np.abs(XYZ[...,0:1])
        q = np.pi/12.0 * np.where(order,XYZ[...,1:2],XYZ[...,0:1]) \
                       / np.where(order,XYZ[...,0:1],XYZ[...,1:2])
        c = np.cos(q)
        s = np.sin(q)
        q = R1*2.0**0.25/beta/ np.sqrt(np.sqrt(2.0)-c) \
          * np.where(order,XYZ[...,0:1],XYZ[...,1:2])

        T = np.block([ (np.sqrt(2.0)*c - 1.0), np.sqrt(2.0) * s]) * q

        # transform to sphere grid (inverse Lambert)
        c = np.sum(T**2,axis=-1,keepdims=True)
        s = c *         np.pi/24.0 /XYZ[...,2:3]**2
        c = c * np.sqrt(np.pi/24.0)/XYZ[...,2:3]
        q = np.sqrt( 1.0 - s)

        ho = np.where(np.isclose(np.sum(np.abs(XYZ[...,0:2]),axis=-1,keepdims=True),0.0,rtol=0.0,atol=1.0e-16),
                      np.block([np.zeros_like(XYZ[...,0:2]),np.sqrt(6.0/np.pi) *XYZ[...,2:3]]),
                      np.block([np.where(order,T[...,0:1],T[...,1:2])*q,
                                np.where(order,T[...,1:2],T[...,0:1])*q,
                                np.sqrt(6.0/np.pi) * XYZ[...,2:3] - c])
                      )

    ho[np.isclose(np.sum(np.abs(cu),axis=-1),0.0,rtol=0.0,atol=1.0e-16)] = 0.0
    ho = np.take_along_axis(ho,_get_pyramid_order(cu,'backward'),-1)

    return ho



def _get_pyramid_order(xyz,direction=None):
    """
    Get order of the coordinates.

    Depending on the pyramid in which the point is located, the order need to be adjusted.

    Parameters
    ----------
    xyz : numpy.ndarray
       coordinates of a point on a uniform refinable grid on a ball or
       in a uniform refinable cubical grid.

    References
    ----------
    D. Roşca et al., Modelling and Simulation in Materials Science and Engineering 22:075013, 2014
    https://doi.org/10.1088/0965-0393/22/7/075013

    """
    order = {'forward': np.array([[0,1,2],[1,2,0],[2,0,1]]),
             'backward':np.array([[0,1,2],[2,0,1],[1,2,0]])}

    p = np.where(np.maximum(np.abs(xyz[...,0]),np.abs(xyz[...,1])) <= np.abs(xyz[...,2]),0,
                 np.where(np.maximum(np.abs(xyz[...,1]),np.abs(xyz[...,2])) <= np.abs(xyz[...,0]),1,2))

    return order[direction][p]
