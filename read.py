import matplotlib.pyplot as plt
import numpy as np
from opt_einsum import contract
import sys
import os

z = np.array([[1,1,1],[1,-1,-1],[-1,1,-1], [-1,-1,1]])/np.sqrt(3)
y = np.array([[0,-1,1],[0,1,-1],[0,-1,-1], [0,1,1]])/np.sqrt(2)
x = np.array([[-2,1,1],[-2,-1,-1],[2,1,-1], [2,-1,1]])/np.sqrt(6)

coord = np.array([x,y,z])

pyrochore_unit_cell = np.array([[-1,-1,-1],[-1,1,1],[1,-1,1],[1,1,-1]])/8
basis = np.array([[0,1,1],[1,0,1],[1,1,0]])/2
BasisBZA = np.array([2*np.pi*np.array([-1,1,1]),2*np.pi*np.array([1,-1,1]),2*np.pi*np.array([1,1,-1])])

dim1 = int(sys.argv[3])
dim2 = int(sys.argv[4])
dim3 = int(sys.argv[5])

coord = np.zeros((dim1*dim2*dim3*4,3))

for i in range(dim1):
    for j in range(dim2):
        for k in range(dim3):
            for u in range(4):
                coord[i*dim2*dim3*4+j*dim3*4+k*4+u] = basis[0]*i + basis[1]*j + basis[2]*k + pyrochore_unit_cell[u]


def readTwoBody(filename):
    f = open(filename, 'r')

dir = sys.argv[1]


# ind = np.array([1,3,5,7], dtype=int)
# spin_info = f[:,ind]ƒ

def genBZ(d, m=1):
    dj = d*1j
    b = np.mgrid[0:m:dj, 0:m:dj, 0:m:dj].reshape(3,-1).T
    return contract('ij, jk->ik', b, BasisBZA)

def gSF(q, v):
    M = np.zeros((len(q),4,4))
    a = np.cross(q, v)
    a = contract('ia, i->ia', a, 1/np.linalg.norm(a, axis=1))
    for i in range(4):
        for j in range(4):
            M[:,i,j] = contract('a, ka, b, kb->k',z[i], a, z[j], a)
    return M

def gNSF(q, v):
    M = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            M[i,j] = contract('a,a, b, b->',z[i], v, z[j], v)
    M = contract('k ,ab->kab', np.ones(len(q)), M)
    return M

def proj(k):
    delta = contract('ar, br->ab', z, z)
    return delta - contract('ar, ir, bp, ip, i->iab', z, k,
                    z, k, 1/contract('ik,ik->i', k ,k))
def projx(k):
    delta = contract('ar, br->ab', x, x)
    return delta - contract('ar, ir, bp, ip, i->iab', x, k,
                    x, k, 1/contract('ik,ik->i', k ,k))
def projy(k):
    delta = contract('ar, br->ab', y, y)
    return delta - contract('ar, ir, bp, ip, i->iab', y, k,
                    y, k, 1/contract('ik,ik->i', k ,k))

## Expectation value
#Note Sz = (C^\dagger_up C_up - C^\dagger_down C_down)/2
#Sp = C^\dagger_up C_down
#Sm = C^\dagger_down C_up
def Sz(k, f):
    indpos = np.arange(0, len(f), 4)
    indneg = np.arange(3, len(f), 4)
    CupCup = f[indpos,4] + 1j * f[indpos,5]
    CupCdown = f[indneg,4] + 1j * f[indneg,5]

    Szz = (CupCup - CupCdown)/2

    ind_pos_1 = np.array(f[indpos,0], dtype=int)
    sublattice_1 = np.mod(ind_pos_1, 4)

    here_coord_1 = coord[ind_pos_1]

    ffact = np.exp(1j*contract('ik, nk->in', k, here_coord_1))
    output = contract('n,in->in', Szz, ffact)/np.sqrt(dim1*dim2*dim3*4)
    sublattice_out = np.zeros((len(k),4),dtype=np.complex128)
    for i in range(len(ind_pos_1)):
        sublattice_out[:, sublattice_1[i]] = sublattice_out[:, sublattice_1[i]] + output[:,i]
    return sublattice_out

def Sp(k, f):
    indpos = np.arange(1, len(f), 4)
    CupCdown = f[indpos,4] + 1j * f[indpos,5]

    Sp = CupCdown

    ind_pos_1 = np.array(f[indpos,0], dtype=int)
    sublattice_1 = np.mod(ind_pos_1, 4)

    here_coord_1 = coord[ind_pos_1]

    ffact = np.exp(1j*contract('ik, nk->in', k, here_coord_1))
    output = contract('n,in->in', Sp, ffact)/np.sqrt(dim1*dim2*dim3*4)
    sublattice_out = np.zeros((len(k),4),dtype=np.complex128)
    for i in range(len(ind_pos_1)):
        sublattice_out[:, sublattice_1[i]] = sublattice_out[:, sublattice_1[i]] + output[:,i]
    return sublattice_out



def SzSz(k,f):
    Sk = Sz(k, f)
    return contract('i,i->i',Sk, np.conj(Sk))


def Szz(k, f):
    indpos = np.concatenate((np.arange(0, len(f), 8), np.arange(3, len(f), 8)))
    indneg = np.concatenate((np.arange(1, len(f), 8), np.arange(2, len(f), 8)))
    CupCup = f[indpos,8] + 1j * f[indpos,9]
    CupCdown = f[indneg,8] + 1j * f[indneg,9]

    Szz = (CupCup - CupCdown)/4

    ind_pos_1 = np.array(f[indpos,0], dtype=int)
    ind_pos_2 = np.array(f[indpos,4], dtype=int)
    # print(len(ind_pos_1))

    sublattice_1 = np.mod(ind_pos_1, 4)
    sublattice_2 = np.mod(ind_pos_2, 4)

    here_coord_1 = coord[ind_pos_1]
    here_coord_2 = coord[ind_pos_2]

    ffact= np.exp(1j*contract('ik, nk->in', k, here_coord_1-here_coord_2))

    output = np.real(contract('n,in->in', Szz, ffact))/(dim1*dim2*dim3*4)
    sublattice_out = np.zeros((len(k),4,4))
    for i in range(len(ind_pos_1)):
        sublattice_out[:, sublattice_1[i], sublattice_2[i]] = sublattice_out[:, sublattice_1[i], sublattice_2[i]] + output[:,i]
    return sublattice_out
def Spm(k, f):
    indpos = np.arange(5, len(f), 8)
    Spm_Smp = f[indpos,8] + 1j * f[indpos,9]

    ind_pos_1 = np.array(f[indpos,0], dtype=int)
    ind_pos_2 = np.array(f[indpos,4], dtype=int)

    here_coord_1 = coord[ind_pos_1]
    here_coord_2 = coord[ind_pos_2]

    sublattice_1 = np.mod(ind_pos_1, 4)
    sublattice_2 = np.mod(ind_pos_2, 4)


    ffact = np.exp(1j*contract('ik, nk->in', k, here_coord_1-here_coord_2))

    output = np.real(contract('n,in->in', Spm_Smp, ffact))/(dim1*dim2*dim3*4)
    sublattice_out = np.zeros((len(k),4,4))
    for i in range(len(ind_pos_1)):
        sublattice_out[:, sublattice_1[i], sublattice_2[i]] = sublattice_out[:, sublattice_1[i], sublattice_2[i]] + output[:,i]
    return sublattice_out

def Smp(k, f):
    indpos = np.arange(4, len(f), 8)
    Spm_Smp = f[indpos,8] + 1j * f[indpos,9]

    ind_pos_1 = np.array(f[indpos,0], dtype=int)
    ind_pos_2 = np.array(f[indpos,4], dtype=int)

    here_coord_1 = coord[ind_pos_1]
    here_coord_2 = coord[ind_pos_2]

    ffact = np.exp(1j*contract('ik, nk->in', k, here_coord_1-here_coord_2))

    return contract('n,in->i', Spm_Smp, ffact)
def Spp(k, f):
    indpos = np.arange(7, len(f), 8)
    Spp_Smm = f[indpos,8] + 1j * f[indpos,9]

    ind_pos_1 = np.array(f[indpos,0], dtype=int)
    ind_pos_2 = np.array(f[indpos,4], dtype=int)

    here_coord_1 = coord[ind_pos_1]
    here_coord_2 = coord[ind_pos_2]

    ffact = np.exp(1j*contract('ik, nk->in', k, here_coord_1-here_coord_2))

    return contract('n,in->i', Spp_Smm, ffact)
def Smm(k, f):
    indpos = np.arange(6, len(f), 8)
    Spp_Smm = f[indpos,8] + 1j * f[indpos,9]

    ind_pos_1 = np.array(f[indpos,0], dtype=int)
    ind_pos_2 = np.array(f[indpos,4], dtype=int)

    here_coord_1 = coord[ind_pos_1]
    here_coord_2 = coord[ind_pos_2]

    ffact = np.exp(1j*contract('ik, nk->in', k, here_coord_1-here_coord_2))

    return contract('n,in->i', Spp_Smm, ffact)

def Szz_global(k, f):
    indpos = np.concatenate((np.arange(0, len(f), 8), np.arange(3, len(f), 8)))
    indneg = np.concatenate((np.arange(1, len(f), 8), np.arange(2, len(f), 8)))
    CupCup = f[indpos,8] + 1j * f[indpos,9]
    CupCdown = f[indneg,8] + 1j * f[indneg,9]

    Szz = (CupCup - CupCdown)/4

    ind_pos_1 = np.array(f[indpos,0], dtype=int)
    ind_pos_2 = np.array(f[indpos,4], dtype=int)

    P = proj(k)
    sublat_ind_1 = np.mod(ind_pos_1, 4)
    sublat_ind_2 = np.mod(ind_pos_2, 4)

    realProj = np.zeros((len(k),len(sublat_ind_1)))

    for i in range(len(sublat_ind_1)):
        realProj[:, i] = P[:,sublat_ind_1[i], sublat_ind_2[i]]

    here_coord_1 = coord[ind_pos_1]
    here_coord_2 = coord[ind_pos_2]

    ffact = np.exp(1j*contract('ik, nk->in', k, here_coord_1-here_coord_2))

    return np.real(contract('n,in, in->i', Szz, ffact, realProj))

def Szz_NSF(k, f, v):
    indpos = np.concatenate((np.arange(0, len(f), 8), np.arange(3, len(f), 8)))
    indneg = np.concatenate((np.arange(1, len(f), 8), np.arange(2, len(f), 8)))
    CupCup = f[indpos,8] + 1j * f[indpos,9]
    CupCdown = f[indneg,8] + 1j * f[indneg,9]

    Szz = (CupCup - CupCdown)/4

    ind_pos_1 = np.array(f[indpos,0], dtype=int)
    ind_pos_2 = np.array(f[indpos,4], dtype=int)

    P = gNSF(k, v)
    sublat_ind_1 = np.mod(ind_pos_1, 4)
    sublat_ind_2 = np.mod(ind_pos_2, 4)

    realProj = np.zeros((len(k),len(sublat_ind_1)))

    for i in range(len(sublat_ind_1)):
        realProj[:,i] = P[:,sublat_ind_1[i], sublat_ind_2[i]]

    here_coord_1 = coord[ind_pos_1]
    here_coord_2 = coord[ind_pos_2]

    ffact = np.exp(1j*contract('ik, nk->in', k, here_coord_1-here_coord_2))

    return np.real(contract('n,in, in->i', Szz, ffact, realProj))


def Spm_global(k, f,proj_method):
    indpos = np.arange(5, len(f), 8)
    Spm_Smp = f[indpos,8] + 1j * f[indpos,9]

    ind_pos_1 = np.array(f[indpos,0], dtype=int)
    ind_pos_2 = np.array(f[indpos,4], dtype=int)

    P = proj_method(k)
    sublat_ind_1 = np.mod(ind_pos_1, 4)
    sublat_ind_2 = np.mod(ind_pos_2, 4)

    realProj = np.zeros((len(k),len(sublat_ind_1)))

    for i in range(len(sublat_ind_1)):
        realProj[:, i] = P[:,sublat_ind_1[i], sublat_ind_2[i]]

    here_coord_1 = coord[ind_pos_1]
    here_coord_2 = coord[ind_pos_2]

    ffact = np.exp(1j*contract('ik, nk->in', k, here_coord_1-here_coord_2))

    return contract('n,in,in->i', Spm_Smp, ffact, realProj)
def Smp_global(k, f, proj_method):
    indpos = np.arange(4, len(f), 8)
    Spm_Smp = f[indpos,8] + 1j * f[indpos,9]

    ind_pos_1 = np.array(f[indpos,0], dtype=int)
    ind_pos_2 = np.array(f[indpos,4], dtype=int)
    P = proj_method(k)
    sublat_ind_1 = np.mod(ind_pos_1, 4)
    sublat_ind_2 = np.mod(ind_pos_2, 4)

    realProj = np.zeros((len(k),len(sublat_ind_1)))

    for i in range(len(sublat_ind_1)):
        realProj[:, i] = P[:,sublat_ind_1[i], sublat_ind_2[i]]
    here_coord_1 = coord[ind_pos_1]
    here_coord_2 = coord[ind_pos_2]

    ffact = np.exp(1j*contract('ik, nk->in', k, here_coord_1-here_coord_2))

    return contract('n,in,in->i', Spm_Smp, ffact, realProj)
def Spp_global(k, f, proj_method):
    indpos = np.arange(7, len(f), 8)
    Spp_Smm = f[indpos,8] + 1j * f[indpos,9]

    ind_pos_1 = np.array(f[indpos,0], dtype=int)
    ind_pos_2 = np.array(f[indpos,4], dtype=int)

    P = proj_method(k)
    sublat_ind_1 = np.mod(ind_pos_1, 4)
    sublat_ind_2 = np.mod(ind_pos_2, 4)

    realProj = np.zeros((len(k),len(sublat_ind_1)))

    for i in range(len(sublat_ind_1)):
        realProj[:, i] = P[:,sublat_ind_1[i], sublat_ind_2[i]]

    here_coord_1 = coord[ind_pos_1]
    here_coord_2 = coord[ind_pos_2]

    ffact = np.exp(1j*contract('ik, nk->in', k, here_coord_1-here_coord_2))

    return contract('n,in, in->i', Spp_Smm, ffact, realProj)
def Smm_global(k, f, proj_method):
    indpos = np.arange(6, len(f), 8)
    Spp_Smm = f[indpos,8] + 1j * f[indpos,9]

    ind_pos_1 = np.array(f[indpos,0], dtype=int)
    ind_pos_2 = np.array(f[indpos,4], dtype=int)

    P = proj_method(k)
    sublat_ind_1 = np.mod(ind_pos_1, 4)
    sublat_ind_2 = np.mod(ind_pos_2, 4)

    realProj = np.zeros((len(k),len(sublat_ind_1)))

    for i in range(len(sublat_ind_1)):
        realProj[:, i] = P[:,sublat_ind_1[i], sublat_ind_2[i]]

    here_coord_1 = coord[ind_pos_1]
    here_coord_2 = coord[ind_pos_2]

    ffact = np.exp(1j*contract('ik, nk->in', k, here_coord_1-here_coord_2))

    return contract('n,in, in->i', Spp_Smm, ffact, realProj)
def Sxx(k, f):
    return np.real(Spm(k,f) + Smp(k,f) + Spp(k, f) + Smm(k, f))/4
def Syy(k, f):
    return np.real(Spm(k,f) + Smp(k,f) - Spp(k, f) - Smm(k, f))/4
def Sxy(k, f):
    return np.real((-Spm(k,f) + Smp(k,f) + Spp(k, f) - Smm(k, f))/(4j))
def Syx(k, f):
    return np.real((Spm(k,f) - Smp(k,f) + Spp(k, f) - Smm(k, f))/(4j))
def Sxx_global(k, f):
    return np.real(Spm_global(k,f, projx) + Smp_global(k,f, projx) + Spp_global(k, f, projx) + Smm_global(k, f, projx))/4
def Syy_global(k, f):
    return np.real(Spm_global(k,f, projy) + Smp_global(k,f, projy) - Spp_global(k, f, projy) - Smm_global(k, f, projy))/4
def hhltoK(H, L):
    return np.einsum('ij,k->ijk',H, 2*np.array([np.pi,np.pi,0])) \
        + np.einsum('ij,k->ijk',L, 2*np.array([0.,0.,np.pi]))

def hk2ktoK(H, L):
    return np.einsum('ij,k->ijk',H, 2*np.array([np.pi,-np.pi,0])) \
        + np.einsum('ij,k->ijk',L, 2*np.array([np.pi,np.pi,-2*np.pi]))
def hk0toK(H, L):
    return np.einsum('ij,k->ijk',H, 2*np.array([np.pi,0,0])) \
        + np.einsum('ij,k->ijk',L, 2*np.array([0.,np.pi,0.]))

def genHHLK(Hr, Lr, nK):
    H = np.linspace(-Hr, Hr, nK)
    L = np.linspace(-Lr, Lr, nK)
    A, B = np.meshgrid(H, L)
    return hhltoK(A, B).reshape((nK * nK, 3))
def genHnH2KK(Hr, Lr, nK):
    H = np.linspace(-Hr, Hr, nK)
    L = np.linspace(-Lr, Lr, nK)
    A, B = np.meshgrid(H, L)
    return hk2ktoK(A, B).reshape((nK * nK, 3))
def genHK0(Hr, Lr, nK):
    H = np.linspace(-Hr, Hr, nK)
    L = np.linspace(-Lr, Lr, nK)
    A, B = np.meshgrid(H, L)
    return hk0toK(A, B).reshape((nK * nK, 3))

nK = 100
fielddir = np.zeros(3)
if sys.argv[2] == "111":
    Hr = 3
    Lr = 3
    K = genHnH2KK(Hr, Lr, nK)
    fielddir = np.array([1,1,1])/np.sqrt(3) 
elif sys.argv[2] == "110":
    Hr = 2.5
    Lr = 2.5
    K = genHHLK(Hr, Lr, nK)
    fielddir = np.array([1,1,0])/np.sqrt(2)
else:
    Hr = 2.5
    Lr = 2.5
    K = genHK0(Hr, Lr, nK)
    fielddir = np.array([0,0,1])

def plot_line(A, B, color, ax):
    temp = np.array([A,B]).T
    ax.plot(temp[0], temp[1], color,zorder=5)

def plot_text(A, text, ax, color='black', offset_adjust = np.array([0,0])):
    temp = A + 0.08*np.array([1,1.4]) + offset_adjust
    ax.text(temp[0,0],temp[0,1],text, color=color)


def plot_BZ_hhl(offset, boundary, color, ax):
    B = boundary+offset
    plot_line(B[0],B[1],color,ax)
    plot_line(B[1],B[2],color,ax)
    plot_line(B[2],B[3],color,ax)
    plot_line(B[3],B[4],color,ax)
    plot_line(B[4],B[5],color,ax)
    plot_line(B[5],B[0],color,ax)

def plot_BZ_hkk(offset, boundary, color):
    B = boundary+offset
    plot_line(B[0],B[1],color)
    plot_line(B[1],B[2],color)
    plot_line(B[2],B[3],color)
    plot_line(B[3],B[4],color)
    plot_line(B[4], B[5], color)
    plot_line(B[5], B[6], color)
    plot_line(B[6], B[7], color)
    plot_line(B[7], B[0], color)

def SSSFgenhelper(d, hhl, ax, fig, showBZ=False):

    if hhl == "110":
        c = ax.imshow(d, interpolation="lanczos", origin='lower', extent=[-2.5, 2.5, -2.5, 2.5], aspect='equal')
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
        if showBZ:
            Gamms = np.array([[0,0],[1,1],[-1,1],[1,-1],[-1,-1],[2,0],[0,2],[-2,0],[0,-2],[2,2],[-2,2],[2,-2],[-2,-2]])
            Ls = np.array([[0.5,0.5]])
            Xs = np.array([[0,1]])
            Us = np.array([[0.25,1]])
            Ks = np.array([[0.75,0]])


            Boundary = np.array([[0.25, 1],[-0.25,1],[-0.75,0],[-0.25,-1],[0.25,-1],[0.75,0]])

            plot_BZ_hhl(Gamms[0], Boundary, 'w:',ax)

            ax.scatter(Gamms[0,0], Gamms[0,1],zorder=6)
            ax.scatter(Ls[:,0], Ls[:,1],zorder=6)
            ax.scatter(Xs[:, 0], Xs[:, 1],zorder=6)
            ax.scatter(Ks[:, 0], Ks[:, 1],zorder=6)
            ax.scatter(Us[:, 0], Us[:, 1],zorder=6)
            plot_text(Gamms,r'$\Gamma$',ax)
            plot_text(Ls,r'$L$',ax)
            plot_text(Xs,r'$X$',ax, offset_adjust=np.array([-0.3,0]))
            plot_text(Us,r'$U$',ax)
            plot_text(Ks,r'$K$',ax)
            ax.set_xlabel(r"$(h,h,0)$")
            ax.set_ylabel(r"$(0,0,l)$")
    elif hhl == "111":
        c = ax.imshow(d, interpolation="lanczos", origin='lower', extent=[-3, 3, -3, 3], aspect='auto')
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
    else:
        c = ax.imshow(d, interpolation="lanczos", origin='lower', extent=[-2.5, 2.5, -2.5, 2.5], aspect='auto')
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
    return c

def pedantic_two_body_correlation(Observables, filename,observablename):
    POT = np.array([[0,0,0],[0,0,2*np.pi]])
    twobody = np.loadtxt(dir+"/output/zvo_cisajscktalt_eigen0.dat", dtype=np.float64)
    SSSF = Observables(K, twobody)
    SSSF_POT = Observables(POT, twobody)

    SSSF = SSSF.reshape((nK,nK,4,4))
    if not os.path.isdir(dir+filename):
        os.mkdir(dir+filename)
    for i in range(4):
        for j in range(4):
            np.savetxt(dir+filename+str(i)+str(j)+".dat", SSSF[:,:,i,j])
            fig, ax = plt.subplots()
            c = SSSFgenhelper(SSSF[:,:,i,j],"110",ax,fig,True)
            fig.colorbar(c,ax=ax)
            ax.set_title(r"$"+observablename+"|_{\Gamma}="+str(SSSF_POT[0,i,j])+"$\n $"+observablename+"|_{X}="+str(SSSF_POT[1,i,j])+"$")
            plt.savefig(dir+filename+str(i)+str(j)+".pdf")
            plt.clf()
            plt.close()
    SSSF = contract('ijab->ij',SSSF)
    SSSF_POT = contract('iab->i',SSSF_POT)
    np.savetxt(dir+filename+"total.dat", SSSF)
    fig, ax = plt.subplots()
    c = SSSFgenhelper(SSSF,"110",ax,fig,True)
    fig.colorbar(c,ax=ax)
    ax.set_title(r"$"+observablename+"|_{\Gamma=(0,0,0)}="+str(SSSF_POT[0])+"$\n $"+observablename+"|_{X=(1,0,0)}="+str(SSSF_POT[1])+"$")
    plt.savefig(dir+filename+"total.pdf")
    plt.clf()
    plt.close()


def pedantic_one_body_correlation(Observables, filename,observablename):
    POT = np.array([[0,0,0],[0,0,2*np.pi]])
    onebody = np.loadtxt(dir+"/output/zvo_cisajs_eigen0.dat", dtype=np.float64)
    SSSF = np.real(Observables(K, onebody))
    SSSF_POT = Observables(POT, onebody)

    SSSF = SSSF.reshape((nK,nK,4))
    if not os.path.isdir(dir+filename):
        os.mkdir(dir+filename)
    for i in range(4):
        np.savetxt(dir+filename+str(i)+".dat", SSSF[:,:,i])
        fig, ax = plt.subplots()
        c = SSSFgenhelper(SSSF[:,:,i],"110",ax,fig,True)
        fig.colorbar(c,ax=ax)
        ax.set_title(r"$"+observablename+"|_{\Gamma}="+str(SSSF_POT[0,i])+"$\n $"+observablename+"|_{X}="+str(SSSF_POT[1,i])+"$")
        plt.savefig(dir+filename+str(i)+".pdf")
        plt.clf()
        plt.close()
    SSSF = contract('ija->ij',SSSF)
    SSSF_POT = contract('ia->i',SSSF_POT)
    np.savetxt(dir+filename+"total.dat", SSSF)
    fig, ax = plt.subplots()
    c = SSSFgenhelper(SSSF,"110",ax,fig,True)
    fig.colorbar(c,ax=ax)
    ax.set_title(r"$"+observablename+"|_{\Gamma=(0,0,0)}="+str(SSSF_POT[0])+"$\n $"+observablename+"|_{X=(1,0,0)}="+str(SSSF_POT[1])+"$")
    plt.savefig(dir+filename+"total.pdf")
    plt.clf()
    plt.close()

##Finite temperature using TPQ

def finite_temp_observables(Observables, POT, steps, num_sets, filename, observablename):
    temp = np.loadtxt(dir+"/output/Flct_rand0.dat")[:,0][19::steps]
    temp = 1/temp
    def SSSFhelper(i, set=0):
        twobody = np.loadtxt(dir+"/output/zvo_cisajscktalt_set"+str(set)+"step"+str(i)+".dat", dtype=np.float64)
        SSSF_POT = Observables(POT, twobody)
        SSSF_POT = SSSF_POT.reshape((4,4))
        return SSSF_POT

    SSSF = np.zeros((len(temp),4,4))
    if not os.path.isdir(dir+filename):
        os.mkdir(dir+filename)
    for i in range (len(temp)):
        for j in range(num_sets):
            SSSF[i] = SSSF[i] + SSSFhelper(i*20, j)/num_sets

    for i in range(4):
        for j in range(4):
            plt.plot(temp, SSSF[:,i,j])
            plt.xlabel(r"$\beta$")
            plt.xscale("log")
            plt.ylabel(r"$"+observablename+"|_{\Gamma=(0,0,0)}$")
            np.savetxt(dir+filename+str(i)+str(j)+".dat", SSSF[:,i,j])
            plt.savefig(dir+filename+str(i)+str(j)+".pdf")
            plt.clf()
            plt.close()
            
    SSSF = contract('iab->i',SSSF)
    np.savetxt(dir+filename+"total.dat", SSSF)
    plt.plot(temp, SSSF)
    plt.xlabel(r"$\beta$")
    plt.xscale("log")
    plt.ylabel(r"$"+observablename+"|_{\Gamma=(0,0,0)}$")
    plt.savefig(dir+filename+"total.pdf")
    plt.clf()
    plt.close()


finite_temp_observables(Szz, np.array([[0,0,0]]), 20, 5, "/Szz_finite_temp/", "\mathcal{S}^{zz}")

##Finite temperature calculation. Requires operator acting on eigenstates.
##In HPhi, first of all, eigenvec has dimension 2^N. In HPhi, it is enumerated by permuting the last site first i.e.
##0 0 0 0 0 0
##0 0 0 0 0 1
##0 0 0 0 1 0
##0 0 0 0 1 1
##0 0 0 1 0 0
##0 0 0 1 0 1
## ....
#Notice that this is exactly binary 0-2^N.

##creates the basis for the eigenvector output

##Let us think about all the operators we need to compute
##First of all, the one body correlation functions S^z_i = (C^\dagger_up C_up - C^\dagger_down C_down)/2
##This is easy, since this simply just gives expectation value of 1/2 if site is 0 and -1/2 if site is 1
##Strategy: for some eigenbasis |n>,  S^z_i|n> = foo(eigen_basis[n,i])|n>

##Secondly, the one body correlation functions S^+_i = C^\dagger_up C_down
##This actually changes the eigenstate by fliping the downspin to upspin. If it's an up spin to begin with it's 0
##Strategy: S^+_i|n>. 
## 1. Find eigenbasis[n]. Bit flip at site i. If spin at site i is 0 already, return 0. else proceed to step 2 with eigenvalue of 1.
## 2. Find index n' in the eigenbasis that matches the new eigenstate.
## 3. return |n'>

##Similarly, S^-_i = C^\dagger_down C_up
## 1. Find eigenbasis[n]. Bit flip at site i. If spin at site i is 1 already, return 0. else proceed to step 2 with eigenvalue of 1.
## 2. Find index n' in the eigenbasis that matches the new eigenstate.
## 3. return |n'>

# eigenbasis = np.zeros((4*dim1*dim2*dim3, 2**(4*dim1*dim2*dim3)))

# def Sz_op(eigenvec, i):
#     return 

# pedantic_two_body_correlation(Szz, "/Szz/","\mathcal{S}^{zz}")
# pedantic_two_body_correlation(Spm, "/Spm/","\mathcal{S}^{\pm}")
# pedantic_one_body_correlation(Sz, "/Sz/","\mathcal{S}^{z}")
# pedantic_one_body_correlation(Sp, "/Sp/","\mathcal{S}^{+}")

# fp = np.loadtxt(dir+"/output/zvo_eigenvec_0_rank_0.dat")
# print(fp)
# SSSF = Szz_global(K, twobody)

# SSSF = SSSF.reshape((nK,nK))
# np.savetxt(dir+"/Szz_global.dat", SSSF)

# C = plt.imshow(SSSF, interpolation="lanczos", origin='lower', extent=[-Hr, Hr, -Lr, Lr], aspect='equal')
# # plt.colorbar(C)
# # plt.ylabel(r'$(K,K,-2K)$')
# # plt.xlabel(r'$(H,-H,0)$')
# plt.xlim([-Hr, Hr])
# plt.ylim([-Lr, Lr])
# plt.savefig(dir+"/Szz_global.png")
# plt.clf()


# SSSF = Szz_NSF(K, twobody, fielddir)

# SSSF = SSSF.reshape((nK,nK))
# np.savetxt(dir+"/Szz_NSF.dat", SSSF)

# C = plt.imshow(SSSF, interpolation="lanczos", origin='lower', extent=[-Hr, Hr, -Lr, Lr], aspect='equal')
# plt.colorbar(C)
# # plt.ylabel(r'$(K,K,-2K)$')
# # plt.xlabel(r'$(H,-H,0)$')
# plt.xlim([-Hr, Hr])
# plt.ylim([-Lr, Lr])
# plt.savefig(dir+"/Szz_NSF.png")
# plt.clf()


# Bz = genBZ(30)

# Spin = Sz(Bz, onebody).reshape((1,1))
# np.savetxt(dir+"/output/Sz.dat", Spin)
