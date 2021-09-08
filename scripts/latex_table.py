from tabulate import tabulate
import glob
import numpy as np
pth = './shearflow_savez_0030/shear_updated/'

def info(files):
    u = 0.5; d = np.pi/3; beta = 1
    data = np.load(files)
    try:
        nsav = data['nsav']
    except KeyError:
        nsav = data['nhst']
    dy = data['dx']; nx = data['nx']; Ly = 2.*np.pi
    m = data['m']; alpha = data['eta']; D = data['D']
    navgy = np.sum(nsav[-1,:,:]*dy,axis=0)/Ly; ntot = np.sum(navgy*dy)*Ly
    Pe = (u*d*beta)/D; Pesh = (u*beta*d**2*m*alpha)/(2.*np.pi*D)
    return navgy, ntot, dy, Pe, Pesh, alpha, m

table = []
files = sorted(glob.glob(pth+'pst-n-pe-*.npz'))
for i,j in enumerate(files):
    if i == 0:
        navgy, ntot, dy, Pe, Pesh, alpha, m = info(j)
        n0y = navgy/np.max(navgy); sigmasq = 0
    else:
        navgy, ntot, dy, Pe, Pesh, alpha, m = info(j)
        navgy = navgy/np.max(navgy)
        sigmasq = np.sum((navgy-n0y)**2*dy)
    # interesting to divide Pesh by Pe perhaps...
    tmp_table = [i,m,alpha,int(Pe),int(Pesh),'{:.2g}'.format(ntot),'{:.2g}'.format(np.sqrt(sigmasq))]
    table.append(tmp_table)
print(tabulate(table,headers=['Case \#','$m$','$/alpha$','Pe','Pes','$n_{/text{tot}}$','$\sigma_x$'],tablefmt='latex_raw'))