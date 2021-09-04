from tabulate import tabulate
import glob
import numpy
from rayleigh_diagnostics import G_Avgs, Shell_Avgs, GridInfo, AZ_Avgs

pth = '/pl/active/rayleigh_lib/centrifugal'
ar_dir = [pth+'/AR_0.1/',pth+'/AR_0.25/',pth+'/AR_0.5/',pth+'/AR_0.75/']

# read main_input file
def read_main(direct):
    alist = [1e-3,1e-4,1e6,1e7]
    blist = ['$10^{-3}$','$10^{-4}$','$10^6$','$10^7$']
    tmp_main = direct+'/main_input'
    file = open(tmp_main,'r')
    y = file.readlines()
    nr = int(y[1].split('=')[1])
    nt = int(y[2].split('=')[1])
    ar = float(y[5].split('=')[1].replace('d','e'))
    tst = float(y[18].split('=')[1].replace('d','e'))
    ek_tmp = float(y[105].split('=')[1].replace('d','e'))
    for i in range(len(alist)):
        if ek_tmp == alist[i]:
            ek = blist[i]
        else:
            pass
    ra_tmp = float(y[106].split('=')[1].replace('d','e'))
    for i in range(len(alist)):
        if ra_tmp == alist[i]:
            ra = blist[i]
        else:
            pass
    fr = float(y[112].split('=')[1].replace('d','e'))
    return nr,nt,ar,tst,ek,ek_tmp,ra,ra_tmp,fr

# output definitions
def rossby(path,ekman):
    tmp_gcomp = path+'/gcomp.dat'
    ga = G_Avgs(tmp_gcomp,path='')
    ke = ga.lut[401]
    ketilde = ga.lut[409]
    ketot = ga.vals[-1,ke]
    keturb = ga.vals[-1,ketilde]
    rossby_turb = '{:.4g}'.format(numpy.sqrt(2*keturb)*ekman/2)
    rossby_tot = '{:.4g}'.format(numpy.sqrt(2*ketot)*ekman/2)
    return rossby_tot, rossby_turb

def renolds(path):
    tmp_gcomp = path+'/gcomp.dat'
    ga = G_Avgs(tmp_gcomp,path='')
    ke = ga.lut[401]
    ketilde = ga.lut[409]
    ketot = ga.vals[-1,ke]
    keturb = ga.vals[-1,ketilde]
    re_tot = '{:.4g}'.format(numpy.sqrt(2*ketot))
    re_turb = '{:.4g}'.format(numpy.sqrt(2*keturb))
    return re_tot, re_turb

def nu_calc(path):
    sacomp = sorted(glob.glob(path+'/sa*'))
    s = Shell_Avgs(sacomp[-1],path='')
    radius = s.radius
    ro = radius[0]
    ri = radius[-1]
    A = (ri**2)*(ro**2)/(ro**2 - ri**2)
    # A = ro*ri # not sure if this is correct DC w/ Nick
    F = A/s.radius**2
    nr = s.nr
    weights=numpy.zeros(s.nr,dtype='float64')
    weights[0] = (s.radius[0]-s.radius[1])*0.5*radius[0]**2
    weights[nr-1] = (s.radius[nr-2]-s.radius[nr-1])*0.5*radius[nr-1]**2
    for j in range(1,nr-1):
        dr = radius[j-1]-radius[j+1]
        weights[j]=dr*radius[j]**2
    cond=s.vals[:,0,s.lut[1470]].reshape(s.nr)
    heat=s.vals[:,0,s.lut[1440]].reshape(s.nr)
    num = cond+heat
    denom = F
    num_int = numpy.sum(weights*num)
    denom_int = numpy.sum(weights*denom)
    #print(num_int, denom_int)
    nu = round(num_int/denom_int,4)
    return nu

def nu_calc_nick(path):
    grid = GridInfo(path+'/grid_info',path='')
    nr = grid.nr
    azcomp = sorted(glob.glob(path+'/azcomp*'))
    az = AZ_Avgs(azcomp[-1],path='')
    nq = az.nq
    svals = numpy.zeros((nr,4,nq),dtype='float64')
    for q in range(nq):
        for i in range(nr):
            svals[i,0,q] = numpy.sum(grid.tweights[:]*az.vals[:,i,q,0])
    az.svals=svals[:]
    s = az
    ro = s.radius[0]
    ri = s.radius[-1]
    A = (ri**2)*(ro**2)/(ro**2 - ri**2)
    fpr = 4.0*numpy.pi*s.radius[:]*s.radius[:]
    F = A/s.radius**2
    nr = s.nr
    radius = s.radius
    weights=numpy.zeros(s.nr,dtype='float64')
    weights[0] = (s.radius[0]-s.radius[1])*0.5*radius[0]**2
    weights[nr-1] = (s.radius[nr-2]-s.radius[nr-1])*0.5*radius[nr-1]**2
    for j in range(1,nr-1):
        dr = radius[j-1]-radius[j+1]
        weights[j]=dr*radius[j]**2
    cond=s.svals[:,0,s.lut[1470]].reshape(s.nr)
    heat=s.svals[:,0,s.lut[1440]].reshape(s.nr)
    num = cond+heat
    denom = F
    num_int = numpy.sum(weights*num)
    denom_int = numpy.sum(weights*denom)
    #print(num_int, denom_int)
    nu = num_int/denom_int
    return nu


table = []
case = 0
for i in range(len(ar_dir)):
    tmp_1 = sorted(glob.glob(ar_dir[i]+'ek*/'))
    for h in range(len(tmp_1)):
        tmp_2 = sorted(glob.glob(tmp_1[h]+'g0_ra*/'))
        for q in range(len(tmp_2)):
            fr_dir = sorted(glob.glob(tmp_2[q]+'fr*'))
            for k in range(len(fr_dir)):
                case += 1
                nr,nt,ar,tst,ek,ek_tmp,ra,ra_tmp,fr = read_main(fr_dir[k])
                rossby_tot,rossby_turb = rossby(fr_dir[k],ek_tmp)
                re_tot,re_turb = renolds(fr_dir[k])
                if ar==0.25 and ra_tmp==1e7:
                        nu = nu_calc_nick(fr_dir[k])
                else:
                        nu = nu_calc(fr_dir[k])
                tmp_table = [case,'$'+fr_dir[k]+'$',ar,ek,ra,fr,str(nr)+'x'+str(nt)+'x'+str(2*nt),re_tot,re_turb,rossby_tot,rossby_turb,nu]
                table.append(tmp_table)

# print(tabulate(table,headers=['AR','E','Ra','Fr','$\delta t$','nr x nt x np','Ro','$\tilde{Ro}$']))

# Output in LaTex form:
print(tabulate(table,headers=['Case \#','Path','\chi','E','Ra','Fr','$n_r$ x $n_{\\theta}$ x $n_{\phi}$','Re','$\widetilde{\\text{Re}}$','Ro','$\widetilde{\\text{Ro}}$','Nu'],tablefmt='latex_raw'))
