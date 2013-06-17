"""
CausticMass.py contains 3 classes/objects each with a list of attributes and functions

PhaseData:
    functions: angulardistance(), findangle(), set_sample(), gaussian_kernel()
    attributes: self.clus_ra, self.clus_dec, self.clus_z, self.r200, self.r, self.v, self.data, self.data_set,
                self.ang_d, self.angle, self.x_scale, self.y_scale, self.x_range, self.y_range, self.ksize_x, 
                self.ksize_y, self.img, self.img_grad, self.img_inf

CausticSurface:
    functions: findvdisp(), findvesc(), findphi(), findAofr(), restrict_gradient2(), identifyslot(), NFWfit()
    attributes: self.levels, self.r200, self.halo_scale_radius, self.halo_scale_radius_e, self.gal_vdisp,
                self.vvar, self.vesc, self.skr, self.level_elem, self.level_final, self.Ar_finalD,
                self.halo_scale_density, self.halo_scale_density_e, self.vesc_fit

MassCalc:
    functions:
    attributes: self.crit, self.g_b, self.conc, self.f_beta, self.massprofile, self.avg_density, self.r200_est,
                self.M200

"""

import numpy as np
import astropy

c = 300000.0

class PhaseData:
    """
    Required input: Galaxy RA,DEC,Z which must be first 3 columns in data input
    
    Optional input: Galaxy mags,memberflag   Cluster RA,DEC,Z,rlimit,vlimit,H0
    
    - if the optional Cluster inputs are not given, average values are calculated. It is far better for the user
    to calculate their own values and feed them to the module than rely on these estimates. The defaults for 
    rlimit = 4 and vlimit = +/- 3500km/s
    
    - User can submit a 2D data array if there are additional galaxy attribute columns not offered by default
    that can be carried through in the opperations for later.
    """
    
    def __init__(self,data,gal_mags=None,gal_memberflag=None,clus_ra=None,clus_dec=None,clus_z=None,r200=2.0,rlimit=4.0,vlimit=3500,q=10.0,H0=100.0,cut_sample=True):
        self.clus_ra = clus_ra
        self.clus_dec = clus_dec
        self.clus_z = clus_z
        self.r200 = r200
        if self.clus_ra == None:
            #calculate average ra from galaxies
            self.clus_ra = np.average(data[:,0])
        if self.clus_dec == None:
            #calculate average dec from galaxies
            self.clus_dec = np.average(data[:,1])
        
        #Reduce data set to only valid redshifts
        data_spec = data[np.where((np.isfinite(data[:,2])) & (data[:,2] > 0.0) & (data[:,2] < 5.0))]

        if self.clus_z == None:
            #calculate average z from galaxies
            self.clus_z = np.average(data_spec[:,2])
        
        #calculate angular diameter distance. 
        #Variable self.ang_d
        self.ang_d,self.lum_d = self.zdistance(self.clus_z,H0) 
        
        #calculate the spherical angles of galaxies from cluster center.
        #Variable self.angle
        self.angle = self.findangle(data_spec[:,0],data_spec[:,1],self.clus_ra,self.clus_dec)


        self.r = self.angle*self.ang_d
        self.v = c*(data_spec[:,2] - self.clus_z)/(1+self.clus_z)
        
        #package galaxy data, USE ASTROPY TABLE HERE!!!!!
        self.data_table = np.vstack((self.r,self.v,data_spec.T,gal_memberflag)).T
        
        #reduce sample within limits
        if cut_sample == True:
            self.data_set = set_sample(self.data_table,rlimit=rlimit,vlimit=vlimit)
        else:
            self.data_set = self.data_table

        gaussian_kernel(self.data_set[:,0],self.data_set[:,1],self.r200,normalization=H0,scale=q,xres=200,yres=220)
        self.img_tot = self.img/np.max(np.abs(img))
        self.img_grad_tot = self.img_grad/np.max(np.abs(self.img_grad))
        self.img_inf_tot += self.img_inf/np.max(np.abs(self.img_inf))

        
    def zdistance(self,clus_z,H0=100.0):
        """
        Finds the angular diameter distance for an array of cluster center redshifts.
        Instead, use angular distance file precalculated and upload.
        """
        cosmo = {'omega_M_0':0.3,'omega_lambda_0':0.7,'h':H0/100.0}
        cosmo = cd.set_omega_k_0(cosmo)
        ang_d = cd.angular_diameter_distance(clus_z,**cosmo)
        lum_d = cd.luminosity_distance(clus_z,**cosmo)
        return ang_d,lum_d
       
    def findangle(self,ra,dec,clus_RA,clus_DEC):
        """
        Calculates the angles between the galaxies and the estimated cluster center.
        The value is returned in radians.
        """
        zsep = np.sin(clus_DEC*np.pi/180.0)*np.sin(np.array(dec)*np.pi/180.0)
        xysep = np.cos(clus_DEC*np.pi/180.0)*np.cos(np.array(dec)*math.pi/180.0)*np.cos(np.pi/180.0*(clus_RA-np.array(ra)))
        angle = np.arccos(zsep+xysep)
        return angle

    def set_sample(self,data,rlimit=4.0,vlimit=3500):
        """
        Reduces the sample by selecting only galaxies inside r and v limits.
        The default is to use a vlimit = 3500km/s and rlimit = 4.0Mpc.
        Specify in parameter file.
        """
        data_set = data[np.where((data[:,0] < rlimit) & (np.abs(data[:,1]) < vlimit))]
        return data_set

    def gaussian_kernel(self,xvalues,yvalues,r200,normalization=100,scale=10,xres=200,yres=220,adj=20):
        """
        Uses a 2D gaussian kernel to estimate the density of the phase space.
        As of now, the maximum radius extends to 6Mpc and the maximum velocity allowed is 5000km/s
        The "q" parameter is termed "scale" here which we have set to 10 as default, but can go as high as 50.
        "normalization" is simply H0
        "x/yres" can be any value, but are recommended to be above 150
        "adj" is a custom value and changes the size of uniform filters when used (not normally needed)
        """
        self.x_scale = xvalues/6.0*res
        self.y_scale = ((yvalues+5000)/(normalization*scale))/(10000.0/(normalization*scale))*yres

        img = np.zeros((xres+1,yres+1))
        self.x_range = np.linspace(0,6,xres+1)
        self.y_range = np.linspace(-5000,5000,yres+1) 

        for j in range(xvalues.size):
            img[self.x_scale[j],self.y_scale[j]] += 1
        
        #Estimate kernel sizes
        #Uniform
        #self.ksize = 3.12/(xvalues.size)**(1/6.0)*((np.var(self.x_scale[xvalues<r200])+np.var(self.y_scale[xvalues<r200]))/2.0)**0.5/adj
        #if self.ksize < 3.5:
        #    self.ksize = 3.5
        #Gaussian
        self.ksize_x = (4.0/(3.0*xvalues.size))**(1/5.0)*np.std(self.x_scale[xvalues<r200])
        self.ksize_y = (4.0/(3.0*yvalues.size))**(1/5.0)*np.std(self.y_scale[xvalues<r200])
        
        #smooth with estimated kernel sizes
        #img = ndi.uniform_filter(img, (self.ksize,self.ksize))#,mode='reflect')
        self.img = ndi.gaussian_filter(img, (self.ksize_y,self.ksize_x))#,mode='reflect')
        self.img_grad = ndi.gaussian_gradient_magnitude(img, (self.ksize_y,self.ksize_x))
        self.img_inf = ndi.gaussian_gradient_magnitude(ndi.gaussian_gradient_magnitude(img, (self.ksize_y,self.ksize_x)), (self.ksize_y,self.ksize_x))




class CausticSurface:
    """
    - For now if r200 is not supplied I am using a default value of 2Mpc 
    
    - If a scale radius is not given for the cluster, then I am using a default value of r200/5.0 with uncertainty 0.01Mpc

    CausticSurface(self,r,v,ri,vi,Zi,memberflags=None,r200=2.0,maxv=5000,halo_scale_radius=None,halo_scale_radius_e=0.01,halo_vdisp=None,bin=None):

        r/v - rvalues/vvalues of galaxies

        ri/vi - x_range/y_range of grid

        Zi - density map

        memberflags = None - indices of known member galaxies to calculate a velocity dispersion

        r200 = 2.0 - critical radius of the cluster

        maxv = 5000km/s -  maximum velocity allowed

        halo_scale_radius - scale radius (default is r200/5.0)

        halo_scale_radius_e = 0.01 - uncertainty in scale radius

        halo_vdisp = None - velocity dispersion

        bin = None - if doing multiple halos, can assign an ID number
    """
    
    def __init__(self,r,v,ri,vi,Zi,memberflags=None,r200=2.0,maxv=5000.0,halo_scale_radius=None,halo_scale_radius_e=0.01,halo_vdisp=None,bin=None,plotphase=True):
        #first guess at the level
        kappaguess = np.max(Zi)
        #create levels (kappas) to try out
        self.levels = np.linspace(0.00001,kappaguess,100)[::-1]
        #when fitting an NFW (later), this defines the r range to fit within
        fitting_radii = np.where((ri>=r200/3.0) & (ri<=r200))
        
        self.r200 = r200

        if halo_scale_radius is None:
            self.halo_scale_radius = self.r200/5.0
        else:
            self.halo_scale_radius = halo_scale_radius
            self.halo_scale_radius_e = halo_scale_radius_e
        
        #Calculate velocity dispersion with either members, fed value, or estimate using 3.5sigma clipping
        if memberflags is not None:
            vvarcal = v[np.where(memberflags==True)]
            self.gal_vdisp = astStats.biweightScale(vvarcal,9.0)
            self.vvar = self.gal_vdisp**2
        if halo_vdisp is not None:
            self.gal_vdisp = use_vdisp
            self.vvar = self.gal_vdisp**2
        else:
            #Variable self.gal_vdisp
            self.findvdisp(r,v,r200,maxv)
            self.vvar = self.gal_vdisp**2
        
        #initilize arrays
        self.vesc = np.zeros(self.levels.size)
        Ar_final_opt = np.zeros((self.levels.size,ri[np.where((ri<r200) & (ri>=0))].size))
        
        #find the escape velocity for all level (kappa) guesses
        for i in range(self.vesc.size):
            self.vesc[i],Ar_final_opt[i] = self.findvesc(self.levels[i],ri,vi,Zi,r200)
        
        #difference equation to search for minimum value
        self.skr = (self.vesc-4.0*vvar)**2
        
        try:
            self.level_elem = np.where(self.skr == np.min(self.skr[np.isfinite(self.skr)]))[0][0]
            self.level_final = self.levels[level_elem]
            self.Ar_finalD = np.zeros(ri.size)
            for k in range(self.Ar_finalD.size):
                self.Ar_finalD[k] = self.findAofr(level_final,Zi[k],vi)
                if k != 0:
                    self.Ar_finalD[k] = self.restrict_gradient2(np.abs(self.Ar_finalD[k-1]),np.abs(self.Ar_finalD[k]),ri[k-1],ri[k])
        
        #This exception occurs if self.skr is entirely NAN. A flag should be raised for this in the output table
        except ValueError:
            self.Ar_finalD = np.zeros(ri.size)
        
        #fit an NFW to the resulting caustic profile.
        self.NFWfit(ri[fitting_radii],self.Ar_finalD[fitting_radii]*np.sqrt(self.gb),self.halo_scale_radius)

        if plotphase == True:
            s =figure()
            ax = s.add_subplot(111)
            ax.plot(r,v,'k.')
            ax.plot(ri,self.Ar_finalD,c='blue')
            ax.set_xlim(0,3500)

   
    def findvdisp(self,r,v,r200,maxv):
        """
        Use astLib.astStats biweight sigma clipping Scale estimator for the velocity dispersion
        """
        v_cut = v[np.where((r<r200) & (np.abs(v)<maxv))]
        self.gal_vdisp = astStats.biweightClipped(v_cut,9.0,sigmaCut=3.5)['biweightScale']

    def findvesc(self,level,ri,vi,Zi,norm,r200,g_b):
        """
        Calculate vesc^2 by first calculating the integrals in Diaf 99 which are not labeled but in 
        between Eqn 18 and 19
        """
        useri = ri[np.where((ri<r200) & (ri>=0))] #look only inside r200
        Ar = np.zeros(useri.size)
        phir = np.zeros(useri.size)
        #loop through each dr and find the caustic amplitude for the given level (kappa) passed to this function
        for i in range(useri.size):
            Ar[i] = self.findAofr(level,Zi[np.where((ri<r200) & (ri>=0))][i],vi)
            if i > -1:  #to fix the fact that the first row of Zi may be 'nan'
                #The Serra paper also restricts the gradient when the ln gradient is > 2. We use > 3
                Ar[i] = self.restrict_gradient2(np.abs(Ar[i-1]),np.abs(Ar[i]),useri[i-1],useri[i])
                philimit = np.abs(Ar[i]) #phi integral limits
                phir[i] = self.findphir(Zi[i][np.where((vi<philimit) & (vi>-philimit))],vi[np.where((vi<philimit) & (vi>-philimit))])
        return (np.trapz(Ar**2*phir,useri)/np.trapz(phir,useri),Ar)

    def findphir(self,shortZi,shortvi):
        short2Zi = np.ma.masked_array(shortZi)
        #print 'test',shortvi[np.ma.where(np.ma.getmaskarray(short2Zi)==False)]
        vi = shortvi[np.ma.where(np.ma.getmaskarray(short2Zi)==False)]
        Zi = short2Zi[np.ma.where(np.ma.getmaskarray(short2Zi)==False)]
        
        vi = vi[np.isfinite(Zi)]
        Zi = Zi[np.isfinite(Zi)]
        x = np.trapz(Zi.compressed(),vi)
        return x

    def findAofr(self,level,Zi,vgridvals):
        """
        Finds the velocity where kappa is
        """
        dens0 = Zi[np.where(vgridvals>=0)][0]
        if dens0 >= level:
            maxdens = 0.0 #v value we are centering on
            highvalues = Zi[np.where(vgridvals >= maxdens)] #density values above the center v value maxdens
            lowvalues = Zi[np.where(vgridvals < maxdens)] #density values below the center v value maxdens
            highv = vgridvals[np.where(vgridvals >= maxdens)] #v values above the center v value maxdens
            lowv = vgridvals[np.where(vgridvals < maxdens)] #v values below the center v value maxdens
            highslot = self.identifyslot(highvalues,level) #identify the velocity
            flip_lowslot = self.identifyslot(lowvalues[::-1],level)
            lowslot = lowvalues.size - flip_lowslot
            if len(lowv) == 0 or len(highv) == 0: #probably all zeros
                highamp = lowamp = 0
                return highamp
            if highslot == highv.size:
                highamp = highv[-1]
            if lowslot ==0:
                lowamp = lowv[0]
            if highslot == 0 or lowslot == lowv.size:
                highamp = lowamp = 0
            if highslot != 0 and highslot != highv.size:
                highamp = highv[highslot]-(highv[highslot]-highv[highslot-1])*(1-(highvalues[highslot-1]-level)/(highvalues[highslot-1]-highvalues[highslot]))
            if lowslot != 0 and lowslot != lowv.size:
                lowamp = lowv[lowslot-1]-(lowv[lowslot-1]-lowv[lowslot])*(1-(lowvalues[lowslot]-level)/(lowvalues[lowslot]-lowvalues[lowslot-1]))
            if np.abs(highamp) >= np.abs(lowamp):
                return lowamp
            if np.abs(highamp) < np.abs(lowamp):
                return highamp
        else: return 0 #no maximum density exists

    def restrict_gradient2(self,pastA,newA,pastr,newr):
        """
        It is necessary to restrict the gradient the caustic can change at in order to be physical
        """
        if pastA <= newA:
            if (np.log(newA)-np.log(pastA))/(np.log(newr)-np.log(pastr)) > 3.0:
                #print newA,pastA
                dr = np.log(newr)-np.log(pastr)
                return np.exp(np.log(pastA) + 2*dr)
            else: return newA
        if pastA > newA:
            if (np.log(newA)-np.log(pastA))/(np.log(newr)-np.log(pastr)) < -3.0 and pastA != 0:
                #print newA,pastA
                dr = np.log(newr)-np.log(pastr)
                return np.exp(np.log(pastA) - 2*dr)
            else: return newA

    def identifyslot(self,dvals,level):
        '''This function takes the density values for a given r grid value either above or below
        the v grid value that corresponds to the maximum density at the r slice and returns the indici
        where the level finally falls below the given level. Density values should be in order
        starting with the corresponding value to the v value closest to the maximum and working toward
        the edges (high to low density in general).'''
        slot = dvals.size - 1
        for i in range(dvals.size):
            if level >= dvals[i]:
                if i != 0:
                    slot = i-1
                    break
                    else:
                        slot = i
                        break

    def NFWfit(self,ri,Ar,halo_srad):
        min_func = lambda x,d0: np.sqrt(2*4*np.pi*4.5e-48*d0*(halo_srad)**2*np.log(1+x/halo_srad)/(x/halo_srad))*3.08e19
        v0 = np.array([1e15])
        out = curve_fit(min_func,ri,Ar,v0[:],maxfev=2000)
        self.halo_scale_density = out[0][0]
        try:
            self.halo_scale_density_e = np.sqrt(out[1][0][0])
        except:
            self.halo_scale_density_e = 1e14
        self.vesc_fit = np.sqrt(2*4*np.pi*4.5e-48*self.halo_scale_density*(halo_srad)**2*np.log(1+ri/halo_srad)/(ri/halo_srad))*3.08e19



class MassCalc:
    """
    MassCalc(self,ri,A,vdisp,r200=None,conc1=None,beta=None,fbr=None):

        ri - rgrid values

        A - caustic profile values

        vdisp - galaxy velocity dispersion

        r200 = 2.0 - critical radius of cluster. Default is 2.0, but advised to take the output r200 and rerun
        the analysis with this better estimate.

        conc1 = None - concentration of cluster. If None given then calculated from relationship

        beta = 0.2 - Anisotrpy parameter. Default value is 0.2, but a profile can be given that has same xvalues as ri.

        fbr = None - An exact guess of Fbeta by whatever means. Usually not used.

        H0 = 100.0 - Hubble constant
    """
    
    def __init__(self,ri,A,vdisp,clus_z,r200=None,conc1=None,beta=0.2,fbr=None,H0=100.0):
        "Calculate the mass profile"
        G = 6.67E-11
        solmass = 1.98892e30
        self.crit = 2.774946e11*(H0/100.0)**2.0*(0.25*(1+clus_z)**3.0 + 0.75)
        r2 = ri[ri>=0]
        A2 = A[ri>=0]
        kmMpc = 3.08568025e19
        #sum = np.zeros(A2.size)
        self.g_b = (3-2.0*beta)/(1-beta)
        if conc1 == None:
            self.conc = 4.0*(vdisp/700.0)**(-0.306)
        else:
            self.conc = conc1
        if fbr is None:
            self.f_beta = 0.5*((r2/r200*self.conc)**2)/((1+((r2/r200*self.conc)))**2*np.log(1+((r2/r200*self.conc))))*self.g_b
            self.f_beta[0] = 0
            #for i in range(A2.size-1):
            #    i += 1    
            #    sum[i] = np.trapz(f_beta[1:i+1]*(A2[1:i+1]*1000)**2,(r2[1:i+1])*kmMpc*1000)
            #    #sum[i] = np.trapz((A2[:i+1]*1000)**2,(r2[:i+1])*kmMpc*1000)
            sum = integrate.cumtrapz(self.f_beta*(A2[:f_beta.size]*1000)**2,r2[:f_beta.size]*kmMpc*1000,initial=0.0)
        else:
            self.f_beta = fbr
            self.f_beta[0] = 0
            #for i in range(A2[ri<1.9].size-1):
            #    i += 1    
            #    sum[i] = np.trapz(f_beta[1:i+1]*(A2[1:i+1]*1000)**2,(r2[1:i+1])*kmMpc*1000)
            #    #sum[i] = np.trapz((A2[:i+1]*1000)**2,(r2[:i+1])*kmMpc*1000)
            sum = integrate.cumtrapz(self.f_beta*(A2[:f_beta.size]*1000)**2,r2[:f_beta.size]*kmMpc*1000,initial=0.0)
        self.massprofile = sum/(G*solmass)
        
        #return the caustic r200
        self.avg_density = self.massprofile/(4.0/3.0*np.pi*(ri[:f_beta.size])**3.0)
        self.r200_est = (ri[:f_beta.size])[np.where(self.avg_density >= 200*self.crit)[0]+1][-1]

        if r200 is None:
            self.M200 = self.massprofile[np.where(ri[:f_beta.size] <= self.r200_est)[0][-1]]
        else:
            self.M200 = self.massprofile[np.where(ri[:f_beta.size] <= r200)[0][-1]]
            

        
