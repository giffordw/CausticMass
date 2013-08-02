"""
CausticMass.py contains 3 classes/objects each with a list of attributes and functions

Caustic:
    functions: zdistance(), findangle(), set_sample(), shiftgapper(), gaussian_kernel()
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
import matplotlib
matplotlib.use('Agg')
import numpy as np
import cosmolopy.distance as cd
from matplotlib.pyplot import *
from astLib import astStats
import scipy.ndimage as ndi
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


c = 300000.0

class Caustic:
    """
    Required input: Galaxy RA,DEC,Z which must be first 3 columns in data input
    
    Optional input: Galaxy mags,memberflag   Cluster RA,DEC,Z,rlimit,vlimit,H0
    
    - if the optional Cluster inputs are not given, average values are calculated. It is far better for the user
    to calculate their own values and feed them to the module than rely on these estimates. The defaults for 
    rlimit = 4 and vlimit = +/- 3500km/s
    
    - User can submit a 2D data array if there are additional galaxy attribute columns not offered by default
    that can be carried through in the opperations for later.

    data -- 2d array with columns starting with RA,DEC,Z
    """
    
    def __init__(self,data,gal_mags=None,gal_memberflag=None,clus_ra=None,clus_dec=None,clus_z=None,gal_r=None,gal_v=None,r200=2.0,rlimit=4.0,vlimit=3500,q=10.0,H0=100.0,xmax=6.0,ymax=5000.0,cut_sample=True,gapper=True,mirror=True):
        self.clus_ra = clus_ra
        self.clus_dec = clus_dec
        self.clus_z = clus_z
        self.r200 = r200
        if gal_r == None:
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
        else:
            data_spec = data[np.where(np.isfinite(gal_v))]
            self.r = gal_r
            self.v = gal_v
        
        #package galaxy data, USE ASTROPY TABLE HERE!!!!!
        if gal_memberflag is None:
            self.data_table = np.vstack((self.r,self.v,data_spec.T)).T
        else:
            self.data_table = np.vstack((self.r,self.v,data_spec.T,gal_memberflag)).T
        
        #reduce sample within limits
        if cut_sample == True:
            self.data_set = self.set_sample(self.data_table,rlimit=rlimit,vlimit=vlimit)
        else:
            self.data_set = self.data_table

        #further select sample via shifting gapper
        if gapper == True:
            self.data_set = self.shiftgapper(self.data_set)

        if mirror == True:
            print 'Calculating Density w/Mirrored Data'
            self.gaussian_kernel(np.append(self.data_set[:,0],self.data_set[:,0]),np.append(self.data_set[:,1],-self.data_set[:,1]),self.r200,normalization=H0,scale=q,xmax=xmax,ymax=ymax,xres=200,yres=220)
        else:
            print 'Calculating Density'
            self.gaussian_kernel(self.data_set[:,0],self.data_set[:,1],self.r200,normalization=H0,scale=q,xmax=xmax,ymax=ymax,xres=200,yres=220)
        self.img_tot = self.img/np.max(np.abs(self.img))
        self.img_grad_tot = self.img_grad/np.max(np.abs(self.img_grad))
        self.img_inf_tot = self.img_inf/np.max(np.abs(self.img_inf))

        #Identify initial caustic surface and members within the surface
        print 'Calculating initial surface'
        if gal_memberflag is None:
            Caustics = CausticSurface(self.data_set,self.x_range,self.y_range,self.img_tot,r200=r200)
        else:
            Caustics = CausticSurface(self.data_set,self.x_range,self.y_range,self.img_tot,memberflags=self.data_set[:,-1],r200=r200)

        self.caustic_profile = Caustics.Ar_finalD
        self.caustic_fit = Caustics.Arfit
        self.gal_vdisp = Caustics.gal_vdisp
        self.memflag = Caustics.memflag

        #Estimate the mass based off the caustic profile, beta profile (if given), and concentration (if given)
        if clus_z is not None:
            Mass = MassCalc(self.x_range,self.caustic_profile,self.gal_vdisp,self.clus_z,r200=r200)

            self.r200_est = Mass.r200_est
            self.M200_est = Mass.M200_est

            print 'r200 estimate: ',Mass.r200_est
            print 'M200 estimate: ',Mass.M200_est

            self.Ngal = self.data_set[np.where((self.memflag==1)&(self.data_set[:,0]<=self.r200_est))].shape[0]

            #calculate velocity dispersion
        try:
            self.vdisp_gal = astStats.biweightScale(self.data_set[:,1][self.memflag==1],9.0)
        except:
            try:
                self.vdisp_gal = np.std(self.data_set[:,1][self.memflag==1],ddof=1)
            except:
                self.vdisp_gal = 0.0
        '''
        self.err = 0
        for k in range(4):
            try:
                #Identify caustic surface and members within the surface
                Caustics = CausticSurface(self.data_set,self.x_range,self.y_range,self.img_tot,memberflags=self.memflag,r200=self.r200_est)
                self.caustic_profile = Caustics.Ar_finalD
                self.gal_vdisp = Caustics.gal_vdisp
                self.memflag = Caustics.memflag
                #Estimate the mass based off the caustic profile, beta profile (if given), and concentration (if given)
                Mass = MassCalc(self.x_range,self.caustic_profile,self.gal_vdisp,self.clus_z,r200=self.r200_est)
                self.r200_est = Mass.r200_est
                self.M200_est = Mass.M200_est
                print 'r200 estimate: ',Mass.r200_est
                print 'M200 estimate: ',Mass.M200_est
            except:
                #Identify caustic surface and members within the surface
                Caustics = CausticSurface(self.data_set,self.x_range,self.y_range,self.img_tot,r200=r200)
                self.caustic_profile = Caustics.Ar_finalD
                self.gal_vdisp = Caustics.gal_vdisp
                self.memflag = Caustics.memflag
                #Estimate the mass based off the caustic profile, beta profile (if given), and concentration (if given)
                Mass = MassCalc(self.x_range,self.caustic_profile,self.gal_vdisp,self.clus_z,r200=r200)
                self.r200_est = Mass.r200_est
                self.M200_est = Mass.M200_est
                print 'r200 estimate: ',Mass.r200_est
                print 'M200 estimate: ',Mass.M200_est
                self.err = 1
                break

            #calculate velocity dispersion
        try:
            self.vdisp_gal = astStats.biweightScale(self.data_set[:,1][self.memflag==1],9.0)
        except:
            try:
                self.vdisp_gal = np.std(self.data_set[:,1][self.memflag==1],ddof=1)
            except:
                self.vdisp_gal = 0.0
        '''



        
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
        xysep = np.cos(clus_DEC*np.pi/180.0)*np.cos(np.array(dec)*np.pi/180.0)*np.cos(np.pi/180.0*(clus_RA-np.array(ra)))
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

    def shiftgapper(self,data):
        npbin = 25
        gap_prev = 2000 #initialize gap size for initial comparison (must be larger to start).
        nbins = np.int(np.ceil(data[:,0].size/(npbin*1.0)))
        origsize = data[:,0].shape[0]
        data = data[np.argsort(data[:,0])] #sort by r to ready for binning
        #print 'NBINS FOR GAPPER = ', nbins
        for i in range(nbins):
            #print 'BEGINNING BIN:',str(i)
            databin = data[npbin*i:npbin*(i+1)]
            datanew = None
            nsize = databin[:,0].size
            datasize = nsize-1
            if nsize > 5:
                while nsize - datasize > 0 and datasize >= 5:
                    #print '    ITERATING'
                    nsize = databin[:,0].size
                    databinsort = databin[np.argsort(databin[:,1])] #sort by v
                    f = (databinsort[:,1])[databinsort[:,1].size-np.int(np.ceil(databinsort[:,1].size/4.0))]-(databinsort[:,1])[np.int(np.ceil(databinsort[:,1].size/4.0))]
                    gap = f/(1.349)
                    #print '    GAP SIZE', str(gap)
                    if gap < 500.0:
                        gap = 500.0
                    if gap >= 2.0*gap_prev: 
                        gap = gap_prev
                        #print '   Altered gap = %.3f'%(gap)
                    databelow = databinsort[databinsort[:,1]<=0]
                    gapbelow =databelow[:,1][1:]-databelow[:,1][:-1]
                    dataabove = databinsort[databinsort[:,1]>0]
                    gapabove = dataabove[:,1][1:]-dataabove[:,1][:-1]
                    try:
                        if np.max(gapbelow) >= gap: vgapbelow = np.where(gapbelow >= gap)[0][-1]
                        else: vgapbelow = -1
                        #print 'MAX BELOW GAP',np.max(gapbelow)
                        try: 
                            datanew = np.append(datanew,databelow[vgapbelow+1:],axis=0)
                        except:
                            datanew = databelow[vgapbelow+1:]
                    except ValueError:
                        pass
                    try:
                        if np.max(gapabove) >= gap: vgapabove = np.where(gapabove >= gap)[0][0]
                        else: vgapabove = 99999999
                        #print 'MAX ABOVE GAP',np.max(gapabove)
                        try: 
                            datanew = np.append(datanew,dataabove[:vgapabove+1],axis=0)
                        except:
                            datanew = dataabove[:vgapabove+1]
                    except ValueError:
                        pass
                    databin = datanew
                    datasize = datanew[:,0].size
                    datanew = None
                #print 'DATA SIZE OUT', databin[:,0].size
                if gap >=500.0:
                    gap_prev = gap
                else:
                    gap_prev = 500.0
                
            try:
                datafinal = np.append(datafinal,databin,axis=0)
            except:
                datafinal = databin
        #print 'GALAXIES CUT =',str(origsize-datafinal[:,0].size)
        return datafinal

    
    def gaussian_kernel(self,xvalues,yvalues,r200,normalization=100,scale=10,xres=200,yres=220,xmax=6.0,ymax=5000.0,adj=20):
        """
        Uses a 2D gaussian kernel to estimate the density of the phase space.
        As of now, the maximum radius extends to 6Mpc and the maximum velocity allowed is 5000km/s
        The "q" parameter is termed "scale" here which we have set to 10 as default, but can go as high as 50.
        "normalization" is simply H0
        "x/yres" can be any value, but are recommended to be above 150
        "adj" is a custom value and changes the size of uniform filters when used (not normally needed)
        """
        self.x_scale = xvalues/xmax*xres
        self.y_scale = ((yvalues+ymax)/(normalization*scale))/((ymax*2.0)/(normalization*scale))*yres

        img = np.zeros((xres+1,yres+1))
        self.x_range = np.linspace(0,xmax,xres+1)
        self.y_range = np.linspace(-ymax,ymax,yres+1) 

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
        self.img = ndi.gaussian_filter(img, (self.ksize_y,self.ksize_x),mode='reflect')
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
    
    def __init__(self,data,ri,vi,Zi,memberflags=None,r200=2.0,maxv=5000.0,halo_scale_radius=None,halo_scale_radius_e=0.01,halo_vdisp=None,bin=None,plotphase=True,gb=None):
        #first guess at the level
        kappaguess = np.max(Zi)
        #create levels (kappas) to try out
        self.levels = np.linspace(0.00001,kappaguess,100)[::-1]
        #when fitting an NFW (later), this defines the r range to fit within
        fitting_radii = np.where((ri>=r200/3.0) & (ri<=r200))
        if gb is None:
            self.gb = (3-2.0*0.2)/(1-0.2)

        self.r200 = r200

        if halo_scale_radius is None:
            self.halo_scale_radius = self.r200/5.0
        else:
            self.halo_scale_radius = halo_scale_radius
            self.halo_scale_radius_e = halo_scale_radius_e
        
        #Calculate velocity dispersion with either members, fed value, or estimate using 3.5sigma clipping
        if memberflags is not None:
            vvarcal = data[:,1][np.where(memberflags==1)]
            try:
                self.gal_vdisp = astStats.biweightScale(vvarcal[np.where(np.isfinite(vvarcal))],9.0)
            except:
                self.gal_vdisp = np.std(vvarcal,ddof=1)
            self.vvar = self.gal_vdisp**2
        elif halo_vdisp is not None:
            self.gal_vdisp = use_vdisp
            self.vvar = self.gal_vdisp**2
        else:
            #Variable self.gal_vdisp
            try:
                self.findvdisp(data[:,0],data[:,1],r200,maxv)
            except:
                self.gal_vdisp = np.std(data[:,1][np.where((data[:,0]<r200) & (np.abs(data[:,1])<maxv))],ddof=1)
            self.vvar = self.gal_vdisp**2
        
        #initilize arrays
        self.vesc = np.zeros(self.levels.size)
        Ar_final_opt = np.zeros((self.levels.size,ri[np.where((ri<r200) & (ri>=0))].size))
        
        #find the escape velocity for all level (kappa) guesses
        for i in range(self.vesc.size):
            self.vesc[i],Ar_final_opt[i] = self.findvesc(self.levels[i],ri,vi,Zi,r200)
        
        #difference equation to search for minimum value
        self.skr = (self.vesc-4.0*self.vvar)**2
        
        try:
            self.level_elem = np.where(self.skr == np.min(self.skr[np.isfinite(self.skr)]))[0][0]
            self.level_final = self.levels[self.level_elem]
            self.Ar_finalD = np.zeros(ri.size)
            for k in range(self.Ar_finalD.size):
                self.Ar_finalD[k] = self.findAofr(self.level_final,Zi[k],vi)
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
            ax.plot(data[:,0],data[:,1],'k.')
            for t in range(Ar_final_opt.shape[0]):
                ax.plot(ri[:Ar_final_opt[t].size],Ar_final_opt[t],c='0.4',alpha=0.5)
                ax.plot(ri[:Ar_final_opt[t].size],-Ar_final_opt[t],c='0.4',alpha=0.5)
            ax.plot(ri,self.Ar_finalD,c='blue')
            ax.plot(ri,-self.Ar_finalD,c='blue')
            ax.set_ylim(-3500,3500)
            s.savefig('plotphase.png')
            close()

        ##Output galaxy membership
        kpc2km = 3.09e16
        #fitfunc = lambda x,a,b,c,d: a*x**3 + b*x**2 + c*x + d
        try:
            fitfunc = lambda x,a,b: np.sqrt(2*4*np.pi*6.67e-20*a*(b*kpc2km)**2*np.log(1+x/b)/(x/b))
            popt,pcov = curve_fit(fitfunc,ri[np.where((ri>10)&(ri<50))],self.Ar_finalD[np.where((ri>10)&(ri<50))])
            self.Arfit = fitfunc(ri,popt[0],popt[1])
        except:
            fitfunc = lambda x,a: np.sqrt(2*4*np.pi*6.67e-20*a*(30.0*kpc2km)**2*np.log(1+x/30.0)/(x/30.0))
            popt,pcov = curve_fit(fitfunc,ri[np.where((ri>10)&(ri<50))],self.Ar_finalD[np.where((ri>10)&(ri<50))])
            self.Arfit = fitfunc(ri,popt[0])
        #self.Arfit = popt[0]*ri**3+popt[1]*ri**2+popt[2]*ri+popt[3]
        self.memflag = np.zeros(data.shape[0])
        fcomp = interp1d(ri,self.Ar_finalD)
        for k in range(self.memflag.size):
            vcompare = fcomp(data[k,0])
            if np.abs(vcompare) >= np.abs(data[k,1]):
                self.memflag[k] = 1

    def causticmembership(self,data,ri,caustics):
        self.memflag = np.zeros(data.shape[0])
        for k in range(self.memflag.size):
            diff = data[k,0]-ri
            xrange_up = ri[np.where(ri > data[k,0])][0]
            xrange_down = ri[np.where(ri <= data[k,0])][-1]
            c_up = np.abs(caustics[np.where(ri > data[k,0])])[0]
            c_down = np.abs(caustics[np.where(ri<= data[k,0])])[-1]
            slope = (c_up-c_down)/(xrange_up-xrange_down)
            intercept = c_up - slope*xrange_up
            vcompare = slope*data[k,0]+intercept
            if vcompare >= np.abs(data[k,1]):
                self.memflag[k] = 1
   
    def findvdisp(self,r,v,r200,maxv):
        """
        Use astLib.astStats biweight sigma clipping Scale estimator for the velocity dispersion
        """
        v_cut = v[np.where((r<r200) & (np.abs(v)<maxv))]
        try:
            self.gal_vdisp = astStats.biweightScale(v_cut,9.0)
        except:
            self.gal_vdisp = np.std(v_cut,ddof=1)

    def findvesc(self,level,ri,vi,Zi,r200):
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
                dr = np.log(newr)-np.log(pastr)
                return np.exp(np.log(pastA) + 2*dr)
            else: return newA
        if pastA > newA:
            if (np.log(newA)-np.log(pastA))/(np.log(newr)-np.log(pastr)) < -3.0 and pastA != 0:
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
        return slot

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
        sumtot = np.zeros(A2.size)
        self.g_b = (3-2.0*beta)/(1-beta)
        if conc1 == None:
            self.conc = 4.0*(vdisp/700.0)**(-0.306)
        else:
            self.conc = conc1
        if fbr is None:
            self.f_beta = 0.5*((r2/r200*self.conc)**2)/((1+((r2/r200*self.conc)))**2*np.log(1+((r2/r200*self.conc))))*self.g_b
            self.f_beta[0] = 0
            for i in range(A2.size-1):
                i += 1    
                sumtot[i] = np.trapz(self.f_beta[1:i+1]*(A2[1:i+1]*1000)**2,(r2[1:i+1])*kmMpc*1000)
                #sum[i] = np.trapz((A2[:i+1]*1000)**2,(r2[:i+1])*kmMpc*1000)
            #sum = integrate.cumtrapz(self.f_beta*(A2[:f_beta.size]*1000)**2,r2[:f_beta.size]*kmMpc*1000,initial=0.0)
        else:
            self.f_beta = fbr
            self.f_beta[0] = 0
            for i in range(A2[ri<1.9].size-1):
                i += 1    
                sumtot[i] = np.trapz(self.f_beta[1:i+1]*(A2[1:i+1]*1000)**2,(r2[1:i+1])*kmMpc*1000)
                #sum[i] = np.trapz((A2[:i+1]*1000)**2,(r2[:i+1])*kmMpc*1000)
            #sum = integrate.cumtrapz(self.f_beta*(A2[:f_beta.size]*1000)**2,r2[:f_beta.size]*kmMpc*1000,initial=0.0)
        self.massprofile = sumtot/(G*solmass)
        
        #return the caustic r200
        self.avg_density = self.massprofile/(4.0/3.0*np.pi*(ri[:self.f_beta.size])**3.0)
        try:
            #self.r200_est = (ri[:self.f_beta.size])[np.where(self.avg_density >= 200*self.crit)[0]+1][-1]
            finterp = interp1d(self.avg_density[::-1],ri[:self.f_beta.size][::-1])
            self.r200_est = finterp(200*self.crit)
        except IndexError:
            self.r200_est = 0.0

        
        self.M200_est = self.massprofile[np.where(ri[:self.f_beta.size] <= self.r200_est)[0][-1]]
        finterp = interp1d(ri[:self.f_beta.size],self.massprofile)
        self.M200_est = finterp(self.r200_est)
        self.M200 = self.massprofile[np.where(ri[:self.f_beta.size] <= r200)[0][-1]]
            

        
