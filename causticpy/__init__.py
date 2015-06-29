"""
Notes: Currently uses an NFW fit as the caustic surface*
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

Github: https://github.com/giffordw/CausticMass
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import cosmolopy.distance as cd
from cosmolopy import magnitudes, fidcosmo
from matplotlib.pyplot import *
import astStats
import scipy.ndimage as ndi
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from skimage import measure
import pdb
import warnings

warnings.filterwarnings('ignore')

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

    def __init__(self):
        pass
    
    def run_caustic(self,data,gal_mags=None,gal_memberflag=None,clus_ra=None,clus_dec=None,clus_z=None,gal_r=None,gal_v=None,r200=None,clus_vdisp=None,rlimit=4.0,vlimit=3500,q=10.0,H0=100.0,xmax=6.0,ymax=5000.0,cut_sample=True,edge_int_remove=False,gapper=True,mirror=True,absflag=False,inflection=False,edge_perc=0.1,fbr=0.65):
        self.S = CausticSurface()
        self.clus_ra = clus_ra
        self.clus_dec = clus_dec
        self.clus_z = clus_z
        self.fbr=fbr
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

        #calculate H(z)
        self.Hz = H0*np.sqrt(0.25*(1+self.clus_z)**3 + 0.75)
        self.hz = self.Hz / 100.0  #little h(z)

        
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

        if self.data_set.shape[0] < 2:
            print 'Encountered Error: Data set has too few elements. Check the r and v objects. Could indicate wrong cluster/galaxy positions or redshifts'
            return 0
        

        #further select sample via shifting gapper
        if gapper == True:
            self.data_set = self.shiftgapper(self.data_set)
        print 'DATA SET SIZE',self.data_set[:,0].size
        
        ##tries to identify double groups that slip through the gapper process
        #upper_max = np.max(self.data_set[:,1][np.where((self.data_set[:,1]>0.0)&(self.data_set[:,0]<1.0))])
        #lower_max = np.min(self.data_set[:,1][np.where((self.data_set[:,1]<0.0)&(self.data_set[:,0]<1.0))])
        #if np.max(np.array([upper_max,-lower_max])) > 1000.0+np.min(np.array([upper_max,-lower_max])):
        #    self.data_set = self.data_set[np.where(np.abs(self.data_set[:,1])<1000.0+np.min(np.array([upper_max,-lower_max])))]
        
        
        #measure Ngal above mag limit
        try:
            if absflag:
                abs_mag = self.data_table[:,5]
            else:
                abs_mag = self.data_table[:,7] - magnitudes.distance_modulus(self.clus_z,**fidcosmo)
            self.Ngal_1mpc = self.r[np.where((abs_mag < -19.55) & (self.r < 0.5) & (np.abs(self.v) < 3500))].size
        except IndexError:
            abs_mag = np.zeros(self.data_table[:,0].size)
            self.Ngal_1mpc = None
        
        if r200 == None:
            vdisp_prelim = astStats.biweightScale(self.data_set[:,1][np.where(self.data_set[:,0]<3.0)],9.0)
            if np.sum(abs_mag) == 0:
                r200_mean_prelim = 0.002*vdisp_prelim + 0.40
                self.r200 = r200_mean_prelim/1.7
            else:
                self.r200 = self.Ngal_1mpc**0.51*np.exp(-1.86)
            
            ##original r200 est
            #rclip,vclip = self.shiftgapper(np.vstack((self.r[np.where((self.r<3.0) & (np.abs(self.v)<3500.0))],self.v[np.where((self.r<3.0) & (np.abs(self.v)<3500.0))])).T).T
            #vdisp_prelim_1 = astStats.biweightClipped(vclip,9.0,3.0)['biweightScale']
            #rclip,vclip = self.shiftgapper(np.vstack((self.r[np.where((self.r<1.5) & (np.abs(self.v)<3500.0))],self.v[np.where((self.r<1.5) & (np.abs(self.v)<3500.0))])).T).T
            #vdisp_prelim_2 = astStats.biweightClipped(vclip,9.0,3.0)['biweightScale']
            #if vdisp_prelim_2 < 0.6*vdisp_prelim_1: vdisp_prelim = vdisp_prelim_2
            #else: vdisp_prelim = vdisp_prelim_1
            #r200_mean_prelim = 0.002*vdisp_prelim + 0.40
            #self.r200 = r200_mean_prelim/1.7
            
            if self.r200 > 3.0:
                self.r200 = 3.0
            if 3.0*self.r200 < 6.0:
                rlimit = 3.0*self.r200
            else:
                rlimit = 5.5

        else:
            self.r200 = r200
            if self.r200 > 3.0:
                self.r200 = 3.0
        print 'Pre_r200=',self.r200

        if mirror == True:
            print 'Calculating Density w/Mirrored Data'
            self.gaussian_kernel(np.append(self.data_set[:,0],self.data_set[:,0]),np.append(self.data_set[:,1],-self.data_set[:,1]),self.r200,normalization=self.Hz,scale=q,xmax=xmax,ymax=ymax)
        else:
            print 'Calculating Density'
            self.gaussian_kernel(self.data_set[:,0],self.data_set[:,1],self.r200,normalization=self.Hz,scale=q,xmax=xmax,ymax=ymax)
        self.img_tot = self.img/np.max(np.abs(self.img))
        self.img_grad_tot = self.img_grad/np.max(np.abs(self.img_grad))
        self.img_inf_tot = self.img_inf/np.max(np.abs(self.img_inf))
        
        if clus_vdisp is None:
            #self.pre_vdisp = 9.15*self.Ngal_1mpc+350.32
            #print 'Pre_vdisp=',self.pre_vdisp
            #print 'Ngal<1Mpc=',self.Ngal_1mpc
            v_cut = self.data_set[:,1][np.where((self.data_set[:,0]<self.r200) & (np.abs(self.data_set[:,1])<vlimit))]
            try:
                self.pre_vdisp2 = astStats.biweightScale(v_cut[np.where(np.isfinite(v_cut))],9.0)
            except:
                self.pre_vdisp2 = np.std(v_cut,ddof=1)
            print 'Vdisp from galaxies=',self.pre_vdisp2
            if self.data_set[:,0].size < 15: 
                self.v_unc = 0.35
                self.c_unc_sys = 0.75
                self.c_unc_int = 0.35
            elif self.data_set[:,0].size < 25 and self.data_set[:,0].size >= 15: 
                self.v_unc = 0.30
                self.c_unc_sys = 0.55
                self.c_unc_int = 0.22
            elif self.data_set[:,0].size < 50 and self.data_set[:,0].size >= 25: 
                self.v_unc = 0.23
                self.c_unc_sys = 0.42
                self.c_unc_int = 0.16
            elif self.data_set[:,0].size < 100 and self.data_set[:,0].size >= 50: 
                self.v_unc = 0.18
                self.c_unc_sys = 0.34
                self.c_unc_int = 0.105
            else: 
                self.v_unc = 0.15
                self.c_unc_sys = 0.29
                self.c_unc_int = 0.09
            
            #if self.pre_vdisp2 > 1.75*self.pre_vdisp: self.pre_vdisp_comb = 9.15*self.Ngal_1mpc+450.32
            #else:
            self.pre_vdisp_comb = self.pre_vdisp2
            
            #if self.data_set[:,1][np.where(self.data_set[:,0]<self.r200)].size >= 10:
            #    self.pre_vdisp_comb = astStats.biweightScale(self.data_set[:,1][np.where(self.data_set[:,0]<self.r200)],9.0)
            #else:
            #    self.pre_vdisp_comb = np.std(self.data_set[:,1][np.where(self.data_set[:,0]<self.r200)],ddof=1)
            #    #self.pre_vdisp_comb = (self.pre_vdisp*(self.pre_vdisp2*self.v_unc)**2+self.pre_vdisp2*118.14**2)/(118.14**2+(self.pre_vdisp2*self.v_unc)**2)
            
        else:
            self.pre_vdisp_comb = clus_vdisp
        print 'Combined Vdisp=',self.pre_vdisp_comb

        self.beta = 0.5*self.x_range/(self.x_range + self.r200/4.0)
        #Identify initial caustic surface and members within the surface
        print 'Calculating initial surface'
        if inflection == False:
            if gal_memberflag is None:
                self.S.findsurface(self.data_set,self.x_range,self.y_range,self.img_tot,r200=self.r200,halo_vdisp=self.pre_vdisp_comb,beta=None,mirror=mirror,edge_perc=edge_perc,Hz=self.Hz,edge_int_remove=edge_int_remove,q=q,plotphase=False)
            else:
                self.S.findsurface(self.data_set,self.x_range,self.y_range,self.img_tot,memberflags=self.data_set[:,-1],r200=self.r200,mirror=mirror,edge_perc=edge_perc,Hz=self.Hz,q=q)
        else:
            if gal_memberflag is None:
                self.S.findsurface_inf(self.data_set,self.x_range,self.y_range,self.img_tot,self.img_inf,r200=self.r200,halo_vdisp=self.pre_vdisp_comb,beta=None,Hz=self.Hz,q=q)
            else:
                self.S.findsurface_inf(self.data_set,self.x_range,self.y_range,self.img_tot,self.img_inf,memberflags=self.data_set[:,-1],r200=self.r200,Hz=self.Hz,q=q)

        self.caustic_profile = self.S.Ar_finalD
        self.caustic_fit = self.S.vesc_fit
        self.caustic_edge = np.abs(self.S.Ar_finalE)
        self.caustic_fit_edge = self.S.vesc_fit_e
        self.gal_vdisp = self.S.gal_vdisp
        self.memflag = self.S.memflag

        #Estimate the mass based off the caustic profile, beta profile (if given), and concentration (if given)
        if clus_z is not None:
            self.Mass = MassCalc(self.x_range,self.caustic_profile,self.gal_vdisp,self.clus_z,r200=self.r200,fbr=None,H0=H0)
            self.Mass2 = MassCalc(self.x_range,self.caustic_profile,self.gal_vdisp,self.clus_z,r200=self.r200,fbr=fbr,H0=H0)
            self.MassE = MassCalc(self.x_range,self.caustic_edge,self.gal_vdisp,self.clus_z,r200=self.r200,fbr=fbr,H0=H0)
            self.MassF = MassCalc(self.x_range,self.caustic_fit,self.gal_vdisp,self.clus_z,r200=self.r200,fbr=fbr,H0=H0)
            self.MassFE = MassCalc(self.x_range,self.caustic_fit_edge,self.gal_vdisp,self.clus_z,r200=self.r200,fbr=fbr,H0=H0)

            self.mprof = self.Mass.massprofile
            self.mprof_fbeta = self.Mass2.massprofile
            self.mprof_edge = self.MassE.massprofile
            self.r200_est = self.Mass.r200_est
            self.r200_est_fbeta = self.Mass2.r200_est
            self.r200_est_edge = self.MassE.r200_est
            self.r500_est = self.Mass.r500_est
            self.r500_est_fbeta = self.Mass2.r500_est
            self.M200_est = self.Mass.M200_est
            self.M200_est_fbeta = self.Mass2.M200_est
            self.M200_fbeta = self.Mass2.M200
            self.M200_edge = self.MassE.M200
            self.M200_edge_est = self.MassE.M200_est
            self.M200_fit = self.MassF.M200
            self.M200_fit_est = self.MassF.M200_est
            self.M200_fit_edge = self.MassFE.M200
            self.M200_fit_edge_est = self.MassFE.M200_est
            self.M500_est = self.Mass.M500_est
            self.M500_est_fbeta = self.Mass2.M500_est

            print 'r200 estimate: ',self.Mass2.r200_est
            print 'M200 estimate: ',self.Mass2.M200_est
            
            self.Ngal = self.data_set[np.where((self.memflag==1)&(self.data_set[:,0]<=self.r200_est_fbeta))].shape[0]
        
        #calculate velocity dispersion
        try:
            self.vdisp_gal = stats.biweightScale(self.data_set[:,1][self.memflag==1],9.0)
        except:
            try:
                self.vdisp_gal = np.std(self.data_set[:,1][self.memflag==1],ddof=1)
            except:
                self.vdisp_gal = 0.0
        return 1

        
    def zdistance(self,clus_z,H0=100.0):
        """
        Finds the angular diameter distance for an array of cluster center redshifts.
        Instead, use angular distance file precalculated and upload.
        """
        cosmo = {'omega_M_0':0.25,'omega_lambda_0':0.75,'h':H0/100.0}
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
        gap_prev = 2000.0 #initialize gap size for initial comparison (must be larger to start).
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
                    #print i,'    GAP SIZE', str(gap)
                    if gap < 500.0: break
                        #gap = 500.0
                    #if gap >= 2.0*gap_prev: 
                    #    gap = gap_prev
                    #    #print '   Altered gap = %.3f'%(gap)
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

    def gaussian_kernel(self,xvalues,yvalues,r200,normalization=100.0,scale=10.0,xres=200,yres=220,xmax=6.0,ymax=5000.0):
        """
        Uses a 2D gaussian kernel to estimate the density of the phase space.
        As of now, the maximum radius extends to 6Mpc and the maximum velocity allowed is 5000km/s
        The "q" parameter is termed "scale" here which we have set to 10 as default, but can go as high as 50.
        "normalization" is simply H0
        "x/yres" can be any value, but are recommended to be above 150
        "adj" is a custom value and changes the size of uniform filters when used (not normally needed)

        Parameters
        ----------
        xvalues : x-coordinates of points in phase space

        yvalues : y-coordinates of points in phase space

        r200 : Required estimate of r200 to calculate a rough dispersion

        normalization = 100 : This is equivalent to H0. Default is H0=100

        scale = 10 : "q" parameter in Diaferio 99. Literature says this can be between 10-50

        xres = 200 : x-grid resolution

        yres = 220 : y-grid resolution

        xmax = 6.0 : Maximum x-grid value. If data points exceed this amount either increase
                     this value or cut sample to be within this value.

        ymax = 5000 : Maximum/minimum y-grid value. If data points exceed this amount either increase
                     this value or cut sample to be within this value.

        Returns
        -------
        self.x_range : array of x-grid values
        self.y_range : array of y-grid values
        self.img : smoothed density image
        self.img_grad : first derivative of img
        self.img_inf : second derivative of img
        """
        if np.max(xvalues) >= xmax:
            raise Exception('Bounding Error: Please either increase your xmax value or trim your sample to be x < '+str(xmax))
        if np.max(np.abs(yvalues)) >= ymax:
            raise Exception('Bounding Error: Please either increase your ymax value or trim your sample to be y < '+str(ymax))

        yvalues = yvalues/(normalization*scale)

        self.x_range = np.arange(0,xmax,0.05)
        self.x_range_bin = np.arange(0,xmax+0.05,0.05)
        xres = self.x_range.size
        self.y_range = np.arange(-ymax/(normalization*scale),ymax/(normalization*scale),0.05)*normalization*scale
        self.y_range_bin = np.arange(-ymax/(normalization*scale),ymax/(normalization*scale)+0.05,0.05)*normalization*scale
        yres = self.y_range.size
        self.x_scale = (xvalues/xmax)*xres
        self.y_scale = ((yvalues*(normalization*scale)+ymax)/(ymax*2.0))*self.y_range.size
        #self.ksize_x = (4.0/(3.0*xvalues.size))**(1/5.0)*np.std(self.x_scale[xvalues<r200])
        self.ksize_x =  (4.0/(3.0*xvalues.size))**(1/5.0)*np.sqrt((astStats.biweightScale((self.x_scale[xvalues<r200]).copy(),9.0)**2 + astStats.biweightScale((self.y_scale[xvalues<r200]).copy(),9.0)**2)/2.0)
        self.ksize_x *= 1.0
        self.ksize_y = self.ksize_x#(4.0/(3.0*xvalues.size))**(1/5.0)*np.std(self.y_scale[xvalues<r200])
        self.imgr,xedge,yedge = np.histogram2d(xvalues,yvalues,bins=[self.x_range_bin,self.y_range_bin/(normalization*scale)])
        self.img = ndi.gaussian_filter(self.imgr, (self.ksize_x,self.ksize_y),mode='reflect')
        self.img_grad = ndi.gaussian_gradient_magnitude(self.imgr, (self.ksize_x,self.ksize_y))
        self.img_inf = ndi.gaussian_gradient_magnitude(ndi.gaussian_gradient_magnitude(self.imgr, (self.ksize_x,self.ksize_y)), (self.ksize_x,self.ksize_y))


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
    def __init__(self):
        pass
    
    def findsurface(self,data,ri,vi,Zi,memberflags=None,r200=2.0,maxv=5000.0,halo_scale_radius=None,halo_scale_radius_e=0.01,halo_vdisp=None,bin=None,plotphase=False,beta=None,mirror=True,q=10.0,Hz = 100.0,edge_perc=0.1,edge_int_remove=False):
        kappaguess = np.max(Zi) #first guess at the level
        #self.levels = np.linspace(0.00001,kappaguess,100)[::-1] #create levels (kappas) to try out
        self.levels = 10**(np.linspace(np.log10(np.min(Zi[Zi>0]/5.0)),np.log10(kappaguess),200)[::-1]) 
        fitting_radii = np.where((ri>=r200/3.0) & (ri<=r200)) #when fitting an NFW (later), this defines the r range to fit within

        self.r200 = r200

        if halo_scale_radius is None:
            self.halo_scale_radius = self.r200/5.0
        else:
            self.halo_scale_radius = halo_scale_radius
            self.halo_scale_radius_e = halo_scale_radius_e

        if beta is None:
            self.beta = 0.2+np.zeros(ri.size)
        else: self.beta = beta
        self.gb = (3-2.0*self.beta)/(1-self.beta)
        
        #Calculate velocity dispersion with either members, fed value, or estimate using 3.5sigma clipping
        if memberflags is not None:
            vvarcal = data[:,1][np.where(memberflags==1)]
            try:
                self.gal_vdisp = astStats.biweightScale(vvarcal[np.where(np.isfinite(vvarcal))],9.0)
                print 'O ya! membership calculation!'
            except:
                self.gal_vdisp = np.std(vvarcal,ddof=1)
            self.vvar = self.gal_vdisp**2
        elif halo_vdisp is not None:
            self.gal_vdisp = halo_vdisp
            self.vvar = self.gal_vdisp**2
        else:
            #Variable self.gal_vdisp
            try:
                self.findvdisp(data[:,0],data[:,1],r200,maxv)
            except:
                self.gal_vdisp = np.std(data[:,1][np.where((data[:,0]<r200) & (np.abs(data[:,1])<maxv))],ddof=1)
            self.vvar = self.gal_vdisp**2
        
        ##initilize arrays
        #self.vesc = np.zeros(self.levels.size)
        #self.Ar_final_opt = np.zeros((self.levels.size,ri[np.where((ri<r200) & (ri>=0))].size))
        #
        ##find the escape velocity for all level (kappa) guesses
        #for i in range(self.vesc.size):
        #    self.vesc[i],self.Ar_final_opt[i] = self.findvesc(self.levels[i],ri,vi,Zi,r200)
        #
        ##optimization equation to search for minimum value
        #self.skr = (self.vesc-4.0*self.vvar)**2

        #try:
        #    self.level_elem = np.where(self.skr == np.min(self.skr[np.isfinite(self.skr)]))[0][0]
        #    self.level_final = self.levels[self.level_elem]
        #    self.Ar_finalD = np.zeros(ri.size)
        #    for k in range(self.Ar_finalD.size):
        #        self.Ar_finalD[k] = self.findAofr(self.level_final,Zi[k],vi)
        #        if k != 0:
        #            self.Ar_finalD[k] = self.restrict_gradient2(np.abs(self.Ar_finalD[k-1]),np.abs(self.Ar_finalD[k]),ri[k-1],ri[k])
        #
        ##This exception occurs if self.skr is entirely NAN. A flag should be raised for this in the output table
        #except ValueError:
        #    self.Ar_finalD = np.zeros(ri.size)
        #

        #find contours (new)
        self.Ar_finalD = self.findcontours(Zi,self.levels,ri,vi,r200,self.vvar,Hz,q)
        
        data_e = data
        #remove outliers from edge calculation
        if edge_int_remove:
            try:
                data_e = self.edge_outlier_clip(data_e,ri,vi,Zi)
                print 'completed edge_outlier_clip'
            except: 
                data_e = data

        #Identify sharp phase-space edge
        numbins = 6
        perc_top = edge_perc #what percent of top velocity galaxies per/bin used to identify surface
        numrval = (data_e[:,0][data_e[:,0]< r200]).size #number of galaxies less than r200
        size_bin = int(np.ceil(numrval*1.0/numbins)) #how many galaxies are in each bin
        rsort = data_e[:,0][np.argsort(data_e[:,0])] #sort r positions
        if mirror == True:
            vsort = np.abs(data_e[:,1][np.argsort(data_e[:,0])]) #sort absolute value of velocities by r position
        else:
            vsort = data_e[:,1][np.argsort(data_e[:,0])] #same as above but not abs
        self.data_e = data_e
        mid_rbin = np.array([])
        avgmax = np.array([])
        avgmin = np.array([])
        mincomp = np.array([])
        for nn in range(numbins):
            vbin = vsort[nn*size_bin:(nn+1)*size_bin] #pick velocities in bin # nn
            if vbin.size==0:
                if nn >= 4: break
            rbin = rsort[nn*size_bin:(nn+1)*size_bin] #pick radii in bin # nn
            vemax = (vbin[np.argsort(vbin)][::-1])[:int(np.ceil(vbin[vbin>0.0].size*perc_top))] #sort by velocity -> flip array from max-min -> take first edge_perc values where v>0
            vemin = (vbin[np.argsort(vbin)])[:int(np.ceil(vbin[vbin<0.0].size*perc_top))] #sort by velocity -> take first edge_perc values where v<0
            avgmax = np.append(avgmax,np.average(vemax)) #add average of top edge_perc velocities to max array
            avgmin = np.append(avgmin,np.average(vemin)) #same as above but min array
            #take the minimum of either the above || below zero caustic
            if np.isnan(avgmax)[-1] == True: break
            if np.min(vbin) >= 0: mincomp = np.append(mincomp,avgmax[nn]) #if no negative velocities (aka, mirrored)
            else: mincomp = np.append(mincomp,np.min([np.abs(avgmin[nn]),avgmax[nn]])) #else take the minimum extreme
            mid_rbin = np.append(mid_rbin,np.median(rbin)) #take median rvalue of bin
        chi = np.array([])
        #loop through contours and find squared difference with edge extreme
        for nn in range(len(self.contours)):
            fint = interp1d(ri[ri<r200],self.contours[nn][ri<r200]) #interpolate contour
            Ar_comp = fint(mid_rbin[mid_rbin<np.max(ri[ri<r200])]) #interpolated contour
            chi = np.append(chi,np.median(np.abs(Ar_comp-mincomp[mid_rbin<np.max(ri[ri<r200])]))) #measure squared distance
        try:
            self.Ar_finalE = np.array(self.contours)[np.isfinite(chi)][np.where(chi[np.isfinite(chi)] == np.min(chi[np.isfinite(chi)]))][0] #find level with min chi value
            #self.level_finalE = ((self.levels[np.isfinite(chi)])[np.where(chi[np.isfinite(chi)] == np.min(chi[np.isfinite(chi)]))])[0] #find level with min chi value
            #self.Ar_finalE = np.zeros(ri.size)
            #for k in range(self.Ar_finalE.size):
            #    self.Ar_finalE[k] = self.findAofr(self.level_finalE,Zi[k],vi)
            #    if k != 0:
            #        self.Ar_finalE[k] = self.restrict_gradient2(np.abs(self.Ar_finalE[k-1]),np.abs(self.Ar_finalE[k]),ri[k-1],ri[k])
        except ValueError:
            self.Ar_finalE = np.zeros(ri.size)
        
        #fit an NFW to the resulting caustic profile.
        self.vesc_fit = self.NFWfit(ri[fitting_radii],self.Ar_finalD[fitting_radii]*np.sqrt(self.gb[fitting_radii]),self.halo_scale_radius,ri,self.gb)
        self.vesc_fit_e = self.NFWfit(ri[fitting_radii],self.Ar_finalE[fitting_radii]*np.sqrt(self.gb[fitting_radii]),self.halo_scale_radius,ri,self.gb)
        #set first element (which is NaN) equal to the second value
        self.vesc_fit[0] = self.vesc_fit[1]
        self.vesc_fit_e[0] = self.vesc_fit_e[1]

        if plotphase == True:
            s,ax = subplots(1,figsize=(10,7))
            #ax.pcolormesh(ri,vi,Zi.T)
            ax.plot(data[:,0],data[:,1],'k.',markersize=0.5,alpha=0.8)
            for t,con in enumerate(self.contours):
                ax.plot(ri,con,c='0.4',alpha=0.5)
                ax.plot(ri,-con,c='0.4',alpha=0.5)
            ax.plot(ri,self.Ar_finalD,c='red')
            ax.plot(ri,-self.Ar_finalD,c='red')
            ax.plot(ri,self.Ar_finalE,c='blue')
            #ax.plot(mid_rbin,avgmax,c='r')
            ax.set_ylim(0,5000)
            ax.set_xlim(0,4)
            s.savefig('plotphase.png')
            close()
            #show()

        ##Output galaxy membership
        kpc2km = 3.09e16
        try:
            fitfunc = lambda x,a,b: np.sqrt(2*4*np.pi*6.67e-20*a*(b*kpc2km)**2*np.log(1+x/b)/(x/b))
            self.popt,self.pcov = curve_fit(fitfunc,ri,self.Ar_finalD,p0=[5e14,1])
            self.Arfit = fitfunc(ri,self.popt[0],self.popt[1])
        except:
            fitfunc = lambda x,a: np.sqrt(2*4*np.pi*6.67e-20*a*(30.0*kpc2km)**2*np.log(1+x/30.0)/(x/30.0))
            self.popt,pcov = curve_fit(fitfunc,ri,self.Ar_finalD)
            self.Arfit = fitfunc(ri,self.popt[0])
        self.memflag = np.zeros(data.shape[0])
        #fcomp = interp1d(ri,self.Ar_finalD)
        #print ri.size, self.vesc_fit.size
        fcomp = interp1d(ri,self.vesc_fit)
        for k in range(self.memflag.size):
            vcompare = fcomp(data[k,0])
            if np.abs(vcompare) >= np.abs(data[k,1]):
                self.memflag[k] = 1

    def edge_outlier_clip(self,data_e,ri,vi,Zi):
            r_inside = []
            v_inside = []
            i = 0
            while ri[i] <= np.max(data_e[:,0]):
                inner_el = i
                outer_el = i + 5
                inner_r = ri[inner_el]
                outer_r = ri[outer_el]
                '''
                dens = np.average(Zi[inner_el:outer_el],axis=0)
                roots = np.sort(np.abs(vi[dens>0.05]))
                databinned = data_e[np.where((data_e[:,0]>=inner_r)&(data_e[:,0]<outer_r))]
                if len(roots) == 0:
                    root = 2 * astStats.biweightScale(databinned[:,1].copy(),9.0)
                elif np.abs(roots[-1]) < 500.0:
                    root = 2 * astStats.biweightScale(databinned[:,1].copy(),9.0)
                elif np.abs(roots[-1]) > 3500.0:
                    root = 3500.0
                else:
                    root = np.abs(roots[-1])
                r_inside.extend(databinned[:,0][np.where(np.abs(databinned[:,1])<root)])
                v_inside.extend(databinned[:,1][np.where(np.abs(databinned[:,1])<root)])
                i += 5
            data_e = np.vstack((np.array(r_inside),np.array(v_inside))).T
            return data_e
                '''
                deriv = (np.average(Zi[inner_el:outer_el],axis=0)[1:]-np.average(Zi[inner_el:outer_el],axis=0)[:-1]) \
                            /(vi[1:]-vi[:-1])
                roots = np.sort(np.abs(vi[((np.average(Zi[inner_el:outer_el],axis=0)[1:]- \
                    np.average(Zi[inner_el:outer_el],axis=0)[:-1])/(vi[1:]- \
                    vi[:-1]))[1:]*((np.average(Zi[inner_el:outer_el],axis=0)[1:]- \
                    np.average(Zi[inner_el:outer_el],axis=0)[:-1])/(vi[1:]-vi[:-1]))[:-1] < 0]))
                databinned = data_e[np.where((data_e[:,0]>=inner_r)&(data_e[:,0]<outer_r))]
                if len(roots) > 1:
                    if roots[1] < 1000.0:
                        if len(roots) > 2:
                            if roots[2] < 1000.0:
                                root = 3 * astStats.biweightScale(databinned[:,1].copy(),9.0)
                            else:
                                root = roots[2]
                        else: root = 3 * astStats.biweightScale(databinned[:,1].copy(),9.0)
                    else: root = roots[1]
                else: root = 3500.0
                r_inside.extend(databinned[:,0][np.where(np.abs(databinned[:,1])<root)])
                v_inside.extend(databinned[:,1][np.where(np.abs(databinned[:,1])<root)])
                i += 5
            data_e = np.vstack((np.array(r_inside),np.array(v_inside))).T
            return data_e
    

    def findsurface_inf(self,data,ri,vi,Zi,Zi_inf,memberflags=None,r200=2.0,maxv=5000.0,halo_scale_radius=None,halo_scale_radius_e=0.01,halo_vdisp=None,beta=None):
        """
        Identifies the caustic surface using the iso-density contours in phase space, 
        as well as the second derivative of the density (aptly named the inflection technique).
        This technique attempts to rid the caustic technique of the dreaded velocity dispersion
        calibration that is used to pick a surface.
        

        Parameters
        ----------
        data : first and second columns must be radius and velocity

        ri : x-grid values

        vi : y-grid values

        Zi : density image

        Zi_inf : second derivative of the density image

        memberflags = None : array of 1's if member 0's if not

        r200 = 2.0 : r200 value

        maxv = 5000.0 : maximum y-value

        halo_scale_radius = None : The default is actually a concentration of 5.0 
                                   which is applied later if None is given.

        halo_scale_radius_e=0.01 : error in halo_scale_radius

        halo_vdisp = None : supply cluster velocity dispersion

        beta = None : The default is actually 0.2 which is applied later in the code
                      although as of now beta is not used in this function

        Variables
        ---------
        
        """
        kappaguess = np.max(Zi)   #first thing is to guess at the level
        self.levels = np.linspace(0.00001,kappaguess,100)[::-1] #create levels (kappas) to try out
        fitting_radii = np.where((ri>=r200/3.0) & (ri<=r200))
        
        self.r200 = r200

        if halo_scale_radius is None:
            self.halo_scale_radius = self.r200/5.0
        else:
            self.halo_scale_radius = halo_scale_radius
            self.halo_scale_radius_e = halo_scale_radius_e
        
        #c_guess = np.array([halo_srad])#np.linspace(1.0,12.0,100)
        #density_guess = np.linspace(1e13,5e16,1000)
        
        if beta is None:
            self.beta = 0.2+np.zeros(ri.size)
        else: self.beta = beta
        self.gb = (3-2.0*self.beta)/(1-self.beta)
        
        #Calculate velocity dispersion with either members, fed value, or estimate using 3.5sigma clipping
        if memberflags is not None:
            vvarcal = data[:,1][np.where(memberflags==1)]
            try:
                self.gal_vdisp = astStats.biweightScale(vvarcal[np.where(np.isfinite(vvarcal))],9.0)
                print 'O ya! membership calculation!'
            except:
                self.gal_vdisp = np.std(vvarcal,ddof=1)
            self.vvar = self.gal_vdisp**2
        elif halo_vdisp is not None:
            self.gal_vdisp = halo_vdisp
            self.vvar = self.gal_vdisp**2
        else:
            #Variable self.gal_vdisp
            try:
                self.findvdisp(data[:,0],data[:,1],r200,maxv)
            except:
                self.gal_vdisp = np.std(data[:,1][np.where((data[:,0]<r200) & (np.abs(data[:,1])<maxv))],ddof=1)
            self.vvar = self.gal_vdisp**2
        
        self.Ar_final_opt = np.zeros((self.levels.size,ri[np.where((ri<r200) & (ri>=0))].size)) #2D array: density levels x velocity profile
        self.inf_vals = np.zeros((self.levels.size,ri[np.where((ri<r200) & (ri>=0))].size)) #2D array: density levels x inflection profile
        #s = figure()
        #ax = s.add_subplot(111)
        for i in range(self.levels.size): # find the escape velocity for all level (kappa) guesses
            self.Ar_final_opt[i],self.inf_vals[i] = self.findvesc2(self.levels[i],ri,vi,Zi,Zi_inf,r200)
            #ax.plot(ri[np.where((ri<r200) & (ri>=0))],np.abs(self.Ar_final_opt[i]),c='black',alpha=0.4) #plot each density contour
        self.inf_avg = np.average(self.inf_vals.T[fitting_radii],axis=0) #average inflection along each contour surface
        self.Ar_avg = np.average(self.Ar_final_opt,axis=1) #average velocity along each contour surface inside r200
        
        #Need to identify maximum average inflection, so smooth the measurement. Might want to do this a non-parametric way
        #tryfit = np.polyfit(self.levels,self.inf_avg,7)
        #self.infyvals = tryfit[0]*self.levels**7+tryfit[1]*self.levels**6+tryfit[2]*self.levels**5+tryfit[3]*self.levels**4+tryfit[4]*self.levels**3+tryfit[5]*self.levels**2+tryfit[6]*self.levels+tryfit[7]
        tryfit = np.polyfit(self.Ar_avg,self.inf_avg,7)
        self.infyvals = tryfit[0]*self.Ar_avg**7+tryfit[1]*self.Ar_avg**6+tryfit[2]*self.Ar_avg**5+tryfit[3]*self.Ar_avg**4+tryfit[4]*self.Ar_avg**3+tryfit[5]*self.Ar_avg**2+tryfit[6]*self.Ar_avg+tryfit[7]
        
        self.inf_std = np.std(self.inf_vals.T[fitting_radii],axis=0) #std of inflection along each caustic surface
        #self.level_elem = (self.levels[Ar_avg>np.sqrt(vvar)])[np.where(self.inf_avg[Ar_avg>np.sqrt(vvar)] == np.max(self.inf_avg[Ar_avg>np.sqrt(vvar)]))]
        self.level_elem = self.levels[np.where(self.inf_avg == np.max(self.inf_avg))][0]
        #low_zone = np.where((np.average(np.abs(self.Ar_final_opt),axis=1)>np.max(v)/2.0) & (np.average(np.abs(self.Ar_final_opt),axis=1)<np.max(v)))
        high_zone = np.where((np.average(np.abs(self.Ar_final_opt),axis=1)>np.max(data[:,1])/2.0))
        #level_elem_low = self.levels[low_zone][np.where(self.inf_avg[low_zone] == np.min(self.inf_avg[low_zone]))][-1]
        #level_elem_high = self.levels[high_zone][np.where(self.inf_avg[high_zone] == np.max(self.inf_avg[high_zone]))][-1]
        try:
            self.level_elem_high = (self.levels[1:-1][np.where((self.infyvals[1:-1]>self.infyvals[2:])&(self.infyvals[1:-1]>self.infyvals[:-2]))])[-1]
        except IndexError:
            self.level_elem_high = self.levels[0]
        self.Ar_final_high = np.zeros(ri.size)
        #self.Ar_final_low = np.zeros(ri.size)
        for i in range(ri.size):
            self.Ar_final_high[i] = self.findAofr(self.level_elem_high,Zi[i],vi)
            #self.Ar_final_low[i] = self.findAofr(level_elem_low,Zi[i],vi)
            if i > 0:
                self.Ar_final_high[i] = self.restrict_gradient2(np.abs(self.Ar_final_high[i-1]),np.abs(self.Ar_final_high[i]),ri[i-1],ri[i])
                #self.Ar_final_low[i] = self.restrict_gradient2(np.abs(self.Ar_final_low[i-1]),np.abs(self.Ar_final_low[i]),ri[i-1],ri[i])
        #Ar_final = self.Ar_final_opt[np.where(self.inf_avg == np.max(self.inf_avg))][0]
        #self.Ar_final = (self.Ar_final_high+self.Ar_final_low)/2.0
        self.Ar_finalD = self.Ar_final_high

        ##Output galaxy membership
        kpc2km = 3.09e16
        try:
            fitfunc = lambda x,a,b: np.sqrt(2*4*np.pi*6.67e-20*a*(b*kpc2km)**2*np.log(1+x/b)/(x/b))
            self.popt,self.pcov = curve_fit(fitfunc,ri,self.Ar_final)
            self.vesc_fit = fitfunc(ri,self.popt[0],self.popt[1])
        except:
            fitfunc = lambda x,a: np.sqrt(2*4*np.pi*6.67e-20*a*(30.0*kpc2km)**2*np.log(1+x/30.0)/(x/30.0))
            self.popt,self.pcov = curve_fit(fitfunc,ri,self.Ar_finalD)
            self.vesc_fit = fitfunc(ri,self.popt[0])

        self.memflag = np.zeros(data.shape[0])
        #fcomp = interp1d(ri,self.Ar_finalD)
        #print ri.size, self.vesc_fit.size
        fcomp = interp1d(ri,self.vesc_fit)
        for k in range(self.memflag.size):
            vcompare = fcomp(data[k,0])
            if np.abs(vcompare) >= np.abs(data[k,1]):
                self.memflag[k] = 1
        
        #ax.plot(ri,np.abs(self.Ar_final),c='red',lw=2)
        #ax.plot(ri,vesc_fit,c='green',lw=2)
        #ax.plot(r,v,'k.')
        #pcolormesh(ri,vi,Zi_inf.T)
        #ax.set_ylim(0,3500)
        #savefig('/nfs/christoq_ls/giffordw/flux_figs/surfacetests/nideal/'+str(bin-1)+'.png')
        #close()

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
            self.gal_vdisp = astStats.biweightScale(v_cut[np.where(np.isfinite(v_cut))],9.0)
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

    def findvesc2(self,level,ri,vi,Zi,Zi_inf,r200):
        """
        Used by findsurface_inf to identify caustic surfaces

        Parameters
        ----------
        level = density value

        ri = x-grid values

        vi = y-grid values

        Zi = density image

        Zi_inf = second derivative of density image

        r200 = r200 of cluster

        Returns
        -------
        (Ar,inf_val)

        Ar = caustic surface

        inf_val = inflection values along caustic surface
        """
        useri = ri[np.where((ri<r200) & (ri>=0))] #look only inside r200
        Ar = np.zeros(useri.size)
        inf_val = np.zeros(useri.size)
        for i in range(useri.size):
            Ar[i] = self.findAofr(level,Zi[np.where((ri<r200) & (ri>=0))][i],vi)
            if i >0:
                Ar[i] = self.restrict_gradient2(np.abs(Ar[i-1]),np.abs(Ar[i]),useri[i-1],useri[i])
            inf_val[i] = Zi_inf[i][np.where(np.abs(vi-Ar[i]) == np.min(np.abs(vi-Ar[i])))][0]
        return Ar,inf_val

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
        #dens0 = Zi[np.where(vgridvals>=0)][0]
        dens0 = np.max(Zi)
        #if dens0:#dens0 >= level:
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
        else:
            return 0 #no maximum density exists

    def restrict_gradient2(self,pastA,newA,pastr,newr):
        """
        It is necessary to restrict the gradient the caustic can change at in order to be physical
        """
        gradu = 0.5
        gradd = 2.0
        if pastA <= newA:
            if (np.log(newA)-np.log(pastA))/(np.log(newr)-np.log(pastr)) > gradu and pastA != 0:
                dr = np.log(newr)-np.log(pastr)
                return np.exp(np.log(pastA) + gradu*dr)
            else: return newA
        if pastA > newA:
            if (np.log(newA)-np.log(pastA))/(np.log(newr)-np.log(pastr)) < -gradd and pastA != 0:
                dr = np.log(newr)-np.log(pastr)
                return np.exp(np.log(pastA) - gradd*dr)
            else: return newA

    def identifyslot(self,dvals,level):
        '''This function takes the density values for a given r grid value either above or below
        the v grid value that corresponds to the maximum density at the r slice and returns the indici
        where the level finally falls below the given level. Density values should be in order
        starting with the corresponding value to the v value closest to the maximum and working toward
        the edges (high to low density in general).'''
        
        slot = dvals.size - 1
        if len(dvals[dvals>level])== 0:
            slot = 0
            return slot
        for i in range(dvals.size):
            if dvals[i] == 0.0:
                continue
            if i < np.where(dvals>level)[0][0]:
                continue
            if level >= dvals[i]:
                if i != 0:
                    slot = i-1
                    break
                else:
                    slot = i
                    break
        return slot
        

    def NFWfit(self,ri,Ar,halo_srad,ri_full,g_b):
        min_func = lambda x,d0: np.sqrt(2*4*np.pi*4.5e-48*d0*(halo_srad)**2*np.log(1+x/halo_srad)/(x/halo_srad))*3.08e19
        v0 = np.array([1e15])
        out = curve_fit(min_func,ri,Ar,v0[:],maxfev=2000)
        self.halo_scale_density = out[0][0]
        try:
            self.halo_scale_density_e = np.sqrt(out[1][0][0])
        except:
            self.halo_scale_density_e = 1e14
        return np.sqrt(2*4*np.pi*4.5e-48*self.halo_scale_density*(halo_srad)**2*np.log(1+ri_full/halo_srad)/(ri_full/halo_srad))*3.08e19/np.sqrt(g_b)

    def findcontours(self,Zi,levels,ri,vi,r200,vvar,Hz=100.0,q=10):
        '''This function will use skimage find_contours() to locate escape surfaces'''
        self.contours = [] #initialize contour array
        rspace = ri[1] - ri[0] #find r spacing
        for i,level in enumerate(levels):
            fcontours = measure.find_contours(Zi, level)
            
            for j,contour in enumerate(fcontours): #sometimes 1 level has more than one contour
                #rescale x & y
                xcont = contour[:, 0]*rspace
                ycont = (contour[:, 1]-vi.size/2.0 - 1)*Hz*q*rspace
                
                #only consider contours that are "full" and don't loop back only in positive or negative space
                if np.max(xcont) >= 0.4 and np.min(xcont) <=0.05 and np.max(ycont) > 0 and np.min(ycont) < 0:
                    xcont_u, ycont_u = xcont[ycont > 0],ycont[ycont > 0] #find positive/negative contours
                    xcont_d, ycont_d = xcont[ycont < 0],ycont[ycont < 0]
                    y_u = np.zeros(ri.size) #initialize positive, negative, and final arrays
                    y_d = np.zeros(ri.size)
                    y_f = np.zeros(ri.size)
                    
                    for k,xplace in enumerate(ri): #loop over r grid values
                        #match contour grid to r grid (nearest neighbor interpolate)
                        try: y_u[k] = ycont_u[np.where((xcont_u>xplace-0.01)&(xcont_u<xplace+0.01))].max()
                        except: y_u[k] = 0.0

                        try: y_d[k] = ycont_d[np.where((xcont_d>xplace-0.01)&(xcont_d<xplace+0.01))].max()
                        except: y_d[k] = 0.0
                        
                        #apply gradient restriction for positive and negative cases.
                        if k != 0:
                            y_u[k] = self.restrict_gradient2(np.abs(y_u[k-1]),np.abs(y_u[k]),ri[k-1],ri[k])
                            y_d[k] = self.restrict_gradient2(np.abs(y_d[k-1]),np.abs(y_d[k]),ri[k-1],ri[k])

                        y_f[k] = np.min([y_u[k],np.abs(y_d[k])]) #take minimum value of positive and negative arrays
                    self.contours.append(y_f)

        #now I need to do the average calculation in Diaferio 99
        #because an integral is involved, I don't want to do this for all contours.
        #instead I select the 25% around the preliminary closest average and do
        #the full calculation for them
        avg_contours = np.average(np.array(self.contours).T[ri <= r200]**2.0,axis=0) #prelim avg
        avg_cont_diff = (avg_contours - 4.0*vvar)**2.0 #prelim diff calc
        i_sort = np.argsort(avg_cont_diff) #sort indices based on prelim diff
        i_sort_small = i_sort[:np.int(i_sort.size/4.0)]
        tot_avg = np.zeros(i_sort_small.size)
        for i,isrt in enumerate(i_sort_small):
            Ar = self.contours[isrt]
            lessr200 = np.where(ri <= r200)
            useri = ri[lessr200]
            Ar = Ar[lessr200]
            phir = np.zeros(useri.size)
            for j in range(useri.size):
                philimit = np.abs(Ar[j]) #phi integral limits
                phir[j] = np.sum(Zi[j][np.where((vi<philimit) & (vi>-philimit))])
                #phir[j] = self.findphir(Zi[j][np.where((vi<philimit) & (vi>-philimit))],vi[np.where((vi<philimit) & (vi>-philimit))])
            #print np.trapz(phir,useri)
            #tot_avg[i] = np.trapz(Ar**2*phir,useri) / np.trapz(phir,useri)
            tot_avg[i] = np.sum(Ar**2 * phir) / np.sum(phir)
        final_contour = self.contours[i_sort_small[((tot_avg - 4.0*vvar)**2.0).argmin()]]
        print 'complete'
        return final_contour
        


        



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
    
    def __init__(self,ri,A,vdisp,clus_z,r200=None,conc1=None,beta=0.25,fbr=None,H0=100.0):
        "Calculate the mass profile"
        G = 6.67E-11
        solmass = 1.98892e30
        self.crit = 2.7745946e11*(H0/100.0)**2.0*(0.25*(1+clus_z)**3.0 + 0.75)
        r2 = ri[ri>=0]
        A2 = A[ri>=0]
        kmMpc = 3.08568025e19
        sumtot = np.zeros(A2.size)
        #print 'Using beta = %.2f'%(beta)
        if conc1 == None:
            #self.conc = 4.0*(vdisp/700.0)**(-0.306)
            self.conc = 5.0 + np.random.normal(0,2.0)
            if self.conc <= 0: self.conc = 5.0
        else:
            self.conc = conc1
        beta = 0.5*(ri/(ri+r200/self.conc))
        self.g_b = (3-2.0*beta)/(1-beta)
        if fbr is None:
            self.f_beta = 0.5*((r2/r200*self.conc)**2)/((1+((r2/r200*self.conc)))**2*np.log(1+((r2/r200*self.conc))))*self.g_b
            self.f_beta[0] = 0
            for i in range(A2.size-1):
                i += 1    
                sumtot[i] = np.trapz(self.f_beta[1:i+1]*(A2[1:i+1]*1000)**2,(r2[1:i+1])*kmMpc*1000)
                #sum[i] = np.trapz((A2[:i+1]*1000)**2,(r2[:i+1])*kmMpc*1000)
            #sum = integrate.cumtrapz(self.f_beta*(A2[:f_beta.size]*1000)**2,r2[:f_beta.size]*kmMpc*1000,initial=0.0)
        else:
            if type(fbr) == float or type(fbr) == int or type(fbr) == np.float64:
                self.f_beta = np.zeros(A2.size)+fbr*1.0
            else:
                self.f_beta = fbr
            self.f_beta[0] = 0
            for i in range(A2.size-1):
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
            self.r500_est = finterp(500*self.crit)
        except IndexError:
            self.r200_est = 0.0
            self.r500_est = 0.0
        #self.M200_est = self.massprofile[np.where(ri[:self.f_beta.size] <= self.r200_est)[0][-1]]
        finterp = interp1d(ri[:self.f_beta.size],self.massprofile)
        self.M200_est = finterp(self.r200_est)
        self.M500_est = finterp(self.r500_est)
        self.M200 = self.massprofile[np.where(ri[:self.f_beta.size] <= r200)[0][-1]]
            

        
