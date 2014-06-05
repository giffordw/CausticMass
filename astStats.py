# -*- coding: utf-8 -*-
"""module for performing statistical calculations.

(c) 2007-2011 Matt Hilton 

U{http://astlib.sourceforge.net}

This module (as you may notice) provides very few statistical routines. It does, however, provide
biweight (robust) estimators of location and scale, as described in Beers et al. 1990 (AJ, 100,
32), in addition to a robust least squares fitting routine that uses the biweight transform.

Some routines may fail if they are passed lists with few items and encounter a `divide by zero'
error. Where this occurs, the function will return None. An error message will be printed to the
console when this happens if astStats.REPORT_ERRORS=True (the default). Testing if an
astStats function returns None can be used to handle errors in scripts. 

For extensive statistics modules, the Python bindings for GNU R (U{http://rpy.sourceforge.net}), or
SciPy (U{http://www.scipy.org}) are suggested.

"""

import math
import numpy
import sys

REPORT_ERRORS=True

#---------------------------------------------------------------------------------------------------
def mean(dataList):
    """Calculates the mean average of a list of numbers.
    
    @type dataList: list
    @param dataList: input data, must be a one dimensional list
    @rtype: float
    @return: mean average
    
    """
    sum=0
    for item in dataList:
        sum=sum+float(item)
    if len(dataList)>0:
        mean=sum/float(len(dataList))
    else:
        mean=0
    return mean
    
#---------------------------------------------------------------------------------------------------
def weightedMean(dataList):
    """Calculates the weighted mean average of a two dimensional list (value, weight) of
    numbers.
    
    @type dataList: list
    @param dataList: input data, must be a two dimensional list in format [value, weight]
    @rtype: float
    @return: weighted mean average
    
    """
    sum=0
    weightSum=0
    for item in dataList:
        sum=sum+float(item[0]*item[1])
        weightSum=weightSum+item[1]
    if len(dataList)>0:
        mean=sum/weightSum
    else:
        mean=0
    return mean

#---------------------------------------------------------------------------------------------------
def stdev(dataList):
    """Calculates the (sample) standard deviation of a list of numbers.
    
    @type dataList: list
    @param dataList: input data, must be a one dimensional list
    @rtype: float
    @return: standard deviation
    
    """
    listMean=mean(dataList)
    sum=0
    for item in dataList:
        sum=sum+(float(item-listMean)*float(item-listMean))
    if len(dataList)>0:
        stdev=math.sqrt(sum/(float(len(dataList))-1))
    else:
        stdev=0
    return stdev
    
#---------------------------------------------------------------------------------------------------
def rms(dataList):
    """Calculates the root mean square of a list of numbers.
    
    @type dataList: list
    @param dataList: input data, must be a one dimensional list
    @rtype: float
    @return: root mean square
    
    """
    dataListSq=[]
    for item in dataList:
        dataListSq.append(item*item)
    listMeanSq=mean(dataListSq)
    rms=math.sqrt(listMeanSq)

    return rms
        
#---------------------------------------------------------------------------------------------------
def weightedStdev(dataList):
    """Calculates the weighted (sample) standard deviation of a list of numbers. 
    
    @type dataList: list
    @param dataList: input data, must be a two dimensional list in format [value, weight]
    @rtype: float
    @return: weighted standard deviation
    
    @note: Returns None if an error occurs.
    
    """
    listMean=weightedMean(dataList)
    sum=0
    wSum=0
    wNonZero=0
    for item in dataList:
        if item[1]>0.0:
            sum=sum+float((item[0]-listMean)/item[1])*float((item[0]-listMean)/item[1])
            wSum=wSum+float(1.0/item[1])*float(1.0/item[1])
            
    if len(dataList)>1:
        nFactor=float(len(dataList))/float(len(dataList)-1)
        stdev=math.sqrt(nFactor*(sum/wSum))
    else:
        if REPORT_ERRORS==True:
            print """ERROR: astStats.weightedStdev() : dataList contains < 2 items."""
        stdev=None
    return stdev
        
#---------------------------------------------------------------------------------------------------
def median(dataList):
    """Calculates the median of a list of numbers.
    
    @type dataList: list
    @param dataList: input data, must be a one dimensional list
    @rtype: float
    @return: median average
    
    """
    dataList.sort()
    midValue=float(len(dataList)/2.0)
    fractPart=math.modf(midValue)[0]

    if fractPart==0.5:		# if odd number of items
        midValue=math.ceil(midValue)
        
    # Doesn't like it when handling a list with only one item in it!	
    if midValue<len(dataList)-1:
        median=dataList[int(midValue)]
        
        if fractPart!=0.5:	# if even
            prevItem=dataList[int(midValue)-1]
            median=(median+prevItem)/2.0

    else:
        median=dataList[0]
        
    return median
    
#---------------------------------------------------------------------------------------------------
def modeEstimate(dataList):
    """Returns an estimate of the mode of a set of values by mode=(3*median)-(2*mean).
    
    @type dataList: list
    @param dataList: input data, must be a one dimensional list
    @rtype: float
    @return: estimate of mode average
    
    """
    mode=(3*median(dataList))-(2*mean(dataList))

    return mode

#---------------------------------------------------------------------------------------------------
def MAD(dataList):
    """Calculates the Median Absolute Deviation of a list of numbers.
    
    @type dataList: list
    @param dataList: input data, must be a one dimensional list
    @rtype: float
    @return: median absolute deviation
    
    """
    listMedian=median(dataList)
    
    # Calculate |x-M| values
    diffModuli=[]
    for item in dataList:
        diffModuli.append(math.fabs(item-listMedian))
    diffModuli.sort()

    midValue=float(len(diffModuli)/2.0)
    fractPart=math.modf(midValue)[0]

    if fractPart==0.5:		# if odd number of items
        midValue=math.ceil(midValue)
        
    # Doesn't like it when handling a list with only one item in it!	
    if midValue<len(diffModuli)-1:
        MAD=diffModuli[int(midValue)]
        
        if fractPart!=0.5:	# if even
            prevItem=diffModuli[int(midValue)-1]
            MAD=(MAD+prevItem)/2.0

    else:
        MAD=diffModuli[0]
        
    return MAD

#---------------------------------------------------------------------------------------------------
def biweightLocation(dataList, tuningConstant):
    """Calculates the biweight location estimator (like a robust average) of a list of
    numbers.
    
    @type dataList: list
    @param dataList: input data, must be a one dimensional list
    @type tuningConstant: float
    @param tuningConstant: 6.0 is recommended.
    @rtype: float
    @return: biweight location
    
    @note: Returns None if an error occurs.	
    
    """	
    C=tuningConstant
    listMedian=median(dataList)
    listMAD=MAD(dataList)
    if listMAD!=0:
        uValues=[]
        for item in dataList:
            uValues.append((item-listMedian)/(C*listMAD))
                
        top=0		# numerator equation (5) Beers et al if you like
        bottom=0	# denominator
        for i in range(len(uValues)):
            if math.fabs(uValues[i])<=1.0:
                top=top+((dataList[i]-listMedian) \
                    *(1.0-(uValues[i]*uValues[i])) \
                    *(1.0-(uValues[i]*uValues[i])))
            
                bottom=bottom+((1.0-(uValues[i]*uValues[i])) \
                    *(1.0-(uValues[i]*uValues[i])))
    
        CBI=listMedian+(top/bottom)
        
    else:
        if REPORT_ERRORS==True:
            print """ERROR: astStats: biweightLocation() : MAD() returned 0."""
        return None
    
    return CBI

#---------------------------------------------------------------------------------------------------
def biweightScale(dataList, tuningConstant):
    """Calculates the biweight scale estimator (like a robust standard deviation) of a list
    of numbers. 
    
    @type dataList: list
    @param dataList: input data, must be a one dimensional list
    @type tuningConstant: float
    @param tuningConstant: 9.0 is recommended.
    @rtype: float
    @return: biweight scale
    
    @note: Returns None if an error occurs.
        
    """	
    C=tuningConstant
    
    # Calculate |x-M| values and u values
    listMedian=median(dataList)
    listMAD=MAD(dataList)
    diffModuli=[]
    for item in dataList:
        diffModuli.append(math.fabs(item-listMedian))
    uValues=[]
    for item in dataList:
        try:
            uValues.append((item-listMedian)/(C*listMAD))
        except ZeroDivisionError:
            if REPORT_ERRORS==True:
                print """ERROR: astStats.biweightScale() : divide by zero error."""
            return None
        
    top=0		# numerator equation (9) Beers et al
    bottom=0
    valCount=0	# Count values where u<1 only
    
    for i in range(len(uValues)):
        # Skip u values >1
        if math.fabs(uValues[i])<=1.0:
            u2Term=1.0-(uValues[i]*uValues[i])
            u4Term=math.pow(u2Term, 4)
            top=top+((diffModuli[i]*diffModuli[i])*u4Term)
            bottom=bottom+(u2Term*(1.0-(5.0*(uValues[i]*uValues[i]))))
            valCount=valCount+1
    
    top=math.sqrt(top)
    bottom=math.fabs(bottom)

    SBI=math.pow(float(valCount), 0.5)*(top/bottom)
    return SBI

#---------------------------------------------------------------------------------------------------
def biweightClipped(dataList, tuningConstant, sigmaCut):
    """Iteratively calculates biweight location and scale, using sigma clipping, for a list
    of values. 	The calculation is performed on the first column of a multi-dimensional
    list; other columns are ignored.
    
    @type dataList: list
    @param dataList: input data
    @type tuningConstant: float
    @param tuningConstant: 6.0 is recommended for location estimates, 9.0 is recommended for
    scale estimates	
    @type sigmaCut: float
    @param sigmaCut: sigma clipping to apply
    @rtype:	dictionary 
    @return: estimate of biweight location, scale, and list of non-clipped data, in the format
    {'biweightLocation', 'biweightScale', 'dataList'}
    
    @note: Returns None if an error occurs.

    """		
    
    iterations=0
    clippedValues=[]
    for row in dataList:
        if type(row)==list:
            clippedValues.append(row[0])
        else:
            clippedValues.append(row)
        
    while iterations<11 and len(clippedValues)>5:
        
        cbi=biweightLocation(clippedValues, tuningConstant)	
        sbi=biweightScale(clippedValues, tuningConstant)
        
        # check for either biweight routine falling over
        # happens when feed in lots of similar numbers
        # e.g. when bootstrapping with a small sample		
        if cbi==None or sbi==None:
            
            if REPORT_ERRORS==True:
                print """ERROR: astStats : biweightClipped() :
                divide by zero error."""
            
            return None
            
        else:
            
            clippedValues=[]
            clippedData=[]
            for row in dataList:
                if type(row)==list:
                    if row[0]>cbi-(sigmaCut*sbi) \
                    and row[0]<cbi+(sigmaCut*sbi):
                        clippedValues.append(row[0])
                        clippedData.append(row)
                else:
                    if row>cbi-(sigmaCut*sbi) \
                    and row<cbi+(sigmaCut*sbi):
                        clippedValues.append(row)
                        clippedData.append(row)
            
        iterations=iterations+1
            
    return {	'biweightLocation':cbi ,
            'biweightScale':sbi,
            'dataList':clippedData}

#---------------------------------------------------------------------------------------------------
def biweightTransform(dataList, tuningConstant):
    """Calculates the biweight transform for a set of values. Useful for using as weights in
    robust line fitting.
    
    @type dataList: list
    @param dataList: input data, must be a one dimensional list
    @type tuningConstant: float
    @param tuningConstant: 6.0 is recommended for location estimates, 9.0 is recommended for
    scale estimates	
    @rtype: list
    @return: list of biweights	
    
    """
    C=tuningConstant
    
    # Calculate |x-M| values and u values
    listMedian=abs(median(dataList))
    cutoff=C*listMedian
    biweights=[]
    for item in dataList:
        if abs(item)<cutoff:
            biweights.append([item,
            (1.0-((item/cutoff)*(item/cutoff))) \
            *(1.0-((item/cutoff)*(item/cutoff)))])
        else:
            biweights.append([item, 0.0])
    
    return biweights
    
#---------------------------------------------------------------------------------------------------
def OLSFit(dataList):
    """Performs an ordinary least squares fit on a two dimensional list of numbers.
    Minimum number of data points is 5.
    
    @type dataList: list
    @param dataList: input data, must be a two dimensional list in format [x, y]
    @rtype: dictionary
    @return: slope and intercept on y-axis, with associated errors, in the format
    {'slope', 'intercept', 'slopeError', 'interceptError'}
    
    @note: Returns None if an error occurs.	
        
    """
    sumX=0
    sumY=0
    sumXY=0
    sumXX=0
    n=float(len(dataList))
    if n > 2:
        for item in dataList:
            sumX=sumX+item[0]
            sumY=sumY+item[1]
            sumXY=sumXY+(item[0]*item[1])
            sumXX=sumXX+(item[0]*item[0])	
        m=((n*sumXY)-(sumX*sumY))/((n*sumXX)-(sumX*sumX))
        c=((sumXX*sumY)-(sumX*sumXY))/((n*sumXX)-(sumX*sumX))
        
        sumRes=0
        for item in dataList:
        
            sumRes=sumRes+((item[1]-(m*item[0])-c) \
            *(item[1]-(m*item[0])-c))
            
        sigma=math.sqrt((1.0/(n-2))*sumRes)
        
        mSigma=(sigma*math.sqrt(n))/math.sqrt((n*sumXX)-(sumX*sumX))
        cSigma=(sigma*math.sqrt(sumXX))/math.sqrt((n*sumXX)-(sumX*sumX))
    else:
        if REPORT_ERRORS==True:
            print """ERROR: astStats.OLSFit() : dataList contains < 3 items."""
            
        return None
        
    return {'slope':m,
            'intercept':c,
            'slopeError':mSigma,
            'interceptError':cSigma}

#---------------------------------------------------------------------------------------------------
def clippedMeanStdev(dataList, sigmaCut = 3.0, maxIterations = 10.0):
    """Calculates the clipped mean and stdev of a list of numbers.
    
    @type dataList: list
    @param dataList: input data, one dimensional list of numbers
    @type sigmaCut: float
    @param sigmaCut: clipping in Gaussian sigma to apply
    @type maxIterations: int
    @param maxIterations: maximum number of iterations
    @rtype: dictionary
    @return: format {'clippedMean', 'clippedStdev', 'numPoints'}
    
    """
    
    listCopy=[]
    for d in dataList:
        listCopy.append(d)
    listCopy=numpy.array(listCopy)
    
    iterations=0
    while iterations < maxIterations and len(listCopy) > 4:
        
        m=listCopy.mean()
        s=listCopy.std()
        
        listCopy=listCopy[numpy.less(abs(listCopy), abs(m+sigmaCut*s))]
        
        iterations=iterations+1
    
    return {'clippedMean': m, 'clippedStdev': s, 'numPoints': listCopy.shape[0]}
    
#---------------------------------------------------------------------------------------------------
def clippedWeightedLSFit(dataList, sigmaCut):
    """Performs a weighted least squares fit on a list of numbers with sigma clipping. Minimum number of data
    points is 5.
    
    @type dataList: list
    @param dataList: input data, must be a three dimensional list in format [x, y, y weight]
    @rtype: dictionary
    @return: slope and intercept on y-axis, with associated errors, in the format
    {'slope', 'intercept', 'slopeError', 'interceptError'}
    
    @note: Returns None if an error occurs.	
    
    """
    
    iterations=0
    clippedValues=[]
    for row in dataList:
        clippedValues.append(row)
        
    while iterations<11 and len(clippedValues)>4:
        
        fitResults=weightedLSFit(clippedValues, "errors")
        
        if fitResults['slope'] == None:
            
            if REPORT_ERRORS==True:
                print """ERROR: astStats : clippedWeightedLSFit() :
                divide by zero error."""
            
            return None
            
        else:
            
            clippedValues=[]
            for row in dataList:
                
                # Trim points more than sigmaCut*sigma away from the fitted line
                fit=fitResults['slope']*row[0]+fitResults['intercept']
                res=row[1]-fit
                if abs(res)/row[2] < sigmaCut:
                    clippedValues.append(row)
            
        iterations=iterations+1
    
    # store the number of values that made it through the clipping process
    fitResults['numDataPoints']=len(clippedValues)
    
    return fitResults
    
#---------------------------------------------------------------------------------------------------
def weightedLSFit(dataList, weightType):
    """Performs a weighted least squares fit on a three dimensional list of numbers [x, y, y error].
    
    @type dataList: list
    @param dataList: input data, must be a three dimensional list in format [x, y, y error]
    @type weightType: string
    @param weightType: if "errors", weights are calculated assuming the input data is in the
    format [x, y, error on y]; if "weights", the weights are assumed to be already calculated and
    stored in a fourth column [x, y, error on y, weight] (as used by e.g. L{astStats.biweightLSFit})
    @rtype: dictionary
    @return: slope and intercept on y-axis, with associated errors, in the format
    {'slope', 'intercept', 'slopeError', 'interceptError'}
    
    @note: Returns None if an error occurs.	
            
    """
    if weightType == "weights":
        sumW=0
        sumWX=0
        sumWY=0
        sumWXY=0
        sumWXX=0
        n=float(len(dataList))
        if n > 4:
            for item in dataList:
                W=item[3]
                sumWX=sumWX+(W*item[0])
                sumWY=sumWY+(W*item[1])
                sumWXY=sumWXY+(W*item[0]*item[1])
                sumWXX=sumWXX+(W*item[0]*item[0])
                sumW=sumW+W
                #print sumW, sumWXX, sumWX
        
            try:
                m=((sumW*sumWXY)-(sumWX*sumWY)) \
                /((sumW*sumWXX)-(sumWX*sumWX))
            except ZeroDivisionError:
                if REPORT_ERRORS == True:
                    print "ERROR: astStats.weightedLSFit() : divide by zero error."
                return None
        
            try:
                c=((sumWXX*sumWY)-(sumWX*sumWXY)) \
                /((sumW*sumWXX)-(sumWX*sumWX))
            except ZeroDivisionError:
                if REPORT_ERRORS == True:
                    print "ERROR: astStats.weightedLSFit() : divide by zero error."
                return None
            
            sumRes=0
            for item in dataList:
            
                sumRes=sumRes+((item[1]-(m*item[0])-c) \
                *(item[1]-(m*item[0])-c))
                
            sigma=math.sqrt((1.0/(n-2))*sumRes)
            
            # Can get div0 errors here so check
            # When biweight fitting converges this shouldn't happen
            if (n*sumWXX)-(sumWX*sumWX)>0.0: 
            
                mSigma=(sigma*math.sqrt(n)) \
                    /math.sqrt((n*sumWXX)-(sumWX*sumWX))
        
                cSigma=(sigma*math.sqrt(sumWXX)) \
                    /math.sqrt((n*sumWXX)-(sumWX*sumWX))
                
            else:
                
                if REPORT_ERRORS==True:
                    print """ERROR: astStats.weightedLSFit()
                    : divide by zero error."""
                return None
                
        else:
            if REPORT_ERRORS==True:
                print """ERROR: astStats.weightedLSFit() :
                dataList contains < 5 items."""
            return None
            
    elif weightType == "errors":
        sumX=0
        sumY=0
        sumXY=0
        sumXX=0
        sumSigma=0
        n=float(len(dataList))
        for item in dataList:
            sumX=sumX+(item[0]/(item[2]*item[2]))
            sumY=sumY+(item[1]/(item[2]*item[2]))
            sumXY=sumXY+((item[0]*item[1])/(item[2]*item[2]))
            sumXX=sumXX+((item[0]*item[0])/(item[2]*item[2]))
            sumSigma=sumSigma+(1.0/(item[2]*item[2]))
        delta=(sumSigma*sumXX)-(sumX*sumX)	
        m=((sumSigma*sumXY)-(sumX*sumY))/delta
        c=((sumXX*sumY)-(sumX*sumXY))/delta
        mSigma=math.sqrt(sumSigma/delta)
        cSigma=math.sqrt(sumXX/delta)
        
    return {'slope':m,
            'intercept':c,
            'slopeError':mSigma,
            'interceptError':cSigma}
    
#---------------------------------------------------------------------------------------------------
def biweightLSFit(dataList, tuningConstant, sigmaCut = None):
    """Performs a weighted least squares fit, where the weights used are the biweight
    transforms of the residuals to the previous best fit .i.e. the procedure is iterative,
    and converges very quickly (iterations is set to 10 by default). Minimum number of data
    points is 10.
    
    This seems to give slightly different results to the equivalent R routine, so use at your
    own risk!
    
    @type dataList: list
    @param dataList: input data, must be a three dimensional list in format [x, y, y weight]
    @type tuningConstant: float
    @param tuningConstant: 6.0 is recommended for location estimates, 9.0 is recommended for
    scale estimates
    @type sigmaCut: float
    @param sigmaCut: sigma clipping to apply (set to None if not required)	
    @rtype: dictionary
    @return: slope and intercept on y-axis, with associated errors, in the format
    {'slope', 'intercept', 'slopeError', 'interceptError'}
    
    @note: Returns None if an error occurs.
        
    """

    dataCopy=[]
    for row in dataList:
        dataCopy.append(row)
        
    # First perform unweighted fit, then calculate residuals
    results=OLSFit(dataCopy)
    origLen=len(dataCopy)
    for k in range(10):
        m=results['slope']
        c=results['intercept']
        res=[]
        for item in dataCopy:
            res.append((m*item[0]+c)-item[1])
            
        if len(res)>5:
            # For clipping, trim away things >3 sigma 
            # away from median
            if sigmaCut != None:
                absRes=[]
                for item in res:
                    absRes.append(abs(item))
                sigma=stdev(absRes)
                count=0
                for item in absRes:
                    if item>(sigmaCut*sigma) \
                    and len(dataCopy)>2:
                        del dataCopy[count]
                        del res[count]
                        
                        # Index of datalist gets out of
                        # sync with absRes as we delete
                        # items
                        count=count-1 
                        
                    count=count+1
                        
            # Biweight transform residuals
            weights=biweightTransform(res, tuningConstant)
                        
            # Perform weighted fit, using biweight transforms 
            # of residuals as weight
            wData=[]
            for i in range(len(dataCopy)):
                wData.append([dataCopy[i][0], dataCopy[i][1], dataCopy[i][2], weights[i][1]])
            
            results=weightedLSFit(wData, "weights")

    return results
    
#---------------------------------------------------------------------------------------------------
def cumulativeBinner(data, binMin, binMax, binTotal):
    """Bins the input data cumulatively.
    
    @param data: input data, must be a one dimensional list
    @type binMin: float
    @param binMin: minimum value from which to bin data
    @type binMax: float
    @param binMax: maximum value from which to bin data	
    @type binTotal: int
    @param binTotal: number of bins 
    @rtype: list
    @return: binned data, in format [bin centre, frequency]
        
    """
    #Bin data
    binStep=float(binMax-binMin)/binTotal
    bins=[]
    totalItems=len(data)
    for i in range(binTotal):
        bins.append(0)
        for item in data:
            if item>(binMin+(i*binStep)):
                bins[i]=bins[i]+1.0/totalItems
                
    # Gnuplot requires points at bin midpoints
    coords=[]
    for i in range(binTotal):
        coords.append([binMin+(float(i+0.5)*binStep), bins[i]])
    
    return coords

#---------------------------------------------------------------------------------------------------
def binner(data, binMin, binMax, binTotal):
    """Bins the input data..
    
    @param data: input data, must be a one dimensional list
    @type binMin: float
    @param binMin: minimum value from which to bin data
    @type binMax: float
    @param binMax: maximum value from which to bin data	
    @type binTotal: int
    @param binTotal: number of bins 
    @rtype: list
    @return: binned data, in format [bin centre, frequency]
        
    """
    #Bin data
    binStep=float(binMax-binMin)/binTotal
    bins=[]
    for i in range(binTotal):
        bins.append(0)
        for item in data:
            if item>(binMin+(i*binStep)) \
            and item<=(binMin+((i+1)*binStep)):
                bins[i]=bins[i]+1
                
    # Gnuplot requires points at bin midpoints
    coords=[]
    for i in range(binTotal):
        coords.append([binMin+(float(i+0.5)*binStep), bins[i]])
    
    return coords

#---------------------------------------------------------------------------------------------------
def weightedBinner(data, weights, binMin, binMax, binTotal):
    """Bins the input data, recorded frequency is sum of weights in bin.
    
    @param data: input data, must be a one dimensional list
    @type binMin: float
    @param binMin: minimum value from which to bin data
    @type binMax: float
    @param binMax: maximum value from which to bin data	
    @type binTotal: int
    @param binTotal: number of bins 
    @rtype: list
    @return: binned data, in format [bin centre, frequency]
        
    """
    #Bin data
    binStep=float(binMax-binMin)/binTotal
    bins=[]
    for i in range(binTotal):
        bins.append(0.0)
        for item, weight in zip(data, weights):
            if item>(binMin+(i*binStep)) \
            and item<=(binMin+((i+1)*binStep)):
                bins[i]=bins[i]+weight
                
    # Gnuplot requires points at bin midpoints
    coords=[]
    for i in range(binTotal):
        coords.append([binMin+(float(i+0.5)*binStep), bins[i]])
    
    return coords
    
#---------------------------------------------------------------------------------------------------
