"""
Sequential Importance Resampling particle filter implementation.
"""

__version__ = '1.0'
__author__ = 'Saul Alonso-Monsalve'
__email__ = "saul.alonso.monsalve@cern.ch"

import numpy as np
import matplotlib.pyplot as plt
import math
import pickle as pk
from numpy.random import uniform
from modules.constants import *

class FittingSIRPF:
    
    def __init__(self, allMeasurements, true_nodes, N):
        self.dim = 4 # pos + charge
        self.N = N   # number of particles
        
        # read histogram
        with open(HIST_PATH, "rb") as fd:
            self.H, self.edges = pk.load(fd)

        self.forward_track = {"avg":[], "cov":[]}
        self.backward_track = {"avg":[], "cov":[]}

        # state average and covariance
        self.stateAvg = np.zeros(shape=(self.dim,))
        self.stateCov = np.zeros(shape=(self.dim, self.dim))
        
        # calculate axis the particle is travelling through
        lenX = allMeasurements[:,0].max()-allMeasurements[:,0].min()
        lenY = allMeasurements[:,1].max()-allMeasurements[:,1].min()
        lenZ = allMeasurements[:,2].max()-allMeasurements[:,2].min()
        lenMax = max(lenX, lenY, lenZ)
        if lenMax==lenX:
            self.axis = 0
        elif lenMax==lenY:
            self.axis = 1
        elif lenMax==lenZ:
            self.axis = 2
        self.allMeasurements = allMeasurements[allMeasurements[:,self.axis].argsort()]
        self.true_nodes = true_nodes[true_nodes[:,self.axis].argsort()]
        self.dir = 1 if allMeasurements[-1,self.axis] > allMeasurements[0,self.axis] else -1
        
        # keep a variable to check events where the fitting failed (bin not found in the histogram)
        self.failed = False

    # from cartesian to spherical coordinates
    def cart2spherical(self, xyz):
        xy = xyz[:,0]**2 + xyz[:,1]**2
        r = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        phi = np.arctan2(xyz[:,1], xyz[:,0])
        return r, theta, phi

    # fill the particle vector with prior states. All the hits are ordered based on the 
    # longest axis along the measurements, so the prior is based on the first measurement
    # regarding that order (if more than one, select the one with the highest energy).
    def makePrior(self, forward=True):
        # Initialise particles
        self.particles = np.zeros(shape=(self.N, self.dim))
        
        # check longest axis and select candidate with highest charge
        cands = (self.allMeasurements[:,self.axis]==self.allMeasurements[0,self.axis])
        self.nb_meas = cands.sum() # number of candidates
        best_cand = (self.allMeasurements[cands,IDX_SCHA]).argmax()
        best_cand = np.full(shape=(self.N,), fill_value=best_cand)
        
        # fill the position asumming 1 cm cubes
        self.particles[:, IDX_SPOS:IDX_EPOS] = self.allMeasurements[best_cand, IDX_SPOS:IDX_EPOS] +\
                                                        uniform(-CUBE_SIZE, CUBE_SIZE, size=(self.N,3))

        # fill the hit charge
        self.particles[:, IDX_SCHA:IDX_ECHA] = self.allMeasurements[best_cand, IDX_SCHA:IDX_ECHA] # no noise

        # Set all the particles to have a uniform weight (normalised to 1)
        self.weights = np.ones(self.N) / self.N

        return

    # calculate the average and covariance for the current state vector
    def makeAverage(self):    
        self.stateCov.fill(0)
        
        # find the averages
        self.stateAvg[:] = np.average(self.particles, axis=0, weights=self.weights)

        # find the covariance (brute force)
        diff = (self.particles-self.stateAvg).reshape(self.N, self.dim, 1)
        self.stateCov += (np.matmul(diff, diff.swapaxes(2, 1)).T * self.weights).sum(axis=-1)

        return

    # the particles are propagated by randomly throwing particles in the next K cubes 
    # from "self.nb_meas". The update can work in either the forward or backward direction.
    def propagate(self, hits=15):        
        max_nb_meas = min(self.nb_meas+hits, len(self.allMeasurements))
        cands = np.random.choice(range(self.nb_meas, max_nb_meas), self.N)
        
        # update the position asumming 1 cm cubes
        self.particles[:,IDX_SPOS:IDX_EPOS] = self.allMeasurements[cands,IDX_SPOS:IDX_EPOS] +\
                                                        uniform(-CUBE_SIZE, CUBE_SIZE, size=(self.N,3))

        # update the hit charge
        self.particles[:, IDX_SCHA:IDX_ECHA] = self.allMeasurements[cands,IDX_SCHA:IDX_ECHA] # no noise

        return

    # implement a likelihood based on the calculated histogram
    def likelihood(self, forward=True):
        # calculate delta X, Y, Z
        deltaXYZ = self.particles[:,IDX_SPOS:IDX_EPOS]-self.stateAvg[IDX_SPOS:IDX_EPOS]
        deltaX = deltaXYZ[:, 0]
        deltaY = deltaXYZ[:, 1]
        deltaZ = deltaXYZ[:, 2]
        _, deltaTheta, _ = self.cart2spherical(deltaXYZ)
        deltaPE = (np.log(self.particles[:,IDX_SCHA:IDX_ECHA])-\
                   np.log(self.stateAvg[IDX_SCHA:IDX_ECHA])).reshape(-1,)

        if not forward:
            # negative deltas for backward fitting
            deltaX = -deltaX
            deltaY = -deltaY
            deltaZ = -deltaZ
            deltaTheta = -deltaTheta
            deltaPE = -deltaPE

        # find indexes of bins in histogram
        indexes_deltaX = np.digitize(deltaX, self.edges[0])-1
        indexes_deltaY = np.digitize(deltaY, self.edges[1])-1
        indexes_deltaZ = np.digitize(deltaZ, self.edges[2])-1
        indexes_deltaTheta = np.digitize(deltaTheta, self.edges[3])-1
        indexes_deltaPE = np.digitize(deltaPE, self.edges[4])-1

        # find values outside the ranges of the edges of the histogram
        wrong_values_deltaX = np.logical_or(indexes_deltaX==-1, indexes_deltaX==self.H.shape[0])
        wrong_values_deltaY = np.logical_or(indexes_deltaY==-1, indexes_deltaY==self.H.shape[1])
        wrong_values_deltaZ = np.logical_or(indexes_deltaZ==-1, indexes_deltaZ==self.H.shape[2])
        wrong_values_deltaTheta = np.logical_or(indexes_deltaTheta==-1, indexes_deltaTheta==self.H.shape[3])
        wrong_values_deltaPE = np.logical_or(indexes_deltaPE==-1, indexes_deltaPE==self.H.shape[4])
        wrong_values_deltas = np.logical_or.reduce((wrong_values_deltaX, wrong_values_deltaY, wrong_values_deltaZ,\
                                                     wrong_values_deltaTheta, wrong_values_deltaPE))

        # give a constant bin index to wrong values
        indexes_deltaX[wrong_values_deltaX] = 0
        indexes_deltaY[wrong_values_deltaY] = 0
        indexes_deltaZ[wrong_values_deltaZ] = 0
        indexes_deltaTheta[wrong_values_deltaTheta] = 0
        indexes_deltaPE[wrong_values_deltaPE] = 0

        # retrieve the bins
        res = self.H[indexes_deltaX, indexes_deltaY, indexes_deltaZ, indexes_deltaTheta, indexes_deltaPE]

        # give a 0 result to wrong likelihood values
        res[wrong_values_deltas] = 0

        # none of the particles found in the histogram! (very rare if enough particles)
        if res.sum()==0:
            res[:] = np.random.normal(loc=0, scale=1, size=len(res)) # random weights
            self.failed = True
            
        return res
            
    # propagate the particles and updated the weights
    def updateParticles(self, forward=True):
        # Propagate
        self.propagate()

        # Measure
        self.weights = self.likelihood(forward)
        self.weights/=self.weights.sum() # norm
        
        # check where is the updated particle and update self.nb_meas accordinly
        selectedXYZ = self.particles[self.weights.argmax(), self.axis]
        if self.dir==1:
            self.nb_meas = (self.allMeasurements[:, self.axis] < selectedXYZ).sum()
        elif self.dir==-1:
            self.nb_meas = (self.allMeasurements[:, self.axis] > selectedXYZ).sum()
        
        return
    
    # weighted average of forward and backward results
    def forwardBackwardSmoothing(self):        
        # get the cube positions within longest axis 
        # (due to possible different number of fitted nodes in forward/backward)
        cubes_for = ((self.forward_track["avg"][:,self.axis]-\
                      DETECTOR_RANGES[self.axis][0])/(CUBE_SIZE*2)).astype(int)
        cubes_bac = ((self.backward_track["avg"][:,self.axis]-\
                      DETECTOR_RANGES[self.axis][0])/(CUBE_SIZE*2)).astype(int)
        all_cubes = np.unique(np.concatenate((cubes_for,cubes_bac),axis=0))
        
        final_track = np.zeros(shape=(all_cubes.shape[0], 4))
            
        # estimate the influence of the prior on the current states
        influence = 1.0/5.0
        
        # iterate over cubes
        for i, cube_id in enumerate(all_cubes):
            # indexes of cubes in forward and backward arrays
            for_idx = np.where(cubes_for==cube_id)[0]
            bac_idx = np.where(cubes_bac==cube_id)[0]
            
            # retrieve cube averages
            for_avg = self.forward_track["avg"][for_idx]
            bac_avg = self.backward_track["avg"][bac_idx]
            
            # retrieve cube covariances
            for_cov = self.forward_track["cov"][for_idx]
            bac_cov = self.backward_track["cov"][bac_idx]
            for_cov[for_cov==0] = 1e-7
            bac_cov[bac_cov==0] = 1e-7
            
            r, wSum = 0, 0
            for j, fMeas in enumerate(for_idx):
                # invert the state covariance to turn it into an error matrix               
                forwErr = np.linalg.inv(for_cov[j]) 
                
                forwardWeight = influence*(fMeas+1.0)
                forwardWeight = 1.0 - math.exp(-0.5*forwardWeight*forwardWeight)
                
                # weighted average of the states and set the new state value
                w = forwErr.diagonal()*forwardWeight
                r += w*for_avg[j]
                wSum += w
                
            for j, bMeas in enumerate(bac_idx):
                # invert the state covariance to turn it into an error matrix
                backErr = np.linalg.inv(bac_cov[j])
                
                backwardWeight = influence*(bMeas+1.0)
                backwardWeight = 1.0 - math.exp(-0.5*backwardWeight*backwardWeight)
                
                # weighted average of the states and set the new state value
                w = backErr.diagonal()*backwardWeight
                r += w*bac_avg[j]
                wSum += w
            
            final_track[i] = r / wSum
            
        return final_track
            
    # main function of the filter
    def run_filter(self):    
        # make prior for forward fitting
        priorMeasurements = self.allMeasurements
        self.makePrior()

        # make average (for first prop)
        self.makeAverage()
        
        # add starting point to forward track
        self.forward_track["avg"].append(self.stateAvg.copy())
        self.forward_track["cov"].append(self.stateCov.copy())
        
        # forward fitting
        for i in range(len(self.allMeasurements)):
            # exit loop if already checked all the measurements
            if self.nb_meas >= len(self.allMeasurements)-1:
                break
                
            # update particles and average state
            self.updateParticles(forward=True)
            self.makeAverage()

            # add state to forward track
            self.forward_track["avg"].append(self.stateAvg.copy())
            self.forward_track["cov"].append(self.stateCov.copy())

        self.forward_track["avg"] = np.array(self.forward_track["avg"])
        self.forward_track["cov"] = np.array(self.forward_track["cov"])
        
        # reverse measurements for backward fitting
        self.allMeasurements = self.allMeasurements[::-1]
        self.dir*=-1
        
        # make prior for backward fitting
        self.makePrior()

        # make average (for first prop)
        self.makeAverage()
        
        # add starting point to backward track
        self.backward_track["avg"].append(self.stateAvg.copy())
        self.backward_track["cov"].append(self.stateCov.copy())

        # backward fitting
        for i in range(len(self.allMeasurements)):
            # exit loop if already checked all the measurements
            if self.nb_meas >= len(self.allMeasurements)-1:
                break
                
            # update particles and average state
            self.updateParticles(forward=False)
            self.makeAverage()
            
            # add state to backward track
            self.backward_track["avg"].append(self.stateAvg.copy())
            self.backward_track["cov"].append(self.stateCov.copy())
            
        self.backward_track["avg"] = np.array(self.backward_track["avg"])
        self.backward_track["cov"] = np.array(self.backward_track["cov"])
        
        # perform forward-backward smoothing
        final_track = self.forwardBackwardSmoothing()

        return final_track, self.failed
