# -*- coding: utf-8 -*-
__author__ = 'Jinze Yu'

import numpy as np
from scipy.spatial.distance import cdist
import openravepy
from mujindetection.shared.scene import camerautil
from scipy.spatial import cKDTree
import time



class SVDICP(object):
    def __init__(self, targetpointcloud):
        self._kdtree = cKDTree(targetpointcloud)
        
    def best_fit_transform(self, A, B):
        '''
        Calculates the least-squares best-fit transform between corresponding 3D points A->B
        Input:
          A: Nx3 numpy array of corresponding 3D points
          B: Nx3 numpy array of corresponding 3D points
        Returns:
          T: 4x4 homogeneous transformation matrix
          R: 3x3 rotation matrix
          t: 3x1 column vector
        '''

        assert len(A) == len(B)

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
           Vt[2,:] *= -1
           R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R,centroid_A.T)

        # homogeneous transformation
        T = np.identity(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t

        return T, R, t


    def _MatchPointCloud(self, srcPointCloud):
        distances, indices = self._kdtree.query(srcPointCloud)
        return self._dstpointcloud[indices]

    def nearest_neighbor(self, src):
        '''
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Nx3 array of points
            dst: Nx3 array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        '''
        return self._kdtree.query(src)
        
        # all_dists = cdist(src, dst, 'euclidean')
        # indices = all_dists.argmin(axis=1)
        # distances = all_dists[np.arange(all_dists.shape[0]), indices]
        # return distances, indices

    def icp(self, A, B, init_pose=None, max_iterations=200, tolerance=0.001):
        '''
        The Iterative Closest Point method
        Input:
            A: Nx3 numpy array of source 3D points
            B: Nx3 numpy array of destination 3D point
            init_pose: 4x4 homogeneous transformation
            max_iterations: exit algorithm after max_iterations
            tolerance: convergence criteria
        Output:
            T: final homogeneous transformation
            distances: Euclidean distances (errors) of the nearest neighbor
        '''

        tolerance = tolerance * max(A.max(axis=0) - A.min(axis=0))

        #from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

        env = openravepy.Environment()
        env.SetViewer("qtosg")
        h = []
        h.append(env.plot3(A, 1, (1,0,0)))
        h.append(env.plot3(B, 1, (0,1,0)))

        # from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

        # make points homogeneous, copy them so as to maintain the originals
        src = np.ones((4,A.shape[0]))
        dst = np.ones((4,B.shape[0]))
        src[0:3,:] = np.copy(A.T)
        dst[0:3,:] = np.copy(B.T)

        # apply the initial pose estimation
        if init_pose is not None:
            src = np.dot(init_pose, src)

        prev_error = 0

        for i in range(max_iterations):
            # find the nearest neighbours between the current source and destination points
            #distances, indices = self.nearest_neighbor(src[0:3,:].T, dst[0:3,:].T)
            
            distances, indices = self.nearest_neighbor(src[0:3,:].T)
            
            # compute the transformation between the current source and nearest destination points
            T,_,_ = self.best_fit_transform(src[0:3,:].T, dst[0:3,indices].T)

            # update the current source
            src = np.dot(T, src)

            # check error
            mean_error = np.sum(distances) / distances.size
            if abs(prev_error-mean_error) < tolerance:
                break
            prev_error = mean_error
            
            # tpts = camerautil.Transform3DPoints(T, A)
            # th = env.plot3(tpts, 1, (0, 0, 1))
            # time.sleep(0.2)
            # th = None 

        # calculate final transformation
        # T,_,_ = best_fit_transform(A, src[0:3,:].T)

        # finalPoints = camerautil.Transform3DPoints(T, A)
        # h.append(env.plot3(finalPoints, 1, (0, 0, 1)))

        # from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())
        return T, distances
