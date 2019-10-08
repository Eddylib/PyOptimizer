import sys
sys.path.append('.')
sys.path.append('..')


from utils import numeric, ProblemLoader
from backend import Graph
from backend import Problem
import numpy as np
import pyquaternion
import scipy.linalg as scp
import random
import math
import time
import os


def vec3_to_quat(vec3:np.ndarray)->pyquaternion.Quaternion:
    norm = np.linalg.norm(vec3)
    axix = vec3/norm
    retQ = pyquaternion.Quaternion._from_axis_angle(axix,norm)
    return retQ


def vec4_to_quat(vec4:np.ndarray):
    """
    :param vec4: x,y,z,w
    :return:
    """
    retQ = pyquaternion.Quaternion(vec4[3],vec4[0],vec4[1],vec4[2])
    return retQ


def quat_to_vec4(quat: pyquaternion.Quaternion):
    retV: np.ndarray = np.array([[quat.x, quat.y, quat.z, quat.w]], np.float).transpose()
    return retV

class Frame(Graph.Vertex):
    def __init__(self, Rwc, twc, pointsObs: dict):
        super().__init__(7,6,ifDense=True)
        self.Rwc = Rwc
        self.twc = twc
        self.setState(self.RtToState()) #dimension, [0,3) translaton, [3,7) rotation
        self.pointsObs = pointsObs # 相机对于各个地图点的观测
        self.initState = self.RtToState()


    def RtToState(self):
        # state 是全局维度，前三维是平移，后四维是旋转四元数
        state = np.zeros((7,1))
        state[0:3] = self.twc
        qState = pyquaternion.Quaternion(matrix=self.Rwc)
        state[3] = qState.x
        state[4] = qState.y
        state[5] = qState.z
        state[6] = qState.w
        return state


    def stateToRt(self):
        trans = self.state[0:3]
        R = vec4_to_quat(self.state[3:7]).rotation_matrix
        return R, trans


    def rotatePw(self, Pw: np.ndarray):
        return self.Rwc.transpose().dot(Pw-self.twc)


    def plus(self, delta: np.ndarray):
        """
        add local dimension delta to state of this vertex, the state is global dimension\n
        for translation just add.\n
        for rotation apply right multiplication. \n
        :param delta:  6x1, 6 is local dimension, [0,3) translaton, [3,6) rotation
        :return:
        """
        self.state[0:3] += delta[0:3]
        stateRotation: np.ndarray = vec4_to_quat(self.state[3:7]).rotation_matrix

        deltaRotation = scp.expm(numeric.hat(delta[3:6]))

        spd = stateRotation.dot(deltaRotation)
        spdQ = pyquaternion.Quaternion(matrix=spd)

        self.state[3:7] = quat_to_vec4(spdQ)




class Point(Graph.Vertex):
    def __init__(self, state, ifDense = True):
        super().__init__(1, 1, ifDense)
        self.setState(state)


class EdgeReprojection(Graph.Edge):
    def __init__(self,vertices:[], measurements: []):
        assert len(vertices) == 3 #必须与三个节点相关联，第一个为地图点，第二个为相机i，第三个为相机j
        assert len(measurements) == 2 #观测数据为两个，分布为host，target帧上的相机归一化坐标
        super().__init__(2, vertices)
        self.measurement_i: np.ndarray = measurements[0]
        self.measurement_j: np.ndarray = measurements[1]

    def computeJacobians(self):
        point:Point = self.getVertices()[0]
        frame_i: Frame = self.getVertices()[1]
        frame_j: Frame = self.getVertices()[2]

        inv_dep_i = float(point.state)
        rotationI, transI = frame_i.stateToRt()
        rotationJ, transJ = frame_j.stateToRt()

        pts_camera_i = self.measurement_i / inv_dep_i # 观测点重投影到3D空间
        pts_w = rotationI.dot(pts_camera_i)+transI
        pts_camera_j = rotationJ.transpose().dot(pts_w-transJ) # 变换到相机坐标

        dep_j = pts_camera_j[2]
        reduce = np.array([[float(1.0 / dep_j),    0,          float(-pts_camera_j[0]/(dep_j*dep_j))],
                           [0,              float(1.0/dep_j),  float(-pts_camera_j[1]/(dep_j*dep_j))]],  dtype=np.float)
        jacobian_pose_i = np.zeros((2,6), dtype=np.float)
        jaco_i = np.zeros((3,6), dtype=np.float)
        jaco_i[:,0:3] = rotationJ.transpose()
        jaco_i[:,3:6] = rotationJ.transpose().dot(rotationI).dot(numeric.hat(-pts_camera_i))
        jacobian_pose_i[:,:] = reduce.dot(jaco_i)

        jacobian_pose_j = np.zeros((2,6), dtype=np.float)
        jaco_j = np.zeros((3,6), dtype=np.float)
        jaco_j[:,0:3] = -rotationJ.transpose()
        jaco_j[:,3:6] = numeric.hat(pts_camera_j)
        jacobian_pose_j[:,:] = reduce.dot(jaco_j)

        jacobian_point = np.zeros((2,1), np.float)
        jacobian_point[:,:] = -1.0/(inv_dep_i*inv_dep_i)*reduce.dot(rotationJ.transpose()).dot(rotationI).dot(self.measurement_i)

        self.jacobians[0][:,:] = jacobian_point
        self.jacobians[1][:,:] = jacobian_pose_i
        self.jacobians[2][:,:] = jacobian_pose_j
        pass

    def computeResidual(self):
        point:Point = self.getVertices()[0]
        frame_i: Frame = self.getVertices()[1]
        frame_j: Frame = self.getVertices()[2]

        inv_dep_i = float(point.state)
        rotationI, transI = frame_i.stateToRt()
        rotationJ, transJ = frame_j.stateToRt()

        pts_camera_i = self.measurement_i / inv_dep_i # 观测点重投影到3D空间
        pts_w = rotationI.dot(pts_camera_i)+transI
        pts_camera_j = rotationJ.transpose().dot(pts_w-transJ) # 变换到相机坐标
        estimation_j = pts_camera_j/pts_camera_j[2] # 投影到归一化平面
        self.residual[:,:] = (estimation_j - self.measurement_j)[0:2]
        pass


def generateSImDataInWordFrame(nFeatures = 20, nFrames = 3):
    radius = 8.0
    points = []
    frames = []
    for n in range(nFrames):
        theta = n*2.0*math.pi/(nFrames * 4)
        Rwc = pyquaternion.Quaternion._from_axis_angle(np.array([0.0,0.0,1.0]),theta).rotation_matrix
        twc = np.array([[radius*math.cos(theta)-radius, radius*math.sin(theta), math.sin(2.0*theta)]]).transpose()
        frames.append([Rwc,twc, dict()])
        # print(Rotation.rotation_matrix)
        # print(translation)
    for jj in range(nFeatures):
        Pw = np.array([[random.uniform(-4,4), random.uniform(-4,4), random.uniform(4,8)]], dtype=float).transpose()
        points.append(Pw)
        for ii in range(nFrames):
            Rwc: np.ndarray = frames[ii][0]
            twc: np.ndarray = frames[ii][1]
            observation: dict = frames[ii][2]
            Pc = Rwc.transpose().dot(Pw - twc)
            Pc = Pc / Pc[2] # pc 是归一化像平面坐标
            Pc[0] += random.normalvariate(0,1.0/1000.0)
            Pc[1] += random.normalvariate(0,1.0/1000.0)
            observation[jj] = Pc
    return frames, points

def formAndSolveProblem(ifPointDense, nPoints, nCameras):
    frames, points = generateSImDataInWordFrame(nFeatures=nPoints, nFrames=nCameras)
    print('problem form done')
    problem = Problem.Problem()

    vertexFrames = []
    for ii in range(len(frames)):
        vertexFrame = Frame(frames[ii][0], frames[ii][1], frames[ii][2])
        if(ii == 0):
            vertexFrame.setFix(True)
        problem.addVertex(vertexFrame)
        vertexFrames.append(vertexFrame)

    vertexPoints = []
    noise_invd = []
    for ii in range(len(points)):
        Pw = points[ii]
        Pc = vertexFrames[0].rotatePw(Pw)
        noise = random.normalvariate(0,1.0)
        inverse_depth = 1.0 / (Pc[2] + noise)
        noise_invd.append(inverse_depth)

        vertexPoint: Point = Point(np.array([[float(inverse_depth)]], dtype=np.float),ifPointDense)

        problem.addVertex(vertexPoint)
        vertexPoints.append(vertexPoint)

        for jj in range(1,len(frames)):
            pt_i: np.ndarray = vertexFrames[0].pointsObs.get(ii)
            pt_j: np.ndarray = vertexFrames[jj].pointsObs.get(ii)
            edge = EdgeReprojection([vertexPoint,vertexFrames[0], vertexFrames[jj]], [pt_i,pt_j])
            problem.addEdge(edge)
    print('graph build done')
    problem.Solve(10)
    print('points status')
    for ii in range(len(points)):
        print("point", ii, "gt", 1./points[ii][2], "noise", noise_invd[ii], "opt", vertexPoints[ii].state)
    print('frame status')
    for ii in range(len(frames)):
        print("point", ii, "gt", frames[ii][1].transpose(), 'init', vertexFrames[ii].initState[0:3].transpose(), 'opt', vertexFrames[ii].state[0:3].transpose())
    import matplotlib.pyplot as plt
    plt.imshow(problem.getHessionDense())
    plt.title('hession matrix')
    plt.figure()
    plt.imshow(problem.getHessionDense() > 0.01)
    plt.title('hession value greater than 0.01')
    plt.show()
    normdelta = np.abs(problem.getHessionDense() - problem.getHessionDense().transpose())
    print(np.sum(normdelta), normdelta.max(), normdelta.min(), normdelta.mean())

if __name__ == '__main__':
    random.seed(0)
    time_start = time.time()
    formAndSolveProblem(True,nPoints=2000, nCameras=10)
    time_end = time.time()
    print('totally cost', time_end - time_start)