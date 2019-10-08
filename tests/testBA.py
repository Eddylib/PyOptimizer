from utils import numeric, ProblemLoader
from backend import Graph
from backend import Problem
import numpy as np
import pyquaternion
import scipy.linalg as scp


class BACameraVertex(Graph.Vertex):
    def __init__(self, initState:np.ndarray):
        """
        初始化相机参数，只需要初始状态

        :param initState: 初始状态9x1
        """
        super().__init__(9, 9, True)
        self.state = initState
    pass

    def plus(self, delta: np.ndarray):
        """
        apply delta to state, the state is global state,
        but the delta is local delta .

        :param delta:  A 9x1 numpy array
        :return:
        """
        rotateDelta = scp.expm(numeric.hat(delta[0:3]))
        rotateDelta = pyquaternion.Quaternion(matrix=rotateDelta)

        angleAxis = self.state[0:3]
        angle = np.linalg.norm(angleAxis)
        axis = angleAxis/angle
        rotateState = pyquaternion.Quaternion(axis=axis, angle=angle)

        stateMDeltaRotation = rotateDelta*rotateState
        stateMdeltaTrans = rotateDelta.rotate(self.state[3:6]) + delta[3:6]

        self.state[0:3] = stateMDeltaRotation.axis*stateMDeltaRotation.angle
        self.state[3:6] = stateMdeltaTrans
        self.state[6] += delta[6]
        self.state[7] += delta[7]
        self.state[8] += delta[8]
        pass


class BAPointVertex(Graph.Vertex):
    def __init__(self, initState: np.ndarray):
        """
        初始化地图点参数，只需要初始地图点状态

        :param initState: 初始状态3x1
        """
        super().__init__(3, 3, True)
        self.state = initState
    pass


class BAEdge(Graph.Edge):
    def __init__(self, vCamera: Graph.Vertex, vPoint: Graph.Vertex, observe: np.ndarray):
        """
        在edge中，第一个vertex为相机，第二个vertex为地图点

        :param vCamera: 相机图节点
        :param vPoint: 地图点图节点
        :param observe: 观测，即地图点在图像上的位置
        """
        super().__init__(2, [vCamera, vPoint])
        assert observe.shape[0] == 2
        self.observation = observe

    def computeResidual(self):
        vertices: [Graph.Vertex] = self.getVertices()
        camVertex: BACameraVertex = vertices[0]
        pointVertex: BAPointVertex = vertices[1]
        assert camVertex is BACameraVertex
        assert pointVertex is BAPointVertex
        angleAxis = camVertex.state[0:3]
        angle = np.linalg.norm(angleAxis)
        axis = angleAxis/angle
        self.rotation = pyquaternion.Quaternion(axis=axis, angle=angle)
        translation = camVertex.state[3:6]
        Pw = pointVertex.state
        self.Pc = self.rotation.rotate(Pw) + translation
        self.pc = np.array([-self.Pc[0]/self.Pc[2], -self.Pc[1]/self.Pc[2]], dtype=np.float)
        self.focal = float(camVertex.state[6])
        self.l1 = float(camVertex.state[7])
        self.l2 = float(camVertex.state[8])
        self.r2 = self.pc.dot(self.pc)
        self.distortion = 1.0 + self.r2 * (self.l1 + self.l2 * self.r2)
        self.pfinal = self.focal * self.distortion * self.pc
        self.residual = self.pfinal - self.observation
        pass

    def computeJacobians(self):

        A = self.r2
        B = (self.l1 + A * self.l2)
        jdrdpc = self.focal * (
                2. * (B + A * self.l2) * self.pc * self.pc.transpose()
                + self.distortion * np.identity(2)
        )

        jdpcdPc = np.array([
            [-1./self.Pc[2],  0,          self.Pc[0]/(self.Pc[2]*self.Pc[2])],
            [0,          -1./self.Pc[2],  self.Pc[1]/(self.Pc[2]*self.Pc[2])]], dtype=np.float)

        jdrdPc = jdrdpc*jdpcdPc
        jdPcdxi = np.zeros((3,6))
        jdPcdxi[:, 0:3] = np.identity(3)
        jdPcdxi[:, 3:6] = numeric.hat(-self.Pc)
        self.jacobians[0][:,0:6] = jdrdPc*jdPcdxi
        self.jacobians[0][:, 6] = self.distortion * self.pc
        self.jacobians[0][:, 7] = self.focal * self.r2 * self.pc
        self.jacobians[0][:, 8] = self.focal * self.r2 * self.r2 * self.pc

        self.jacobians[1] = jdrdPc.dot(self.rotation.rotation_matrix)
        pass

def test_func():
    loader = ProblemLoader.ProblemLoader('data/problem-49-7776-pre.txt')
    problem = Problem.Problem(loader.num_cameras+loader.num_points)
    for ii in range(loader.num_cameras):
        camV: BACameraVertex = BACameraVertex(loader.camera_data[ii])
        camV.setIdx(ii)
        problem.addVertex(camV)
    for ii in range(loader.num_points):
        pointV: BAPointVertex = BAPointVertex(loader.point_data[ii])
        pointV.setIdx(ii+loader.num_cameras)
        problem.addVertex(pointV)

    for ii in range(loader.num_observations):
        camIdx = loader.observations[ii][0]
        ptIdx = loader.observations[ii][1] + loader.num_cameras
        observation = loader.observations[ii][2]
        edge: BAEdge = BAEdge(problem.getVertex(camIdx), problem.getVertex(ptIdx), observation)
        problem.addEdge(edge)
    problem.Solve(100)
    pass