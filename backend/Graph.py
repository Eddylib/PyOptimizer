from abc import ABC, abstractmethod
import numpy as np


class Edge(ABC):
    __EdgeIndex = 0

    def __init__(self, residualDim: int, vertices: []):
        self.__idx = Edge.__EdgeIndex
        self.__residualDim = residualDim
        Edge.__EdgeIndex += 1

        # 残差和雅克比
        self.jacobians = []
        self.__hessions = []
        self.__vertices = vertices
        self.residual = np.zeros([residualDim,1],dtype=np.float)
        for ii in range(0, len(vertices)):
            self.jacobians.append(np.zeros([self.__residualDim, vertices[ii].getLocalDimension()]))
            self.__hessions.append(np.zeros([vertices[ii].getLocalDimension(),
                                             vertices[ii].getLocalDimension()]))
            # 添加索引
            vertices[ii].addEdge(self)

        self.__information = np.identity(self.__residualDim)
        pass

    def getResidual(self):
        return self.residual

    def errorNorm2(self):
        return self.residual.transpose().dot(self.__information).dot(self.residual)

    def getIdx(self):
        return self.__idx

    def getInformation(self):
        return self.__information

    def getJacobians(self):
        return self.jacobians

    def getHessions(self):
        return self.__hessions

    def setHession(self, hession: np.ndarray, idx: int):
        self.__hessions[idx] = hession

    def getVertices(self) -> []:
        return self.__vertices

    def __str__(self):
        return "Edge: " + str(self.getIdx())

    @abstractmethod
    def computeResidual(self):
        pass

    @abstractmethod
    def computeJacobians(self):
        pass


class VertexABC(ABC):
    __vertexIndex = 0

    def __init__(self, dimension, localdimension=-1, ifDense=True):
        """
        :param dimension: 数据存储维度
        :param localdimension: 雅克比计算维度
        :param ifDense: 是否为舒尔补的稠密部分的节点
        """
        self.__idx = VertexABC.__vertexIndex
        self.__ifDense = ifDense
        self.__dimension = dimension
        self.__local_dimension = localdimension
        self.__ifFix = False
        self.__edges = dict()
        self.__window_idx = 0
        if self.__local_dimension < 0:
            self.__local_dimension = dimension
        VertexABC.__vertexIndex += 1

        self.state = np.zeros((self.__dimension, 1), dtype=np.float)
        pass

    def setWindowIdx(self, idx: int):
        self.__window_idx = idx

    def getWindowIdx(self):
        return self.__window_idx

    def getState(self):
        return self.state

    def setState(self, state: np.ndarray):
        assert state.shape == self.state.shape
        self.state = state

    def getIdx(self):
        return self.__idx

    def setIdx(self, idx: int):
        self.__idx = idx


    def ifDense(self):
        return self.__ifDense

    def getLocalDimension(self):
        return self.__local_dimension

    def setFix(self, ifFix: bool):
        self.__ifFix = ifFix

    def ifFix(self):
        return self.__ifFix

    def addEdge(self, edge):
        find = self.__edges.get(edge.getIdx())
        assert find is None
        self.__edges[edge.getIdx()] = edge

    def __str__(self):
        return "Vertex: " + str(self.getIdx())


    @abstractmethod
    def plus(self, delta: np.ndarray):
        """
        :param delta: local dimension of delta
        :return: None

        plus local tangent space delta to the global space state
        """
        pass


class Vertex(VertexABC):
    def plus(self, delta: np.ndarray):
        self.state += delta
