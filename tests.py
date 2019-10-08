import unittest
import backend.Problem as Problem
import backend.Graph as Graph
import numpy as np
import random
import sys


def generateObservation():
    a = 1.0
    b = 2.0
    c = 1.0
    N = 2000
    w_sigma = 2.0
    obx = []
    oby = []
    for ii in range(0, N):
        x: float = float(ii) / 100.0
        noise = random.normalvariate(0, w_sigma)
        y: float = a * x * x * x + b * x * x + c * x + noise
        obx.append(x)
        oby.append(y)
    return obx, oby


class CFVertex(Graph.Vertex):
    def __init__(self, a, b, c):
        super().__init__(3, 3, True)
        self.state[0] = a
        self.state[1] = b
        self.state[2] = c


class CFEdge(Graph.Edge):
    def __init__(self, vertex, x: float, y: float):
        """
        curve: :math:`y=exp(ax^2 + bx +c)`

        :param vertex: vertex of dimension 3, for a, b, c
        :param x: observation x
        :param y: observation y
        """
        super().__init__(1, [vertex])
        self.__x = x
        self.__y = y

    def computeResidual(self):
        """
        curve: :math:`y=exp(ax^2 + bx +c)`

        :return: None
        """
        state: np.ndarray = self.getVertices()[0].state
        self.residual[0] = state[0] * self.__x * self.__x * self.__x + state[1] * self.__x * self.__x + state[
            2] * self.__x - self.__y
        pass

    def computeJacobians(self):
        """
        jacobians of curve: :math:`y=exp(ax^2 + bx +c)`. \n
        :math:`dy/da = exp(ax^2 + bx +c)x^2` \n
        :math:`dy/db = exp(ax^2 + bx +c)x` \n
        :math:`dy/dc = exp(ax^2 + bx +c)` \n

        :return: None
        """
        state: np.ndarray = self.getVertices()[0].state

        self.jacobians[0][0, 0] = self.__x * self.__x * self.__x
        self.jacobians[0][0, 1] = self.__x * self.__x
        self.jacobians[0][0, 2] = 1.0 * self.__x
        pass


class TestFunc(unittest.TestCase):
    # 继承自unittest.TestCase
    # 重写TestCase的setUp()、tearDown()方法：在每个测试方法执行前以及执行后各执行一次
    def setUp(self):
        print("do something before test : prepare environment")
        random.seed(0)

    def tearDown(self):
        print("do something after test : clean up ")

    def main(self):
        print('test main')
        sys.stdout.flush()
        problem = Problem.Problem(10)
        vAdd = CFVertex(0, 0, 0)
        problem.addVertex(vAdd)

        obx, oby = generateObservation()
        for ii in range(len(obx)):
            eAdd = CFEdge(vAdd, obx[ii], oby[ii])
            problem.addEdge(eAdd)
        problem.Solve(100)
        print(vAdd.state.transpose())
        sys.stdout.flush()

    def loader(self):
        print('test loader')
        from utils import ProblemLoader
        loader = ProblemLoader.ProblemLoader('data/problem-49-7776-pre.txt')

    def simpleBA(self):
        print('simple BA')
        import tests.testBA as thistest
        thistest.test_func()



if __name__ == '__main__':
    tests = [TestFunc('main')]
    suite = unittest.TestSuite()
    suite.addTests(tests)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
