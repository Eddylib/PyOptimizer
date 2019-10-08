import numpy as np
from backend.Graph import Vertex as Vertex
from backend.Graph import Edge as Edge


class Problem:
    def __init__(self, max_parm_window: int = 0):
        """

        :param max_parm_window:  滑窗优化的最大窗口大小（未实现）
        """
        self.__max_parm_window = max_parm_window
        self.__edges = dict()  # id to edge instance
        self.__vertices_dense = dict()  # id to vertex instance
        self.__vertices_sparse = dict()

        self.__hessian_dense = None  # J^T E J
        self.__hessian_sparse_C = None
        self.__hessian_sparse_E = None
        self.__hessian_vertex_idx_dense_and_sparse = None  #存着所有节点的id，包括dense和sparse
        self.__hessian_block_idx_dense_and_sparse = None
        self.__b_dense_and_sparse = None  # -J * r
        self.__delta_all = None


        self.__lm_curr = 0
        self.__total_err = 0
        self.__lm_ni = 0
        self.__lm_err_stop = 0


    def addVertex(self, vertex: Vertex):
        if(vertex.ifDense()):
            finded = self.__vertices_dense.get(vertex.getIdx())
            assert (finded is None and "vertex: exists in problem")
            self.__vertices_dense[vertex.getIdx()] = vertex
        else:
            finded = self.__vertices_sparse.get(vertex.getIdx())
            assert (finded is None and "vertex: exists in problem")
            self.__vertices_sparse[vertex.getIdx()] = vertex

    def addEdge(self, edge: Edge):
        finded = self.__edges.get(edge.getIdx())
        assert (finded is None and "Edge: exists in problem")
        self.__edges[edge.getIdx()] = edge

    def getVertex(self, idx: int) -> Vertex:
        finded = self.__vertices_dense.get(idx)
        return finded

    def getEdge(self, idx: int)->Edge:
        finded = self.__edges.get(idx)
        return finded

    def __str__(self):
        return 'Problem with ' + str(len(self.__edges)) + ' edges and ' + str(len(self.__vertices_dense)) + ' vertices'

    def __setOrdering(self):
        """
        统计Hession内的节点的id，存于self.__hessian_vertex_idx。\n
        统计Hession内每个节点的数组起始下标，存于self.__hessian_block_idx。\n
        统计Hession大小，存于self.__hDenseSize\n
        :return:
        """
        cnt = 0
        nDenseVertex = len(self.__vertices_dense)
        nSparseVertex = len(self.__vertices_sparse)
        hDenseSize = 0
        self.__hessian_vertex_idx_dense_and_sparse = np.zeros((nDenseVertex + nSparseVertex), dtype=np.int)
        self.__hessian_block_idx_dense_and_sparse = np.zeros((nDenseVertex + nSparseVertex + 1), dtype=np.int)
        for vertex_key in self.__vertices_dense.keys():
            assert self.__vertices_dense[vertex_key].ifDense()
            vertex: Vertex = self.__vertices_dense[vertex_key]
            vertex.setWindowIdx(cnt)
            self.__hessian_vertex_idx_dense_and_sparse[cnt] = vertex.getIdx()
            self.__hessian_block_idx_dense_and_sparse[cnt] = hDenseSize
            hDenseSize += vertex.getLocalDimension()
            cnt += 1
        hSparseSize = 0
        for vertex_key in self.__vertices_sparse.keys():
            assert not self.__vertices_sparse[vertex_key].ifDense()
            vertex: Vertex = self.__vertices_sparse[vertex_key]
            vertex.setWindowIdx(cnt)
            self.__hessian_vertex_idx_dense_and_sparse[cnt] = vertex.getIdx()
            self.__hessian_block_idx_dense_and_sparse[cnt] = hDenseSize + hSparseSize
            hSparseSize += vertex.getLocalDimension()
            cnt += 1
        self.__hessian_block_idx_dense_and_sparse[cnt] = hDenseSize + hSparseSize
        self.__hDenseSize = hDenseSize
        self.__hSparseSize = hSparseSize
        return


    def getHessionDense(self):
        return self.__hessian_dense

    def __makeHession(self):
        """
        H:\n
        B,  E\n

        E^T, C\n
        E: nSparseVertex个块\n
        :math:`E_{i,k}= \sum_r (dr/dd_i)^T * (dr/ds_k ) k\in [0,nSparseVertex)`，注意实现中按i索引而不是按k，
        这样避免超长向量的出现\n
        :math:`C_{k,k}= \sum_r (dr/ds_k)^T * (dr/dps_k ) k\in [0,nSparseVertex)`\n
        :return:
        """
        nDenseVertex = len(self.__vertices_dense)
        nSparseVertex = len(self.__vertices_sparse)
        self.__hessian_dense = np.zeros([self.__hDenseSize, self.__hDenseSize])
        self.__b_dense_and_sparse = np.zeros([self.__hDenseSize + self.__hSparseSize, 1])
        self.__hessian_sparse_C = [] #every element of C is a matirx of DimSparseVertex x DimSparseVertex
        self.__hessian_sparse_E = [] #every element of E is a matirx of DimDenseVertex x self.__hSparseSize, very long
        for ii in range(nSparseVertex):
            vertex: Vertex = self.__vertices_sparse.get(self.__hessian_vertex_idx_dense_and_sparse[nDenseVertex + ii])
            self.__hessian_sparse_C.append(np.zeros([vertex.getLocalDimension(), vertex.getLocalDimension()], dtype=np.float))
        for ii in range(nDenseVertex):
            vertex: Vertex = self.__vertices_dense.get(self.__hessian_vertex_idx_dense_and_sparse[ii])
            self.__hessian_sparse_E.append(np.zeros([vertex.getLocalDimension(),self.__hSparseSize], dtype=np.float))

        for edge_key in self.__edges.keys():
            # edge = Edge.Edge(1,[])
            edge = self.__edges[edge_key]
            edge.computeResidual()
            edge.computeJacobians()
            vertices = edge.getVertices()
            jacobians = edge.getJacobians()
            nVertexOfThisEdge = len(vertices)
            for ii in range(0, nVertexOfThisEdge):
                vertex_i: Vertex = vertices[ii]
                if vertex_i.ifFix():
                    continue
                jacobian_i = jacobians[ii]
                jtw = jacobian_i.transpose().dot( edge.getInformation())
                idx_i = vertex_i.getWindowIdx()
                block_start_i = self.__hessian_block_idx_dense_and_sparse[idx_i]
                block_end_i = self.__hessian_block_idx_dense_and_sparse[idx_i] + vertex_i.getLocalDimension()
                self.__b_dense_and_sparse[block_start_i:block_end_i] -= jtw.dot(edge.getResidual())
                if not vertex_i.ifDense():
                    #由于E和B分开存放，E的稀疏部分的列下标应该减去稠密部分的总行数
                    # 但是b不用这样做
                    block_start_i -= self.__hDenseSize
                    block_end_i -= self.__hDenseSize
                for jj in range(ii, nVertexOfThisEdge):
                    vertex_j: Vertex = vertices[jj]
                    if vertex_j.ifFix():
                        continue
                    idx_j = vertex_j.getWindowIdx()
                    jacobian_j = jacobians[jj]
                    hessian = jtw.dot(jacobian_j)
                    # 以下四个都以0开始
                    block_start_j = self.__hessian_block_idx_dense_and_sparse[idx_j]
                    block_end_j = self.__hessian_block_idx_dense_and_sparse[idx_j] + vertex_j.getLocalDimension()
                    if not vertex_j.ifDense():
                        block_start_j -= self.__hDenseSize
                        block_end_j -= self.__hDenseSize

                    if vertex_i.ifDense() and vertex_j.ifDense():# 如果vi,vj都是dense，那么直接放到hessian矩阵里即可
                        self.__hessian_dense[block_start_i :block_end_i, block_start_j:block_end_j] += hessian
                        if ii != jj:
                            self.__hessian_dense[block_start_j :block_end_j, block_start_i:block_end_i] += hessian.transpose()
                        pass
                    elif vertex_i.ifDense() and not vertex_j.ifDense():# 如果vi是dense， vj是sparse， 那么放到E里，
                        self.__hessian_sparse_E[vertex_i.getWindowIdx()][:,block_start_j:block_end_j] += hessian
                        pass
                    elif not vertex_i.ifDense() and vertex_j.ifDense():# 如果vi是sparse， vj是dense， 这种情况有可能出现，也放到E里
                        self.__hessian_sparse_E[vertex_j.getWindowIdx()][:,block_start_i:block_end_i] += hessian.transpose()
                        pass
                    elif not vertex_i.ifDense() and not vertex_j.ifDense():# 如果vi,vj都是sparse， 那么放到C里
                        # sparse点和sparse点不能产生关联！一个edge只能连接一个sparse点，否则会在sparse点间产生关联
                        # todo 如果不标记节点是否稀疏，该如何检测哪些点是稀疏的？似乎是个NP难问题？
                        assert ii == jj
                        self.__hessian_sparse_C[vertex_i.getWindowIdx()-nDenseVertex] += hessian
        pass

    def __initLM(self):
        """
        choose max element of diagonal of hessian as initial lm factur $\mu$
        :return: None
        """
        self.__lm_ni = 2.0
        self.__lm_curr = -1.0
        self.__total_err = 0.0

        for edge_key in self.__edges.keys():
            # edge = Edge.Edge(0,[])
            edge = self.__edges[edge_key]
            self.__total_err += edge.errorNorm2()
        self.__lm_err_stop = 1e-30 * self.__total_err

        maxDiag = 0
        for ii in range(self.__hDenseSize):
            maxDiag = max(maxDiag, self.__hessian_dense[ii, ii])
        for kk in range(len(self.__vertices_sparse)):
            for jj in range(self.__hessian_sparse_C[kk].shape[0]):
                maxDiag = max(maxDiag,self.__hessian_sparse_C[kk][jj,jj])

        tau = 1e-5
        self.__lm_curr = tau * maxDiag

    def __applyLM(self):
        for ii in range(self.__hDenseSize):
            self.__hessian_dense[ii, ii] += self.__lm_curr
        for ii in range(len(self.__hessian_sparse_C)):
            assert(self.__hessian_sparse_C[ii].shape[0] == self.__hessian_sparse_C[ii].shape[1])
            for jj in range(self.__hessian_sparse_C[ii].shape[0]):
                self.__hessian_sparse_C[ii][jj,jj] += self.__lm_curr

    def __removeLM(self):
        for ii in range(self.__hDenseSize):
            self.__hessian_dense[ii, ii] -= self.__lm_curr
        for ii in range(len(self.__hessian_sparse_C)):
            assert(self.__hessian_sparse_C[ii].shape[0] == self.__hessian_sparse_C[ii].shape[1])
            for jj in range(self.__hessian_sparse_C[ii].shape[0]):
                self.__hessian_sparse_C[ii][jj,jj] -= self.__lm_curr

    def __isGoodLMStep(self):
        L0_sub_Ldx = self.__delta_all.transpose().dot(self.__lm_curr * self.__delta_all + self.__b_dense_and_sparse)
        L0_sub_Ldx += 0.3
        F_x_plus_dx = 0
        for edge_key in self.__edges.keys():
            # edge = Edge.Edge(0,[])
            edge = self.__edges[edge_key]
            edge.computeResidual()
            F_x_plus_dx += edge.errorNorm2()

        rho = float((self.__total_err - F_x_plus_dx) / L0_sub_Ldx)
        # miu增大， 减速， miu减小， 加速！
        if (rho > 0 and np.isfinite(F_x_plus_dx)):  # 如果误差在下降，降低miu，使得系统更接近高斯牛顿
            alpha = min(1.0 - (2.0 * rho - 1) ** 3, 2.0 / 3.0)
            factor_final = max(1.0 / 3.0, alpha)
            self.__lm_curr *= factor_final
            self.__lm_ni = 2
            self.__total_err = F_x_plus_dx
            return True
        else:  # 否则增加miu使得更接近最速下降法，且步长相当于1除以miu，被降低了
            self.__lm_curr *= self.__lm_ni
            self.__lm_ni *= 2
            return False


    def __solveLinearSystemByMatrixInverse(self):
        # import matplotlib.pyplot as plt
        # plt.imshow(self.__hessian_dense)
        # plt.show()
        # np.save('__solveLinearSystemByMatrixInverse_h',self.__hessian_dense)
        # np.save('__solveLinearSystemByMatrixInverse_b', self.__b_dense_and_sparse)
        # exit(0)
        self.__delta_all = np.zeros([self.__hDenseSize+self.__hSparseSize,1], dtype=np.float)
        self.__delta_all[0:self.__hDenseSize] = np.linalg.inv(self.__hessian_dense).dot(self.__b_dense_and_sparse)

    def __solveLinearSystemByReconstructedMatrixInverse(self):
        nSparseVertex = len(self.__vertices_sparse)
        nDenseVertex = len(self.__vertices_dense)
        hessianSize = self.__hSparseSize+self.__hDenseSize
        hessianAll = np.zeros([hessianSize,hessianSize], np.float)
        hessianAll[0:self.__hDenseSize, 0:self.__hDenseSize] = self.__hessian_dense

        for kk in range(0,nSparseVertex):
            block_start_k = self.__hessian_block_idx_dense_and_sparse[kk+nDenseVertex]
            block_end_k = self.__hessian_block_idx_dense_and_sparse[kk+nDenseVertex+1]
            hessianAll[block_start_k:block_end_k,block_start_k:block_end_k] = self.__hessian_sparse_C[kk]
        for ii in range(0, nDenseVertex):
            block_start_i = self.__hessian_block_idx_dense_and_sparse[ii]
            block_end_i = self.__hessian_block_idx_dense_and_sparse[ii+1]
            # data = self.__hessian_sparse_E[ii]
            hessianAll[block_start_i:block_end_i, self.__hDenseSize:] = self.__hessian_sparse_E[ii]
            hessianAll[self.__hDenseSize:,block_start_i:block_end_i] = self.__hessian_sparse_E[ii].transpose()
            pass

        self.__delta_all = np.zeros([self.__hDenseSize+self.__hSparseSize,1], dtype=np.float)
        self.__delta_all[:] = np.linalg.inv(hessianAll).dot(self.__b_dense_and_sparse)

        # import matplotlib.pyplot as plt
        # plt.imshow(hessianAll)
        # plt.show()
        # np.save('__solveLinearSystemByReconstructedMatrixInverse_h',hessianAll)
        # np.save('__solveLinearSystemByReconstructedMatrixInverse_b', self.__b_dense_and_sparse)
        # exit(0)
        pass

    def __solveLinearSystemBySparseSchur(self):
        nDenseVertex = len(self.__vertices_dense)
        nSparseVertex = len(self.__vertices_sparse)
        reserveH = np.zeros(self.__hessian_dense.shape)
        # equationRight分两段， 第一段是稠密部分解方程组的右边，第二段是解稀疏部分的方程右边的E^T \Delta x_d
        equationRight = np.zeros((self.__hDenseSize+self.__hSparseSize,1))
        # todo 这里可以进行大量加速
        for ii in range(nDenseVertex):
            block_start_i = self.__hessian_block_idx_dense_and_sparse[ii]
            block_end_i = self.__hessian_block_idx_dense_and_sparse[ii + 1]
            EiCinv = np.zeros(self.__hessian_sparse_E[ii].shape)
            for kk in range(nSparseVertex):
                block_start_k = self.__hessian_block_idx_dense_and_sparse[kk+nDenseVertex] - self.__hDenseSize
                block_end_k = self.__hessian_block_idx_dense_and_sparse[kk+nDenseVertex + 1] - self.__hDenseSize
                EiCinv[:,block_start_k:block_end_k] += self.__hessian_sparse_E[ii][:,block_start_k:block_end_k].dot(np.linalg.inv(self.__hessian_sparse_C[kk]))
            for jj in range(nDenseVertex):
                block_start_j = self.__hessian_block_idx_dense_and_sparse[jj]
                block_end_j = self.__hessian_block_idx_dense_and_sparse[jj + 1]
                reserveH[block_start_i:block_end_i, block_start_j:block_end_j] += EiCinv.dot(self.__hessian_sparse_E[jj].transpose())
            equationRight[block_start_i:block_end_i] += EiCinv.dot(self.__b_dense_and_sparse[self.__hDenseSize:self.__hDenseSize+self.__hSparseSize])

        # equationRight分两段， 第一段是稠密部分解方程组的右边，第二段是解稀疏部分的方程右边的E^T \Delta x_d
        self.__delta_all = np.zeros([self.__hDenseSize+self.__hSparseSize,1], dtype=np.float)
        self.__delta_all[0:self.__hDenseSize] = np.linalg.inv(self.__hessian_dense-reserveH).dot(
            self.__b_dense_and_sparse[0:self.__hDenseSize]-equationRight[0:self.__hDenseSize])

        if nSparseVertex == 0:
            return

        # equationRight分两段， 第一段是稠密部分解方程组的右边，第二段是解稀疏部分的方程右边的E^T \Delta x_d
        for ii in range(nDenseVertex):
            block_start_i = self.__hessian_block_idx_dense_and_sparse[ii]
            block_end_i = self.__hessian_block_idx_dense_and_sparse[ii + 1]
            equationRight[self.__hDenseSize:self.__hDenseSize + self.__hSparseSize] += self.__hessian_sparse_E[
                ii].transpose().dot(
                self.__delta_all[block_start_i:block_end_i]
            )
        # E^T \Delta x_d 组合完成，接下来用b_sparse减之
        equationRight[self.__hDenseSize: self.__hDenseSize + self.__hSparseSize] = \
            self.__b_dense_and_sparse[self.__hDenseSize: self.__hDenseSize + self.__hSparseSize] -\
            equationRight[self.__hDenseSize: self.__hDenseSize + self.__hSparseSize]

        # 此时的equationRight的稀疏部分左乘C^{-1}就是稀疏节点的改变量
        for kk in range(nSparseVertex):
            block_start_k = self.__hessian_block_idx_dense_and_sparse[kk+nDenseVertex]
            block_end_k = self.__hessian_block_idx_dense_and_sparse[kk+nDenseVertex + 1]
            self.__delta_all[block_start_k:block_end_k] = np.linalg.inv(self.__hessian_sparse_C[kk]).dot(equationRight[block_start_k:block_end_k])
        pass

    def __updataStates(self):
        for ii in range(0, len(self.__hessian_vertex_idx_dense_and_sparse)):
            vertex = self.__vertices_dense.get(self.__hessian_vertex_idx_dense_and_sparse[ii])
            if vertex is None:
                vertex = self.__vertices_sparse[self.__hessian_vertex_idx_dense_and_sparse[ii]]
            vertex.plus(self.__delta_all[self.__hessian_block_idx_dense_and_sparse[ii]:self.__hessian_block_idx_dense_and_sparse[ii+1]])

    def __rollbackStates(self):
        for ii in range(0, len(self.__hessian_vertex_idx_dense_and_sparse)):
            vertex = self.__vertices_dense.get(self.__hessian_vertex_idx_dense_and_sparse[ii])
            if vertex is None:
                vertex = self.__vertices_sparse[self.__hessian_vertex_idx_dense_and_sparse[ii]]
            vertex.plus(-self.__delta_all[self.__hessian_block_idx_dense_and_sparse[ii]:self.__hessian_block_idx_dense_and_sparse[ii+1]])

    def __solveDenseOrSchur(self, numIterations: int):
        # set ordering
        self.__setOrdering()
        # make hession and right vector
        self.__makeHession()
        # exit(0)
        # lm init
        self.__initLM()
        # lm doing
        stop = False
        iter = 0
        if len(self.__vertices_sparse) == 0:
            linear_solver = self.__solveLinearSystemByMatrixInverse
        else:
            linear_solver = self.__solveLinearSystemBySparseSchur

        # linear_solver = self.__solveLinearSystemByReconstructedMatrixInverse

        while not stop and iter < numIterations:
            print("iter", iter, "err_total", self.__total_err, "miu", self.__lm_curr)
            oneStepSucess = False
            falseCnt = 0
            # solve dense part
            while not oneStepSucess:
                self.__applyLM()
                linear_solver()
                self.__removeLM()
                # 终止条件1： deltax 很小
                if np.linalg.norm(self.__delta_all) < 1e-6 or falseCnt > 30:
                    print("stop: delta small or many false try: false tries: ",
                          falseCnt, " norm of delta ",
                          np.linalg.norm(self.__delta_all))
                    stop = True
                    break
                self.__updataStates()
                oneStepSucess = self.__isGoodLMStep()
                if oneStepSucess:
                    self.__makeHession()
                    falseCnt = 0
                else:
                    falseCnt += 1
                    self.__rollbackStates()
            iter += 1

            # 终止条件2：误差下降了1e6倍， 终止
            if np.sqrt(self.__total_err) <= self.__lm_err_stop:
                print("stop: error decreased for 1e6 times")
                stop = True
        print("iter", "final", "err_total", self.__total_err, "miu", self.__lm_curr)

    def Solve(self, numIterations: int):
        self.__solveDenseOrSchur(numIterations)