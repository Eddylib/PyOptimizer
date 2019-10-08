import numpy as np
class ProblemLoader:
    def __init__(self, file_name):
        """
        elf.observations: 观测， 分别为：相机索引， 点的索引， 图像位置x,y
        """
        with open(file_name, 'r') as f:
            lines = f.readlines()
        self.num_cameras, self.num_points, self.num_observations = [int(item) for item in lines[0].split(' ')]
        obsdata = lines[1:self.num_observations+1]
        self.observations = [[int(item.split()[0]), int(item.split()[1]), np.array([float(item.split()[2]), float(item.split()[3])], dtype=np.float).reshape((2,1))] for item in obsdata]

        cameradata = lines[self.num_observations+1:self.num_observations+1+9*self.num_cameras]
        pointsdata = lines[self.num_observations+1+9*self.num_cameras:]
        assert  len(pointsdata)/3 == self.num_points

        cameras_data_np = np.array(cameradata, dtype=np.float)
        points_data_np = np.array(pointsdata, dtype=np.float)
        self.camera_data = np.reshape(cameras_data_np, (int(cameras_data_np.shape[0]/9), 9))
        self.point_data = np.reshape(points_data_np, (int(points_data_np.shape[0]/3), 3))
        pass