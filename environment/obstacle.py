import os
import inspect

class Obstacle():

    def __init__(self, client, init_pos):
        self.client = client
        self.currPos = init_pos
        self.orn = [0,0,1,1]

    def loadURDF(self):
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        urdfPath = currentdir + "/urdf/sphere.urdf"
        self.point = self.client.loadURDF(urdfPath,
                        self.currPos,
                        useFixedBase=True,
                        globalScaling=20)
        return self.point

    def resetPosition(self, newPos):
        self.currPos = newPos
        self.client.resetBasePositionAndOrientation(self.point, self.currPos, self.orn)

    def getCurrPos(self):
        self.currPos, _ =  self.client.getBasePositionAndOrientation(self.point)
        self.currPos = list(self.currPos)
        return self.currPos