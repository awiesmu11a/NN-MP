import numpy as np

class Point():

    def __init__(self, client, init_pos):
        self.client = client
        self.currPos = init_pos
        self.orn = [0,0,1,1]

    def loadURDF(self):
        urdfPath = "urdf/sphere.urdf"
        self.point = self.client.loadURDF(urdfPath,
                        self.currPos,
                        useFixedBase=False,
                        globalScaling=5)
        return self.point

    def resetPosition(self, newPos):
        self.currPos = newPos
        self.client.resetBasePositionAndOrientation(self.point, self.currPos, self.orn)

    def act_force_continous(self, control):
        force = [10*control[0]*np.cos(np.pi*control[1]), 10*control[0]*np.sin(np.pi*control[1]), 0]
        self.currPos, _ = self.client.getBasePositionAndOrientation(self.point)
        self.client.applyExternalForce(self.point, -1, force, self.currPos, self.client.WORLD_FRAME)

    def act_force(self, control):
        self.currPos, _ = self.client.getBasePositionAndOrientation(self.point)
        if control==0:
            self.client.applyExternalForce(self.point, -1, [10,0,0], self.currPos, self.client.WORLD_FRAME)
        elif control==1:
            self.client.applyExternalForce(self.point, -1, [-10,0,0], self.currPos, self.client.WORLD_FRAME)
        elif control==2:
            self.client.applyExternalForce(self.point, -1, [0,10,0], self.currPos, self.client.WORLD_FRAME)
        elif control==3:
            self.client.applyExternalForce(self.point, -1, [0,-10,0], self.currPos, self.client.WORLD_FRAME)
        elif control==4:
            pass
        else:
            print("Invalid control")
            return

    def act_movement(self, control):
        if control==0:
            self.currPos[0] += 1.0
        elif control==1:
            self.currPos[0] -= 1.0
        elif control==2:
            self.currPos[1] += 1.0
        elif control==3:
            self.currPos[1] -= 1.0
        else:
            print("Invalid control")
            return
        
        self.client.resetBasePositionAndOrientation(self.point, self.currPos, self.orn)
        
    def getVel(self):
        vel, _ = self.client.getBaseVelocity(self.point)
        return [vel[0], vel[1]]

    # def checkMaxVelPos(self):
    #     max_vel_x, max_vel_y = 40,40
    #     vel_x,vel_y = self.getVel()
    #     if ((vel_x <= max_vel_x) and (vel_y <= max_vel_y)):
    #         return True
    #     return False
    
    # def checkMaxVelNeg(self):
    #     max_vel_x, max_vel_y = -40,-40
    #     vel_x,vel_y = self.getVel()
    #     if ((vel_x >= max_vel_x) and (vel_y >= max_vel_y)):
    #         return True
    #     return False

    def getCurrPos(self):
        # print(self.currPos)
        self.currPos, _ =  self.client.getBasePositionAndOrientation(self.point)
        self.currPos = list(self.currPos)
        # print(self.currPos)
        return self.currPos
