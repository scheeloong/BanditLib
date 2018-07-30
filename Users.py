import numpy as np
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
import json
from random import choice, randint

class User():
    '''
    Each user has id
    parameter that is disjoint for itself
    parameter that is shared
    '''
    def __init__(self, id, theta=None, CoTheta=None):
        self.id = id # Unique id for this user
        self.theta = theta # parameter for this user
        self.CoTheta = CoTheta # Shared parameter for this user

class UserManager():
    '''
    Keep track of ...
    '''
    def __init__(self, dimension, userNum, UserGroups, thetaFunc, argv=None):
        '''
        '''
        self.dimension = dimension # 
        self.thetaFunc = thetaFunc # 
        self.userNum = userNum # Number of users
        self.UserGroups = UserGroups # The different possible user groups
        self.argv = argv
        self.signature = "A-" + "+PA" + "+TF-" + self.thetaFunc.__name__

    def saveUsers(self, users, filename, force=False):
        '''
        Save all users and their respective parameter values 
        in a json format in file.
        '''
        # Output warning message if going to overwrite
        # Otherwise, output warning if not supposed to overwrite and an existing file exist 
        fileOverWriteWarning(filename, force)
        with open(filename, 'w') as f:
            # Loop through all users
            for i in range(len(users)):
                # print users[i].theta
                # Write a user and it's parameter values in a json list
                f.write(json.dumps((users[i].id, users[i].theta.tolist())) + '\n')

    def loadUsers(self, filename):
        '''
        Load the saved users from the json file
        '''
        users = []
        with open(filename, 'r') as f:
            for line in f:
                id, theta = json.loads(line)
                users.append(User(id, np.array(theta)))
        return users

    def generateMasks(self):
        '''
        '''
        mask = {}
        for i in range(self.UserGroups):
            mask[i] = np.random.randint(2, size=self.dimension)
        return mask

    def simulateThetafromUsers(self):
        '''
        '''
        usersids = {}
        users = []
        mask = self.generateMasks()
        if (self.UserGroups == 0):
            for key in range(self.userNum):
                thetaVector = self.thetaFunc(self.dimension, argv=self.argv)
                l2_norm = np.linalg.norm(thetaVector, ord=2)
                users.append(User(key, thetaVector / l2_norm))
        else:
            for i in range(self.UserGroups):
                usersids[i] = range(self.userNum * i / self.UserGroups,
                                    (self.userNum * (i + 1)) / self.UserGroups)

                for key in usersids[i]:
                    thetaVector = np.multiply(
                        self.thetaFunc(self.dimension, argv=self.argv),
                        mask[i])
                    l2_norm = np.linalg.norm(thetaVector, ord=2)
                    users.append(User(key, thetaVector / l2_norm))
        return users
