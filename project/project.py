from __future__ import division
import concurrent.futures
import math, random
import numpy as np
from numpy import linalg as LA
from scipy import constants
import time
import sys, traceback
import matplotlib.pyplot as plt, numpy as np
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import imageio
from gfycat.client import GfycatClient
from gfycat.error import GfycatClientError

#Number of particles to simulate (randomly sampled from file)
num_particles = 60000
#Smallest leaf (partition)
smallest_leaf = 7500
#Number of timesteps to take (and image)
num_images = 200
#Epsilon in force equation
soften = 5000
#Timestep per simulation step
global_timestep = 0.25
#Increase the effect of mass in the force equation
mass_effect = 4
#Multiply the initial velocity in particles by this value
initial_velo_factor = 1
#Used in the opening angle criterion
theta = 10000

random.seed()
np.seterr(all='raise')

# Gives the size of the NBody, num particles in it, and it's mass
class NBody(object):
    def __init__(self, particles):
        self.minX = float('Inf')
        self.minY = float('Inf')
        self.minZ = float('Inf')
        self.maxX = -float('Inf')
        self.maxY = -float('Inf')
        self.maxZ = -float('Inf')

        self.midX = 0
        self.midY = 0
        self.midZ = 0
        self.numParticles = len(particles)
        self.area = 0
        self.particles = particles

        #Find the size of the nbody
        for particle in particles:
            if particle.location[0] < self.minX:
                self.minX = particle.location[0]
            if particle.location[1] < self.minY:
                self.minY = particle.location[1]
            if particle.location[2] < self.minZ:
                self.minZ = particle.location[2]
            if particle.location[0] > self.maxX:
                self.maxX = particle.location[0]
            if particle.location[1] > self.maxY:
                self.maxY = particle.location[1]
            if particle.location[2] > self.maxZ:
                self.maxZ = particle.location[2]
        self.midX = (self.minX + self.maxX) / 2
        self.midY = (self.minY + self.maxY) / 2
        self.midZ = (self.minZ + self.maxZ) / 2
        self.area = (self.maxX - self.minX) * (self.maxY - self.minY) * (self.maxZ - self.minZ)
        
class Particle(object):
    location = np.array([])
    def __init__(self, line):
        self.location = np.array([float(line[0]), float(line[1]), float(line[2])])
        self.velocity = np.array([float(line[3]), float(line[4]), float(line[5])]) * initial_velo_factor
        self.mass = float(line[6])

        #Color based on particle type (mass is only indicator)
        if self.mass == float(1.0463387E-03):
            self.color = "blue"
        else:
            self.color = "green"
        self.acceleration = 0
    def print_particle(self):
        print "Particle - location: {0}, velocity: {1}, mass: {2}, acceleration: {3}".format(
           self.location, self.velocity, self.mass, self.acceleration)

class Tree(object):
    def __init__(self, particles):
        self.treeNode = None
        self.hasChildren = False
        if len(particles) == 0:
            return
        else:
            self.children = [None, None, None, None, None, None, None, None]
            self.treeNode = (calc_centre_of_mass(particles))
            self.treeNode.nbody = NBody(particles)

            #split 3d nbody into 8 boxes (smallest box must have atleast 32 particles
            
            if len(particles) < (smallest_leaf * 1.1):
                return
            if len(particles) > smallest_leaf:
                self.hasChildren = True

                self.children[0] = Tree([x for x in particles if x.location[0] < self.treeNode.nbody.midX
                                                    and x.location[1] < self.treeNode.nbody.midY
                                                    and x.location[2] < self.treeNode.nbody.midZ])
                
                self.children[1] = Tree([x for x in particles if x.location[0] >= self.treeNode.nbody.midX
                                                    and x.location[1] < self.treeNode.nbody.midY
                                                    and x.location[2] < self.treeNode.nbody.midZ])
                
                self.children[2] = Tree([x for x in particles if x.location[0] < self.treeNode.nbody.midX
                                                    and x.location[1] >= self.treeNode.nbody.midY
                                                    and x.location[2] < self.treeNode.nbody.midZ])
                
                self.children[3] = Tree([x for x in particles if x.location[0] < self.treeNode.nbody.midX
                                                    and x.location[1] < self.treeNode.nbody.midY
                                                    and x.location[2] >= self.treeNode.nbody.midZ])
                
                self.children[4] = Tree([x for x in particles if x.location[0] >= self.treeNode.nbody.midX
                                                    and x.location[1] >= self.treeNode.nbody.midY
                                                    and x.location[2] < self.treeNode.nbody.midZ])
                
                self.children[5] = Tree([x for x in particles if x.location[0] >= self.treeNode.nbody.midX
                                                    and x.location[1] < self.treeNode.nbody.midY
                                                    and x.location[2] >= self.treeNode.nbody.midZ])

                self.children[6] = Tree([x for x in particles if x.location[0] < self.treeNode.nbody.midX
                                                    and x.location[1] >= self.treeNode.nbody.midY
                                                    and x.location[2] >= self.treeNode.nbody.midZ])

                self.children[7] = Tree([x for x in particles if x.location[0] >= self.treeNode.nbody.midX
                                                    and x.location[1] >= self.treeNode.nbody.midY
                                                    and x.location[2] >= self.treeNode.nbody.midZ])

                for child in self.children:
                    if child.treeNode is None or child.treeNode.nbody.numParticles < smallest_leaf:
                        child = None


    def calculate_force_on_particle(self, particle):
        r = LA.norm(particle.location - self.treeNode.location)
        A = self.treeNode.nbody.area

        force = 0

        #Soften nearby particles
        if r < 5:
            return 0
        
        if  (A/r) < theta or not self.hasChildren:
            force += calc_force_due_to_nbody(particle, self.treeNode)
        else:
            for child in self.children:
                if child is None or child.treeNode is None:
                    continue
                try:
                    force += child.calculate_force_on_particle(particle)
                except Exception:
                    print "Exception in user code:"
                    print '-'*60
                    traceback.print_exc(file=sys.stdout)
                    print '-'*60
        particle.acceleration = force 
        #print "Accel: {0}, Loc: {1}, Velo: {2}".format(particle.acceleration, particle.location, particle.velocity)
        return force

    def calc_force_on_leaves(self, base_tree):
        if self.treeNode is None:
            return
        if self.hasChildren is False:
            force = self.calculate_force_on_treenode(self.treeNode, base_tree)
            for particle in self.treeNode.nbody.particles: particle.acceleration = force
        else:
            for child in self.children:
                if child is None or child.treeNode is None:
                    continue
                else:
                    child.calc_force_on_leaves(base_tree)

    def calculate_force_on_treenode(self, node, tree):
        r = LA.norm(node.location - tree.treeNode.location)
        A = tree.treeNode.nbody.area

        force = 0

        #Soften nearby particles
        if r < 5:
            return 0
        
        if  (A/r) < theta or not tree.hasChildren:
            force += calc_force_due_to_nbody(node, self.treeNode)
        else:
            for child in tree.children:
                if child is None or child.treeNode is None:
                    continue
                try:
                    force += child.calculate_force_on_treenode(node, child)
                except Exception:
                    print "Exception in user code:"
                    print '-'*60
                    traceback.print_exc(file=sys.stdout)
                    print '-'*60

        return force
        
    def print_tree(self):
        if self.treeNode is None:
            return
        
        self.treeNode.print_node()

        if self.treeNode.nbody.numParticles is 1:
            return
        
        for child in self.children:
            if child is None or child.treeNode is None:
                continue
            else:
                child.print_tree()
    
class TreeNode(object):
    def __init__(self, location, mass, nbody):
        self.location = location
        self.mass = mass
        self.nbody = None
    def print_node(self):
        print "TreeNode - Location: {0}, Mass: {1}, numParticles: {2}".format(self.location, self.mass, self.nbody.numParticles)

def calc_force_due_to_nbody(particle, treeNode):
        G = 40000
        if (treeNode.location is particle.location):
            return 0
        acceleration = 0
        
        x_ij = np.power((treeNode.location - particle.location) + soften, 2)
        
        #Handles rounding errors when x_ij approaches 0
        if np.count_nonzero(x_ij) is not 3:
            return 0
        
        vector = treeNode.location - particle.location
        unit_vector = vector / LA.norm(vector)
        #print "x_ij: {0}, unit_vector: {1}, x_hat: {2}, mass: {3}, norm: {4}".format(x_ij, unit_vector, vector, treeNode.mass, LA.norm(vector))

        acceleration += ( (treeNode.mass * mass_effect) / x_ij) * unit_vector * G
        #print "Accel: {0}, Location: {1}, Destination: {2}, Direction {3}, Velocity: {4}".format(acceleration, particle.location, treeNode.location, unit_vector, particle.velocity)
        return acceleration

def advance_particles(particles, timestep):
    for particle in particles:
        particle.location += particle.velocity * timestep + (0.5 * particle.acceleration * timestep * timestep)
        particle.velocity += (particle.acceleration * timestep)

def populate_from_file(fname, particles):
    with open(fname) as f:
        content = f.readlines()

    indices = random.sample(range(1, 60001), num_particles)
    sample = [content[i] for i in sorted(indices)]
    for i in range(num_particles):
        particles.append(Particle(sample[i].split()))

def calc_centre_of_mass(particles):
    if (len(particles) is 0):
        return None
    masses = np.array([[0, 0, 0, 0]])
    totalMass = 0
    for particle in particles:
        masses = np.append(masses, [np.append(particle.location, particle.mass)], axis=0)
        totalMass += particle.mass
    nonZeroMasses = masses[np.nonzero(masses[:,3])]
    CM = np.average(nonZeroMasses[:,:3], axis=0, weights=nonZeroMasses[:,3])
    return TreeNode(CM, totalMass, None)


def update_particles(timestep):
    tree = Tree(particles)
    #for particle in particles:
    #    tree.calculate_force_on_particle(particle)
    tree.calc_force_on_leaves(tree)
    advance_particles(particles, timestep)

def save_image(i):
    fig = plt.figure()
    v= np.array([o.location for o in particles])
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-300, 300])
    ax.set_ylim([-300, 300])
    ax.set_zlim([-300, 300])
    colors= [o.color for o in particles]
    ax.scatter(v[:,0],v[:,1],v[:,2], color=colors, marker='.')
    fig.savefig('images/' + str(i) + '.png')
    plt.close(fig)
    
def run_sim():
    save_image(0)

    for i in range(0, num_images):
        curr = time.clock()
        update_particles(global_timestep)
        print i,
        print time.clock() - curr
        
        #print "Distance: {0}".format(LA.norm(particles[0].location - particles[1].location))

        #Save an image of current state
        save_image(i + 1)
        
    save_to_gif()
        

#Save image as a gif
def save_to_gif():
    images = []
    for i in range(num_images):
        images.append(imageio.imread('images/' + str(i) + '.png'))
    imageio.mimsave('sim.gif', images)

    # Save to gfycat
    client = GfycatClient()
    try:
        print client.upload_from_file('sim.gif')
    except GfycatClientError as e:
        print(e.error_message)
        print(e.status_code)

def normalizeTotalVelocity():
    totalVelo = 0
    for p in particles:
        totalVelo += p.velocity

    for p in particles:
        for i in range(3):
            p.velocity[i] -= (totalVelo[i] / num_particles)

particles = []
populate_from_file("galaxy.ascii", particles)
normalizeTotalVelocity()

run_sim()
