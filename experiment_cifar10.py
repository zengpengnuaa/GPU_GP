import math
import random
import time
import gp_restrict
import numpy
import gp_tree
import operator
import functionSet as fs
from deap import creator, base, gp, tools
import cuda_func_1
import evalGP, test
from STGPdataType import Double, Filter, Pixelset, KerSize, Img, Vector, Row, Col, X, Y, Size, Filter_size, Filter_type
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from numba import cuda
from fitnesseval import fitness_func_1
import multiprocessing
import importlib
import torchvision, torch

randomSeed = 315

# cifar10
dataset_path = './Dataset/cifar'
from torchvision import transforms

gray_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
])
train_dataset = torchvision.datasets.CIFAR10(dataset_path, train=True)
test_dataset = torchvision.datasets.CIFAR10(dataset_path, train=False)
x_train = train_dataset.data
x_train = numpy.transpose(torch.tensor(x_train), (0, 3, 1, 2))
x_train = gray_transform(x_train)
x_train = x_train.float()
x_train = torch.reshape(x_train, shape=[50000, 32, 32])
x_train = x_train.detach().numpy()
y_train = train_dataset.targets
y_train = numpy.asarray(y_train)
x_test = test_dataset.data
x_test = numpy.transpose(torch.tensor(x_test), (0, 3, 1, 2))
x_test = gray_transform(x_test)
x_test = x_test.float()
x_test = torch.reshape(x_test, shape=[10000, 32, 32])
x_test = x_test.detach().numpy()
y_test = test_dataset.targets
y_test = numpy.asarray(y_test)

# data process
x_train = test.data_process(x_train)
x_test = test.data_process(x_test)
print(x_train.shape)

# XSGP parameters
population = 500
generation = 50
cxProb = 0.5
mutProb = 0.49
elitismProb = 0.01
totalRuns = 1
initialMinDepth = 2
initialMaxDepth = 6
maxDepth = 10
height, width = x_train[1, :, :].shape

# XSGP tree structure, function set and terminal set
pset = gp_tree.PrimitiveSetTyped('MAIN', [Img], Vector)
pset.addPrimitive(fs.add, [Img, Row, Col, Row, Col], float, 'add')
pset.addPrimitive(fs.sub, [Img, Row, Col, Row, Col], float, 'sub')
pset.addPrimitive(fs.mul, [Img, Row, Col, Row, Col], float, 'mul')
pset.addPrimitive(fs.div, [Img, Row, Col, Row, Col], float, 'div')
pset.addPrimitive(fs.num_add, [float, float], float, 'num_add')
pset.addPrimitive(fs.num_sub, [float, float], float, 'num_sub')
pset.addPrimitive(fs.num_mul, [float, float], float, 'num_mul')
pset.addPrimitive(fs.protected_div, [float, float], float, 'protected_div')
pset.addPrimitive(fs.if_else, [float, float, float], float, 'if_else')
pset.addPrimitive(fs.neg, [float], float, name='neg')
pset.addPrimitive(fs.mixadd, [float, Double, float, Double], float, name="mixadd")
pset.addPrimitive(fs.mixsub, [float, Double, float, Double], float, name='mixsub')
pset.addPrimitive(fs.regions, [Img, X, Y, Size], Pixelset, name='regions')
pset.addPrimitive(fs.regionr, [Img, X, Y, Size, Size], Pixelset, name='regionr')
pset.addPrimitive(fs.mean, [Pixelset], float, name='mean')
pset.addPrimitive(fs.std, [Pixelset], float, name='std')
pset.addPrimitive(fs.mean, [Img], float, name='mean')
pset.addPrimitive(fs.std, [Img], float, name='std')
pset.addPrimitive(fs.conv, [Pixelset, Filter], Pixelset, name='conv')
pset.addPrimitive(fs.conv, [Img, Filter], Pixelset, name='conv')
pset.addPrimitive(fs.gen_filter, [Filter_size, Filter_type], Filter, name='gen_filter')
pset.addPrimitive(fs.maxP, [Pixelset, KerSize, KerSize], Pixelset, name='maxP')
pset.addPrimitive(fs.maxP, [Img, KerSize, KerSize], Pixelset, name='maxP')
pset.addPrimitive(fs.relu, [Pixelset], Pixelset, name='relu')
pset.addPrimitive(fs.concat, [float for i in range(10)], Vector, name='concat')
pset.addEphemeralConstant('row', lambda: random.randint(0, height - 1), Row)
pset.addEphemeralConstant('col', lambda: random.randint(0, width - 1), Col)
pset.addEphemeralConstant('randomD', lambda: round(random.random(), 3), Double)
pset.addEphemeralConstant('X', lambda: random.randint(0, width - 20), X)
pset.addEphemeralConstant('Y', lambda: random.randint(0, height - 20), Y)
pset.addEphemeralConstant('Size', lambda: random.randint(10, height + 1), Size)
pset.addEphemeralConstant('filtersize', lambda: random.choice([3, 5]), Filter_size)
pset.addEphemeralConstant('filtertype', lambda: random.choice(
    ['mean', 'gauss', 'sharpen', 'prewitt_r', 'prewitt_ru', 'sobel_u', 'sobel_lu', 'laplace_4', 'laplace_8']),
                          Filter_type)
pset.addEphemeralConstant('kernelSize', lambda: random.randrange(2, 5, 2), KerSize)

# XSGP creator
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# XSGP toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# genetic operators
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=6)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("mutate_eph", gp.mutEphemeral, mode='all')
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))


def GPmain(randomSeed):
    random.seed(randomSeed)
    pop = toolbox.population(population)
    hof = tools.HallOfFame(10)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=lambda ind: ind.height)
    mstats = tools.MultiStatistics(fitness=stats_fit, size_tree=stats_size_tree)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log = evalGP.eaSimple(randomSeed, pop, toolbox, cxProb, mutProb, elitismProb, generation,
                               stats=mstats, halloffame=hof, verbose=True)
    return pop, log, hof


def init(l):
    global lock
    lock = l


if __name__ == "__main__":
    lock = multiprocessing.Manager().Lock()
    toolbox.register("evaluate", fitness_func_1, lock=lock)
    num_processes = 8
    pool = multiprocessing.Pool(processes=num_processes, initializer=init, initargs=(lock,))
    toolbox.register("map", pool.map)
    print('Conducting calculation with {} cpu workers'.format(num_processes))
    beginTime = time.time()
    pop, log, hof = GPmain(randomSeed)
    endTime = time.time()
    trainTime = endTime - beginTime

    acc = test.predict(toolbox, hof[0], x_train, y_train, x_test, y_test)
    testTime = time.time() - endTime
    # pool.close()
    print('Best individual: ', hof[0])
    print('Test Results: ', acc)
    print('Train time  ', trainTime)
    print('Test time  ', testTime)
    print('End')
