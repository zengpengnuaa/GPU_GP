import random
from deap import tools
import matplotlib.pyplot as plt


def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb = cxpb / (cxpb + mutpb)
    new_mutpb = mutpb / (cxpb + mutpb)
    i = 1
    pb = random.random()
    while i < len(offspring):
        if pb < new_cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        elif pb < new_mutpb+new_cxpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            i += 1
        else:
            offspring[i], = toolbox.mutate_eph(offspring[i])
            del offspring[i].fitness.values
            i += 1
    return offspring


def eaSimple(randomseed, population, toolbox, cxpb, mutpb, elitpb, ngen,
             stats=None, halloffame=None, verbose=__debug__):
    fd = open('Parallel/CIFAR10.txt', mode='a')
    logbook = tools.Logbook()
    logbook.header = ['gen', 'popsize'] + (stats.fields if stats else [])
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
    hof_store = tools.HallOfFame(5 * len(population))
    hof_store.update(population)
    record = stats.compile(population) if stats else {}
    print('Innitial population fitnesses evaluate finished, Best is {}'.format(record['fitness']['max']))
    logbook.record(gen=0, popsize=len(population), **record)
    if verbose:
        fd.write(logbook.stream + '\n')

    for gen in range(1, ngen + 1):
        elitismNum = int(elitpb * len(population))
        population_for_eli = [toolbox.clone(ind) for ind in population]
        offspringE = toolbox.selectElitism(population_for_eli, k=elitismNum)
        offspring = toolbox.select(population, len(population) - elitismNum)
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        for i in offspring:
            index = 0
            while index < len(hof_store):
                if i == hof_store[index]:
                    i.fitness.values = hof_store[index].fitness.values
                    index = len(hof_store)
                else:
                    index += 1
        invalid_indi = [indi for indi in offspring if not indi.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_indi)
        for indi, fit in zip(invalid_indi, fitnesses):
            indi.fitness.values = fit

        offspring[0:0] = offspringE

        if halloffame is not None:
            halloffame.update(offspring)
        copy_pop = offspring.copy()
        hof_store.update(offspring)
        for i in hof_store:
            copy_pop.append(i)
        population[:] = offspring
        record = stats.compile(population) if stats else {}
        print('{}th population fitnesses evaluate finished, Best is {}'.format(gen, record['fitness']['max']))
        logbook.record(gen=gen, popsize=len(offspring), **record)
        if verbose:
            fd.write(logbook.stream + '\n')

    fd.close()
    return population, logbook
