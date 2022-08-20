import random
import math
import numpy as np
from utils import loss_gt, dice_coefficient
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from keras_surgeon import delete_channels
import time
import pandas as pd
from paretoset import paretoset


# Create initial population
# 1. wise population

def wise_initialization(chromosome_len, population_size, filter, layers_dic, target):
    wise_pop = []
    number_of_wise = 1
    threshold = int(
        round(chromosome_len / 4))  # when we remove half of the filters we should divide chromosome_len by 2
    if threshold < 8:  # As we have group normalization layer we need to prune factors of group size
        threshold = 8

    # removed_thresholds = random.sample(range(0, threshold), number_of_wise)
    # 0 --> all filters will be preserved
    # 3--> three least important filters will be removed

    for i in range(number_of_wise):
        fil_np = np.array(filter)
        rm = []
        t = threshold * (i + 1)  # removed_thresholds[i] # This is mutiplied by (i+1) to create different chromosomes
        for j in layers_dic[target][:t]:
            rm.append(j[0])
        fil_np[rm] = 0
        wise_pop.append(fil_np)

    return wise_pop


# 2. random initialization

def random_initialization(chromosome_len, population_size, filter):
    number_of_random = population_size - 1
    rand_pop = []
    threshold = int(
        round(chromosome_len / 4))  # when we remove half of the filters we should divide chromosome_len by 2
    if threshold < 8:  # As we have group normalization layer we need to prune factors of group size
        threshold = 8

    for i in range(number_of_random):
        removed_filters = random.sample(range(0, chromosome_len), threshold)
        # print(removed_filters)
        fil_np = np.array(filter)
        fil_np[removed_filters] = 0
        rand_pop.append(fil_np)

    return rand_pop


# Evaluation of chromosomes

# 1. calculate the number of parameters

def total_parameters(model):
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
    return trainable_count + non_trainable_count


# 2. calculate fitness

def fitness_runtime(model, target, remove_indices, validation_generator):
    conv_layer = model.get_layer(index=target)
    new_model = delete_channels(model, conv_layer, remove_indices)
    new_model.compile(optimizer=Adam(lr=5e-5), loss=[loss_gt()], metrics=[dice_coefficient],
                      experimental_run_tf_function=False)
    para = total_parameters(new_model)
    start = time.time()
    print('Start evaluation of Chromosome')
    loss, acc = new_model.evaluate(validation_generator)
    print('Done!')
    end = time.time()
    run_time = end - start
    return acc, run_time, para


# 3. evaluate population

def evaluate_popualtion(population_size, population, model, target, validation_generator):
    evaluated_pop = []
    for i in range(population_size):
        mem = np.array(population[i])
        indices = np.where(mem < 1)
        removed_filters_indices = indices[0].tolist()  # indices of filters which will be removed
        acc, run_time, para = fitness_runtime(model, target, removed_filters_indices, validation_generator)
        evaluated_pop.append([mem, acc, run_time, para])

    return evaluated_pop


# Find best members

def pareto(evaluated_pop, population_size):
    run_times = []
    accuracies = []

    for i in range(population_size):
        member = evaluated_pop[i]
        acc = member[1]
        run_time = member[2]
        accuracies.append(acc)
        run_times.append(run_time)

    fitnesses = pd.DataFrame({"acc": accuracies, "run_time": run_times})
    # we convert it to dataframe in order to calculate paretoset (As we have multiobjective optimization)

    mask = paretoset(fitnesses, sense=["max", "min"])
    paretoset_fitnesses = fitnesses[mask]

    index = paretoset_fitnesses.index
    b_indices = list(index)  # indices of best chromosomes
    return b_indices


def diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))


def weighted_sum(w1, w2, population_size, b_indices, evaluated_pop):  # Aggregation fitness for ranking

    final_fitnesses = {}
    run_times = []
    all_chrom = list(range(population_size))
    remain_chr = diff(all_chrom, b_indices)  # to find chromosmes which are not ranked by paretoset

    for i in remain_chr:  # for normalization of run times
        member = evaluated_pop[i]
        run_time = member[2]
        run_times.append(run_time)

    for i in remain_chr:
        member = evaluated_pop[i]
        acc = member[1]
        run_time = member[2] / sum(run_times)
        final_fitness = w1 * acc + w2 * run_time  # we should maximize this
        final_fitnesses[i] = final_fitness

    ranked_chromosome = sorted(final_fitnesses.items(), key=lambda kv: kv[1], reverse=True)

    return ranked_chromosome


# Ranked-based selection and Stochastic Universal Sampling (SUS) ( having multiple fixed points) read more :
# https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm


def final_ranked_fitness(population_size, b_indices, ranked_chromosome):
    ns = population_size
    ranked_fitnesses = {}
    for i in b_indices:
        ranked_fitnesses[i] = ns
        ns = ns - 1

    for i in ranked_chromosome:
        mem = i[0]
        ranked_fitnesses[mem] = ns
        ns = ns - 1

    # lets create a list that matches the fitnesses of each member

    final_fitnesses = [1] * population_size
    for key in ranked_fitnesses:
        final_fitnesses[key] = ranked_fitnesses[key]

    final_fitnesses = [i / sum(final_fitnesses) for i in final_fitnesses]

    return final_fitnesses


def selection(population_size, final_fitnesses):
    # for multiple identifier
    cumulative_fitness = [1] * population_size

    for i in range(population_size):
        h = i + 1
        if h < population_size:
            cumulative_fitness[i] = round(sum(final_fitnesses[:h]), 3)
        else:
            cumulative_fitness[i] = round(sum(final_fitnesses), 3)

    cumulative_fitness = np.array(cumulative_fitness)

    selected_pop = []
    first_choice = random.uniform(0, 1)
    print('First random number : ', first_choice)

    first_individial = list(np.where(cumulative_fitness >= first_choice)[0])[
        0]  # [i for i, e in enumerate(cumulative_fitness) if e >= first_choice]
    selected_pop.append(first_individial)
    gap = 1 / population_size
    print('Gap : ', gap)
    next_choice = first_choice
    for i in range(population_size - 1):
        next_choice = next_choice + gap
        if next_choice > 1:
            next_choice = next_choice - 1
            next_individual = list(np.where(cumulative_fitness >= next_choice)[0])[
                0]  # [i for i, e in enumerate(cumulative_fitness) if e >= next_choice]
        else:
            next_individual = list(np.where(cumulative_fitness >= next_choice)[0])[
                0]  # [i for i, e in enumerate(cumulative_fitness) if e >= next_choice]

        selected_pop.append(next_individual)

    return selected_pop


# one-point cross over

def one_point_cross_over(population_size, selected_pop, pc, evaluated_pop, chromosome_len):
    children = []
    parents = []
    j = 0
    threshold = population_size / 2
    while j < threshold:
        p_indices = random.sample(range(0, population_size), 2)
        # print(p_indices)
        idx1 = p_indices[0]
        idx2 = p_indices[1]
        p1 = selected_pop[idx1]
        p2 = selected_pop[idx2]
        pair = [p1, p2]
        if pair not in parents:
            parents.append(pair)
            j = j + 1
    # print(parents)
    # apply pc
    cr_parents = []
    unchanged_parents = []
    for i in parents:
        p = random.uniform(0, 1)
        if p < pc:
            cr_parents.append(i)
        else:
            unchanged_parents.append(i)
    # print('cr_parents',cr_parents)
    # print('unchanged_parents',unchanged_parents)
    # now we implement cross over for parents which are eligible
    for i in cr_parents:
        idx1 = i[0]
        idx2 = i[1]
        parent1 = evaluated_pop[idx1][0].tolist()
        parent2 = evaluated_pop[idx2][0].tolist()
        l = int(chromosome_len / 8)
        point = random.randint(l, 3 * l)  # to ensure that chromosomes changes
        # print('Point',point)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        # print('Members')
        # print(parent1,parent2)
        # print(child1,child2)
        print('----------------------------------------------')
        # We need this part because we use group normalization and the number of elements should be a factor of group size
        ch1_zeros = np.where(np.array(child1) == 0)[0]
        ch_z_list = ch1_zeros.tolist()
        ch2_zeros = np.where(np.array(child2) == 0)[0]
        ch2_z_list = ch2_zeros.tolist()
        # print('Zero location')
        # print(ch_z_list,ch2_z_list)
        parent_zeros = np.where(np.array(parent1) == 0)[0]
        nb_zeros = len(parent_zeros.tolist())
        diff = abs(len(ch2_z_list) - nb_zeros)
        # print('Diff with parent',diff)
        if len(ch2_z_list) < len(ch_z_list):
            # print('Second chromosome is shorter')
            unique_items = list(set(ch_z_list) - set(ch2_z_list))
            th = round(len(unique_items) / 2)
            elements = random.sample(unique_items, th)
        elif len(ch2_z_list) > len(ch_z_list):
            # print('First chromosome is shorter')
            unique_items = list(set(ch2_z_list) - set(ch_z_list))
            th = round(len(unique_items) / 2)
            elements = random.sample(unique_items, th)
        for i in range(diff):  # we consider this as mutation, so we don't apply mutation anymore
            j = i + 1
            if len(ch2_z_list) < len(ch_z_list):
                for i in range(len(elements)):
                    elem = elements[i]
                    if elem not in ch2_z_list:
                        ch2_z_list.append(elem)
                        ch_z_list.remove(elem)
                        break
            elif len(ch2_z_list) > len(ch_z_list):
                for i in range(len(elements)):
                    elem = elements[i]
                    if elem not in ch_z_list:
                        ch_z_list.append(elem)
                        ch2_z_list.remove(elem)
                        break
        fil = [1] * len(parent1)
        new_child1 = np.array(fil)
        new_child1[ch_z_list] = 0
        new_child2 = np.array(fil)
        new_child2[ch2_z_list] = 0
        # print('New Children')
        # print(new_child1,new_child2)
        # ---------------------------------------------
        children.append(new_child1)
        children.append(new_child2)
    # Now we add unchanged children
    for i in unchanged_parents:
        idx1 = i[0]
        idx2 = i[1]
        parent1 = evaluated_pop[idx1][0].tolist()
        parent2 = evaluated_pop[idx2][0].tolist()
        child1 = parent1
        child2 = parent2
        children.append(child1)
        children.append(child2)

    return children


# mutation

def mutation(children, pm, chromosome_len):
    m_children = []
    unchanged_children = []
    for i in children:
        p = random.uniform(0, 1)
        if p < pm:
            m_children.append(i)
        else:
            unchanged_children.append(i)

    print('Number of unchanged children : ', len(unchanged_children))
    print('Number of mutated children : ', len(m_children))

    l = math.ceil(chromosome_len / 8)  # as we don't want to change chromosomes a lot

    for i in m_children:
        nb_m = random.randint(1, l)  # number of genes which will be flipped
        # print(nb_m)
        p_indices = random.sample(range(0, chromosome_len), nb_m)  # indices of genes which will be flipped
        # print(p_indices)
        for j in p_indices:
            if i[j] == 1:
                i[j] = 0
            else:
                i[j] = 1

    print("Mutation completed!")
    final_children = m_children + unchanged_children
    return final_children


# replacement

def replacement(children, parents, population_size, w1, w2):
    final_fitnesses_dict = {}
    next_generation = []
    population = children + parents  # we combine all of them to choose best ones
    pop_size = len(population)

    best_indices = pareto(population, pop_size)
    ranked_population = weighted_sum(w1, w2, pop_size, best_indices, population)
    final_fitnesses = final_ranked_fitness(pop_size, best_indices, ranked_population)

    for i in range(pop_size):
        final_fitnesses_dict[i] = final_fitnesses[i]

    sorted_fitnesses = sorted(final_fitnesses_dict.items(), key=lambda kv: kv[1], reverse=True)

    next_generation_indices = sorted_fitnesses[
                              0:population_size]  # Now we want to remove individuals with the least fitness
    for i in next_generation_indices:
        idx = i[0]
        next_generation.append(population[idx])

    return next_generation
