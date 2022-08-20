from tensorflow import math
import math
import time
import numpy as np
from utils import extract_dataset, create_path
from model import build_model
from genetic_algorithm import wise_initialization, weighted_sum, random_initialization, evaluate_popualtion, pareto
from genetic_algorithm import final_ranked_fitness, selection, one_point_cross_over, replacement
import pickle
from tensorflow.keras.optimizers import Adam
from utils import my_custom_generator_segmentation, loss_gt, dice_coefficient
from keras_surgeon import delete_channels


# Order layers for pruning

def order_layers(model):
    conv_layers_indexes = []
    layers_dic = {}
    unordered_layers = {}

    for i in range(len(model.layers)):
        if model.layers[i].__class__.__name__ == 'Conv3D':
            conv_layers_indexes.append(i)

    for i in conv_layers_indexes:  # i = 0 3 6
        weight = model.layers[i].get_weights()[0]
        filter_shape = weight.shape
        nb_of_fils = filter_shape[-1]
        weight_dict = {}
        # computation of l2 norm for a wise initialization
        for j in range(nb_of_fils):
            square = np.square(abs(weight[:, :, :, :, j]))
            sum = np.sum(square)
            l2 = math.sqrt(sum)
            # l1 = np.sum(abs(weight[:,:,:,:,j])) # change this because not all networks have this dimension
            weight_dict[j] = l2.numpy()

        weight_dict_sorted = sorted(weight_dict.items(), key=lambda kv: kv[1])
        layers_dic[i] = weight_dict_sorted
        print('Layer ' + str(i))
        print(weight_dict_sorted)

    for x in layers_dic:
        ordered_filters = layers_dic[x]
        lowest_value = ordered_filters[0][-1]  # ordered_filters[0] -->(28, 26.955196)
        unordered_layers[x] = lowest_value

    ordered_layers = sorted(unordered_layers.items(), key=lambda kv: kv[1])
    return ordered_layers, layers_dic


def pruning(model, target, layers_dic, population_size, w1, w2, pc, pm, itr_threshold, gen_threshold, validation_generator):
    filters, biases = model.layers[target].get_weights()
    w = filters.shape
    chromosome_len = w[-1]
    fil = [1] * chromosome_len  # in the first step all filters are used

    print('Create initial population')
    # Create initial population
    wise_pop = wise_initialization(chromosome_len, population_size, fil, layers_dic, target)
    rand_pop = random_initialization(chromosome_len, population_size, fil)
    pop = wise_pop + rand_pop  # this is the final mask to remove filters
    evaluated_pop = evaluate_popualtion(population_size, pop, model, target)
    # loop
    itr = 0
    num_gen = 0
    best_members = []
    best_results = [0]
    start = time.time()
    pop = evaluated_pop
    print('Lets start evolution')

    while (num_gen != gen_threshold) and (itr < itr_threshold):
        print('---------------------------------------------------------------------')
        print('Iteretion : ', itr + 1)
        # Find best chromosome
        print('pop', pop)
        b_indices = pareto(pop, population_size)
        # print('B_indices',b_indices)
        best_choromosome_idx = b_indices[0]
        best_chromosome = pop[best_choromosome_idx]

        if best_chromosome[1] >= (max(best_results) + 0.00001):
            best_members.append(best_chromosome)
            best_results.append(best_chromosome[1])
        print('Best chromosome : ', best_chromosome)
        # Rank chromosomes
        ranked_chromosome = weighted_sum(w1, w2, population_size, b_indices, pop)
        # print('Ranked chromosomes',ranked_chromosome)

        final_fitnesses = final_ranked_fitness(population_size, b_indices, ranked_chromosome)

        # Selection
        selected_pop = selection(population_size, final_fitnesses)

        # Crossover
        children = one_point_cross_over(population_size, selected_pop, pc, pop, chromosome_len)

        # Mutation
        # final_children = mutation(children,pm,chromosome_len)

        # Evaluate
        final_children_np = [np.array(i) for i in children]
        evaluated_children = evaluate_popualtion(population_size, children, model, target, validation_generator)

        # Replacement
        next_generation = replacement(evaluated_children, pop, population_size, w1, w2)
        # print('Next gen: ',next_generation )
        best_member = next_generation[0]
        # print('Best member of next gen: ', best_member )
        # print('Ex best member', best_chromosome )
        # if np.array_equal(best_chromosome[0],best_member[0]):
        if best_chromosome[1] == best_member[1]:  # I think this one is better
            print('Equal')
            num_gen = num_gen + 1
            print('Num gen:', num_gen)
        else:
            num_gen = 0
            print('Num_gen‌: ', num_gen)

        pop = next_generation
        itr = itr + 1
        print('---------------------------------------------------------------------')
    end = time.time()
    r = end - start
    hours, rem = divmod(r, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Preprocessing time :")
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    print('Completed!')
    return best_members[-1], best_members


def main(config_file):
    dataset_path = config_file['path']['dataset']
    extract_dataset(dataset_path, 'Data')
    data_paths = create_path('./Data', train=True)
    best_model = config_file['path']['best_model']

    labels_val = []
    paths_val = []
    for i, data_path in enumerate(data_paths):
        path = []
        path.append(data_path['flair'])
        path.append(data_path['t1'])
        path.append(data_path['t2'])
        path.append(data_path['t1ce'])
        label = data_path['seg']
        paths_val.append(path)
        labels_val.append(label)

    batch_size = 1
    validation_generator = my_custom_generator_segmentation(paths_val, labels_val, batch_size)

    model = build_model(input_shape=(4, 128, 128, 128), output_channels=3, gradient_accumulation='True', n_gradients=8)

    model.load_weights(best_model)

    # for i, layer in enumerate(model.layers):
    #     print('layer:', i, '\tName:', layer.name, '\tClass:', layer.__class__.__name__)


    w1 = 0.8  # weight for accuracy
    w2 = -0.2  # weight for run time
    pc = 0.9  # higher cross over probability --> more exploitation
    pm = 0.1  # higher mutation probability --> more exploration
    itr_threshold = 25
    gen_threshold = 4  # if the best member didn't change, this parameter would stop the algorithm
    layers = [5, 12]
    nb_filters = [32, 128]
    ratio = 0.25

    ordered_layers, layers_dic = order_layers(model)  # put your model name here
    final_layers = {}
    # for i in range(len(ordered_layers)):
    #   layers.append(ordered_layers[i][0])
    best_over_gens = []

    for i, target in enumerate(layers):
        population_size = int((ratio * nb_filters[i]))
        print('Target => ', target)
        final_layer, best_over_gen = pruning(model, target, layers_dic, population_size, w1, w2, pc, pm, itr_threshold,
                                             gen_threshold, validation_generator)
        final_layers[target] = final_layer
        best_over_gens.append(best_over_gen)
        print('Target:', target, 'Best over generation', best_over_gen)

    # final_layers
    with open('/content/drive/MyDrive/final_layers_64_64layers', 'wb') as fp:
        pickle.dump(final_layers, fp)

    new_model = model
    for key in final_layers:
        best_mask = final_layers[key][0]
        indices = np.where(best_mask < 1)
        removed_filters_indices = indices[0].tolist()  # indices of filters which will be removed
        conv_layer = new_model.get_layer(index=key)
        new_model = delete_channels(new_model, conv_layer,removed_filters_indices)
        new_model.compile(optimizer=Adam(lr=1e-4), loss=[loss_gt()], metrics=[dice_coefficient],
                          experimental_run_tf_function=False)

    return new_model
