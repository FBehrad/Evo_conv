import time
import numpy as np
from segmentation.model import build_model
from genetic_algorithm import wise_initialization, weighted_sum, random_initialization, evaluate_popualtion, pareto
from genetic_algorithm import final_ranked_fitness, selection, one_point_cross_over, replacement
import pickle
from tensorflow.keras.optimizers import Adam
from segmentation.utils import my_custom_generator_segmentation, loss_gt, dice_coefficient, create_path
from keras_surgeon import delete_channels
import yaml
from utils import order_layers, creat_pair_path
import tensorflow as tf


def pruning(model, target, layers_dic, population_size, conf, validation_generator):
    filters, biases = model.layers[target].get_weights()
    w = filters.shape
    chromosome_len = w[-1]
    fil = [1] * chromosome_len  # in the first step all filters are used

    print('Create initial population')
    # Create initial population
    wise_pop = wise_initialization(chromosome_len, population_size, fil, layers_dic, target)
    rand_pop = random_initialization(chromosome_len, population_size, fil)
    pop = wise_pop + rand_pop  # this is the final mask to remove filters
    evaluated_pop = evaluate_popualtion(population_size, pop, model, target, validation_generator)
    
    # loop
    itr = 0
    num_gen = 0
    best_members = []
    best_results = [0]
    start = time.time()
    pop = evaluated_pop
    print('Lets start evolution')

    while (num_gen != conf['gen_threshold']) and (itr < conf['itr_threshold']):
        print('---------------------------------------------------------------------')
        print('Iteretion : ', itr + 1)
        # Find best chromosome
        b_indices = pareto(pop, population_size)
        # print('B_indices',b_indices)
        best_choromosome_idx = b_indices[0]
        best_chromosome = pop[best_choromosome_idx]

        if best_chromosome[1] >= (max(best_results) + 0.00001):
            best_members.append(best_chromosome)
            best_results.append(best_chromosome[1])
        print('Best chromosome : ', best_chromosome)
        # Rank chromosomes
        ranked_chromosome = weighted_sum(conf['w1'], conf['w2'], population_size, b_indices, pop)
        # print('Ranked chromosomes',ranked_chromosome)

        final_fitnesses = final_ranked_fitness(population_size, b_indices, ranked_chromosome)

        # Selection
        selected_pop = selection(population_size, final_fitnesses)
        
        # Crossover
        children = one_point_cross_over(population_size, selected_pop, conf['pc'], pop, chromosome_len)

        # Mutation
        # final_children = mutation(children,pm,chromosome_len)

        # Evaluate
        final_children_np = [np.array(i) for i in children]
        evaluated_children = evaluate_popualtion(population_size, children, model, target, validation_generator)

        # Replacement
        next_generation = replacement(evaluated_children, pop, population_size, conf['w1'], conf['w2'])
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
            print('Num gen: ', num_gen)

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


def genetic_pruning(config, layers, nb_filters, model, data_paths, ratio):
    # create generators
    paths_val, labels_val = creat_pair_path(data_paths)
    validation_generator = my_custom_generator_segmentation(paths_val, labels_val, config['training']['batch_size'])

    ordered_layers, layers_dic = order_layers(model)  # put your model name here
    final_layers = {}
    best_over_gens = []

    for i, target in enumerate(layers):
        population_size = int((ratio * nb_filters[i]))
        print('Target => ', target)
        final_layer, best_over_gen = pruning(model, target, layers_dic, population_size, config['genetic'],
                                             validation_generator)
        best_over_gens.append(best_over_gen)
        final_layers[target] = final_layer
        print('Target:', target, 'Best over generation', best_over_gen)

    return final_layers


def remove_filters(model, final_layers):
    new_model = model
    for key in final_layers:
        best_mask = final_layers[key][0]
        indices = np.where(best_mask < 1)
        removed_filters_indices = indices[0].tolist()  # indices of filters which will be removed
        conv_layer = new_model.get_layer(index=key)
        new_model = delete_channels(new_model, conv_layer, removed_filters_indices)
        new_model.compile(optimizer=Adam(lr=1e-4), loss=[loss_gt()], metrics=[dice_coefficient],
                          experimental_run_tf_function=False)

    return new_model


if __name__ == '__main__':
    path = open('../config.yaml', 'r')
    config = yaml.safe_load(path)
    data_paths = create_path('../preprocessed_data', train=True) + create_path('../augmented_data', aug=True)

    # load model
    best_model = config['path']['best_model']
    input_size = config['preprocessing_seg']['optimal_roi']
    input_size = (4, input_size[0], input_size[1], input_size[2])
    model_param = config['model']
    model = build_model(input_shape=input_size, gradient_accumulation=model_param['accumulated_grad']['enable'],
                        n_gradients=model_param['accumulated_grad']['num_batch'])
    model.load_weights(best_model).expect_partial()

    # The indices of convolutional layers which can be pruned and the number of their filters
    layers = [5, 13, 94, 20, 64, 43, 28, 74, 35, 84]
    nb_filters = [32, 64, 32, 64, 256, 256, 128, 128, 128, 64]
    all_layers = genetic_pruning(config, layers, nb_filters, model, data_paths, ratio=0.25)

    if config['genetic']['version'] == 'third':
        layers = [50, 57]
        nb_filters = [256, 256]
        final_layers = genetic_pruning(config, layers, nb_filters, model, data_paths, ratio=0.5)
    else:
        layers = [50, 57]
        nb_filters = [256, 256]
        final_layers = genetic_pruning(config, layers, nb_filters, model, data_paths, ratio=0.25)

    all_layers = all_layers + final_layers

    # Final_layers
    with open('../final_layers/final_layers', 'wb') as fp:
        pickle.dump(all_layers, fp)

    pruned_model = remove_filters(model, all_layers)

    # Prepare a model for validation
    val_model = build_model(input_shape=(4, 240, 240, 160),
                            gradient_accumulation=model_param['accumulated_grad']['enable'],
                            n_gradients=model_param['accumulated_grad']['num_batch'])

    pruned_val_model = remove_filters(val_model, all_layers)
    pruned_val_model.set_weights(pruned_model.get_weights())

    if config['genetic']['version'] == 'third':
        tf.keras.models.save_model(pruned_val_model, '../Pruned_a_third_model.h5')
    else:
        tf.keras.models.save_model(pruned_val_model, '../Pruned_a_forth_model.h5')

