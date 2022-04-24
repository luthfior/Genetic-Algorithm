import os
import random
import matplotlib.pyplot
import numpy as np
import imageio as imgio
import itertools
import functools
import operator
import numpy

def imgChromosome(img_arr):
    chromosome = numpy.reshape(a=img_arr, newshape=(functools.reduce(operator.mul, img_arr.shape)))
    return chromosome

target = np.array(imgio.imread('img/teknik.png'), dtype=np.float64)
target_chromosome = imgChromosome(target)

pop = 12
p_s = 5
mutasi = .01

def inisial_populasi(img_shape, n_individu=8):
    i_populasi = numpy.empty(shape=(n_individu, functools.reduce(operator.mul, img_shape)), dtype=numpy.uint8)
    for indv_num in range(n_individu):
        i_populasi[indv_num, :] = numpy.random.random(functools.reduce(operator.mul, img_shape))*256
    return i_populasi

def chromosome2img(chromosome, img_shape):
    img_arr = numpy.reshape(a=chromosome, newshape=img_shape)
    return img_arr


def fitness(target_chrom, indiv_chrom):
    kualitas = numpy.mean(numpy.abs(target_chrom-indiv_chrom))
    kualitas = numpy.sum(target_chrom) - kualitas
    return kualitas

def popFitness(target_chrom, pop):
    kualitas = numpy.zeros(pop.shape[0])
    for i in range(pop.shape[0]):
        kualitas[i] = fitness(target_chrom, pop[i, :])
    return kualitas

def mating_pool(pop, kualitas, a_parents):
    parents = numpy.empty((a_parents, pop.shape[1]), dtype=numpy.uint8)
    for parent_a in range(a_parents):
        nilai_max = numpy.where(kualitas == numpy.max(kualitas))
        nilai_max = nilai_max[0][0]
        parents[parent_a, :] = pop[nilai_max, :]
        kualitas[nilai_max] = -1
    return parents


def crossover(parents, img_shape, n_individu=8):
    populasi_baru = numpy.empty(shape=(n_individu, 
                                        functools.reduce(operator.mul, img_shape)),
                                        dtype=numpy.uint8)
    populasi_baru[0:parents.shape[0], :] = parents


    num_newly_generated = n_individu-parents.shape[0]
    parents_permutations = list(itertools.permutations(iterable=numpy.arange(0, parents.shape[0]), r=2))
    selected_permutations = random.sample(range(len(parents_permutations)), 
                                          num_newly_generated)
    
    comb_idx = parents.shape[0]
    for comb in range(len(selected_permutations)):
        selected_comb_idx = selected_permutations[comb]
        selected_comb = parents_permutations[selected_comb_idx]
        
        half_size = numpy.int32(populasi_baru.shape[1]/2)
        populasi_baru[comb_idx+comb, 0:half_size] = parents[selected_comb[0], 
                                                             0:half_size]
        populasi_baru[comb_idx+comb, half_size:] =  parents[selected_comb[1], 
                                                             half_size:]
    
    return populasi_baru

def mutation(populasi, p_s, mut_percent):
    for idx in range(p_s, populasi.shape[0]):
        random_index = numpy.uint32(numpy.random.random(size=numpy.uint32(mut_percent/100*populasi.shape[1]))*populasi.shape[1])
        n_values = numpy.uint8(numpy.random.random(size=random_index.shape[0])*256)
        populasi[idx, random_index] = n_values
    return populasi


def open_images(iterasi, kualitas, populasi_baru, im_shape, 
                save_point, save_dir):

    if(numpy.mod(iterasi, save_point)==0):
        solusi_chrom_terbaik = populasi_baru[numpy.where(kualitas == 
                                                         numpy.max(kualitas))[0][0], :]
        gambar_terbaik = chromosome2img(solusi_chrom_terbaik, im_shape)
        matplotlib.pyplot.imsave(save_dir+'solution_'+str(iterasi)+'.png', gambar_terbaik)

def inds_show(individuals, im_shape):
    index = individuals.shape[0]
    fig_row_col = 1
    for k in range(1, numpy.uint16(individuals.shape[0]/2)):
        if numpy.floor(numpy.power(k, 2)/index) == 1:
            fig_row_col = k
            break
    fig1, axis1 = matplotlib.pyplot.subplots(fig_row_col, fig_row_col)

    current_index = 0
    for index_row in range(fig_row_col):
        for index_column in range(fig_row_col):
            if(current_index>=individuals.shape[0]):
                break
            else:
                curr_img = chromosome2img(individuals[current_index, :], im_shape)
                axis1[index_row, index_column].imshow(curr_img)
                current_index = current_index + 1

populasi_baru = inisial_populasi(img_shape=target.shape, n_individu=pop)

for iteration in range(40001):
    kualitas = popFitness(target_chromosome, populasi_baru)
    print('Kualitas : ', np.max(kualitas), ', Iterasi : ', iteration)
    
    parents = mating_pool(populasi_baru, kualitas, p_s)
    
    populasi_baru = crossover(parents, target.shape, n_individu=pop)

    populasi_baru = mutation(populasi=populasi_baru, p_s = p_s, mut_percent=mutasi)
 
    open_images(iteration, kualitas, populasi_baru, target.shape, save_point=1000, save_dir=os.curdir+'//')

inds_show(populasi_baru, target.shape)