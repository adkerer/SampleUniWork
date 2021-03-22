import math
from random import random
import pandas as pd

def print_map(amap):
    side_length = int(math.sqrt(len(amap)))
  
    for i in range(side_length):
        print(amap[i*side_length:(i+1)*side_length])

    print("\n")
      
def preserve_layer(amap):
    side_length = int(math.sqrt(len(amap)))
    preserve_map = ""
    for i in range(len(amap)):
        if amap[i] in 'FMW':
            #if you cannot build on this land
            preserve_map += 'N'
            #if water, have nothing in front of it 
        elif amap[i] in 'T':    
            #if it is a trainstation
            preserve_map += 'T'
        else:
            preserve_map += 'B'
    print_map(preserve_map)
    return preserve_map

def commercial_layer(amap):
    side_length = int(math.sqrt(len(amap)))
    #set all as open
    commercial_matrix = [['O' for x in range(side_length)] for y in range(side_length)]
    #print(commercial_matrix)
    #change commerical to right
    for i in range(len(amap)):
        if amap[i] in 'T':
            indexes = []
            #set neighbouring tiles to C
            #find 2d index
            row = int(i/side_length)
            col = int(i%side_length)
            if row > 0 and row < side_length:
                indexes.append([col,row-1])
                indexes.append([col,row+1])
            elif row == 0:
                indexes.append([col,row+1])
            else:
                indexes.append([col,row-1])
            if col%side_length == 0:
                #if leftside
                indexes.append([col+1,row])
            elif col+1%side_length == 0:
                #if right side
                indexes.append([col-1,row])
            else:
                #middle of row
                indexes.append([col+1,row])
                indexes.append([col-1,row])
    for i in range(len(indexes)):
        commercial_matrix[indexes[i][1]][indexes[i][0]] = 'C'
        
    #tostring
    commercial_map = ""
    for i in range(len(commercial_matrix)):
        line = "".join(commercial_matrix[i])
        commercial_map += line
    return commercial_map
    
def get_layers(amap):
    preserve_map = preserve_layer(amap)
    commercial_map = commercial_layer(amap)
    return preserve_map, commercial_map

def get_init_map(omap,pmap,cmap):
    #omap: original map. pmap: preserve map
    side_length = int(math.sqrt(len(pmap)))
    init_map = ""
    for i in range(len(pmap)):
        if pmap[i] in 'N':
            #if you cannot build on this land
            init_map += omap[i]
        elif pmap[i] in 'T':    
            #if it is a trainstation
            init_map += 'T'
        elif cmap[i] in 'C':
            #create commercial area
            init_map += 'C'
        else:
            init_map += 'R'
    
    return init_map

def get_zoning(amap):
    preserve_map, commercial_map = get_layers(amap)
    init_map = get_init_map(amap, preserve_map, commercial_map)
    #return_map = add_mixed_use(init_map)
    return init_map


def add_mixed_use(amap):
    mixed_use_char = 'X'
    side_length = int(math.sqrt(len(amap)))
    mixed_map = ""
    for i in range(len(amap)):
        if amap[i] in 'R' and is_next_to(i,'C',amap):
            mixed_map += mixed_use_char
        else:
            mixed_map += amap[i]
    
    distances, all_coords = get_distances(mixed_map, 'R', mixed_use_char)
    
    while max(distances) >= 3:
        mixed_map = distance_optimizer_utilitarian(mixed_map,'R', mixed_use_char)
        distances, all_coords = get_distances(mixed_map, 'R', mixed_use_char)
    return mixed_map
    
def add_parks(amap):
    res_by_water_char = 'V'
    park_char = 'F'
    side_length = int(math.sqrt(len(amap)))
    park_map = ""
    for i in range(len(amap)):
        if amap[i] in 'R' and is_next_to(i,'W',amap):
            park_map += res_by_water_char
        else:
            park_map += amap[i]
    
    park_map = distance_optimizer_utilitarian(park_map,'R', park_char, res_by_water_char)
    distances, all_coords = get_distances(park_map, 'R', park_char, res_by_water_char)
    
    while max(distances) >= 4:
        park_map = distance_optimizer_utilitarian(park_map,'R', park_char, res_by_water_char)
        distances, all_coords = get_distances(park_map, 'R', park_char, res_by_water_char)
    return park_map

def get_population_distribution(amap):
    full_block = 1000
    populations = []
    for i in range(len(amap)):
        coefficient = 1
        if amap[i] in 'TCMWFS':
            #no one lives in these blocks
            coefficient = 0
        elif amap[i] in 'X':
            coefficient *= 4/5
        
        neighbors = get_neighbors(amap, i)
        for j in range(len(neighbors)):
            if neighbors[j] in 'W':
                coefficient *= 0.4
            elif neighbors[j] in 'X':
                coefficient *= 95/100
        pop = int(coefficient * full_block)
        populations.append(pop)
    
    return populations

def place_schools(amap, pop_dist):
    school_map = amap
    percentage_children = 0.123 #retreived from CNN
    population_school = 2000
    full_block = max(pop_dist)
    children_pop = percentage_children*sum(pop_dist)
    num_of_schools = round((children_pop/population_school)/(1+ (percentage_children*full_block/population_school)))
    #print(num_of_schools, children_pop, full_block*percentage_children)
    school_char = 'S'
    for i in range(num_of_schools):
        school_map = distance_optimizer_utilitarian(school_map,'R', school_char) 

    return school_map

def is_next_to(index,char2,amap):
    #coord_str = str(coords[i][0]) + str(coords[i][1])
    side_length = int(math.sqrt(len(amap)))
    boolean = False
    next_to_index= [-side_length, side_length, -1, 1]
    #return_index = -1
    row = int(index/side_length)
    col = int(index%side_length) 
    for  i in range(len(next_to_index)):
        do_check = None
        if i == 0 and row == 0:
            do_check = False
        elif i == 1 and row == side_length -1:
            do_check = False
        elif i == 2 and col%side_length == 0:
            do_check = False
        elif i == 3 and (col+1)%side_length == 0:
            do_check = False
        else:
            do_check = True
        if do_check:
            #print(index, row, col, i, next_to_index[i])
            if amap[index + next_to_index[i]] in char2:
                    boolean = True
    return boolean

def get_distances(amap, char1, char2, add_chars= None):
    char1_coords = get_char_coords(amap,char1)
    char2_coords = get_char_coords(amap,char2)
    if not add_chars == None:
        for i in range(len(add_chars)):
            char_coords = get_char_coords(amap,add_chars[i])
            #print(char_coords)
            char2_coords.extend(char_coords)
            #print(char2_coords)
    
    min_dist = len(amap)
    char1_index = []
    char2_index = []
    distances = []
    coords = []
    for i in range(len(char1_coords)):
        min_dist = len(amap)
        for j in range(len(char2_coords)):
            dist = abs(char1_coords[i][0] - char2_coords[j][0]) + abs(char1_coords[i][1] - char2_coords[j][1])
            if dist < min_dist:
                min_dist = dist
                char1_index = char1_coords[i]
                char2_index = char2_coords[j]
        distances.append(min_dist)
        coords.append(char1_index)
    #return min_dist, char1_index, char2_index
    return distances, coords

def get_char_coords(amap,char):
    coords = []
    side_length = int(math.sqrt(len(amap)))
    for i in range(len(amap)):
        if amap[i] in char:
            row = int(i/side_length)
            col = int(i%side_length)
            coords.append([row,col])
    return coords

def distance_optimizer_utilitarian(amap,char1, char2,char3 = ""):
    summed_distances = []
    indexes = []
    for i in range(len(amap)):
        if amap[i] in char1:
            test_map = amap[0:i] + char2 + amap[i+1:len(amap)]
            distances, coords = get_distances(test_map,char1, char2, char3)
            dist_sum = sum(distances)
            #print(dist_sum,distances)
            summed_distances.append(dist_sum)
            indexes.append(i)
    #print(summed_distances)
    min_value = min(summed_distances)
    #print(min_value, summed_distances)
    num_of_minimums = summed_distances.count(min_value)
    min_indexes = []
    min_values_indexes = [i for i, n in enumerate(summed_distances) if n == min_value]
    rand_ind = int(random()*num_of_minimums)
    final_index = indexes[min_values_indexes[rand_ind]]
    print(num_of_minimums, min_values_indexes, min_values_indexes[rand_ind], summed_distances[min_values_indexes[rand_ind]],final_index)
    return_map = amap[0:final_index] + char2 + amap[final_index+1:len(amap)]
    return return_map

def get_neighbors(amap, index):
    neighbors = []
    side_length = int(math.sqrt(len(amap)))
    next_to_index= [-side_length, side_length, -1, 1]
    #return_index = -1
    row = int(index/side_length)
    col = int(index%side_length) 
    for  i in range(len(next_to_index)):
        do_check = None
        if i == 0 and row == 0:
            do_check = False
        elif i == 1 and row == side_length -1:
            do_check = False
        elif i == 2 and col%side_length == 0:
            do_check = False
        elif i == 3 and (col+1)%(side_length) == 0:
            do_check = False
        else:
            do_check = True
        if do_check:
            neighbors.append(amap[index + next_to_index[i]] )#print(index, row, col, i, next_to_index[i])

    return neighbors

def map_to_csv(amap, csv_name):
    side_length = int(math.sqrt(len(amap)))
    cols = [[] for i in range(side_length)]
    for i in range(side_length):
        for j in range(side_length):
            cols[i].append(amap[i+j*side_length])

    coords = 'ABCDEFGHIJKLMNOPQRSTUV'[0:side_length]
    d = {coords[i]:cols[i] for i in range(side_length)}
    df = pd.DataFrame(data=d) 
    filename = csv_name + '.csv'
    df.to_csv(filename)

def summary():
    filename_map = 'final_map.csv'
    filename_pop = 'population_distribution_map.csv'
    df_map = pd.read_csv(filename_map)
    df_pop = pd.read_csv(filename_pop)
    
    side_length = len(df_map)
    print(side_length, len(df_map))
    coords = 'ABCDEFGHIJKLMNOPQRSTUV'[0:side_length]
    final_map = []
    final_pop = []
    
    
    for i in range(side_length):
        row_map = [df_map[coords[j]][i] for j in range(side_length)]
        row_pop = [df_pop[coords[j]][i] for j in range(side_length)]
        final_map.extend(row_map)
        final_pop.extend(row_pop)
    print_map(final_map)
    print_map(final_pop)

    population = sum(final_pop)
    all_chars = 'RTCXFSVWM'
    char_amounts = []
    for i in range(len(all_chars)):
        counter = 0
        for j in range(len(final_map)):
            if final_map[j] in all_chars[i]:
                counter += 1
        char_amounts.append(counter)
     
    print_array = [(str(all_chars[i]) + ': ' + str(char_amounts[i]) + '. ') for i in range(len(all_chars))]
    print_str =  ""
    for i in range(len(print_array)): print_str += print_array[i]
    print('Total population: ' + str(population))
    print('Total tile amounts ' + print_str)
    
    #print pop, num of all tiles. 


geog_map = 'LLLLLLMMLLLLLLLMLLLLLLLLLTLLLLLLLLLLLLLWLLLLLLLWLLLLLLWWLLLLLWWW'
print_map(geog_map)
init_map = get_zoning(geog_map)
print_map(init_map)
mixed_map =  add_mixed_use(init_map)
print_map(mixed_map)
park_map = add_parks(mixed_map)
print_map(park_map)
pop_dist = get_population_distribution(park_map)
final_map = place_schools(park_map, pop_dist)
pop_dist = get_population_distribution(final_map)
print_map(pop_dist)
print_map(final_map)

map_to_csv(geog_map,'geog_map')
map_to_csv(final_map,'final_map')
map_to_csv(pop_dist,'population_distribution_map')

summary()
