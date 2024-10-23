"""
Contains functions for analyzing the yield and hardness alloy datasets:
generating new alloys, comparing element combinations, etc.
"""

import random


################################################

### GENERATE RANDOM ALLOYS BASED ON CLUSTERS ###

"""
For the given cluster centroid, search for elements with a 
corresponding mole fraction greater than 0.01. Return a list of 
the indices of these "important" elements. 
"""
def get_important_elems(centroid):
    important_elems = [] # Indices of important elements
    for i in range(len(centroid)):
        # If mole fraction of current element > 0.01, 
        # add index to important_elems
        if centroid[i] > 0.01:
            important_elems.append(i)
    return important_elems

"""
Normalize a list of values so that they add to 1
"""
def normalize(list):
    list_sum = sum(list) # Sum of values
    # Divide every value in list by sum
    for i in range(len(list)):
        list[i] /= list_sum
    return list

"""
Helper function for generate_alloy.
Check whether all nonzero mole fractions in an alloy are between min_frac and max_frac
"""
def check_alloy(alloy, min_frac, max_frac):
    for molfrac in alloy:
        if molfrac > 0:
            if molfrac < min_frac:
                return False
            elif molfrac > max_frac:
                return False
    return True

"""
Helper function for generate_alloy. 
Multiply the largest mole fraction of the alloy by 0.9.
"""
def adjust_alloy(alloy):
    max_index = 0
    for index in range(1, len(alloy)):
        if alloy[index] > alloy[max_index]:
            max_index = index
    alloy[max_index] *= 0.9
    return alloy

"""
Generate an alloy with a random composition according to the 
given cluster center (centroid). Adjust the alloy until all 
nonzero mole fraction that are between 0.05 and 0.35. 

Params: 
    centroid: list of coordinates for cluster center
    min_elems: minimum number of elements (default is 3)
    max_elems: maximum number of elements (default is 7)
    min_frac: minimum mole fraction (default is 0.1)
    max_frac: maximum mole fraction (default is 0.35)

Return: list with element mole fractions for generated alloy
"""
def generate_alloy(centroid, min_elems=3, max_elems=7, min_frac=0.05, max_frac=0.35):
    important_elems = get_important_elems(centroid)
    new_alloy = [0]*len(centroid)

    # Change max_elems if it is more than the number of important elements
    if max_elems > len(important_elems):
        max_elems = len(important_elems)
    # Pick number of elements between min_elems and max_elems (inclusive)
    n_elems = random.randint(min_elems,max_elems)

    # Sample `n_elems` elements from `important_elems`; these
    # are the elements to include in alloy
    included_elems = random.sample(important_elems, n_elems)

    for elem_index in included_elems:
        cent_frac = centroid[elem_index] # Centroid mol frac
        # Random scale factor - ranges from 1.0 to 1.5
        scale = random.random()*0.5 + 1.0
        # Scale operation: 0 = divide by scale, 1 = multiple by scale
        scale_op = random.randint(0,1)
        
        # Calculate new mole fraction for element
        if scale_op == 0:
            new_alloy[elem_index] = cent_frac / scale
        else:
            new_alloy[elem_index] = cent_frac * scale
        
    # Normalize to make all mole fractions add to 1
    new_alloy = normalize(new_alloy) 

    # Check if alloy is valid (all nonzero mole fracs between min_frac and max_frac).
    # If not, multiply the largest mole fraction by 0.9 until the alloy is valid.
    while not check_alloy(new_alloy, min_frac, max_frac):
        new_alloy = adjust_alloy(new_alloy)
        new_alloy = normalize(new_alloy)
    
    # Round each mole fraction to 2 decimal places
    for i in range(len(new_alloy)):
        new_alloy[i] = round(new_alloy[i], 2)

    return new_alloy

"""
Generate a dataset of alloys based on the given centroid.

Params:
    centroid: list of coordinates for desired cluster center
    num_alloys: number of alloys to generate

Return: multidimensional list of alloys
"""
def gen_alloy_dataset(centroid, num_alloys):
    alloys = [] # List of alloy lists
    for i in range(num_alloys): 
        new_alloy = generate_alloy(centroid)
        alloys.append(new_alloy)
    return alloys


################################################

### COMPARE ELEMENT COMBINATIONS ###

"""
Get the possible element combos of the given dataset array/list.
Combos are stored in a dictionary: 
    keys = elements of the first element in the combo
    values = set of combos for that starting element
        combos are represented by tuples of indices of the present elements
"""
def get_elem_combos(data):
    combos = {} 
    # Iterate through each alloy in dataset
    for alloy in data:
        elems_list = [] # List of elements in the alloy
        # Iterate through each element in the alloy
        for i in range(len(alloy)):
            # Append element index to elems_list if the element is 
            # present in the alloy
            if alloy[i] > 0:
                elems_list.append(i)
        
        if len(elems_list) > 0:
            key = elems_list[0]
            # Get set of combos corresponding to the alloy's first element
            combo_set = combos.get(key)
            # Add new set to dictionary if first element does not yet exist as a key
            if combo_set == None:
                combo_set = set()
                combos[key] = combo_set

            # Add elems_list to set as a tuple
            # The set can only contain unique values, so duplicates will not be added
            combo_set.add(tuple(elems_list))

    return combos

"""
Count the total number of element combos in the given dictionary

Params:
    combos: dictionary of combos

Return: total number of combos
"""
def count_elem_combos(combos):
    count = 0
    # For each set of combos in the dictionary, 
    # add its number of combos to total count
    for combo_set in combos.values():
        count += len(combo_set)
    return count

"""
HELPER FUNCTION FOR COMPARE_ELEM_COMBOS
Check if the given combo is contained in the given combo dictionary.

Params:
    combo: tuple representing an element combo
    combo_dict: dictionary of combos

Return: True if dictionary contains combo, false otherwise
"""
def check_combo_contain(combo, combo_dict):
    # NOTE: This creates an "int not subscriptable" error if combo only has 1 element.
    # Shouldn't be a problem if we don't consider single-element alloys.
    key = combo[0] # First element = key for dictionary
    if key not in combo_dict:
        return False
    combo_set = combo_dict[key] # Set of combos corresponding to key
    # Compare given combo to every combo in combo_set
    for c in combo_set:
        if combo == c:
            return True
    return False

"""
Compare a collection of element combos with reference combos.

Params:
    ref_combos: dictionary of reference combos
    cmp_combos: dictionary of combos to be compared to references

Return: 
    valid_rate: proportion of compare combos that are valid reference combos
    included_rate: proportion of reference combos that are included in compare combos
"""
def compare_elem_combos(ref_combos, cmp_combos):
    n_ref = count_elem_combos(ref_combos) # Number of reference combos
    n_cmp = count_elem_combos(cmp_combos) # Number of compare combos

    num_valid = 0 # Number of combos in cmp_combos that are in ref_combos
    num_included = 0 # Number of combos in ref_commbos that are in cmp_combos

    ### COUNT NUMBER OF VALID COMBOS
    # Iterate through each set of compare combos
    for combo_set in cmp_combos.values():
        # Iterate through each combo in the current set
        for combo in combo_set:
            # Check if current combo is included in ref_combos
            if(check_combo_contain(combo, ref_combos)):
                num_valid += 1
    
    #### COUNT NUMBER OF INCLUDED COMBOS
    # Iterate through each set of compare combos
    for combo_set in ref_combos.values():
        # Iterate through each combo in the current set
        for combo in combo_set:
            # Check if current combo is included in ref_combos
            if(check_combo_contain(combo, cmp_combos)):
                num_included += 1

    # Proportion of compare combos that are valid reference combos
    valid_rate = num_valid / n_cmp
    # Proportion of reference combos that are included in compare combos
    included_rate = num_included / n_ref
    return valid_rate, included_rate