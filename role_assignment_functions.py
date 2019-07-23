from print_read_functions import *

# This file provides the functions underlying
# several predefined role schemes


def interleave_index(length):
	positions = range(length)
	start_ind = 0
	end_ind = length - 1

	start = True
	to_fill = [0 for _ in positions]
	for elt in positions:
		if start:
			to_fill[start_ind] = elt
			start_ind += 1
			start = False
		else:
			to_fill[end_ind] = elt
			end_ind -= 1
			start = True
	return to_fill

def create_inter_roles(max_length, vocab_size):
	return max_length, lambda x: interleave_index(len(x))



# Bag-of-words roles: All fillers have
# the same role
def create_bow_roles(max_length, vocab_size):
	bow_function = lambda fillers: [0 for i in fillers]
	return 1, bow_function

# Left-to-right roles: Each filler's role is its position
# in the sequence, counting from left to right
def create_ltr_roles(max_length, vocab_size):
	ltr_function = lambda fillers: range(len(fillers))
	return max_length, ltr_function

# Low-to-high roles: Each filler's role is its position
# in the sequence after sorting from low to high (smallest, 2nd smallest, etc)
def create_lth_roles(max_length, vocab_size):
	sort_function = lambda fillers: sorted([[fillers[x], x] for x in range(len(fillers))], key=lambda y: y[0])
	lth_function = lambda fillers: [p[0] for p in sorted([[index, z[1]] for index, z in enumerate(sort_function(fillers))], key=lambda q:q[1])]
	return max_length, lth_function

# Low-to-high roles: Each filler's role is its position
# in the sequence after sorting from low to high (smallest, 2nd smallest, etc)
def create_htl_roles(max_length, vocab_size):
	sort_function = lambda fillers: sorted([[fillers[x], x] for x in range(len(fillers))], key=lambda y: y[0])[::-1]
	htl_function = lambda fillers: [p[0] for p in sorted([[index, z[1]] for index, z in enumerate(sort_function(fillers))], key=lambda q:q[1])]
	return max_length, htl_function

# Low-to-high roles: Each filler's role is its position
# in the sequence after sorting from low to high (smallest, 2nd smallest, etc)
def create_birank_roles(max_length, vocab_size):
	to_add = [0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55]
	sort_function = lambda fillers: sorted([[fillers[x], x] for x in range(len(fillers))], key=lambda y: y[0])
	lth_function = lambda fillers: [p[0] for p in sorted([[index, z[1]] for index, z in enumerate(sort_function(fillers))], key=lambda q:q[1])]
	birank_function = lambda fillers: [x + to_add[len(fillers)] for x in lth_function(fillers)]

	return to_add[max_length + 1], birank_function


# Right-to-left roles: Each filler's role is its position 
# in the sequence, counting from right to left
def create_rtl_roles(max_length, vocab_size):
	rtl_function = lambda fillers: range(len(fillers))[::-1]
	return max_length, rtl_function

# Bidirectional roles: Each filler's role is a 2-tuple 
# containing its left-to-right index and its
# right-to-left index
def create_bidirectional_roles(max_length, vocab_size):
	bi_roles_dict = {}
	top_index_used_bi = 0
	
	# Create a dictionary assigning a numerical
	# index to each bidirectional role
	for length in range(max_length):
		roles = []

		for i in range(length + 1):
			roles.append(top_index_used_bi)
			top_index_used_bi += 1
		
		bi_roles_dict[length + 1] = roles

	bidirectional_function = lambda fillers: bi_roles_dict[len(fillers)]

	return top_index_used_bi, bidirectional_function

# Wickelroles: Each filler has a contextual role representing
# the filler before it and the filler after it
def create_wickel_roles(max_length, vocab_size):
	wickel_dict = {}

	# Create the dictionary assigning a numerical
	# index for each wickelrole	
	num_wickels = 0
	for i in range(vocab_size):
		wickel_dict["#_" + str(i)] = num_wickels
		num_wickels += 1
		wickel_dict[str(i) + "_#"] = num_wickels
		num_wickels += 1

	for i in range(vocab_size):
		for j in range(vocab_size):
			role = str(i) + "_" + str(j)
			wickel_dict[role] = num_wickels
			num_wickels += 1
			
	wickel_dict["#_#"] = num_wickels
	num_wickels += 1

	# Assign wickelroles to a sequence
	def wickel_function(seq):
		prev_num = "#"
		
		wickels = []
		for index, item in enumerate(seq):
			if index == len(seq) - 1:
				next_num = "#"
			else:
				next_num = seq[index + 1]
			
			wickels.append(wickel_dict[str(prev_num) + "_" + str(next_num)])
			prev_num = item
		
		return wickels

	return num_wickels, wickel_function

# Helper function for parsing a sequence of digits
# for use with tree position roles
def parse_digits_helper(digit_seq, tree_so_far):
    if len(digit_seq) == 1:
        return tree_so_far + [[[0]]]

    else:
        min_rep = 100
        index_of_min = -1

        for index, elt in enumerate(digit_seq[:-1]):
            rep = elt
            if rep < min_rep:
                min_rep = rep
                index_of_min = index

        start_vec = range(len(digit_seq))
        new_vec = list(start_vec[:index_of_min]) + list([[index_of_min, index_of_min + 1]]) + list(start_vec[index_of_min + 2:])

        return  parse_digits_helper(digit_seq[:index_of_min] + digit_seq[index_of_min + 1:], tree_so_far + [new_vec])

# Parse a sequence of digits
# At each step, cluster the smallest digit
# with the digit after it, and replace the pair
# with the second digit. Iterate until there's just
# one digit left.
def parse_digits(digit_seq):
    tree_so_far = [range(len(digit_seq))]

    if len(digit_seq) == 1:
        return [[[0]]]

    start_seq = parse_digits_helper(digit_seq, tree_so_far)

    new_seq = []
    for elt in start_seq:
        new_elt = []
        for inner_elt in elt:
            if isinstance(inner_elt, int):
                new_elt.append([inner_elt])
            else:
                new_elt.append(inner_elt)
        new_seq.append(new_elt)

    return new_seq
	
# For a given tree depth, the set of
# possible roles in a tree of that depth
def poss_roles(depth):
	if depth == 1:
		return ["L", "R"]

	else:
		lefts = []
		rights = []

		for elt in poss_roles(depth - 1):
			lefts.append(elt + "L")
			rights.append(elt + "R")

		return lefts + rights

def all_roles(depth):
	roles = []

	for j in range(depth):
		roles += poss_roles(j + 1)
			
	return roles


# Tree position roles: Each filler's role is its
# position in a parse tree defined by the digit-parsing
# algorithm defined above
def create_tree_roles(max_length, vocab_size):
	tree_roles_dict = {}
	depth = max_length - 1
	tree_counter = 0
	
	for role in all_roles(depth):
		tree_roles_dict[role] = tree_counter
		tree_counter += 1
	
	# Given a sequence, assign tree roles to it
	def get_tree_roles(tree_rep):

		if tree_rep == [[[0]]]:
			return [tree_roles_dict["L"]]

		current_list = ["L", "R"]
		this_list = tree_rep[:-2][::-1]

		for item in this_list:
			next_list = []
			for index, elt in enumerate(item):
				if len(elt) == 1:
					next_list.append(current_list[index])
				else:
					next_list.append(current_list[index] + "L")
					next_list.append(current_list[index] + "R")

			current_list = next_list

		final_list = []
		for elt in current_list:
			final_list.append(tree_roles_dict[elt])

		return final_list

	return tree_counter, lambda fillers: get_tree_roles(parse_digits(fillers))



# Tree position roles: Each filler's role is its
# position in a parse tree defined by the digit-parsing
# algorithm defined above
def create_treerev_roles(max_length, vocab_size):
	tree_roles_dict = {}
	
	depth = max_length - 1
	tree_counter = 0
	
	for role in all_roles(depth):
		tree_roles_dict[role] = tree_counter
		tree_counter += 1

	# Given a sequence, assign tree roles to it
	def get_tree_roles(tree_rep):

		if tree_rep == [[[0]]]:
			return [tree_roles_dict["L"]]

		current_list = ["L", "R"]
		this_list = tree_rep[:-2][::-1]

		for item in this_list:
			next_list = []
			for index, elt in enumerate(item):
				if len(elt) == 1:
					next_list.append(current_list[index])
				else:
					next_list.append(current_list[index] + "L")
					next_list.append(current_list[index] + "R")

			current_list = next_list

		final_list = []
		for elt in current_list:
			final_list.append(tree_roles_dict[elt])

		return final_list

	return tree_counter, lambda fillers: get_tree_roles(parse_digits(fillers))[::-1]

# e.g., (lth, tree)
def combine_fillers_roles(roles1, roles2):
	new_roles = []

	for role1 in roles1:
		new_roles.append(roles2[role1])

	return new_roles

	

# Tree position roles: Each filler's role is its
# position in a parse tree defined by the digit-parsing
# algorithm defined above
def create_treelth_roles(max_length, vocab_size):
	tree_roles_dict = {}
	
	depth = max_length - 1
	tree_counter = 0
	
	for role in all_roles(depth):
		tree_roles_dict[role] = tree_counter
		tree_counter += 1

	# Given a sequence, assign tree roles to it
	def get_tree_roles(tree_rep):

		if tree_rep == [[[0]]]:
			return [tree_roles_dict["L"]]

		current_list = ["L", "R"]
		this_list = tree_rep[:-2][::-1]

		for item in this_list:
			next_list = []
			for index, elt in enumerate(item):
				if len(elt) == 1:
					next_list.append(current_list[index])
				else:
					next_list.append(current_list[index] + "L")
					next_list.append(current_list[index] + "R")

			current_list = next_list

		final_list = []
		for elt in current_list:
			final_list.append(tree_roles_dict[elt])

		return final_list
	sort_function = lambda fillers: sorted([[fillers[x], x] for x in range(len(fillers))], key=lambda y: y[0])
	lth_function = lambda fillers: [p[0] for p in sorted([[index, z[1]] for index, z in enumerate(sort_function(fillers))], key=lambda q:q[1])]
	
	
	
	return tree_counter, lambda fillers: combine_fillers_roles(lth_function(fillers), get_tree_roles(parse_digits(fillers)))


# Tree position roles: Each filler's role is its
# position in a parse tree defined by the digit-parsing
# algorithm defined above
def create_treeinter_roles(max_length, vocab_size):
	tree_roles_dict = {}
	
	depth = max_length - 1
	tree_counter = 0
	
	for role in all_roles(depth):
		tree_roles_dict[role] = tree_counter
		tree_counter += 1

	# Given a sequence, assign tree roles to it
	def get_tree_roles(tree_rep):

		if tree_rep == [[[0]]]:
			return [tree_roles_dict["L"]]

		current_list = ["L", "R"]
		this_list = tree_rep[:-2][::-1]

		for item in this_list:
			next_list = []
			for index, elt in enumerate(item):
				if len(elt) == 1:
					next_list.append(current_list[index])
				else:
					next_list.append(current_list[index] + "L")
					next_list.append(current_list[index] + "R")

			current_list = next_list

		final_list = []
		for elt in current_list:
			final_list.append(tree_roles_dict[elt])

		return final_list
	
	
	return tree_counter, lambda fillers: combine_fillers_roles(interleave_index(len(fillers)), get_tree_roles(parse_digits(fillers)))


def create_role_scheme(role_scheme, max_length, vocab_size, role_filea=None, filler_filea=None, role_fileb=None, filler_fileb=None, role_filec=None, filler_filec=None, role_filed=None, filler_filed=None):
    if role_filea is None:
        if role_scheme == "bow":
            n_r, seq_to_roles = create_bow_roles(max_length, vocab_size)
        elif role_scheme == "ltr":
            n_r, seq_to_roles = create_ltr_roles(max_length, vocab_size)
        elif role_scheme == "rtl":
            n_r, seq_to_roles = create_rtl_roles(max_length, vocab_size)
        elif role_scheme == "bi":
            n_r, seq_to_roles = create_bidirectional_roles(max_length, vocab_size)
        elif role_scheme == "wickel":
            n_r, seq_to_roles = create_wickel_roles(max_length, vocab_size)
        elif role_scheme == "tree":
            n_r, seq_to_roles = create_tree_roles(max_length, vocab_size)
        elif role_scheme == "lth":
            n_r, seq_to_roles = create_lth_roles(max_length, vocab_size)
        elif role_scheme == "htl":
            n_r, seq_to_roles = create_htl_roles(max_length, vocab_size)
        elif role_scheme == "birank":
            n_r, seq_to_roles = create_birank_roles(max_length, vocab_size)
        elif role_scheme == "inter":
            n_r, seq_to_roles = create_inter_roles(max_length, vocab_size)
        elif role_scheme == "treerev":
            n_r, seq_to_roles = create_treerev_roles(max_length, vocab_size)
        elif role_scheme == "treelth":
            n_r, seq_to_roles = create_treelth_roles(max_length, vocab_size)
        elif role_scheme == "treeinter":
            n_r, seq_to_roles = create_treeinter_roles(max_length, vocab_size)
        else:
            print("Invalid role scheme")
    else:
        filler_role_dict = {}

        filler_lists = []
        role_lists = []
        filler2int = {}
        int2filler = {}
        for q in range(10):
            filler2int[str(q)] = q
            int2filler[q] = str(q)
        role2int = {}
        int2role = {}
        filler_num = 10
        role_num = 0

        for (ff, rf) in [(filler_filea, role_filea), (filler_fileb, role_fileb), (filler_filec, role_filec), (filler_filed, role_filed)]:
            if ff is not None:
                filler_lists, filler2int, int2filler, filler_num = file_to_integer_lists(ff, filler_lists, filler2int, int2filler, filler_num)
                role_lists, role2int, int2role, role_num = file_to_integer_lists(rf, role_lists, role2int, int2role, role_num)

        for fl, rl in zip(filler_lists, role_lists):
            filler_role_dict[tuple(fl)] = rl

        #print("fr dict", filler_role_dict)
        n_r = role_num
        seq_to_roles = lambda seq: filler_role_dict[tuple(seq)]

        

    return n_r, seq_to_roles




