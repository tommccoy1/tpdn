

def lists_to_file(list_of_lists, fo):
	for lst in list_of_lists:
		fo.write(" ".join([str(x) for x in lst]) + "\n")

def file_to_lists(fi, digits=True):
	list_of_lists = []
	for line in fi:
		elts = line.strip().split()
		if digits:
			elts = [int(elt) for elt in elts]
		list_of_lists.append(elts)

	return list_of_lists

	
def file_to_integer_lists(fi, list_of_lists=[], word2int={}, int2word={}, vocab_size=0):

	for line in fi:
		int_list = []
		words = line.strip().split("\t")[0].split()

		for word in words:
			if word not in word2int:
				word2int[word] = vocab_size
				int2word[vocab_size] = word
				vocab_size += 1

			int_list.append(word2int[word])

		list_of_lists.append(int_list)

	return list_of_lists, word2int, int2word, vocab_size

def file_to_integer_vector_lists(fi, word2int={}, int2word={}, vocab_size=0, max_length=0):
	list_of_lists = []
	encoding_list = []

	for line in fi:
		int_list = []
		parts = line.strip().split("\t")
		seq = parts[0]
		enc = parts[1]
		words = seq.split()
		if len(words) > max_length:
			max_length = len(words)

		for word in words:
			if word not in word2int:
				word2int[word] = vocab_size
				int2word[vocab_size] = word
				vocab_size += 1

			int_list.append(word2int[word])

		list_of_lists.append(int_list)

		encoding = [float(x) for x in enc.split()]
		encoding_list.append(encoding)

	dataset = []
	for index in range(len(list_of_lists)):
		dataset.append([list_of_lists[index], encoding_list[index]])

	return dataset, word2int, int2word, vocab_size, max_length


def create_embedding_dictionary(emb_file, f_dim, filler_to_index, index_to_filler, emb_squeeze=None, unseen_words="zero"):
	embedding_dict = {}
	embed_file = open(emb_file, "r")
	for line in embed_file:
		parts = line.strip().split()
		if len(parts) == f_dim + 1:
			embedding_dict[parts[0]] = list(map(lambda x: float(x), parts[1:]))

	matrix_len = len(filler_to_index.keys())
	if embed_squeeze is not None:
		weights_matrix = np.zeros((matrix_len, embed_squeeze))
	else:
		weights_matrix = np.zeros((matrix_len, f_dim))

	for i in range(matrix_len):
		word = index_to_filler[i]
		if word in embedding_dict:
			weights_matrix[i] = embedding_dict[word]
		else:
			if unseen_words == "random":
				weights_matrix[i] = np.random.normal(scale=0.6, size=(args.filler_dim,))
			elif args.unseen_words == "zero":
				pass # It was initialized as zero, so don't need to do anything
			else:
				print("Invalid choice for embeddings of unseen words")


	return weights_matrix





