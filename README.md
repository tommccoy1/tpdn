# tpdn

This repository provides the code for training a Tensor Product Decomposition Network (TPDN). TPDNs were introduced in [this paper](https://openreview.net/forum?id=BJx0sjC5FX) by [Tom McCoy](https://tommccoy1.github.io/), [Tal Linzen](http://tallinzen.net/), [Ewan Dunbar](http://www.linguist.univ-paris-diderot.fr/~edunbar/), and [Paul Smolensky](https://www.microsoft.com/en-us/research/people/psmo/), and there is an interactive demo available [here](https://tommccoy1.github.io/tpdn/tpr_demo.html).


## Quick start

If you have the dependencies installed (see below), you can train a TPDN by running the following line in the main directory of this repository:

```
python decompose.py --data_prefix example --role_scheme bi --filler_dim 20 --role_dim 20 --test_decoder True --decoder_prefix example --decoder_embedding_size 10
```

This code fits a TPDN to the vector encodings found in ```data/example.data_from_train``` and evaluates the fit on the vector encodings in ```data/example.data_from_train```. The TPDN here uses a bidirectional role scheme, 20-dimensional filler vectors, and 20-dimensional role vectors. These encodings were derived from the encoder-decoder model whose weights are stored in ```models/decoder_example.weights``` and ```models/decoder_example.weights```. The results will be printed to a log file in the directory ```logs```. 

Instead of using the ```--role_scheme``` flag (which uses a Python function to assign roles to fillers), you can use a file specifying the roles with the ```--role_prefix``` flag. Here, we load the bidirectional role files from ```data/example.bi_roles.data_from_train.roles```:

```
python decompose.py --data_prefix example --role_prefix example.bi_roles --filler_dim 20 --role_dim 20 --test_decoder True --decoder_prefix example --decoder_embedding_size 10
```

## Dependencies

This code requires Pytorch, which can be installed from [the Pytorch website](https://pytorch.org/). The code was developed using Pytorch version 0.4.1 but should be compatible with later versions as well. The models can be run on GPUs or on CPUs.


## TPDN options

The TPDN code requires a file listing sequences and their encodings; each line of the file should contain the sequence (with its elements separated by spaces), then a tab, then the encoding (with its elements separated by spaces). For an example, see ```data/example.data_from_train```. Specifically, your dataset should have a single prefix (call it PREFIX), and you should have three files of sequences and their encodings called ```PREFIX.data_from_train```, ```PREFIX.data_from_dev```, and ```PREFIX.data_from_test```.

The ```decompose.py``` function shown in the quick start section has many options specified by command line flags. A full list of options is in ```decompose.py```, but the most important ones are here:

```--data_prefix```: The prefix for the files contain the sequences and their encodings.

```--role_scheme```: Allows you to train the TPDN using a predefined role scheme. The available role schemes are ```ltr``` (left-to-right), ```rtl``` (right-to-left), ```bi``` (bidirectional), ```wickel``` (wickelroles), ```tree``` (tree positions), and ```bow``` (bag-of-words).

```--role_prefix```: Allows you to train the TPDN using a text file that specifies the role for each filler. For each of your files ```PREFIX.data_from_train```, ```PREFIX.data_from_dev```, and ```PREFIX.data_from_test```, there should be corresponding files called ```ROLE_PREFIX.data_from_train.roles```, ```ROLE_PREFIX.data_from_dev.roles```, and ```ROLE_PREFIX.data_from_test.roles```. Each line of the role file should give the roles for the corresponding line in the file listing the encodings; see ```data/example.data_from_train``` and ```data/example.bi_roles.data_from_train.roles``` for an example. Note that ```PREFIX`` and ```ROLE_PREFIX``` do not have to be the same, but can be.

```--filler_dim```: The dimension of the filler embeddings to be trained.

```--role_dim```: The dimension of the role embeddings to be trained.

```--test_decoder```: Whether or not to test the TPDN approximation by passing its encodings to the decoder from the original seq2seq model. (In the paper, we refer to this as the substitution accuracy).

```--decoder_prefix```: If you set ```--test_decoder True```, use this flag to specify the prefix for the decoder to use.

```--decoder_embedding_size```: If you set ```--test_decoder True```, use this flag to specify the embedding size used by the decoder.



## Training a seq2seq model

The examples above show how to run a TPDN on a file of vectors that you provide. It is also possible to use this code to train the seq2seq from which you derive the encodings to be analyzed. The pipeline for doing this is:

1) Generate training, development, and test sets of digit sequences (by default, this will generate 40,000 training sequences, 5,000 development sequences, and 5,000 test sequences, using the digits from 0 to 9 and varying in length from 1 to 6. You can change these settings using the flags specified in the example_maker.py file). This will generate pickled lists of sequences in the ```data/``` directory (e.g. ```data/digits.train.pkl```):
```
python example_maker.py
```

2) Train a sequence-to-sequence model to autoencode these digit sequences. You can also change the training task to something other than autoencoding, and you can vary the architecture type (by default it is a unidirectional encoder and unidirectional decoder, but both the encoder and decoder can be modified). See the model_trainer.py file for the various options. This code will save the model's weights in the ```models/``` directory as something like ```models/decoder_ltr_ltr_auto_0.weights``` and ```models/encoder_ltr_ltr_auto_0.weights```:

```
python model_trainer.py
```

3) Using the trained model, generate encodings for all of the training, development, and test data. The ```--model_prefix``` option should identify the model files that were generated by the previous step; if you have been running the code exactly as specified here with a fresh download of the repository, this should be ```ltr_ltr_auto_0```, but in general this setting will vary. This will generate .txt files in which each line contains first a digit sequence (with the elements separated by spaces), then a tab, and then the encoding of that sequence (with its elements separated by spaces). These files will be in ```data/``` named something like ```data/ltr_ltr_auto_0.data_from_train```, ```data/ltr_ltr_auto_0.data_from_dev```, and ```data/ltr_ltr_auto_0.data_from_test```:

```
python generate_vectors.py --model_prefix ltr_ltr_auto_0
```

4) Fit a tensor product decomposition network to the encodings from the sequence-to-sequence model:
```
python decompose.py --data_prefix ltr_ltr_auto_0 --role_scheme ltr --filler_dim 20 --role_dim 20 --test_decoder True --decoder_prefix ltr_ltr_auto_0 --decoder_embedding_size 10
```
## Descriptions of all scripts

* ```role_assignment_functions.py``` : Defines the functions for assigning roles to sequences of fillers for various predefined role schemes.
* ```training.py``` : Defines functions for training sequence-to-sequence models and tensor product decomposition networks
* ```tasks.py``` : Defines the sequence-manupulation tasks on which our seq2seq models are trained.
* ```model_trainer.py``` : For training seq2seq models on digit manipulation tasks
* ```models.py``` : Defines the seq2seq models and the TPDN
* ```generate_vectors.py``` : Generate encodings for some set of vectors
* ```example_maker.py``` : Generates digit-sequence training, dev, and test sets
* ```evaluation.py``` : Functions for evaluation
* ```binding_operations.py``` : Defines various functions for binding fillers and roles
* ```decompose.py``` : Perform a tensor product decomposition


## Citation

If you make use of this code, please cite the following ([bibtex](https://tommccoy1.github.io/tpdn/tpdn.html)):

R. Thomas McCoy, Tal Linzen, Ewan Dunbar, and Paul Smolensky.  RNNs implicitly implement tensor-product representations.  In *International Conference on Learning Representations*, 2019. URL https://openreview.net/forum?id=BJx0sjC5FX.

*Questions? Comments? Email [tom.mccoy@jhu.edu](mailto:tom.mccoy@jhu.edu).*

