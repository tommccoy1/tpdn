
python example_maker.py
python model_trainer.py
python generate_vectors.py --model_prefix ltr_ltr_auto_4
python decompose.py --data_prefix ltr_ltr_auto_4 --role_scheme ltr --filler_dim 20 --role_dim 20 --test_decoder True --decoder_prefix ltr_ltr_auto_4 --decoder_embedding_size 10


