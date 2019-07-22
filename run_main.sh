CUDA_VISIBLE_DEVICES=1 python main.py --status train \
		--wordemb glove \
		--train data/$1/sod.train \
		--dev data/$1/sod.dev \
		--test data/$1/sod.test \
		--savemodel data/$1/saved_model \


# python main.py --status decode \
# 		--raw data/$1/dev.bmes \
# 		--savedset data/$1/saved_model.lstmcrf.dset \
# 		--loadmodel data/$1/saved_model.lstmcrf.13.model \
# 		--output data/$1/raw.out \
