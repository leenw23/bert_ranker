#!/bin/sh

# # default dataset
# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=False  --setname=dev
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=False  --setname=dev
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=False  --setname=dev

# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=False  --setname=test
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=False  --setname=test
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=False  --setname=test

# Persona dataset
# python eval_selection_model.py --model=select  --use_annotated_testset=False  --setname=dev --corpus=persona
# python eval_selection_model.py --model=mcdrop  --use_annotated_testset=False  --setname=dev --corpus=persona
# python eval_selection_model.py --model=ensemble  --use_annotated_testset=False  --setname=dev --corpus=persona

python eval_selection_model.py --model=select  --use_annotated_testset=False  --setname=test --corpus=persona
python eval_selection_model.py --model=mcdrop  --use_annotated_testset=False  --setname=test --corpus=persona
python eval_selection_model.py --model=ensemble  --use_annotated_testset=False  --setname=test --corpus=persona

# # UW token ratio
# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/UW_att_ratio/UW_att_20.txt --annotated_testset_attribute=UW_att_20 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/UW_att_ratio/UW_att_20.txt --annotated_testset_attribute=UW_att_20 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/UW_att_ratio/UW_att_20.txt --annotated_testset_attribute=UW_att_20 --replace_annotated_testset_into_original=False --is_ic=False

# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/UW_att_ratio/UW_att_40.txt --annotated_testset_attribute=UW_att_40 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/UW_att_ratio/UW_att_40.txt --annotated_testset_attribute=UW_att_40 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/UW_att_ratio/UW_att_40.txt --annotated_testset_attribute=UW_att_40 --replace_annotated_testset_into_original=False --is_ic=False

# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/UW_att_ratio/UW_att_60.txt --annotated_testset_attribute=UW_att_60 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/UW_att_ratio/UW_att_60.txt --annotated_testset_attribute=UW_att_60 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/UW_att_ratio/UW_att_60.txt --annotated_testset_attribute=UW_att_60 --replace_annotated_testset_into_original=False --is_ic=False

# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/UW_att_ratio/UW_att_80.txt --annotated_testset_attribute=UW_att_80 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/UW_att_ratio/UW_att_80.txt --annotated_testset_attribute=UW_att_80 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/UW_att_ratio/UW_att_80.txt --annotated_testset_attribute=UW_att_80 --replace_annotated_testset_into_original=False --is_ic=False

# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/UW_att_ratio/UW_att_100.txt --annotated_testset_attribute=UW_att_100 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/UW_att_ratio/UW_att_100.txt --annotated_testset_attribute=UW_att_100 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/UW_att_ratio/UW_att_100.txt --annotated_testset_attribute=UW_att_100 --replace_annotated_testset_into_original=False --is_ic=False

# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/K_token_ratio/W_token_30.txt --annotated_testset_attribute=W_token_30 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/K_token_ratio/W_token_30.txt --annotated_testset_attribute=W_token_30 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/K_token_ratio/W_token_30.txt --annotated_testset_attribute=W_token_30 --replace_annotated_testset_into_original=False --is_ic=False

# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/K_token_ratio/W_token_35.txt --annotated_testset_attribute=W_token_35 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/K_token_ratio/W_token_35.txt --annotated_testset_attribute=W_token_35 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/K_token_ratio/W_token_35.txt --annotated_testset_attribute=W_token_35 --replace_annotated_testset_into_original=False --is_ic=False

# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/K_token_ratio/W_token_40.txt --annotated_testset_attribute=W_token_40 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/K_token_ratio/W_token_40.txt --annotated_testset_attribute=W_token_40 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/K_token_ratio/W_token_40.txt --annotated_testset_attribute=W_token_40 --replace_annotated_testset_into_original=False --is_ic=False

# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/K_token_ratio/W_token_45.txt --annotated_testset_attribute=W_token_45 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/K_token_ratio/W_token_45.txt --annotated_testset_attribute=W_token_45 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/K_token_ratio/W_token_45.txt --annotated_testset_attribute=W_token_45 --replace_annotated_testset_into_original=False --is_ic=False

# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/K_token_ratio/W_token_50.txt --annotated_testset_attribute=W_token_50 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/K_token_ratio/W_token_50.txt --annotated_testset_attribute=W_token_50 --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./0513_annotated/K_token_ratio/W_token_50.txt --annotated_testset_attribute=W_token_50 --replace_annotated_testset_into_original=False --is_ic=False
