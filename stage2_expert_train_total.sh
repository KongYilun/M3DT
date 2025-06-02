SEED=0
EXPERT_NUM=48
EMBED_DIM=256


CUDA_VISIBLE_DEVICES=0 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 0 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=1 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 1 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=2 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 2 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=3 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 3 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
wait
CUDA_VISIBLE_DEVICES=0 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 4 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=1 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 5 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=2 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 6 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=3 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 7 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
wait
CUDA_VISIBLE_DEVICES=0 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 8 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=1 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 9 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=2 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 10 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=3 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 11 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
wait

CUDA_VISIBLE_DEVICES=0 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 12 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=1 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 13 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=2 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 14 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=3 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 15 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
wait

CUDA_VISIBLE_DEVICES=0 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 16 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=1 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 17 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=2 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 18 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=3 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 19 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
wait
CUDA_VISIBLE_DEVICES=0 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 20 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=1 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 21 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=2 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 22 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=3 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 23 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
wait
CUDA_VISIBLE_DEVICES=0 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 24 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=1 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 25 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=2 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 26 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=3 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 27 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
wait
CUDA_VISIBLE_DEVICES=0 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 28 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=1 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 29 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=2 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 30 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=3 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 31 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &

wait

CUDA_VISIBLE_DEVICES=0 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 32 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=1 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 33 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=2 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 34 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=3 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 35 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
wait
CUDA_VISIBLE_DEVICES=0 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 36 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=1 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 37 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=2 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 38 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=3 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 39 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
wait
CUDA_VISIBLE_DEVICES=0 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 40 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=1 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 41 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=2 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 42 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=3 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 43 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
wait

CUDA_VISIBLE_DEVICES=0 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 44 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=1 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 45 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=2 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 46 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
CUDA_VISIBLE_DEVICES=3 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 47 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM 

wait


# CUDA_VISIBLE_DEVICES=0 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 48 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
# CUDA_VISIBLE_DEVICES=0 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 49 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
# CUDA_VISIBLE_DEVICES=0 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 50 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
# CUDA_VISIBLE_DEVICES=2 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 51 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
# CUDA_VISIBLE_DEVICES=2 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 52 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
# CUDA_VISIBLE_DEVICES=2 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 53 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
# CUDA_VISIBLE_DEVICES=0 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 54 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM &
# CUDA_VISIBLE_DEVICES=2 python stage2_expert_train.py --prefix_name gradient_group_$EXPERT_NUM --seed $SEED --expert_id 55 --embed_dim $EMBED_DIM --expert_num $EXPERT_NUM 

# wait
# CUDA_VISIBLE_DEVICES=0 python step3_moe_train.py --prefix_name mt160_random_48expert_e20w_5mlp_top5 --seed $SEED --expert_num $EXPERT_NUM --embed_dim $EMBED_DIM --model_scale 1

# CUDA_VISIBLE_DEVICES=0 python expert_test.py --prefix_name expert_0 --seed $SEED --expert_id 0 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_0_20w.out  &
# CUDA_VISIBLE_DEVICES=1 python expert_test.py --prefix_name expert_1 --seed $SEED --expert_id 1 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_1_20w.out  &
# CUDA_VISIBLE_DEVICES=2 python expert_test.py --prefix_name expert_2 --seed $SEED --expert_id 2 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_2_20w.out  &
# CUDA_VISIBLE_DEVICES=3 python expert_test.py --prefix_name expert_3 --seed $SEED --expert_id 3 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_3_20w.out  &
# CUDA_VISIBLE_DEVICES=0 python expert_test.py --prefix_name expert_4 --seed $SEED --expert_id 4 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_4_20w.out  &
# CUDA_VISIBLE_DEVICES=1 python expert_test.py --prefix_name expert_5 --seed $SEED --expert_id 5 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_5_20w.out  &
# CUDA_VISIBLE_DEVICES=2 python expert_test.py --prefix_name expert_6 --seed $SEED --expert_id 6 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_6_20w.out  &
# CUDA_VISIBLE_DEVICES=3 python expert_test.py --prefix_name expert_7 --seed $SEED --expert_id 7 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_7_20w.out  &

# wait

# CUDA_VISIBLE_DEVICES=0 python expert_test.py --prefix_name expert_8 --seed $SEED --expert_id 8 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_8_20w.out  &
# CUDA_VISIBLE_DEVICES=1 python expert_test.py --prefix_name expert_9 --seed $SEED --expert_id 9 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_9_20w.out  &
# CUDA_VISIBLE_DEVICES=2 python expert_test.py --prefix_name expert_10 --seed $SEED --expert_id 10 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_10_20w.out  &
# CUDA_VISIBLE_DEVICES=3 python expert_test.py --prefix_name expert_11 --seed $SEED --expert_id 11 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_11_20w.out  &
# CUDA_VISIBLE_DEVICES=0 python expert_test.py --prefix_name expert_12 --seed $SEED --expert_id 12 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_12_20w.out  &
# CUDA_VISIBLE_DEVICES=1 python expert_test.py --prefix_name expert_13 --seed $SEED --expert_id 13 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_13_20w.out  &
# CUDA_VISIBLE_DEVICES=2 python expert_test.py --prefix_name expert_14 --seed $SEED --expert_id 14 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_14_20w.out  &
# CUDA_VISIBLE_DEVICES=3 python expert_test.py --prefix_name expert_15 --seed $SEED --expert_id 15 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_15_20w.out  &
# wait

# CUDA_VISIBLE_DEVICES=0 python expert_test.py --prefix_name expert_16 --seed $SEED --expert_id 16 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_16_20w.out  &
# CUDA_VISIBLE_DEVICES=1 python expert_test.py --prefix_name expert_17 --seed $SEED --expert_id 17 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_17_20w.out  &
# CUDA_VISIBLE_DEVICES=2 python expert_test.py --prefix_name expert_18 --seed $SEED --expert_id 18 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_18_20w.out  &
# CUDA_VISIBLE_DEVICES=3 python expert_test.py --prefix_name expert_19 --seed $SEED --expert_id 19 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_19_20w.out  &
# CUDA_VISIBLE_DEVICES=0 python expert_test.py --prefix_name expert_20 --seed $SEED --expert_id 20 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_20_20w.out  &
# CUDA_VISIBLE_DEVICES=1 python expert_test.py --prefix_name expert_21 --seed $SEED --expert_id 21 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_21_20w.out  &
# CUDA_VISIBLE_DEVICES=2 python expert_test.py --prefix_name expert_22 --seed $SEED --expert_id 22 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_22_20w.out  &
# CUDA_VISIBLE_DEVICES=3 python expert_test.py --prefix_name expert_23 --seed $SEED --expert_id 23 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_23_20w.out  &
# CUDA_VISIBLE_DEVICES=0 python expert_test.py --prefix_name expert_24 --seed $SEED --expert_id 24 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_24_20w.out  &
# CUDA_VISIBLE_DEVICES=1 python expert_test.py --prefix_name expert_25 --seed $SEED --expert_id 25 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_25_20w.out  &
# CUDA_VISIBLE_DEVICES=2 python expert_test.py --prefix_name expert_26 --seed $SEED --expert_id 26 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_26_20w.out  &
# CUDA_VISIBLE_DEVICES=3 python expert_test.py --prefix_name expert_27 --seed $SEED --expert_id 27 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_27_20w.out  &
# CUDA_VISIBLE_DEVICES=0 python expert_test.py --prefix_name expert_28 --seed $SEED --expert_id 28 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_28_20w.out  &
# CUDA_VISIBLE_DEVICES=1 python expert_test.py --prefix_name expert_29 --seed $SEED --expert_id 29 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_29_20w.out  &
# CUDA_VISIBLE_DEVICES=2 python expert_test.py --prefix_name expert_30 --seed $SEED --expert_id 30 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_30_20w.out  &
# CUDA_VISIBLE_DEVICES=3 python expert_test.py --prefix_name expert_31 --seed $SEED --expert_id 31 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_31_20w.out  &

# wait

# CUDA_VISIBLE_DEVICES=0 python expert_test.py --prefix_name expert_32 --seed $SEED --expert_id 32 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_32_20w.out  &
# CUDA_VISIBLE_DEVICES=1 python expert_test.py --prefix_name expert_33 --seed $SEED --expert_id 33 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_33_20w.out  &
# CUDA_VISIBLE_DEVICES=2 python expert_test.py --prefix_name expert_34 --seed $SEED --expert_id 34 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_34_20w.out  &
# CUDA_VISIBLE_DEVICES=3 python expert_test.py --prefix_name expert_35 --seed $SEED --expert_id 35 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_35_20w.out  &
# CUDA_VISIBLE_DEVICES=0 python expert_test.py --prefix_name expert_36 --seed $SEED --expert_id 36 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_36_20w.out  &
# CUDA_VISIBLE_DEVICES=1 python expert_test.py --prefix_name expert_37 --seed $SEED --expert_id 37 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_37_20w.out  &
# CUDA_VISIBLE_DEVICES=2 python expert_test.py --prefix_name expert_38 --seed $SEED --expert_id 38 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_38_20w.out  &
# CUDA_VISIBLE_DEVICES=3 python expert_test.py --prefix_name expert_39 --seed $SEED --expert_id 39 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_39_20w.out  &
# CUDA_VISIBLE_DEVICES=0 python expert_test.py --prefix_name expert_40 --seed $SEED --expert_id 40 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_40_20w.out  &
# CUDA_VISIBLE_DEVICES=1 python expert_test.py --prefix_name expert_41 --seed $SEED --expert_id 41 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_41_20w.out  &
# CUDA_VISIBLE_DEVICES=2 python expert_test.py --prefix_name expert_42 --seed $SEED --expert_id 42 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_42_20w.out  &
# CUDA_VISIBLE_DEVICES=3 python expert_test.py --prefix_name expert_43 --seed $SEED --expert_id 43 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_43_20w.out  &
# CUDA_VISIBLE_DEVICES=0 python expert_test.py --prefix_name expert_44 --seed $SEED --expert_id 44 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_44_20w.out  &
# CUDA_VISIBLE_DEVICES=1 python expert_test.py --prefix_name expert_45 --seed $SEED --expert_id 45 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_45_20w.out  &
# CUDA_VISIBLE_DEVICES=2 python expert_test.py --prefix_name expert_46 --seed $SEED --expert_id 46 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_46_20w.out  &
# CUDA_VISIBLE_DEVICES=3 python expert_test.py --prefix_name expert_47 --seed $SEED --expert_id 47 --model_scale 20 --embed_dim 512 --expert_num $EXPERT_NUM > result/1M_random_group16_seed1/experts_20w/expert_47_20w.out  &

# wait