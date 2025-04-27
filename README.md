python drift_exp_stats.py \
    --datasets cifar10 svhn \
    --models resnet18 densenet121 resnext50 \
    --ft_ratios 0.1 0.2 0.3 \
    --num_runs 30 \
    --train_epochs 10 \
    --ft_epochs 5 \
    --lr_base 0.01 \
    --lr_ft 0.001 \
    --results_dir ./drift_results_combined \
    # --skip_train \  # Add this flag AFTER the first run to load existing baselines
    # --use_pretrained \ # Add this if you want ImageNet pretraining
    # --skip_viz # Add this if you don't need the visualization plots
