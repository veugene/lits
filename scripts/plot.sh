#!/bin/bash

declare -a metrics=(dice_loss_2 dice_2 dice_loss_2_m0 dice_2_m0 dice_loss_1_2 dice_1_2)
if [[ $OUTPUT_NAME == 1 ]]; then
    metrics=(output_0_dice_loss_2 output_0_dice_2 output_0_dice_loss_2_m0 output_0_dice_2_m0 output_1_dice_loss_1_2 output_1_dice_1_2)
fi
if [[ $OUTPUT_NAME == 2 ]]; then
    metrics=(dice fdice masked_dice_loss)
fi
for m in "${metrics[@]}"
do
    min_y=0
    max_y=1
    if [[ $m == *"loss"* ]]; then
        min_y=-1
        max_y=0
    fi
    echo ${m}
    python3 plot.py --max_x $MAX_X --min_y $min_y --max_y $max_y --keys ${m} val_${m} --title "${STAGE}/${experiment_ID}" --source ../results/$STAGE/$experiment_ID/training_log.txt --dest ../results/$STAGE/$experiment_ID/${experiment_ID}_${m}.png
done
