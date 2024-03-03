
#data_root="data"
#OFFSET=$RANDOM
OFFSET=0
for percent in 1 5 10; do
    for fold in 1 2 3 4 5; do
        python tools/dataset/semi_kitti.py --percent ${percent} --seed ${fold} --seed-offset ${OFFSET}
    done
done