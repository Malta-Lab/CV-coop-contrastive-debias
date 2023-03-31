export CUDA_DEVICE_ORDER=PCI_BUS_ID

python get_centroids.py \
--dataset celeba \
--classifier resnet \
--feat_layer layer1 \
--transform 220-180 \
--batch_size 100 \
--dataset_path ./datasets/ \
--gpu 1 \
--name 03-03_Arched_Eyebrows_CENT_layer1 \
--load_path_cl ./checkpoints/classifier/27-02_Arched_Eyebrows_Male_bp0.05/cl_best_model.ckpt \
--centroid_path ./centroid_celeba/03-03_Arched_Eyebrows_CENT_layer1
--is_biased True \
--bias_prop 0.05 \
--target Arched_Eyebrows \
--bias Male \
--seed 42 \