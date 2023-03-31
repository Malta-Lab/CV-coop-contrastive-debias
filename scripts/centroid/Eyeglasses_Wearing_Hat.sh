export CUDA_DEVICE_ORDER=PCI_BUS_ID

python get_centroids.py \
--dataset celeba \
--classifier resnet \
--feat_layer layer1 \
--transform 220-180 \
--batch_size 100 \
--dataset_path ./datasets/ \
--gpu 1 \
--name 09-03_Eyeglasses_Wearing_Hat_CENT_layer1 \
--load_path_cl ./checkpoints/classifier/09-03_Eyeglasses_Wearing_Hat_bp0.05/cl_best_model.ckpt \
--is_biased True \
--bias_prop 0.05 \
--target Eyeglasses \
--bias Wearing_Hat \
--seed 42 \