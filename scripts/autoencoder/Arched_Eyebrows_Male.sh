export CUDA_DEVICE_ORDER=PCI_BUS_ID

for i in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
alpha=$i
theta=$( echo "1-$alpha" | bc )
echo $alpha 
echo $theta

python train_autoencoder.py \
--classifier resnet \
--autoencoder old \
--feat_layer layer1 \
--batch_size 92 \
--dataset celeba \
--dataset_path ./datasets/ \
--transform 220-180 \
--lr 1e-3 \
--epoch 100 \
--early_stop 15 \
--gpu 0 \
--load_path_cl ./checkpoints/classifier/27-02_Arched_Eyebrows_Male_bp0.05/cl_best_model.ckpt \
--load_path_ae ./checkpoints/rec_ae/rec_old_ae_celeba_180_220_1e-4.ckpt \
--centroid_path ./centroid_celeba/03-03_Arched_Eyebrows_CENT_layer1 \
--name 27-02_ae_Arched_Eyebrows_Male_class$alpha\_contrast0$theta\_1e-3_new_rec_ae \
--temperature 0.5 \
--alpha $alpha \
--theta 0$theta \
--recon 0 \
--rec_epoch 0 \
--target Arched_Eyebrows \
--bias Male \
--is_biased True \
--bias_prop 0.05 \
--seed 42 \
--contrast centroid \

done