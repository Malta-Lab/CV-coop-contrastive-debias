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
--gpu 2 \
--load_path_cl ./checkpoints/classifier/Chubby_Double_Chin_bp0_1/cl_best_model.ckpt \
--load_path_ae None \
--name AE_Chubby_Double_Chin_bp0_1_class$alpha\_contrast0$theta \
--temperature 0.5 \
--alpha $alpha \
--theta 0$theta \
--recon 0 \
--rec_epoch 0 \
--target Chubby \
--bias Double_Chin \
--is_biased True \
--bias_prop 0.1 \
--seed 42 \

done