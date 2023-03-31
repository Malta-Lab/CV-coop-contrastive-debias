export CUDA_DEVICE_ORDER=PCI_BUS_ID

python train_rec_ae.py \
--autoencoder old \
--batch_size 128 \
--dataset celeba \
--dataset_path ./datasets/ \
--lr 1e-4 \
--epoch 100 \
--early_stop 6 \
--gpu 1 \
--name rec_old_180_220_1e-4 \
--seed 40
