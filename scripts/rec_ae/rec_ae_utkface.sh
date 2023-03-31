export CUDA_DEVICE_ORDER=PCI_BUS_ID

python train_rec_ae.py \
--autoencoder old \
--batch_size 128 \
--dataset utkface \
--dataset_path ./datasets/ \
--transform default \
--lr 1e-4 \
--epoch 1000 \
--early_stop 10 \
--gpu 1 \
--name rec_old_utkface \
--seed 42
