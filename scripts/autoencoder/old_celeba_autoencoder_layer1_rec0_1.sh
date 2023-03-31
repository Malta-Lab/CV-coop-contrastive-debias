export CUDA_DEVICE_ORDER=PCI_BUS_ID

python train_autoencoder.py \
--alpha 1 \
--theta 1 \
--dataset celeba \
--classifier resnet \
--autoencoder old \
--feat_layer layer1 \
--transform 220-180 \
--batch_size 100 \
--dataset_path ./datasets/ \
--early_stop 10 \
--epoch 100 \
--gpu 0 \
--name 17_12_old_ae_1e-3_dyn_layer1_rec0.1 \
--temperature 0.5 \
--load_path_cl ./checkpoints/celeba/best_model_encoder_old.ckpt \
--load_path_ae ./checkpoints/rec_ae/rec_old_ae_celeba_180_220_1e-4.ckpt \
--recon 0.1 \
--rec_epoch 100 \
--lr 1e-3 \
--seed 50 \
--dyn \


