export CUDA_DEVICE_ORDER=PCI_BUS_ID

python train_classifier.py \
--classifier resnet \
--batch_size 192 \
--dataset celeba \
--dataset_path ./datasets/ \
--target Arched_Eyebrows \
--bias Male \
--transform 220-180 \
--lr 1e-4 \
--epoch 100 \
--early_stop 15 \
--gpu 0 \
--name Arched_Eyebrows_Male_bp0_1 \
--seed 42 \
--bias_prop 0.1 \
--is_biased True