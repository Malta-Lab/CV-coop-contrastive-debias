export CUDA_DEVICE_ORDER=PCI_BUS_ID

python train_classifier.py \
--classifier resnet \
--batch_size 192 \
--dataset celeba \
--dataset_path ./datasets/ \
--target Male \
--bias Arched_Eyebrows \
--transform 220-180 \
--lr 1e-4 \
--epoch 100 \
--early_stop 10 \
--gpu 0 \
--name 22-02_Male_Arched_Eyebrows \
--seed 42 \
--bias_prop 0.05 \
--is_biased True \