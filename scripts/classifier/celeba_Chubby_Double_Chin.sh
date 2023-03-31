export CUDA_DEVICE_ORDER=PCI_BUS_ID

python train_classifier.py \
--classifier resnet \
--batch_size 192 \
--dataset celeba \
--dataset_path ./datasets/ \
--target Chubby \
--bias Double_Chin \
--transform 220-180 \
--lr 1e-4 \
--epoch 100 \
--early_stop 10 \
--gpu 1 \
--name 03-03_Chubby_Double_Chin_bp0.1 \
--seed 42 \
--bias_prop 0.1 \
--is_biased True \