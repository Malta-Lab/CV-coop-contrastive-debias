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
--early_stop 15 \
--gpu 3 \
--name Chubby_Double_Chin_bp0_05 \
--seed 42 \
--bias_prop 0.05 \
--is_biased True