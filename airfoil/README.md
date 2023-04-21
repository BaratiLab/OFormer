### Training

```bash
python ../../train_airfoil.py \
--lr 6e-4 \
--ckpt_every 10000 \
--iters 48000 \
--batch_size 16 \
--train_dataset_path ../../../pde_data2/prcoessed_airfoil_train_data_dt6 \
--test_dataset_path ../../../pde_data2/prcoessed_airfoil_test_data_dt6
```
