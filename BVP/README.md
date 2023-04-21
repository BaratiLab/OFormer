### Training

Electrostatics
```bash
python ../../train_electro.py \
--lr 5e-4 \
--ckpt_every 4000 \
--iters 32000 \
--batch_size 32 \
--train_dataset_path ../../../pde_data2/es_train_no_ma.pkl \   # if using mesh augmentation please change to es_train_ma.pkl
--test_dataset_path ../../../pde_data2/es_test.pkl
```

Magnetostatics
```bash
python ../../train_magneto.py \
--lr 5e-4 \
--ckpt_every 4000 \
--iters 32000 \
--batch_size 32 \
--train_dataset_path ../../../pde_data2/ms_train_no_ma.pkl \
--test_dataset_path ../../../pde_data2/ms_test_shape.pkl
```

#### Recomended hyperparameters
For electro, it is recomended to use following set parameter (change line 45 - 62 in tran_electro.py)
```python
encoder = IrregSpatialEncoder2D(
  input_channels=11,  # the feature provided in the dataset
  in_emb_dim=64,
  out_chanels=64
  heads=1,
  depth=2,
  res=50,
  use_ln=False,
  emb_dropout=0.05
)

decoder = IrregSpatialDecoder2D(
  latent_channels=64,
  out_channels=3,  # the label provided in the dataset, potential and field
  res=50,
  scale=1,
  dropout=0.0,
)

```

For magneto, it is recomended to use following set parameter (slightly larger model)
```python
encoder = IrregSpatialEncoder2D(
        input_channels=11,   # the feature provided in the dataset
        in_emb_dim=64,
        out_chanels=96,  
        heads=1,
        depth=2,
        res=50,
        use_ln=False,
        emb_dropout=0.05
    )

decoder = IrregSpatialDecoder2D(
  latent_channels=96,
  out_channels=3,      # the label provided in the dataset, potential and field
  res=50,
  scale=1,
  dropout=0.0,
)

```

