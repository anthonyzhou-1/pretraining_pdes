# Data Generation

## Usage

To generate 2D combined data:

```
python 2D_combined.py --num_samples=[# samples] --split=['train', 'valid'] 
```

Note that this could take a while and use a couple GB of storage.

The data is stored in the following structure, which is expected by the code
- Heat_Adv_Burgers_train_3072.h5
    - train
        - u (num_samples, n_t, n_x, n_y)
        - x (2, n_x, n_y)
        - t (n_t,)
        - coefficients (e.g. nu, etc.) (num_samples,)
