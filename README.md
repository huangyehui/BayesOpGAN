## BayesOpGAN

BayesOpGAN offers a generative adversarial network for 1D data generation. The network incorporates BayesOpLoss to generate higher-fidelity 1D data and provides a Fourier Distance Tool to measure the authenticity of generated data.
<img width="720" height="405" alt="gan-raman" src="https://github.com/user-attachments/assets/2bf2eb34-e30a-4668-b540-4003e5710023" />
## Dataset
All datasets are available in RRUFF https://rruff.info/

## Usage
### Generate synthetic data
```python
python bayesopgan.py
```
### Measure the generated data with Fourier Distance Tool

```python
python fourier_tool.py
```

## Results
<img width="320" height="240" alt="109600_15 txt" src="https://github.com/user-attachments/assets/a69dd1b9-cb43-4c82-a107-229cfebe6cfd" />

## Citation

If you find BayesOpGAN useful in your research, please consider to cite the following related papers:

```python
@article{BayesOpGAN,
  title={BayesOpGAN: A Bayesian-Optimized GAN Framework with Fourier-Based Evaluation for Quality-Controlled Raman Spectral Data Augmentation},
  author={ Huang, Yehui  and  Zhang, Xintian  and  Wang, Yulin  and  Wang, Yufei  and  Chang, Dongxu  and  Wang, Yong  and  Jiang, Shuqin },
}
```
## Contributing

Main contributors:

- [Yehui Huang], ``huangyehui@gmail.com``
- [Shuqin Jiang] ``jshuqinn@163.com``
