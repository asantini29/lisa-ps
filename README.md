# LISAps: 

`lisaps` is a python library built to analyze stochastic signals in LISA under agnostic modelling assumptions. This is done including Akima splines in both the noise and signal models. 

This package is designed to interact with the `eryn` MCMC sampler (https://github.com/mikekatz04/Eryn). The paper describing the code implementation can be found on [ArXiv](https://arxiv.org/abs/2507.06300): 

## Installation
First, install the GPU-accelerated Akima splines module:
```
git clone https://github.com/asantini29/CudAkima.git
cd CudAkima
python setup.py install
cd ..
```
Details about our splines can be found here: https://github.com/asantini29/CudAkima.

Now install the library:
```
git clone https://github.com/asantini29/lisa-ps.git
cd lisa-ps
python setup.py install
```


## Versioning

We use [SemVer](http://semver.org/) for versioning. 

Current Version: 0.1.0

## Authors

* **Alesandro Santini**

### Contributors
* Martina Muratore
* Olaf Hartwig


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## Citing

If you use `lisa-ps` in your research, you can cite it in the following way:

TODO

## Aknowledgments
We thank Nikolaos Karnesis, Jean-Baptiste Bayle, Mauro Pieroni for discussions.
