# LeADMM reproduction




## About

Были воспроизведены модели ADMM, LeADMM, описанные в статье [Learned reconstructions for practical
mask-based lensless imaging](https://arxiv.org/pdf/1908.11502), а также leADMM + post8, pre8 + leADMM, pre4 + leADMM + post4 из статьи [Towards Robust and Generalizable Lensless Imaging with Modular
Learned Reconstruction](https://arxiv.org/pdf/2502.01102) без psf correction и с anisotropic version of TV regularisation.

Каждая модель была обучена в течении 12000 шагов

Результаты обучения представлены в отчете X

## Structure

Основной код проекта находится в папке `src`, названия папок соотносятся с кодом в них содержашимся, отдельно стоит упомянуть:

Загрузка весов моделей осуществляется при помощи скрипта Y

Протестировать обученные модели вы можете используя `demo.ipynb`

## Instalation

1. Склонируйте репозиторий

```bash
git clone https://github.com/mikkklyubbin/LE_ADDM_reproduction.git
cd ./LE_ADDM_reproduction
```
2. (Опцианально)
создайте conda env
```bash
# create env
conda create -n admm python=3.12

# activate env
conda activate admm
```

3. Установите все требуемые библиотеки
```bash
pip install -r requirements.txt
```

## Reproduction

### Train

1. (Опционально) изменить настройки trian в  `src/configs/train.yaml`

2. Запутстите скрипт

```bash
python train.py
```

### Evaluation

1. (Опционально) изменить настройки trian в  `src/configs/inference.yaml`

2. Запутстите скрипт

```bash
python train.py
```



## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

## Cites

### PyTorch Project Template
authors:
   - family-names: Grinberg

   - given-names: Petr

   - orcid: https://orcid.org/0009-0008-4480-5595

date-released: 2024-09-01

url: https://github.com/Blinorot/pytorch_project_template
