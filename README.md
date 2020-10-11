# About
"[自然発話に頑健な機械翻訳の検討](https://ichiroex.github.io/paper/murakami19nlp.pdf) (村上ら, 言語処理学会 2019)" の再現実装

## requirements
- Python 3.7  
  - Pytorch 1.2.0+
     
     If you use pipenv, you can easily set up the environment as follows.  
     ```
     pipenv install --dev
     ```

## get started
The operation is completed by executing two shell scripts. Place the raw data in the right place before running it (see data / README.md for details).

1. Train Noise labeling model

```bash
./run.sh
```

2. Add noise to Japanese files in the bilingual corpus

```bash
./generate.sh
```
