# SAL-CryptoPulse

The official implementation of the paper "CryptoPulse: Short-Term Cryptocurrency Forecasting with Dual-Prediction and Cross-Correlated Market Indicators". Portions of the code have been adapted from the DLinear implementation.

## 🚀 Features

- **Short-Term Forecasting**: Predict cryptocurrency prices with a model designed for short-term predictions.
- **Batch Processing**: Run predictions for multiple cryptocurrencies at once.
- **Flexible & Easy to Use**: Just install, run, and watch the results roll in!

## ⚙️ Installation

You can install the package directly from GitHub with `pip`:

pip install git+https://github.com/aamitssharma07/SAL-Cryptopulse.git

Alternatively, you can install it manually:
1.Clone the repo:

git clone https://github.com/aamitssharma07/SAL-Cryptopulse.git
cd SAL-Cryptopulse

2.Install dependencies in a virtual environment:

python -m venv env_crypto
env_crypto\Scripts\activate
pip install -e .

## 🎯 Usage
After installation, you can run predictions directly using the package's command line interface if you had installed the  package directly from GitHub with `pip :

For a single run:
cryptopulse --data <crypto-ticker-symbol> --train-epochs 10 --batch-size 32

For batch processing (multiple cryptos at once):
cryptopulse_batch

Alternatively, you can install it using python command
For a single run:
python -m cryptopulse.batch_processor

For batch processing (multiple cryptos at once):
python -m cryptopulse.main --data BTC-USD --train-epochs 10 --batch-size 32

## 📊 Results
All results are saved in the results/cryptopulse_results directory.

## 💡 Contributing
Feel free to use thi research work. Let’s build a smarter CryptoPulse together!

## 📝 Citation
@article{salcryptopulse2024,
  author = {Amit Kumar and Dr. Taoran Ji},
  title = {CryptoPulse: Short-Term Cryptocurrency Forecasting with Dual-Prediction and Cross-Correlated Market Indicators},
  journal = {Conference of IEE BigData 2024},
  year = {2024},
  volume = {5},
  pages = {},
}

