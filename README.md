# Bayes Transition Partial-label Learning

Bayes Transition PLL is a novel algorithm for partial-label learning, inspired by advancements in noisy label learning. This project introduces a Bayesian transition approach to improve robustness and accuracy in handling ambiguous and noisy labels.

##Overview
* Objective: Develop a Bayesian transition matrix-based method to enhance partial-label learning.
* Key Contributions:
  * Innovative algorithm combining noisy label learning techniques with partial-label learning.
  * Focus on robust statistical modeling and rigorous mathematical foundations.
  * Implementation validated on benchmark datasets.

## Repository Structure

```
├── datasets/             # Data files for training and evaluation
├── utils/                # Auxiliary algorithms
├── loss.py               # Custom loss functions using Bayesian transitions
├── main.py               # Main training and evaluation script
├── resnet.py             # Backbone ResNet model
├── resnet_bayes.py       # Bayesian ResNet implementation
└── README.md             # Project documentation
```

## Quick Start

1.	Clone the repository:

  ```bash
  git clone https://github.com/KevinCarpricorn/Transition_Matrix_PLL.git
  cd Transition_Matrix_PLL
  ```

2.	Run training:

```bash
python main.py --epochs 50 --batch_size 32 --learning_rate 0.001
```


## Results

The proposed algorithm demonstrates:
* High accuracy and robustness in partial-label learning tasks.
* * Superior performance in handling label noise compared to baseline methods.

## License

This project is licensed under the MIT License.
