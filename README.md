# Neural Network From Scratch

This project implements a fully connected neural network (FCNN) **from scratch in NumPy**. The objective was to deeply understand forward and backward propagation, including enhancements like Batch Normalization and L2 Regularization.

---

## Project Structure

- **Part 1:** Forward propagation (linear, activation, softmax, batchnorm)
- **Part 2:** Backward propagation (gradients, chain rule, L2 reg)
- **Part 3:** Training loop (with mini-batch, early stopping)
- **Part 4:** MNIST classification (baseline, no batchnorm)
- **Part 5:** MNIST with Batch Normalization
- **Part 6:** L2 Regularization (with and without batchnorm)

---

## Model Config

- Input: Flattened MNIST images (784)
- Architecture: `[784, 20, 7, 5, 10]`
- Optimizer: Mini-batch gradient descent
- Batch size: 128  
- Learning rate: 0.009  
- Early stopping: 100 steps without improvement

---

## Results Summary

| Setup                     | Test Accuracy | Epochs |
|---------------------------|---------------|--------|
| No BatchNorm, No L2       | **87.97%**    | 80     |
| BatchNorm Only            | 83.36%        | 43     |
| L2 Regularization Only    | **90.72%**    | 76     |
| BatchNorm + L2            | 89.69%        | 88     |

---

## Key Takeaways

- BatchNorm speeds up convergence but may reduce accuracy in shallow nets.
- L2 Regularization effectively prevents overfitting and improves generalization.
- Best performance was achieved using **L2 Regularization without BatchNorm**.

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook

