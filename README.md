Mathematical background

The mathematical details of the network are as follows. Given an input
vector, x, of size d × 1 our classifier outputs a vector of probabilities, p
(K × 1), for each possible output label:


$$
\begin{aligned}
{s}_1 &= {W}_ix + {b}_1 \\
h &= \max(0, {s}_1) \\
s &= {W}_2 h + {b}_2 \\
p &= \text{softmax}(s)
\end{aligned}
$$

where the matrix ${W}_1$ and ${W}_2$ have size m × d and K × m respectively and the vectors ${b}_1$ and ${b}_2$ have sizes m × 1 and K × 1. SOFTMAX is defined as

$$
\begin{aligned}
\text{SOFTMAX(s)} &= \frac{e^{s}}{\mathbf{1}^T e^{s}}
\end{aligned}
$$

The predicted class corresponds to the label with the highest probability:

$$
k^* = \arg\max_{1 \leq k \leq k} \; p_k
$$


We have to learn the parameters ${W}_1$, ${W}_2$, ${b}_1$ and ${b}_2$ from our labelled training data.

The model as

$$
\Theta = \{W_1, W_2, b_1, b_2\}
$$

$$
J(D, \lambda, \Theta) =
\frac{1}{|D|}
\sum_{(x,y)\in D}
\ell_{\text{cross}}(x, y, \Theta)
+
\lambda \sum_{l=1}^{2} \sum_{i,j} W_{l,ij}^2
$$

Where 

$$
\ell_{\text{cross}}(x, y, \Theta) = -\log(p_y)
$$

The optimization problem we have to solve is

$$
\Theta^* = \arg\min_{\Theta} J(D, \lambda, \Theta)
$$




