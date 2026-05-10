Background 1: Mathematical background

The mathematical details of the network are as follows. Given an input
vector, x, of size d × 1 our classifier outputs a vector of probabilities, p
(K × 1), for each possible output label:


$$
\begin{aligned}
{s}_i &= {W}_ix + {b}_1 \\
h &= \max(0, {s}_1) \\
s &= {W}_2 h + {b}_2 \\
p &= \test{softmax}(s)
\end{aligned}
$$



