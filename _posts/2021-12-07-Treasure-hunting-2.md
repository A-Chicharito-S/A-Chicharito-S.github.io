<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

---
title: Treasure hunting 2
categories:
- Treasure hunting
feature_image: "https://picsum.photos/2560/600?image=872"
---

## Conditional Poisson Stochastic Beam Search4

Today I'm going to introduce the paper '**Conditional Poisson Stochastic Beam Search**' by *Clara Meister*, *Afra Amini*, *Tim Viera*, and *Ryan Cotterell*, you can find the paper [[here]](https://arxiv.org/abs/2109.11034) 

And different from the previous '**Treasure hunting**' episodes, in this very episode, instead of introducing the paper following its structure, I re-arrange the paper and introduce it in a more logical way that I consider myself better to understand. (**And to be honest the paper is really tough for me !**) Though still some parts of the paper remain unclear for me, I try my best to illustrate its main ideas.

Note that: '**xxx**' means quoted from the paper; **xxx** is to underline; ***sub_title*** is the sub title I named to help understand the structure of the paper; '*xxx*' is the question / comments I raised to help understand the idea/logic of the paper; And the formulas are all from the paper (some minor modification for consistence with my own signs are done), and for writing fluency, I sometimes introduce concepts as 'We define ...' instead of 'The authors define ...'

### Deterministic vs. Stochastic

- ***What is beam search ?***

**Beam search** is a very important decoding strategy for NLP tasks involving generation (e.g., NMT, text generation), they're usually done in the following way: 

For example, we're going to generate a sentence with a vocabulary $\mathcal{V}={I, like, spring, <EOS>}$, where $\langle BOS\rangle$ only indicates the end of the sentence. At the beginning of generation, we start with a $\langle BOS\rangle$ token which indicates the beginning of the sentence. And beam search aims to find the top-$K$ possible decoding options at each time step, thus, at step 1, we may have a probability of ${I=0.6, like=0.1, spring=0.3}$, indicating the likelihood of a word from $\mathcal{V}$ being placed at the step ( at step 1, which means after $\<BOS\>$), and let the $K=2$ , then the decoding process can be roughly illustrated as: <a name='1'></a>

<div>
$$\begin{split} \{<BOS>\} & \stackrel{step 1}{\longrightarrow}\{<BOS>I={\color{red}0.6},<BOS>like=0.1,<BOS>spring={\color{red}0.3},<BOS><EOS>=0.0\} \\& \stackrel{step 2}{\longrightarrow}\{I+like={\color{red}0.6\times 0.9},I+I=0.6\times 0.05,I+spring=0.6\times 0.05,I\,+<EOS>=0.6\times 0.0; \\& \,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,spring+like={\color{red}0.3\times 0.8},spring+I=0.8\times 0.05,spring+spring=0.8\times 0.1, \\& \,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,spring\,+<EOS>=0.8\times 0.05;\} \\& \stackrel{step 3}{\longrightarrow}\,\,\,\,...\end{split}$$
</div>

