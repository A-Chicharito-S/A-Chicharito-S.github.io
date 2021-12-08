---
title: Treasure hunting 2
categories:
- Treasure hunting
feature_image: "https://picsum.photos/2560/600?image=872"
---
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

## Conditional Poisson Stochastic Beam Search5

Today I'm going to introduce the paper '**Conditional Poisson Stochastic Beam Search**' by *Clara Meister*, *Afra Amini*, *Tim Viera*, and *Ryan Cotterell*, you can find the paper [[here]](https://arxiv.org/abs/2109.11034) 

And different from the previous '**Treasure hunting**' episodes, in this very episode, instead of introducing the paper following its structure, I re-arrange the paper and introduce it in a more logical way that I consider myself better to understand. (**And to be honest the paper is really tough for me !**) Though still some parts of the paper remain unclear for me, I try my best to illustrate its main ideas.

Note that: '**xxx**' means quoted from the paper; **xxx** is to underline; ***sub_title*** is the sub title I named to help understand the structure of the paper; '*xxx*' is the question / comments I raised to help understand the idea/logic of the paper; And the formulas are all from the paper (some minor modification for consistence with my own signs are done), and for writing fluency, I sometimes introduce concepts as 'We define ...' instead of 'The authors define ...'

### Deterministic vs. Stochastic

- ***What is beam search ?***

**Beam search** is a very important decoding strategy for NLP tasks involving generation (e.g., NMT, text generation), they're usually done in the following way: 

For example, we're going to generate a sentence with a vocabulary $\mathcal{V}={I, like, spring, EOS}$

