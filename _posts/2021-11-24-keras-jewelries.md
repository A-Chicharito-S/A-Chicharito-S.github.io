---
title: Keras Jewelries
categories:
- Coding Skill
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


After making a horrible mistake which makes a two-week experiment nothing, I start to think maybe it's time to summarize some typical coding skills as well as some reminders for **Keras**

### 1. Wrap in function instead of sub-classing

Consider the following scenario:  you need to get an intermediate output of your model, which can be done by:
```python
new_model = keras.Model(inputs=old_model.input, outputs=old_model.get_layer['wanted_layer'].output)
```
However, if your model is created by sub-classing:
```python
class my_model(keras.Model):
    def __init__(self, 'your_arguments_here'):
        super(my_model, self).__init__()
        'some custom properties here'
        'make sure all layers, e.g.,CNN, are created here, and then later used in call()'
    def call(self, inputs):
        'your defined computation process'
        return y
        'y is the computed output'
```
you'll find that the '**new_model**' will throw an error indicating that since it can't infer the inputs shapes and what's more, if you add '**shape=()**' parameter to the '**call()**' method or add '**Input()**' layer in the '**\__init__()**' method, it will either not work or throw another error indicating you can't add '**Input()**' layer in the '**\__init__()**' method

Thus, I suggest using the following way to create a model:
```python
def make_model('your_arguments_here'):
    x_1 = Input(shape=('shape of input x_1'))
    ...
    x_n = Input(shape=('shape of input x_n'))
    'computation process here'
    model = keras.Model(inputs=[x_1,...,x_n], outputs=[y_1,..,y_m])
    return model
```
In this sense, the model is then able to infer the inputs' shapes when getting the intermediate outputs

### 2. Long directory may lead to failure in saving model

Funny story, when you're trying to save a model:
```python
model.save('your_model_name.hd5')
```
However you can save it in a non-'.hd5' way, see [[here]](https://www.tensorflow.org/guide/keras/custom_layers_and_models). And sometimes (might be with lots of parameters), the saving process will fail and throw an error, which after online search, can be solve by shorten the saving directory, like:

   C:\Users\xxx\projects\tf_projects\keras_projects\my_project $\longrightarrow$  C:\Users\xxx\my_project

### 3. When you're padded, calculating 'Mean' involving its mask

For example, when you're dealing with the padded sentences:

​          input: [34, 52, 1257, 872, 0, 0, 0]  (4 actual words, 3 paddings), with shape: (7, )

​          input's mask: [1, 1, 1, 1, 0, 0, 0]

​          after embedding, it becomes with the shape: (7, embedding_dim)

And now say, you would like to calculate the mean of the embeddings of words, you may write:
```python
mean = tf.reduce_mean(input, axis=0)
```
However this is actually **WRONG** ! Since the calculation of the above code is actually: 

​    ($sum$( **4** embeddings for **words**) + $sum$( **3** embeddings for **paddings**)) / **7**

And what we want to calculate is:

​     $sum$( **4** embeddings for words) / **4**

So what we should do is to make use of the mask:
```python
 mean = tf.reduce_sum(input, axis=0) / tf.reduce_sum(mask, axis=0, keep_dim=True)     
```

### 4. Deal carefully with length

Sometimes you may encounter the following error:

'ValueError: all input arrays must have the same shape'

This is when doing padding, you forget to control the length (e.g., forget to truncate, forget to unify the max length)
