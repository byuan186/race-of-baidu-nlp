#### 1.虽然Beam Search 效果比贪心搜索好一些，但是仍然会面临以下问题：OOV（未登录词）和 词语重复问题
#### 2.指针生成网络首先引入指针网络（Pointer Network），用于从原文中复制单词，解决OOV问题，然后引入Coverage机制，解决摘要中语句重复的问题。
#### 心得体会：
##### 一开始使用pgn的结果才30分左右，成绩甚是尴尬，后来我发现论文中decoder使用context的方法和tensorflow官网上机器翻译模型不一样

+ 机器翻译模型是把context_vector和decoder输入词的词向量拼接后作为gru单元的输入，gru的输出接入一个全连接层，算出预测词的分布
```
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):

    hidden_with_time_axis = tf.expand_dims(query, 1)

    """ 1: 计算注意力分布，形状 == （批大小，最大长度，1） """
    score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
    attention_weights = tf.nn.softmax(score, axis=1)

    """ 2: 计算上下文向量，形状 == （批大小，隐藏层大小）"""
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
```
+ 论文中PGN的gru单元输入就只有输入词的词向量，context_vector是和gru的输出做拼接，再输入全连接层进行预测，论文里是有两层全连接层。
