#### beam search
#### 优点：改进了之前的greedy search algorithm，因为贪心搜索可能会造成达到局部最优解的情况，所以每次返回2~3个可以改进
#### 缺点：（1）概率相乘可能造成数据下溢，使用log改进
####       （2）倾向生成短的序列，使用归一化对数似然函数改进
####       （3）单一性问题，输出句子差异性小，分组加入相似性惩罚项，diverse search

### 这里有两个notebook，一个是参考网上的代码写的base版本，一个是对包Transformers中关于Beam Search 实现的细节进行学习
