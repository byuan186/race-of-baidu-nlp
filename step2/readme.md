### 开始逐步构建seq2seq模型baseline
#### 1.Teachering Forcing
#### 优点：训练阶段，decoder进行预测时防止出现某时间步预测错误导致后面都跟着错
#### 缺点：模型泛化能力差


#### tips：
#### 在训练阶段计算decoder的loss时，刚开始是只mask那些被padding的位置，我后续尝试mask那些UNK，结果分数会提升1%
