# Audio-Attack

# To Do List

### 1: 语音攻击: 设计新的loss function 或者处理数据长度，使得攻击目标的比例变高

2：命令攻击：

2.1：调参与训练

2.2：目前的loss function的曲线可能是由于attack过于容易优化导致，应调整训练时对于振幅的限制，使得限制逐渐变小

2.3：给attack加上振幅的扰动，使其对于振幅的影响更加robust

2.4：思考trade off怎么做，包括怎么在demo中体现，以及是否需要在loss function中体现

# Run

```
python3 test.py model.model_path=librispeech_pretrained_v3.ckpt test_path=data/command_train.json attack=True
```
