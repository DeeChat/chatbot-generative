DeeChat
====

This is a generative-model based chatbot, trained with Chinese corpus. Forked from https://github.com/chiphuyen/tf-stanford-tutorials/tree/master/assignments/chatbot, Seq2Seq with attention mechanism is used. The code is based on TensorFlow r1.2 and has been tested on Python 2.7 and Python 3.6. 

To train your model, run:
```bash
cd chatbot-generative
python chatbot.py --mode=train
```
To chat with your chatbot on terminal, run:
```bash
python chatbot.py --mode=chat
```
To deploy the model on your Wechat, run:
```bash
python wechat.py
```
