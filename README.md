# Danish Pre-trained BERT Tutorial
Tutorial on how to download and convert the [BotXo's pre-trained danish BERT model](https://github.com/botxo/nordic_bert) from TensorFlow to PyTorch. More specifically it is converted to the convenient NLP library [transformers](https://github.com/huggingface/transformers) created by [Huggingface](https://huggingface.co/).

<img src="https://user-images.githubusercontent.com/39537120/92368737-66794280-f0f8-11ea-9ff7-51a72222a668.png" width="350">


# Background
BERT is an NLP model artichecture which is pre-trained on a huge corpus of text and afterwards fine-tuned to more specific problems eg. NLP classification tasks. 

Pre-training BERT is computationally heavy. [AWS writes that BERT can be pre-trained in 62 minutes using 256 p3dn.24xlarge EC2 GPU instances](https://aws.amazon.com/blogs/machine-learning/amazon-web-services-achieves-fastest-training-times-for-bert-and-mask-r-cnn/), each costing 33.7 USD/hour. 
This adds up to ~8627 USD to pre-train BERT!

The transformers library provide pre-trained versions of BERT in many languages, but sadly not in danish. 

<img src="https://user-images.githubusercontent.com/39537120/91264134-d38fee00-e770-11ea-83d9-45ad39fa47d6.png" width="400">

Fortunately [BotXo](https://github.com/botxo) has pre-trained a danish BERT and made it public available. However, this is a TensorFlow version of BERT, and since I use PyTorch I would like to convert it to PyTorch.

# Tutorial
## Step 1
Install virtual environment
```
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Step 2
Download the [danish pre-trained model](https://www.dropbox.com/s/19cjaoqvv2jicq9/danish_bert_uncased_v2.zip?dl=1)
Create a directory called 'bert-base-danish' and put all downloaded files into it.

## Step 3
Use the transformers CLI to convert the model from TensorFlow to PyTorch:
```
transformers-cli convert --model_type bert --tf_checkpoint ./bert-base-danish/bert_model.ckpt --config ./bert-base-danish/bert_config.json --pytorch_dump_output ./bert-base-danish/pytorch_model.bin
```

## Step 4
Rename bert_config.json to config.json

## Step 5
Initialize the BERT and Tokenizer in Python:
```
from transformers import BertConfig, BertTokenizer, BertModel

# Init BERT configuration object
configuration = BertConfig.from_json_file('./bert-base-danish/config.json')

# Init BERT tokenizer
tokenizer = BertTokenizer(vocab_file='./bert-base-danish/vocab.txt')

# Init BERT model from pretrained
model = BertModel(config=configuration).from_pretrained('bert-base-danish')
```

