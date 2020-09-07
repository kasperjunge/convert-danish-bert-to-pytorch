###########################################
#                                         #
#       DOWNLOAD PRE-TRAINED MODEL        #
#                                         # 
###########################################

## Download the pretrained danish BERT Model from botxo's nordic_bert repo (thanks to Jens Dahl Møllerhøj)
# URL: https://github.com/botxo/nordic_bert


###########################################
#                                         #
#   CONVERT FROM TENSORFLOW TO PYTORCH    #
#                                         # 
###########################################

## Convert Model from TensorFlow to PyTorch with Huggingface transformer-cli:
# transformers-cli convert --model_type bert --tf_checkpoint bert_model.ckpt --config bert_config.json --pytorch_dump_output pytorch_model.bin



from transformers import BertConfig, BertTokenizer, BertModel

configuration = BertConfig.from_json_file('./bert-base-danish/config.json')
tokenizer = BertTokenizer(vocab_file='./bert-base-danish/vocab.txt')
model = BertModel(config=configuration).from_pretrained('bert-base-danish')

