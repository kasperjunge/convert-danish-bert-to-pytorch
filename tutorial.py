###########################################
#                                         #
#       DOWNLOAD PRE-TRAINED MODEL        #
#                                         # 
###########################################

## Download the pretrained danish BERT Model from botxo's nordic_bert repo (thanks to Jens Dahl Møllerhøj)
# URL: https://github.com/botxo/nordic_bert

## In this case the weights are stored in S3. Run the following commands in the command line to load the pre-trained bert weights:
# aws --profile=jppol-dfp s3 cp s3://dfp-prod-ekstrabladet-upload/ebml/pretrained-bert/bert_model.ckpt.data-00000-of-00001 bert_model.ckpt.data-00000-of-00001
# aws --profile=jppol-dfp s3 cp s3://dfp-prod-ekstrabladet-upload/ebml/pretrained-bert/vocab.txt vocab.txt
# aws --profile=jppol-dfp s3 cp s3://dfp-prod-ekstrabladet-upload/ebml/pretrained-bert/bert_config.json bert_config.json
# aws --profile=jppol-dfp s3 cp s3://dfp-prod-ekstrabladet-upload/ebml/pretrained-bert/bert_model.ckpt.index bert_model.ckpt.index
# aws --profile=jppol-dfp s3 cp s3://dfp-prod-ekstrabladet-upload/ebml/pretrained-bert/bert_model.ckpt.meta bert_model.ckpt.meta



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

