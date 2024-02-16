from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large") #using the large parameter from GPT to generate larger te# xts
model = GPT2LMHeadModel.from_pretrained("gpt2-large",pad_token_id=tokenizer.eos_token_id).to(device)

sentence = 'The stream of water is heard loudly as others talk'  #input sentence
input_ids = tokenizer.encode(sentence, return_tensors='pt').to(device) #using pt to return as pytorch tensors

output = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True).to(device)
print(tokenizer.decode(output[0], skip_special_tokens=True))
