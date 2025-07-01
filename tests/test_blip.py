# code snippet that works with the revision
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-opt-2.7b", 
    revision="51572668da0eb669e01a189dc22abe6088589a24"
)
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b", 
#     revision="51572668da0eb669e01a189dc22abe6088589a24", # important to use the revision with a older version to prevent the mismatch error
#     torch_dtype=torch.float16,
#     device_map="auto"
# ).to(device)
# # model.enable_model_cpu_offload() # BLIP2 does not support this, we can use device_map="auto" to offload the model to cpu automatically

# int8 usage
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", 
    revision="51572668da0eb669e01a189dc22abe6088589a24", # important to use the revision with a older version to prevent the mismatch error
    load_in_8bit=True, 
    device_map="auto"
)

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# image = Image.open("data/49a82360aa_original/DSC00043_original.png")
image = Image.open("data/fb5a96b1a2_original/DSC02791_original.png")

prompt = ""
# prompt = "Question: how many tables are there? Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
# inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

# another code snippet that works w/o the revision
# from PIL import Image
# import requests
# from transformers import Blip2Processor, Blip2ForConditionalGeneration, AddedToken
# import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
# )

# processor.num_query_tokens = model.config.num_query_tokens
# image_token = AddedToken("<image>", normalized=False, special=True)
# processor.tokenizer.add_tokens([image_token], special_tokens=True)

# model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=64) # pad for efficient computation
# model.config.image_token_index = len(processor.tokenizer) - 1

# # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# # image = Image.open(requests.get(url, stream=True).raw)

# image = Image.open("data/49a82360aa_original/DSC00043_original.png")

# # prompt = "Question: how many cats are there? Answer:"
# prompt = ""
# inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

# generated_ids = model.generate(**inputs, max_new_tokens=20)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
# print(generated_text)





