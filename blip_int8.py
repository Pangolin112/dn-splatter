# code snippet that works with the revision
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-opt-2.7b", 
    revision="51572668da0eb669e01a189dc22abe6088589a24"
)

# int8 usage
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", 
    revision="51572668da0eb669e01a189dc22abe6088589a24", # important to use the revision with a older version to prevent the mismatch error
    load_in_8bit=True, 
    device_map="auto"
)

# image = Image.open("data/49a82360aa_original/DSC00043_original.png")
image = Image.open("data/ScanNetpp_512/fb5a96b1a2/images/DSC02785.png")

prompt = ""
# prompt = "Question: how many tables are there? Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)