from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration,AutoTokenizer
import torch 
from PIL import Image

def custom(model_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map=device)
    processor = LlavaNextProcessor.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf')

    #prompt = "<image>\nUSER: Fill in the blank: this is a photo of a {}, you should select the appropriate categories from the provided labels: [Hallway,Sauna,Surrounding environment and attractions,Business Center,Pets,Restaurant,Fitness Facility,Activities,Pool,Exterior,Bar,Transportation,Bathhouse,Food and Dining,Lobby,Shopping,Parking,Staircase/Elevator,Laundry Room,Room,Banquet Hall,Spa,Hot springs,Terrace/Patio,Entertainment facility,Other]\nASSISTANT:"
    
    conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "Fill in the blank: this is a photo of a {}, you should select the appropriate categories from the provided labels: [Hallway,Sauna,Surrounding environment and attractions,Business Center,Pets,Restaurant,Fitness Facility,Activities,Pool,Exterior,Bar,Transportation,Bathhouse,Food and Dining,Lobby,Shopping,Parking,Staircase/Elevator,Laundry Room,Room,Banquet Hall,Spa,Hot springs,Terrace/Patio,Entertainment facility,Other]"},
          {"type": "image"},
        ],
    },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)


    image = Image.open("food.jpg")
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = inputs.to(device)
    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=100)

    print(processor.decode(output[0], skip_special_tokens=True))


    #print ("res: ", res)

if __name__ == "__main__":
    ## 
    print ("hf model")
    model_id = 'llava-hf/llava-v1.6-mistral-7b-hf'
    custom(model_id)
    print ("cutomer model")
    model_id = 'sft-llava-1.6-7b-hf-customer/checkpoint-20'
    custom(model_id)