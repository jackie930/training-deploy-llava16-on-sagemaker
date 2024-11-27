from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset,load_from_disk

#dataset = load_from_disk('/home/ec2-user/SageMaker/klook/data0527/data.hf')
#dataset = load_dataset('HuggingFaceH4/llava-instruct-mix-vsft')
processor = AutoProcessor.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf', trust_remote_code=True)



if __name__ == "__main__":
    #dataset = load_from_disk('data.hf')
    test =  [{'content': [{'index': None, 'text': 'Fill in the blank: this is a photo of a {}, you should select the appropriate categories from the provided labels: [Hallway,Sauna,Surrounding environment and attractions,Business Center,Pets,Restaurant,Fitness Facility,Activities,Pool,Exterior,Bar,Transportation,Bathhouse,Food and Dining,Lobby,Shopping,Parking,Staircase/Elevator,Laundry Room,Room,Banquet Hall,Spa,Hot springs,Terrace/Patio,Entertainment facility,Other]\n', 'type': 'text'}, {'index': 0, 'text': None, 'type': 'image'}], 'role': 'user'}, {'content': [{'index': None, 'text': 'this is a photo of Surrounding environment and attractions.', 'type': 'text'}], 'role': 'assistant'}]
    processor = AutoProcessor.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf', trust_remote_code=True)
    texts = processor.apply_chat_template(test, tokenize=False) 
    texts = texts.replace('<\s>','</s>')
    print ("test!",texts)