from datasets import Dataset,DatasetDict,Image
import json

def update_json_v16(input_json,output_json):
    with open(input_json, 'r') as file:
        data = json.load(file)

    # Update the data
    for i in data:
        i['images'] = i.pop('image')
        i['messages'] = []
        i['messages'].append({'content': [{'index': None, 'text': i['conversations'][0]['value'].replace('<image>\n','')+'\n', 'type': 'text'},
        {'index': 0, 'text': None, 'type': 'image'}],
        'role': 'user'})
        i['messages'].append({'content': [{'index': None, 'text': i['conversations'][1]['value'], 'type': 'text'}],
        'role': 'assistant'})

    # Write back to the file
    with open(output_json, 'w') as file:
        json.dump(data, file, indent=4)
    return output_json

def custom():
    input_train = update_json_v16('data_v2.json','train_16.json')
    input_test = update_json_v16('test_v2.json','test_16.json')

    train_dataset = Dataset.from_json('train_16.json')
    test_dataset = Dataset.from_json('test_16.json')

    # Combine into a DatasetDict
    dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    dataset = dataset.cast_column("images", Image())
    
    return dataset

if __name__ == "__main__":
    dataset = custom()
    dataset.save_to_disk("data.hf")