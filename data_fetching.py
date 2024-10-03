

import requests
import json

urls="https://datasets-server.huggingface.co/rows?dataset=lavita%2FChatDoctor-HealthCareMagic-100k&config=default&split=train&offset=0&length=100"

response=requests.get(urls)

if response.status_code==200:
    data=json.loads(response.text)

data

import pandas as pd

response=requests.get(urls)

if response.status_code==200:
    data=json.loads(response.text)
    rows=data['rows']
    extracted_data=[]
    for row in rows:
        instruction=row['row'].get('instruction','')
        input_text=row['row'].get('input','')
        output_text=row['row'].get('output','')
        extracted_data.append({
            'instruction':instruction,
            'input':input_text,
            'output':output_text
        })

        df=pd.DataFrame(extracted_data)
    df.to_csv('huggingface_extracted_data.csv',index=False)