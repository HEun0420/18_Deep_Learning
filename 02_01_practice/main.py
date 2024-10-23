from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
from googletrans import Translator
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
translator = Translator() 

class LocationRequest(BaseModel):
    image_url: str
    locations: list[str]

@app.post("/analyze/")
async def analyze_location(request: LocationRequest):
    try:
        image = Image.open(requests.get(request.image_url, stream=True).raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail="URL을 입력해주세요.")

    # 지역 이름 입력받기
    choices = []
    original_choices = []  

    for location in request.locations:
        if location:  
            translated_location = translator.translate(location, src='ko', dest='en').text
            choices.append(translated_location)  # 번역된 지역 이름 추가
            original_choices.append(location)  # 원래 지역 이름 추가

    # 예외처리
    if not choices:
        raise HTTPException(status_code=400, detail="지역을 입력해주세요.")

    # 이미지와 입력한 텍스트를 모델에 전달
    inputs = processor(text=choices, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # 가장 유사도가 큰 값의 인덱스 구하기
    max_index = torch.argmax(logits_per_image).item()

    # 가장 높은 유사도를 가진 지역과 유사도 값
    most_similar_location = original_choices[max_index]  
    similarity = probs[0, max_index].item() * 100  # 유사도 값을 퍼센트로 변환

    if similarity <= 75:
        return {
            "message": (
                f"입력하신 지역들은 이미지와 유사도가 낮아 정확하게 찾아줄 수는 없습니다."
                f"\n그러나 이 이미지는 현재 입력하신 지역 중 {most_similar_location}가 가장 유사도가 높습니다."
                f" 유사도는 {similarity:.2f}%입니다."
            )
        }
    else:
        return {
            "message": f"이 이미지는 {most_similar_location}가 가장 유사도가 높습니다. 유사도는 {similarity:.2f}%입니다."
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
