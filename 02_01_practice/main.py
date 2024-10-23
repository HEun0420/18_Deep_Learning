from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
from googletrans import Translator
from io import BytesIO
from fastapi.responses import HTMLResponse

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


# Root route to serve HTML form
@app.get("/", response_class=HTMLResponse)
async def get_form():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>지역 이미지 분석 프로그램</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>

    <style>
        body {
            background: linear-gradient(135deg, #d5f5e3 0%, #b7e6b1 100%);
        }

        h1 {
            color: #2e7d32;
            font-size: 2.8rem;
            text-shadow: 2px 2px #a8d8a3;
            margin-bottom: 1.5rem;
            text-align: center;
        }

        .container {
            max-width: 700px;
            background-color: #ffffff;
            padding: 40px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
            border-radius: 20px;
            margin-top: 50px;
            border: 3px solid #4caf50;
        }

        .form-group label {
            font-weight: bold;
            color: #2e7d32;
            font-size: 1.2rem;
        }

        .form-control {
            border: 2px solid #c8e6c9;
            border-radius: 10px;
            padding: 10px;
            font-size: 1.1rem;
        }

        .btn-primary {
            background-color: #81c784;
            border: none;
            padding: 12px 25px;
            font-size: 1.2rem;
            border-radius: 30px;
            width: 100%;
            transition: background-color 0.3s ease;
            color: #fff;
        }

        .btn-primary:hover {
            background-color: #66bb6a;
        }

        #result h3, #result h4 {
            color: #388e3c;
        }

        #submittedImage {
            border: 2px solid #c8e6c9;
            border-radius: 15px;
            margin-top: 20px;
        }

        /* Fun border animations */
        .container {
            border-image: linear-gradient(45deg, #4caf50, #81c784) 1;
        }

        /* Extra button styling */
        .btn-primary:focus {
            outline: none;
            box-shadow: 0 0 10px #81c784;
        }

        @media (max-width: 768px) {
            .container {
                padding: 30px;
            }

            h1 {
                font-size: 2.2rem;
            }

            .btn-primary {
                font-size: 1.1rem;
            }
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <h1 class="text-center">지역 이미지 분석 프로그램</h1>
        <form id="analyzeForm" class="mt-4">
            <div class="form-group">
                <label for="image_url">이미지 URL</label>
                <input type="text" class="form-control" id="image_url" name="image_url" placeholder="이미지 URL을 넣어주세요" required>
            </div>
            <div class="form-group">
                <label for="locations">지역 설정 (comma로 구분)</label>
                <input type="text" class="form-control" id="locations" name="locations" placeholder="지역을 적어주세요 (e.g., 서울, 부산, 제주)" required>
            </div>
            <button type="submit" class="btn btn-primary">분석</button>
        </form>

        <div id="result" class="mt-4" style="display:none;">
            <h3>분석 결과:</h3>
            <p id="resultMessage"></p>
            <h4>제공된 이미지:</h4>
            <img id="submittedImage" src="" alt="Submitted Image" class="img-fluid mt-2" style="max-width: 100%; height: auto; display: none;">
        </div>
    </div>

    <script>
        $('#analyzeForm').on('submit', function(event) {
            event.preventDefault();  
            const imageUrl = $('#image_url').val();
            const locations = $('#locations').val();
            $.ajax({
                url: '/analyze/',
                method: 'POST',
                contentType: 'application/x-www-form-urlencoded',
                data: {
                    image_url: imageUrl,
                    locations: locations
                },
                success: function(response) {
                    $('#resultMessage').text(response.message);
                    $('#submittedImage').attr('src', imageUrl).show(); // Display the submitted image
                    $('#result').show();  
                },
                error: function(xhr, status, error) {
                    $('#resultMessage').text('오류가 발생했습니다. 다시 시도해주세요.');
                    $('#submittedImage').hide(); // Hide the image in case of error
                    $('#result').show();
                }
            });
        });
    </script>
</body>
</html>

    """
    return HTMLResponse(content=html_content)



@app.post("/analyze/")
async def analyze_location(image_url: str = Form(...), locations: str = Form(...)):
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail="유효한 이미지 URL을 입력해주세요.")

    # 지역 이름 입력받기
    choices = []
    original_choices = []

    # Split locations by comma and strip spaces
    for location in locations.split(","):
        location = location.strip()
        if location:
            translated_location = translator.translate(
                location, src="ko", dest="en"
            ).text
            choices.append(translated_location)
            original_choices.append(location)

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
