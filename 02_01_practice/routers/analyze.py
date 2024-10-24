from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import HTMLResponse
from service.analyzaion import analyze_image
from service.analyzaion import render_form

router = APIRouter()

# 기본 HTML 페이지 반환
@router.get("/", response_class=HTMLResponse)
async def get_form():
    return render_form()

# 이미지 분석 처리
@router.post("/analyze/")
async def analyze_location(image_url: str = Form(...), locations: str = Form(...)):
    return analyze_image(image_url, locations)
