from fastapi import APIRouter
from models.translation import TranslationRequest
from services.translation import perform_translation


router = APIRouter()

@router.post("/translate")
async def translate(request: TranslationRequest):
    
    result = perform_translation(request.text, request.lang)
    
    print(result)
    
    return {"result": result}