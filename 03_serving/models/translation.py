from pydantic import BaseModel
from utils.language import Language


class TranslationRequest(BaseModel):
    text: str
    lang: Language
    

