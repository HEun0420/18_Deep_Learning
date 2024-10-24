from pydantic import BaseModel
from utils.language import Language
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


class TranslationRequest(BaseModel):
    text: str
    lang: Language
    

# 모델
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Load Tokenizer
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# 토크나이저 한국어 설정
tokenizer.src_lang = "ko_KR"

# # 전처리
#     # 한글 토큰화
#     tokenizer.src_lang = "ko_KR"


def perform_translation(text: str, lang: str) -> str:
    
    #토큰화
    encoded_ko = tokenizer(text, return_tensors="pt")
    
    # 추론
    generated_tokens = model.generate(
        **encoded_ko,
        forced_bos_token_id=tokenizer.lang_code_to_id[lang]
    )
    
    # 후처리
    # 번역 결과 decoding
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    return {"translated_text" : translated_text[0]}