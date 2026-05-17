from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from inference import classifier, predict_emotion

app = FastAPI(
    title="Emotion Classification API",
    description="API для определения эмоции в русском тексте с помощью BERT",
    version="1.0.0"
)

class TextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        pattern=r"\S",
        json_schema_extra={"example": "Сегодня обычный день, ничего особенного."}
    )

class EmotionResponse(BaseModel):
    emotion: str = Field(..., json_schema_extra={"example": "Нейтрально"})
    confidence: float = Field(..., json_schema_extra={"example": 0.859})

@app.post("/predict", response_model=EmotionResponse)
async def predict(request: TextRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    try:
        emotion, confidence = predict_emotion(request.text)
        return EmotionResponse(emotion=emotion, confidence=round(confidence, 4))
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)