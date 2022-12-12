from fastapi import FastAPI
from fastapi import UploadFile, File
from prediction import read_imagefile, predict
import uvicorn

app = FastAPI()

@app.post('/api/predict')
async def predict_image(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ('jpg', 'jpeg', 'png')
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(image)
    return prediction

if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='localhost',)