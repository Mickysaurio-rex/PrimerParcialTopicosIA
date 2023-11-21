import io
import cv2
import csv
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    status,
    Depends,
)
from fastapi.responses import Response, FileResponse
import numpy as np
from PIL import Image, UnidentifiedImageError
import mediapipe as mp
from predictor import ObjectDetector

PERSON_COLOR = (70,223,49)
BOOK_COLOR = (20,37,241)
BED_COLOR = (9,234,200)
COUCH_COLOR = (252, 233, 1)
CHAIR_COLOR = (166, 29, 245)
CAR_COLOR = (246, 18, 187)



app = FastAPI(title="Deteccion y clasificacion de Objetos")

object_predictor = ObjectDetector()
list_predict = []
def get_object_detector():
    return object_predictor

def predict_uploadfile(predictor, file):
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, 
            detail="Not an image"
        )
    img_obj = Image.open(img_stream)
    img_array = np.array(img_obj)
    return predictor.predict_image(img_array), img_array

@app.get("/status")
def root():
    return {"message": "Bienvenido al detector de Objetos", 
            "creador": "Miguel Molina Flores", 
            "load": "Creando y cargando el modelo",
            "status": "OK"}


@app.post("/predecir_y_anotar_objetos", responses={
    200: {"content": {"image/jpeg": {}}}
    })
def detect_objects(
    file: UploadFile = File(...), 
    predictor: ObjectDetector = Depends(get_object_detector)
) -> Response:
    results, img = predict_uploadfile(predictor, file)
    color_adapt = (0,0,0)
    val_dis = 80
    object_count = {}
    for result in results:
        bbox = result['bbox']
        name = result['name']
        score = result['score']
        date = result['date']

        dict ={
            "nombre": str(name[0]),
            "probabilidad": str(score[0]),
            "punto_x": str(bbox[0]),
            "punto_y": str(bbox[1]),
            "alto": str(bbox[2]),
            "ancho": str(bbox[3]),
            "fecha": str(date[0]) 
        }
        list_predict.append(dict)

        if name[0] not in object_count:
            object_count[name[0]] =  1
        else : 
            object_count[name[0]] = object_count[name[0]] + 1
        
        if name[0] == "book":
            color_adapt = BOOK_COLOR
        elif name[0] == "bed":
            color_adapt = BED_COLOR
        elif name[0] == "person":
            color_adapt = PERSON_COLOR
        elif name[0] == "chair":
            color_adapt = CHAIR_COLOR
        elif name[0] == "car":
            color_adapt = CAR_COLOR
        elif name[0] == "couch":
            color_adapt = COUCH_COLOR
        else:
            color_adapt = (0,0,0)


        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                            color_adapt, 2)
        cv2.putText(
            img, 
            name[0], 
            (bbox[0], bbox[1] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color_adapt,
            2,
        )
        
        

    for key, val in object_count.items():
        cv2.putText(
            img, 
            f"{key}: {val}", 
            (20, val_dis), 
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,0),
            2,
        )
        val_dis = val_dis + 45

    
    img_pil = Image.fromarray(img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    
    return Response(content=image_stream.read(), media_type="image/jpeg")


@app.get("/reporte csv", response_class=Response, responses={200: {"content": {"text/csv": {}}}})
def generate_reports(predictor: ObjectDetector = Depends(get_object_detector)):
    if not list_predict:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No se pudo generar el reporte, primero cargue una imagen"
        )

    csv_file_path = "reporteObjectDetection.csv"

    with open(csv_file_path, mode="w", newline="") as csv_file:
        fieldnames = list_predict[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(list_predict)

    return FileResponse(csv_file_path, filename="reporteObjectDetection.csv", media_type="text/csv")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", reload=True)
