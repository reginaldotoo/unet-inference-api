from fastapi import FastAPI, HTTPException
from UNet_Inference import CLASS_NAMES, get_sites, run_inference

app = FastAPI(title='U-Net Geospatial Inference API')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/sites')
def sites():
    return {'sites': list(get_sites().keys())}


@app.get('/model-info')
def model_info():
    return {
        'model': 'TensorFlow/Keras U-Net',
        'input_tile_size': '128x128',
        'classes': CLASS_NAMES,
    }


@app.post('/predict/{site_name}')
def predict(site_name: str):
    try:
        return run_inference(site_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
