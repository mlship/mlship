from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import json
import asyncio

app = FastAPI()

def setup_ui_routes(app, templates):
    """Set up UI routes."""
    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        """Serve the UI."""
        return templates.TemplateResponse("index.html", {
            "request": request,
            "model_info": json.dumps(get_model_info(app.state.model))
        })

    @app.get("/ui", response_class=HTMLResponse)
    async def ui(request: Request):
        """Serve the UI."""
        return templates.TemplateResponse("index.html", {
            "request": request,
            "model_info": json.dumps(get_model_info(app.state.model))
        })

def setup_api_routes(app, model):
    """Set up API routes."""
    # Store model in app state
    app.state.model = model
    
    @app.get("/api/model/info")
    async def get_model_info_api():
        """Get information about the loaded model."""
        return get_model_info(app.state.model)

    @app.post("/api/model/predict")
    async def predict(request: Request):
        """Make a prediction with the model."""
        try:
            data = await request.json()
            inputs = data.get("inputs")
            if inputs is None:
                raise HTTPException(status_code=400, detail="No inputs provided")
            
            # Convert inputs to list of lists if it's not already
            if not isinstance(inputs[0], list):
                inputs = [inputs]
                
            prediction = app.state.model.predict(inputs)
            return {"prediction": prediction[0] if len(prediction) == 1 else prediction}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.websocket("/ws/metrics")
    async def websocket_metrics(websocket: WebSocket):
        """WebSocket endpoint for real-time metrics."""
        await websocket.accept()
        try:
            while True:
                metrics = {
                    "requests": app.state.model.request_count if hasattr(app.state.model, 'request_count') else 0,
                    "avg_latency": app.state.model.average_latency if hasattr(app.state.model, 'average_latency') else 0
                }
                await websocket.send_json(metrics)
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            pass

def get_model_info(model):
    """Get information about the loaded model."""
    if model is None:
        return {}
    try:
        return model.get_info()
    except Exception as e:
        return {"error": str(e)}