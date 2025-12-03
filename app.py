
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
import base64
import requests
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage
import io
import logging
import os
import sys
from PIL import Image
import traceback
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


log_dir = os.path.join(current_dir, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'app.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Passport Name Extractor API",
    description="API for extracting names from passport images using Ollama (Langchain or direct API).",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],  
    allow_headers=["*"],  
)


OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'qwen2.5vl:7b')

class PassportNameExtractorWebhook:
    def __init__(self, ollama_base_url: str = OLLAMA_BASE_URL, model_name: str = OLLAMA_MODEL):
        """
        Initialize the passport name extractor webhook service

        Args:
            ollama_base_url (str): Ollama server URL
            model_name (str): Ollama model name
        """
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name
        self.chat_model = None 

        
        try:
            self.chat_model = ChatOllama(
                base_url=ollama_base_url,
                model=model_name,
                temperature=0.1
            )
            logger.info(f"Initialized Ollama chat model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama chat model. This might be due to Ollama server not being reachable. Error: {str(e)}")
            

    def validate_base64_image(self, base64_string: str) -> bool:
        """
        Validate that the base64 string is a valid image

        Args:
            base64_string (str): Base64 encoded image

        Returns:
            bool: True if valid image, False otherwise
        """
        try:
            
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]

            
            image_data = base64.b64decode(base64_string)

            
            image = Image.open(io.BytesIO(image_data))
            image.verify()

            return True
        except Exception as e:
            logger.error(f"Invalid image data: {str(e)}")
            return False

    async def extract_name_from_base64(self, base64_image: str) -> str:
        """
        Extract full name from passport image using base64 data

        Args:
            base64_image (str): Base64 encoded passport image

        Returns:
            str: Extracted full name in English
        """
        try:
            if self.chat_model is None:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                    detail="Chat model not initialized. Check Ollama connection and logs.")

            
            if not self.validate_base64_image(base64_image):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                    detail="Invalid image data provided. Image validation failed.")

            
            if base64_image.startswith('data:image'):
                base64_image = base64_image.split(',')[1]

            
            prompt = """
            Please analyze this passport image and extract the full name of the person in English.

            Instructions:
            1. Look for the name field in the passport
            2. Extract the complete full name (first name, middle name if present, last name)
            3. Return only the name in English characters
            4. Format: "FirstName MiddleName LastName" (omit middle name if not present)

            Return only the extracted name, nothing else.
            """

            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            )

            
            response = await self.chat_model.ainvoke([message]) 
            
            
            extracted_name = response.content.strip()

            logger.info(f"Successfully extracted name via Langchain: {extracted_name}")
            return extracted_name

        except HTTPException:
            raise 
        except Exception as e:
            logger.error(f"Error extracting name from passport (Langchain): {str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Error extracting name from passport: {str(e)}")

    async def direct_ollama_extract(self, base64_image: str) -> str:
        """
        Direct API call to Ollama without Langchain wrapper

        Args:
            base64_image (str): Base64 encoded image

        Returns:
            str: Extracted name
        """
        try:
            
            if base64_image.startswith('data:image'):
                base64_image = base64_image.split(',')[1]

            
            payload = {
                "model": self.model_name,
                "prompt": "Extract the full name from this passport image in English. Return only the name.",
                "images": [base64_image],
                "stream": False
            }

            
            response = requests.post(f"{self.ollama_base_url}/api/generate", json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                extracted_name = result.get("response", "").strip()
                logger.info(f"Successfully extracted name via direct API: {extracted_name}")
                return extracted_name
            else:
                logger.error(f"Direct Ollama API request failed: {response.status_code} - {response.text}")
                raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                                    detail=f"Ollama API request failed: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError as ce:
            logger.error(f"Connection error to Ollama: {str(ce)}")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail=f"Could not connect to Ollama server at {self.ollama_base_url}. Please ensure it's running.")
        except requests.exceptions.Timeout:
            logger.error(f"Ollama API request timed out.")
            raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                                detail="Ollama API request timed out.")
        except Exception as e:
            logger.error(f"Direct Ollama API error: {str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Direct Ollama API error: {str(e)}")


try:
    extractor = PassportNameExtractorWebhook()
    logger.info("Service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize service: {str(e)}")
    extractor = None

class ImageExtractRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image string. Can include 'data:image/jpeg;base64,' prefix.")
    method: str = Field("langchain", description="Extraction method: 'langchain' or 'direct'. Defaults to 'langchain'.", pattern="^(langchain|direct)$")

class SingleExtractionResponse(BaseModel):
    success: bool
    extracted_name: Optional[str] = None
    method_used: str
    error: Optional[str] = None

class BatchImage(BaseModel):
    id: str = Field(..., description="Unique identifier for the image in the batch.")
    data: str = Field(..., description="Base64 encoded image string for the batch item.")

class BatchExtractRequest(BaseModel):
    images: List[BatchImage] = Field(..., description="List of image objects to process in batch.")
    method: str = Field("langchain", description="Extraction method: 'langchain' or 'direct'. Defaults to 'langchain'.", pattern="^(langchain|direct)$")

class BatchExtractionResult(BaseModel):
    id: str
    success: bool
    extracted_name: Optional[str] = None
    error: Optional[str] = None

class BatchExtractionResponse(BaseModel):
    success: bool
    method_used: str
    results: List[BatchExtractionResult]
    total_processed: int
    successful_extractions: int
    error: Optional[str] = None

@app.get("/", summary="Root endpoint", response_model=dict)
async def root():
    """
    Root endpoint providing service information.
    """
    return JSONResponse(content={
        "service": "Passport Name Extractor API",
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "extract_single": "/extract-name",
            "extract_batch": "/batch-extract"
        }
    })

@app.get("/health", summary="Health check endpoint", response_model=dict)
async def health_check():
    """
    Performs a health check of the service and its dependencies, specifically the Ollama server.
    """
    ollama_status = "unknown"
    try:
        
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            ollama_status = "connected"
        else:
            ollama_status = "error"
    except requests.exceptions.ConnectionError:
        ollama_status = "disconnected"
    except Exception as e:
        ollama_status = f"error: {str(e)}"
    
    return JSONResponse(content={
        "status": "healthy",
        "service": "passport-name-extractor",
        "ollama_url": OLLAMA_BASE_URL,
        "model": OLLAMA_MODEL,
        "ollama_status": ollama_status,
        "extractor_initialized": extractor is not None
    })

@app.post("/extract-name", summary="Extract name from a single passport image", response_model=SingleExtractionResponse)
async def extract_name_webhook(request_data: ImageExtractRequest):
    """
    Webhook endpoint to extract the full name from a single base64 encoded passport image.

    This endpoint accepts a base64 encoded image and uses either the Langchain integration
    or a direct Ollama API call to extract the name.

    Args:
        request_data (ImageExtractRequest): JSON payload containing 'image' and optional 'method'.

    Returns:
        SingleExtractionResponse: A JSON object with the extraction result.
    """
    if extractor is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Service not properly initialized. Check server logs for details.")

    try:
        base64_image = request_data.image
        method = request_data.method.lower()

        if method == 'langchain':
            extracted_name = await extractor.extract_name_from_base64(base64_image)
        else: 
            extracted_name = await extractor.direct_ollama_extract(base64_image)
        
        return SingleExtractionResponse(success=True, extracted_name=extracted_name, method_used=method)

    except HTTPException as e:
        raise e 
    except ValueError as ve:
        logger.error(f"Validation error in /extract-name: {str(ve)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Validation error: {str(ve)}")
    except Exception as e:
        logger.error(f"Extraction error in /extract-name: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Extraction failed: {str(e)}")

@app.post("/batch-extract", summary="Extract names from multiple passport images", response_model=BatchExtractionResponse)
async def batch_extract_webhook(request_data: BatchExtractRequest):
    """
    Webhook endpoint to extract names from multiple base64 encoded passport images in a batch.

    This endpoint processes a list of images, applying the specified extraction method (Langchain or direct Ollama API)
    to each one.

    Args:
        request_data (BatchExtractRequest): JSON payload containing an 'images' array and optional 'method'.

    Returns:
        BatchExtractionResponse: A JSON object with results for each processed image.
    """
    if extractor is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Service not properly initialized. Check server logs for details.")

    results = []
    total_processed = 0
    successful_extractions = 0
    method = request_data.method.lower()

    if not request_data.images:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="No images provided. Include 'images' array with image objects.")

    for img_obj in request_data.images:
        img_id = img_obj.id
        img_data = img_obj.data
        total_processed += 1

        try:
            if not img_data:
                results.append(BatchExtractionResult(id=img_id, success=False, error="No image data provided"))
                continue

            
            if method == 'langchain':
                extracted_name = await extractor.extract_name_from_base64(img_data)
            else: 
                extracted_name = await extractor.direct_ollama_extract(img_data)
            
            results.append(BatchExtractionResult(id=img_id, success=True, extracted_name=extracted_name))
            successful_extractions += 1

        except HTTPException as e:
            logger.error(f"Error processing image {img_id}: {e.detail}")
            results.append(BatchExtractionResult(id=img_id, success=False, error=e.detail))
        except Exception as e:
            logger.error(f"Error processing image {img_id}: {str(e)}")
            results.append(BatchExtractionResult(id=img_id, success=False, error=f"Processing failed: {str(e)}"))
    
    return BatchExtractionResponse(
        success=True,
        method_used=method,
        results=results,
        total_processed=total_processed,
        successful_extractions=successful_extractions
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handles FastAPI's built-in HTTPExceptions."""
    logger.error(f"HTTP Error: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail},
    )

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handles Pydantic validation errors."""
    logger.error(f"Validation Error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"success": False, "error": "Invalid input data", "details": exc.errors()},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handles all other unhandled exceptions."""
    logger.error(f"Unhandled internal server error: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "error": "An unexpected internal server error occurred."},
    )

from a2wsgi import ASGIMiddleware


wsgi_app = ASGIMiddleware(app)


if __name__ == '__main__':
    import uvicorn
    
    HOST = '127.0.0.1'
    PORT = 5000
    
    logger.info(f"Starting Passport Name Extractor Service (Development Mode)")
    logger.info(f"Ollama URL: {OLLAMA_BASE_URL}")
    logger.info(f"Model: {OLLAMA_MODEL}")
    logger.info(f"Server will run on http://{HOST}:{PORT}")
    
    # Run the FastAPI app using Uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True, log_level="info")