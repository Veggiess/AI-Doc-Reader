from PIL import Image
from src.inference import DocumentExtractor
import gradio as gr
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize extractor (lazy loading would be better for production)
try:
    extractor = DocumentExtractor()
    logger.info("Document extractor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize document extractor: {e}")
    raise

def extract_info(image: Image.Image) -> Dict[str, Any]:
    """
    Process document image and return extracted information.
    
    Args:
        image: PIL Image containing document
        
    Returns:
        Dictionary with extracted fields or error message
    """
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
            
        logger.info("Processing document image...")
        result = extractor.predict(image)
        logger.info("Successfully extracted document information")
        return result
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return {
            "error": "Failed to process document",
            "details": str(e)
        }

# Configure Gradio interface
interface = gr.Interface(
    fn=extract_info,
    inputs=gr.Image(label="Upload Document", type="pil"),
    outputs=gr.JSON(label="Extracted Information"),
    title="Document Information Extractor",
    description="Upload a document image (ID card, invoice, etc.) to extract structured information",
    examples=[
        ["examples/id_card_sample.jpg"],
        ["examples/invoice_sample.jpg"]
    ],
    allow_flagging="never"
)

# Launch with production-ready settings
if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for temporary public sharing
        debug=False
    )