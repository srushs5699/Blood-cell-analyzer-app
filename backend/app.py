from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64
import time
import os
from models.yolo_model import BloodCellYOLO
from utils.image_processor import ImageProcessor
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model
blood_cell_model = BloodCellYOLO()
image_processor = ImageProcessor()

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Blood Cell Analyzer API is running"})

@app.route('/api/analyze', methods=['POST'])
def analyze_blood_cells():
    try:
        start_time = time.time()
        
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Process image
        image = Image.open(file.stream)
        image_array = np.array(image)
        
        # Run YOLO detection
        results = blood_cell_model.detect(image_array)
        
        # Process results
        processed_results = image_processor.process_detection_results(results, image_array)
        
        # Calculate metrics
        cell_counts = {
            'WBC': processed_results.get('wbc_count', 0),
            'RBC': processed_results.get('rbc_count', 0),
            'Platelets': processed_results.get('platelet_count', 0)
        }
        
        total_cells = sum(cell_counts.values())
        percentages = {
            cell_type: (count / total_cells * 100) if total_cells > 0 else 0
            for cell_type, count in cell_counts.items()
        }
        
        processing_time = time.time() - start_time
        
        # Encode annotated image
        annotated_image = processed_results.get('annotated_image')
        if annotated_image is not None:
            _, buffer = cv2.imencode('.jpg', annotated_image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
        else:
            img_base64 = None
        
        response = {
            "success": True,
            "analysis_id": f"analysis_{int(time.time())}",
            "processing_time": round(processing_time, 2),
            "cell_counts": cell_counts,
            "percentages": percentages,
            "total_cells_detected": total_cells,
            "confidence_score": processed_results.get('avg_confidence', 0),
            "annotated_image": img_base64,
            "detected_objects": processed_results.get('detections', [])
        }
        
        logger.info(f"Analysis completed in {processing_time:.2f}s - Detected {total_cells} cells")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    try:
        files = request.files.getlist('images')
        if not files:
            return jsonify({"error": "No images provided"}), 400
        
        results = []
        for i, file in enumerate(files):
            if file.filename != '':
                image = Image.open(file.stream)
                image_array = np.array(image)
                
                detection_results = blood_cell_model.detect(image_array)
                processed = image_processor.process_detection_results(detection_results, image_array)
                
                results.append({
                    "image_index": i,
                    "filename": file.filename,
                    "cell_counts": {
                        'WBC': processed.get('wbc_count', 0),
                        'RBC': processed.get('rbc_count', 0),
                        'Platelets': processed.get('platelet_count', 0)
                    },
                    "confidence": processed.get('avg_confidence', 0)
                })
        
        return jsonify({
            "success": True,
            "batch_results": results,
            "total_images_processed": len(results)
        })
        
    except Exception as e:
        return jsonify({"error": f"Batch analysis failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
