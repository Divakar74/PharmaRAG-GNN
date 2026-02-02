"""
Flask API for DDI Prediction System.
"""
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.ddi_predictor import DDIPredictor

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize predictor
predictor = DDIPredictor()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@app.route('/', methods=['GET'])
def serve_frontend():
    """Serve the frontend HTML."""
    return send_from_directory('frontend', 'index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('frontend/static', filename)


@app.route('/predict_ddi', methods=['POST'])
def predict_ddi():
    """API endpoint for DDI prediction."""
    try:
        data = request.get_json()
        drug_a = data.get('drug_a')
        drug_b = data.get('drug_b')
        user_id = data.get('user_id', 'anonymous')

        logger.info(f"Predict DDI called by user {user_id} with drug_a={drug_a}, drug_b={drug_b}")

        if not drug_a or not drug_b:
            return jsonify({'error': 'Both drug_a and drug_b are required'}), 400

        result = predictor.predict_interaction(drug_a, drug_b, user_id)
        logger.info(f"Prediction result for user {user_id}: {result}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in predict_ddi: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/search_drug', methods=['GET'])
def search_drug():
    """Drug search with autocomplete."""
    try:
        query = request.args.get('q', '')
        limit = int(request.args.get('limit', 10))

        if not query:
            return jsonify({'error': 'Query parameter q is required'}), 400

        results = predictor.search_drug(query, limit)
        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/explain', methods=['POST'])
def explain():
    """Tiered explanation endpoint."""
    try:
        data = request.get_json()
        drug_a = data.get('drug_a')
        drug_b = data.get('drug_b')
        probability = data.get('probability', 0.5)
        use_llm = data.get('use_llm', True)

        if not drug_a or not drug_b:
            return jsonify({'error': 'Both drug_a and drug_b are required'}), 400

        explanation = predictor.generate_tiered_explanation(drug_a, drug_b, probability, use_llm)
        return jsonify(explanation)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'DDI Prediction API'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
