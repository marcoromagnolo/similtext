import werkzeug.exceptions
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import settings
import pidman
import logg
from similarity import Similarity


pidman.add_pid_file("similtext.pid")

logger = logg.create_logger('app')

# Set Flask app
app = Flask(__name__)

@app.errorhandler(werkzeug.exceptions.NotFound)
def handle_bad_request(e):
    logger.warning(str(e))
    return 'Not Found', 404


@app.errorhandler(werkzeug.exceptions.BadRequest)
def handle_bad_request(e):
    logger.warning(f"Bad request from ip: {request.remote_addr} {str(e)}")
    return 'Bad Request', 400


@app.errorhandler(Exception)
def handle_generic_error(e):
    logger.error(e, exc_info=True)
    return 'Error', 500


@app.route('/', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return jsonify({"status": "running"}), 200


@app.route('/init', methods=['POST'])
def init():
    """
    Load vectors from json data:
        [
            [1, "This is the first document."],
            [2, "This is the second document."],
            [3, "This is the third document."]
        ]
    """
    # Get JSON payload from the request
    data = request.get_json()
    logger.error(f"Data type: {type(data)}, Data content: {data}")
    if not isinstance(data, list):
        return "Error: Invalid data format. Expected a list of tuples or objects.", 400

    s = Similarity()
    s.build_vectorizer(data)
    return jsonify(s.get_document_ids()), 200


@app.route('/list', methods=['GET'])
def get_list():
    """
    Get the list of documents by id
    """
    s = Similarity()
    return jsonify(s.get_document_ids()), 200


@app.route('/check', methods=['POST'])
def check():
    """
    Check similarity between two articles using cosine similarity on their TF-IDF vectors.
    json data:
        [4, "This is the fourth document."]
    """

    data = request.get_json()
    logger.error(f"Data type: {type(data)}, Data content: {data}")

    if not isinstance(data, list):
        return "Error: Invalid data format. Expected a list of tuples or objects.", 400
    
    try:
        s = Similarity(logger)
        document_id = data[0]
        document_text = data[1]
        similar_document_id = s.get_most_similar_document_id(document_id, document_text)
        logger.info(f"Page {document_id} is most similar to page {similar_document_id}")    
        return jsonify(similar_document_id), 200
    except Exception as e:
        return f"Error: {str(e)}", 500
    

@app.route('/verify', methods=['POST'])
def verify():
    """
    Check if two text speak about the same things.
    json data:
        ["Text number 1.", "Text number 2."]
    """

    data = request.get_json()
    logger.error(f"Data type: {type(data)}, Data content: {data}")

    if not isinstance(data, list):
        return "Error: Invalid data format. Expected a list of tuples or objects.", 400
    
    s = Similarity(logger)
    result = s.compare_texts(data[0], data[1])
    return jsonify(result), 200 


if __name__ == '__main__':
    logger.info('Application start')

    app.run(host=settings.WEB_SETTINGS['host'],
            port=settings.WEB_SETTINGS['port'],
            debug=settings.WEB_SETTINGS['debug'],
            use_reloader=settings.WEB_SETTINGS['use_reloader'])
