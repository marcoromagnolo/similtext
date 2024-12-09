import werkzeug.exceptions
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import settings
import pidman
import logg
from similarity import Similarity
import mysql.connector
from datetime import datetime, timedelta
import schedule
import time
import threading
import html
import w3lib.html


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

    s = Similarity(logger)
    s.build_vectorizer(data)
    return jsonify(s.get_document_ids()), 200


@app.route('/list', methods=['GET'])
def get_list():
    """
    Get the list id of documents
    """
    s = Similarity(logger)
    return jsonify(s.get_document_ids()), 200


@app.route('/scores<int:id>', methods=['GET'])
def get_scores(id):
    """
    Get TF-IDF scores of documents
    """
    s = Similarity(logger)
    return jsonify(s.get_document_scores(id)), 200


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


def run_init():
    logger.info("Run Init")
    # Calculate the date 30 days ago
    thirty_days_ago = (datetime.now() - timedelta(days=settings.INIT_FROM_DAYS)).strftime('%Y-%m-%d %H:%M:%S')

    # SQL query
    query = """
        SELECT id, post_content
        FROM wp_posts
        WHERE post_type = 'post'
          AND post_status IN ('publish', 'future')
          AND post_date >= %s
    """

    try:
        # Connect to the database
        connection = mysql.connector.connect(**settings.DB_SETTINGS)
        cursor = connection.cursor()

        # Execute the query
        cursor.execute(query, (thirty_days_ago,))
        
        # Fetch the results
        posts = cursor.fetchall()
        data = [(post[0], html.unescape(w3lib.html.remove_tags(post[1]))) for post in posts]
        s = Similarity(logger)
        s.build_vectorizer(data)

    except mysql.connector.Error as err:
        logger.error(f"Error: {err}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
        logger.info("Init finished")


def run_scheduler():
    logger.debug('Scheduler running')

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    logger.info('Application start')

    schedule.every().day.at(f"{settings.INIT_SCHEDULE_AT}").do(run_init)

    # Run Flask in a Thread
    threading.Thread(target=run_scheduler, daemon=True).start()

    app.run(host=settings.WEB_SETTINGS['host'],
            port=settings.WEB_SETTINGS['port'],
            debug=settings.WEB_SETTINGS['debug'],
            use_reloader=settings.WEB_SETTINGS['use_reloader'])
