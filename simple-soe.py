import os
import sys
import json
from flask import Flask, request, jsonify, render_template, Response, abort
from twilio.rest import Client
import redis
import traceback
import random
import requests
import re
import jwt
import logging
import cv2
import uuid
import numpy as np
import image_utils
import urllib.parse
from datetime import datetime
import time
from threading import Thread, Event
import resource

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


@app.before_request
def do_something_whenever_a_request_comes_in():
    r = request
    add_log_entry('>>>>>>>>>>>>>>>>>>>> %s %s' % (r.method, r.url))
    # headers = r.headers
    # if len(headers) > 0:
    #     add_log_entry("Request headers: \n%s" % headers)
    args = r.args
    if len(args) > 0:
        add_log_entry("Request query parameters: \n%s" % args)
    # values = r.values
    # if len(values) > 0:
    #     add_log_entry("Request values: \n%s" % values)
    data = r.data
    if len(data) > 0:
        add_log_entry("Data payload: \n%s" % data)

    auth = r.headers.get('authorization')
    if auth is not None:
        jwt.decode(auth, JWT_SECRET, algorithms=["HS256"])

@app.after_request
def do_something_after_a_request_finishes(response):
    # add_log_entry('Max memory used: %s' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return response


@app.errorhandler(Exception)
def handle_bad_request(e):
    add_log_entry('>>>>>>>>>>>>>>>>>>>> Error: ' + str(e))
    add_log_entry(traceback.format_exc())
    add_log_entry('Max memory used: %s' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return str(e)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test_image')
def test_image():
    # Image URLs:
    #     image_url = 'https://i.imgur.com/NnnZcYf.png'
    #     image_url = 'https://i.imgur.com/HXzhPo4.jpg'
    args = dict(request.args)
    image_url = args.get('url', None)
    if image_url is None:
        image_url = 'https://i.imgur.com/NnnZcYf.png'
    message2 = {'payload': {'output': {'entities': [], 'generic':[]}, 'context': {'global': {'system': {'user_id': 'Test'}, 'session_id': 'test_session_id'}, 'skills': {'main skill': {'user_defined': {}}}}}}
    process_input_image2(message2, media_url=image_url)

    return 'Test requested.'


@app.route('/test_solver')
def test_solver():
    image_url = 'https://i.imgur.com/wqPEOF9.jpg'
    http_headers = {'Content-Type': 'application/json',
                    'Accept': 'application/json'}
    data = json.dumps({'inputMatrix':
                           [[1, 2, 3, 4, 5, 6, 7, 8, 9],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]
                            ]})
    response = requests.post(SUDOKU_SOLVER_URL + 'getSolution', headers=http_headers,
                             data=data)
    return 'Test completed: %s' % str(response.json())


@app.route('/redis', defaults={'file_path': ''})
@app.route('/redis/<path:file_path>')
def redis_content(file_path):
    if file_path is None or file_path == '':
        matrix_image_names_as_bytes = runtime_cache.keys()
        matrix_image_names = []
        for entry in matrix_image_names_as_bytes:
            matrix_image_names.append(entry.decode('utf-8'))
        return render_template('redis.html', files=matrix_image_names, title='Images')
    else:
        url_file_path = urllib.parse.quote("/%s" % file_path)
        matrix_bytes = runtime_cache.get(url_file_path)
        if matrix_bytes is None:
            return abort(404)
        else:
            filename, file_extension = os.path.splitext(url_file_path)
            if file_extension == '.json':
                return Response(matrix_bytes, mimetype='application/json', status=200)
            else:
                return Response(matrix_bytes, mimetype='image/png', status=200)


@app.route('/clear_redis')
def clear_redis():
    keys = runtime_cache.keys()
    number_of_entries = len(keys)
    for key in keys:
        runtime_cache.delete(key)
    return Response('Removed %s redis entries' % number_of_entries, mimetype='text/text', status=200)


@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon-96x96.png')

@app.route('/build', methods=['GET', 'POST'])
def build():
    try:
        build_file = open('static/build.txt')
        build_stamp = build_file.readlines()[0]
        build_file.close()
    except FileNotFoundError:
        from datetime import date
        build_stamp = generate_build_stamp()
    results = 'Running %s %s.\nBuild %s.\nPython %s.' % (sys.argv[0], app.name, build_stamp, sys.version)
    return results

def generate_build_stamp():
    from datetime import date
    return 'Development build - %s' % date.today().strftime("%m/%d/%y")


@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.data
    msg = json.loads(data)
    event = msg[EVENT][NAME]
    if event == MESSAGE_RECEIVED:
        add_log_entry('>>>>>>>>>>>>>>>>>>>> Assistant pre-webhook invoked')
        add_log_entry('Context before pre-webhook: %s' % get_context_from_redis(msg))
        msg = pre_process(msg)
        add_log_entry('Context after pre-webhook: %s' % get_context_from_redis(msg))
    elif event == MESSAGE_PROCESSED:
        add_log_entry('>>>>>>>>>>>>>>>>>>>> Assistant post-webhook invoked')
        add_log_entry('Context before post-webhook: %s' % get_context_from_redis(msg))
        msg = post_process(msg)
        add_log_entry('Context after post-webhook: %s' % get_context_from_redis(msg))
    add_log_entry('Webhook results:')
    add_log_entry(msg)
    response = jsonify(msg)
    return response


def pre_process(message):
    input = message[PAYLOAD][INPUT]
    if input is not None and input['message_type'] == 'text':
        set_context(message, PUZZLE_INPUT, input['text'])
    else:
        delete_context(message, PUZZLE_INPUT)

    return message


def post_process(message):
    output = message[PAYLOAD][OUTPUT]
    # This next bit is to deal with situation where we get input to initiate a conversation. Need to
    # distinguish this from, e.g. a phone call, which just starts a conversation with no input
    if message[PAYLOAD][CONTEXT][GLOBAL][SYSTEM][TURN_COUNT] == 1:
        if get_context(message, PUZZLE_ACTION) is not None:
            if get_context(message, PUZZLE_ACTION) == CONVERSATION_START:
                delete_context(message, PUZZLE_ACTION)
            else:
                # Going to insert a greeting before any response text.
                output[GENERIC].insert(0, {RESPONSE_TYPE: TEXT, TEXT: 'Well, Hello. '})

    action = None
    context = message[PAYLOAD][CONTEXT][SKILLS][MAIN_SKILL].get(USER_DEFINED)
    if context is not None:
        action = context.get(PUZZLE_ACTION)
        if action is not None and len(action) == 0:
            action = None
    if action == HINT_OR_FIX:
        if get_context(message, PUZZLE_SOLUTION) is not None:
            provide_hint(message)
        else:
            fix_input(message)
    elif action == PROVIDE_SOLUTION:
        provide_solution_matrix(message)
    elif action == PROCESS_INPUT:
        process_input(message)
    elif action == PROCESS_INPUT_IMAGE:
        process_input_image2(message)
    elif action == ECHO_INPUT:
        provide_input_matrix(message)
    elif action == SOLVE_PUZZLE:
        solve_puzzle(message)
    elif action == CHECK_STATUS:
        check_status(message)
    elif action == CHANGE_NAME:
        change_name(message)
    elif action == START_OVER:
        clean_up(message)
    elif action == BYE:
        clean_up(message)
        terminate_session(message)
    if action is not None:
        message[PAYLOAD][CONTEXT][SKILLS][MAIN_SKILL][USER_DEFINED][PUZZLE_ACTION] = ''

    # padding output text sentences so they show up better through Twilio
    for item in output[GENERIC]:
        if item[RESPONSE_TYPE] == TEXT:
            item[TEXT] = item[TEXT] + ' '

    # Now deal with how I address the caller
    call_me = get_context(message, PUZZLE_CALL_ME)
    if call_me is None:
        insulting_names = ['Fartface', 'Dick nose', 'Butthead', 'Dumb ass', 'Bonehead',
                           'Dipshit', 'Shithead']
        choice = int(random.random() * len(insulting_names))
        call_me = insulting_names[choice]
        set_context(message, PUZZLE_CALL_ME, call_me)
    # Now append this call_me value to the end of every response from Assistant
    add_response_text(message, [call_me])
    return message


def provide_hint(message):

    row = None
    column = None
    proposed_answer = None
    index = 0
    entities = message[PAYLOAD][OUTPUT].get(ENTITIES)
    if entities is not None and len(entities) >= 2:
        while index < len(entities):
            entity = entities[index]
            if entity[ENTITY] == 'ordinal':
                last_value_found = ordinal_to_integer(entity)
                if entities[index + 1][ENTITY] == 'row':
                    row = last_value_found
                    index += 2
                elif entities[index + 1][ENTITY] == 'column':
                    column = last_value_found
                    index += 2
                else:
                    index += 1
            elif entity[ENTITY] == 'row':
                if entities[index + 1][ENTITY] == 'sys-number':
                    row = int(entities[index + 1]['value'])
                    index += 2
                else:
                    index += 1
            elif entity[ENTITY] == 'column':
                if entities[index + 1][ENTITY] == 'sys-number':
                    column = int(entities[index + 1]['value'])
                    index += 2
                else:
                    index += 1
            elif entity[ENTITY] == 'sys-number':
                proposed_answer = int(entity['value'])
                index += 1
            else:
                index += 1
            if row is not None and column is not None and proposed_answer is not None:
                break
    solution = get_context(message, PUZZLE_SOLUTION)
    if row is not None and (row < 1 or row > 9) or column is not None and (column < 1 or column > 9):
        hint = 'Row and column need to be values from 1 to 9.'
    elif row is not None and column is not None:
        answer = solution[row - 1][column - 1]
        if proposed_answer is None:
            hint = 'The value of row %s column %s is %s.' % (row, column, answer)
        else:
            if proposed_answer == answer:
                hint = 'The value of row %s column %s is %s.' % (row, column, proposed_answer)
            else:
                hint = 'The value of row %s column %s is not %s.' % (row, column, proposed_answer)
    elif row is not None:
        hint = 'The value of row %s is %s.' % (row, solution[row - 1])
    elif column is not None:
        column_values = []
        for row in solution:
            column_values.append(row[column - 1])
        hint = 'The value of column %s is %s.' % (column, column_values)
    else:
        # Shouldn't get here
        hint = 'I\'m sorry, I don\'t have enough information to give you a hint.'

    set_response_text(message, [hint])
    return message


def fix_input(message):
    input_matrix = get_context(message, PUZZLE_INPUT_MATRIX)
    if input_matrix is None:
        text = 'I don\'t have an input matrix from you yet.'
    else:
        row = None
        column = None
        new_value = None
        index = 0
        entities = message[PAYLOAD][OUTPUT].get(ENTITIES)
        if entities is not None and len(entities) >= 2:
            while index < len(entities):
                entity = entities[index]
                if entity[ENTITY] == 'ordinal':
                    last_value_found = ordinal_to_integer(entity)
                    if entities[index + 1][ENTITY] == 'row':
                        row = last_value_found
                        index += 2
                    elif entities[index + 1][ENTITY] == 'column':
                        column = last_value_found
                        index += 2
                    else:
                        index += 1
                elif entity[ENTITY] == 'row':
                    if entities[index + 1][ENTITY] == 'sys-number':
                        row = int(entities[index + 1]['value'])
                        index += 2
                    else:
                        index += 1
                elif entity[ENTITY] == 'column':
                    if entities[index + 1][ENTITY] == 'sys-number':
                        column = int(entities[index + 1]['value'])
                        index += 2
                    else:
                        index += 1
                elif entity[ENTITY] == 'sys-number':
                    new_value = int(entity['value'])
                    index += 1
                elif entity[ENTITY] == 'empty':
                    new_value = 0
                    index += 1
                else:
                    index += 1
                if row is not None and column is not None and new_value is not None:
                    break
        if row is not None and column is not None and new_value is not None:
            input_matrix[row - 1][column - 1] = new_value
            set_context(message, PUZZLE_INPUT_MATRIX, input_matrix)
            if new_value == 0:
                new_value = 'blank'
            text = 'OK, row %s, column %s is now %s.' % (row, column, new_value)
        else:
            text = 'I\'m sorry, I don\'t understand you. I don\'t have a solution yet to your puzzle. '
            text += 'Ask me to solve your puzzle of tell me to fix an input at a specific row and column.'
    set_response_text(message, [text])
    return message


def provide_solution_matrix(message):
    solution_matrix = get_context(message, PUZZLE_SOLUTION)
    input_matrix = get_context(message, PUZZLE_INPUT_MATRIX)
    if solution_matrix and input_matrix is not None:
        filename = '%s.%s.png' % (get_context(message, INPUT_IMAGE_ID), 'solution')

        input_image_url = get_context(message, PUZZLE_INPUT_IMAGE_URL)
        input_image_coordinates = get_context(message, PUZZLE_INPUT_IMAGE_COORDINATES)
        if input_image_url is None or input_image_coordinates is None:
            # This means input was not by means of an image or url of an image
            solution_image_url = generate_matrix_image(input_matrix, filename, solution_matrix=solution_matrix)
        else:
            # Input came as a texted image or a url to an image
            input_image_bytes = runtime_cache.get(input_image_url)
            image_bytearray = np.asarray(bytearray(input_image_bytes), dtype="uint8")
            input_image = cv2.imdecode(image_bytearray, cv2.IMREAD_COLOR)
            solution_image_url = generate_matrix_image(input_matrix, filename, input_image=input_image,
                                                       input_image_coordinates=input_image_coordinates,
                                                       solution_matrix=solution_matrix)
        image_response = {'response_type': 'image', 'source': solution_image_url}
        message[PAYLOAD][OUTPUT][GENERIC].append(image_response)
        add_response_text(message, ['Ta Da!'])
        # add_response_text(message, [solution_image_url])
        # add_response_text(message, vocalize_matrix(solution_matrix))
    else:
        input_matrix = get_context(message, PUZZLE_INPUT_MATRIX)
        if input_matrix is None:
            text = 'I don\'t yet have a puzzle to solve for you.'
        else:
            text = 'You haven\'t yet asked me to solve your puzzle.'
        set_response_text(message, [text])
    return message


def process_input(message):
    if get_context(message, PUZZLE_INPUT_MATRIX) is not None:
        set_response_text(message, ['I already have a matrix to solve.',
                                    'If you want me to solve another, tell me to start over.'])
    else:
        text_input = ''
        for entry in message[PAYLOAD][OUTPUT][ENTITIES]:
            if entry[ENTITY] == ENTITY_EMPTY:
                text_input += '0'
            elif entry[ENTITY] == ENTITY_SYS_NUMBER:
                text_input += str(entry['value'])
        if len(text_input) > 81:
            text_input = text_input[0:81]
        else:
            if len(text_input) < 81:
                text_input = text_input.ljust(81, '0')
        # At this point we have a valid sequence of 81 digits representing the input matrix. Now create the matrix
        input_string_index = 0
        input_matrix = []
        for row in range(0, 9):
            new_row = []
            for column in range(0, 9):
                new_row.append(int(text_input[input_string_index]))
                input_string_index += 1
            input_matrix.append((new_row))

        set_context(message, PUZZLE_INPUT_MATRIX, input_matrix)
        provide_input_matrix(message)

    return message


def process_input_image(message, media_url=None):
    def random_response():

        responses = [
            'Hold please, I\'ll just be a minute more.',
            'I\'m thinking.',
            'Look, stop fidgiting; I\'ll figure it out.',
            'OK, I didn\'t expect that.',
            'This is harder than I thought.',
            'Oops, um, give me a moment to fix this.',
            'Do you happen to have a calculator, or even an abacus?',
            'Oooh, that cough medicine is making me a bit woozy.',
            'I probably shouldn\'t have had that third margarita last night.',
            'I\'m sorry this is taking so long, but it\'d take you longer.',
            'Frowning at me isn\'t going to make this go faster.',
            'Um, do you have an  eraser?',
            'I don\'t think my prescription is working.'
        ]

        random_index = int(random.random() * len(responses))
        return responses[random_index]

    def spew_messages(done_event):
        send_sms(message, "OK, this\'ll just be a minute.")
        time.sleep(10)
        while not done_event.is_set():
            send_sms(message, random_response())
            time.sleep(10)
        return

    if media_url is None:
        for entity in message[PAYLOAD][OUTPUT][ENTITIES]:
            if entity[ENTITY] == 'url':
                media_url = get_context(message, PUZZLE_INPUT)[entity['location'][0]:entity['location'][1]]
                break
    if media_url is not None:
        response = requests.get(media_url)
        if response.status_code == 200:
            results = response.content
            done = Event()
            # th = Thread(target=spew_messages, args=(done,))
            # th.start()
            image_bytearray = np.asarray(bytearray(results), dtype="uint8")
            bw_input_puzzle_image = cv2.imdecode(image_bytearray, cv2.IMREAD_GRAYSCALE)
            if bw_input_puzzle_image is None:
                # This means the file or page was not a valid image
                add_response_text(message,
                                  ['I\'m sorry, I don\'t recognize your input as an image.',
                                   'If it\'s a GIF, we don\'t do GIFs, they\'re for losers.'])
                done.set()
            else:
                input_puzzle_image = cv2.imdecode(image_bytearray, cv2.IMREAD_COLOR)
                input_matrix, image_with_ocr, image_with_lines, coordinates = \
                    image_utils.extract_matrix_from_image(bw_input_puzzle_image)
                done.set()

                if input_matrix is None or len(input_matrix) != 9:
                    add_response_text(message,
                                      ['I\'m sorry. I had a problem understanding the matrix_image you sent.'])
                else:
                    set_context(message, INPUT_IMAGE_ID,
                                '%s.%s' % (message[PAYLOAD][CONTEXT][GLOBAL][SESSION_ID], str(uuid.uuid1())[0:8]))
                    now = datetime.now()
                    ocr_image_filename = urllib.parse.quote('/ocr-input/%s.%s.png' % (get_context(message, INPUT_IMAGE_ID), now.strftime('%H-%M-%S')))
                    ocr_image_bytes = cv2.imencode('.png', image_with_ocr)
                    runtime_cache.setex(ocr_image_filename, REDIS_TTL, ocr_image_bytes[1].tobytes())

                    lines_image_filename = urllib.parse.quote(
                        '/ocr-lines/%s.%s.png' % (get_context(message, INPUT_IMAGE_ID),
                                                  now.strftime('%H-%M-%S')))
                    lines_image_bytes = cv2.imencode('.png', image_with_lines)
                    runtime_cache.setex(lines_image_filename, REDIS_TTL, lines_image_bytes[1].tobytes())

                    input_image_filename = urllib.parse.quote(
                        '/input-image/%s.%s.png' % (get_context(message, INPUT_IMAGE_ID),
                                                    now.strftime('%H-%M-%S')))
                    input_image_bytes = cv2.imencode('.png', input_puzzle_image)
                    runtime_cache.setex(input_image_filename, REDIS_TTL, input_image_bytes[1].tobytes())
                    set_context(message, PUZZLE_INPUT_IMAGE_URL, input_image_filename)
                    # Need this step to convert coordinates from int64 to int so they can me converted to string
                    int_coordinates = []
                    for i in range(len(coordinates)):
                        new_row = []
                        for j in range(len(coordinates[i])):
                            new_pair = []
                            for k in range(len(coordinates[i][j])):
                                new_pair.append(int(coordinates[i][j][k]))
                            new_row.append(new_pair)
                        int_coordinates.append(new_row)

                    set_context(message, PUZZLE_INPUT_IMAGE_COORDINATES, int_coordinates)

                    # This next step is to convert from numpy array (which can't be automatically serialized)
                    # to a list, which can
                    matrix_as_list = []
                    for row in input_matrix:
                        new_row = []
                        for number in row:
                            new_row.append(int(number))
                        matrix_as_list.append(new_row)
                    set_context(message, PUZZLE_INPUT_MATRIX, matrix_as_list)
                    provide_input_matrix(message)
                    add_response_text(message, ['Is this the matrix you want me to solve?'])
        else:
            add_response_text(message,
                              ['I\'m sorry. I had a problem understanding the matrix_image you sent.'])
    else:
        add_response_text(message,
                          ['I\'m sorry. I seem to have misplaced your matrix to solve.'])
    return message

def process_input_image2(message, media_url=None):

    def async_image_processing(results, job_id):
        status = {"status": "processing"}
        runtime_cache.setex(job_id, REDIS_TTL, json.dumps(status))

        image_bytearray = np.asarray(bytearray(results), dtype="uint8")
        bw_input_puzzle_image = cv2.imdecode(image_bytearray, cv2.IMREAD_GRAYSCALE)
        if bw_input_puzzle_image is None:
            # This means the file or page was not a valid image
            error_status = {
                'status': 'error',
                'message': 'I\'m sorry, I don\'t recognize your input as an image. If it\'s a GIF, we don\'t do GIFs, they\'re for losers.'
            }
            runtime_cache.setex(job_id, REDIS_TTL, json.dumps(error_status))
        else:
            input_puzzle_image = cv2.imdecode(image_bytearray, cv2.IMREAD_COLOR)
            height, width = bw_input_puzzle_image.shape
            add_log_entry('!!!!!!!!!!!!! Input image dimensions: %s x %s.' % (height, width))
            if height > MAX_IMAGE_HEIGHT:
                reduction = int(np.log2(height / MAX_IMAGE_HEIGHT)) + 1
                new_dim = (int(width / 2 ** reduction), int(height / 2 ** reduction))
                resized_image = cv2.resize(bw_input_puzzle_image, new_dim)
                height, width = resized_image.shape
                add_log_entry("!!!!!!!!!!!!! Reduced input image by factor of %s. New dimensions: %s x %s." % (2**reduction, height, width))
                bw_input_puzzle_image = resized_image

            input_matrix, image_with_ocr, image_with_lines, coordinates = \
                image_utils.extract_matrix_from_image(bw_input_puzzle_image)

            if input_matrix is None or len(input_matrix) != 9:
                error_status = {
                    'status': 'error',
                    'message': 'I\'m sorry. I had a problem understanding the matrix_image you sent.'
                }
                runtime_cache.setex(job_id, REDIS_TTL, json.dumps(error_status))
            else:
                add_log_entry('!!!!!!!!!!!!! finished image processing')
                now = datetime.now()
                ocr_image_filename = urllib.parse.quote('/ocr-input/%s.%s.png' % (get_context(message, INPUT_IMAGE_ID),
                                                                                  now.strftime('%H-%M-%S')))
                ocr_image_bytes = cv2.imencode('.png', image_with_ocr)
                runtime_cache.setex(ocr_image_filename, REDIS_TTL, ocr_image_bytes[1].tobytes())

                lines_image_filename = urllib.parse.quote(
                    '/ocr-lines/%s.%s.png' % (get_context(message, INPUT_IMAGE_ID),
                                              now.strftime('%H-%M-%S')))
                lines_image_bytes = cv2.imencode('.png', image_with_lines)
                runtime_cache.setex(lines_image_filename, REDIS_TTL, lines_image_bytes[1].tobytes())

                input_image_filename = urllib.parse.quote(
                    '/input-image/%s.%s.png' % (get_context(message, INPUT_IMAGE_ID),
                                                now.strftime('%H-%M-%S')))
                input_image_bytes = cv2.imencode('.png', bw_input_puzzle_image)
                runtime_cache.setex(input_image_filename, REDIS_TTL, input_image_bytes[1].tobytes())
                set_context(message, PUZZLE_INPUT_IMAGE_URL, input_image_filename)
                # Need this step to convert coordinates from int64 to int so they can me converted to string
                int_coordinates = []
                for i in range(len(coordinates)):
                    new_row = []
                    for j in range(len(coordinates[i])):
                        new_pair = []
                        for k in range(len(coordinates[i][j])):
                            new_pair.append(int(coordinates[i][j][k]))
                        new_row.append(new_pair)
                    int_coordinates.append(new_row)

                set_context(message, PUZZLE_INPUT_IMAGE_COORDINATES, int_coordinates)

                # This next step is to convert from numpy array (which can't be automatically serialized)
                # to a list, which can
                matrix_as_list = []
                for row in input_matrix:
                    new_row = []
                    for number in row:
                        new_row.append(int(number))
                    matrix_as_list.append(new_row)
                set_context(message, PUZZLE_INPUT_MATRIX, matrix_as_list)

                completion_status = {
                    'status': 'finished',
                    PUZZLE_INPUT_MATRIX: matrix_as_list,
                    PUZZLE_INPUT_IMAGE_COORDINATES: int_coordinates,
                    PUZZLE_INPUT_IMAGE_URL: input_image_filename
                }
                runtime_cache.setex(job_id, REDIS_TTL, json.dumps(completion_status))
    if media_url is None:
        for entity in message[PAYLOAD][OUTPUT][ENTITIES]:
            if entity[ENTITY] == 'url':
                media_url = get_context(message, PUZZLE_INPUT)[entity['location'][0]:entity['location'][1]]
                break
    add_log_entry('Retreiving input image \'%s\'.' % media_url)
    if media_url is not None:
        response = requests.get(media_url)
        if response.status_code == 200:
            add_log_entry('Successfully retrieved input image \'%s\'.' % media_url)
            results = response.content
            set_context(message, INPUT_IMAGE_ID,
                        '%s.%s' % (message[PAYLOAD][CONTEXT][GLOBAL][SESSION_ID], str(uuid.uuid1())[0:8]))
            set_context(message, PUZZLE_ASYNC_JOB_ID,
                        '/job/%s.json' % get_context(message, INPUT_IMAGE_ID))
            add_log_entry('!!!!!!!!!!!!! starting image processing')
            th = Thread(target=async_image_processing, args=(results, get_context(message, PUZZLE_ASYNC_JOB_ID)))
            th.start()
            add_response_text(message,
                              ['This\'ll just take a moment. Ask me in a bit.'])
        else:
            add_log_entry('%s error retreiving input image \'%s\'.' % (response.status_code, media_url))
            add_response_text(message,
                              ['I\'m sorry. I had a problem understanding the matrix_image you sent.'])
    else:
        add_response_text(message,
                          ['I\'m sorry. I seem to have misplaced your matrix to solve.'])
    return message

def provide_input_matrix(message):
    input_matrix = get_context(message, PUZZLE_INPUT_MATRIX)
    if input_matrix is not None:
        input_image_id = get_context(message, INPUT_IMAGE_ID)
        if input_image_id is None:
            set_context(message, INPUT_IMAGE_ID, message[PAYLOAD][CONTEXT][GLOBAL][SESSION_ID])
        filename = '%s.%s.%s.png' % (get_context(message, INPUT_IMAGE_ID), str(uuid.uuid1())[0:8], 'input')

        input_image_url = get_context(message, PUZZLE_INPUT_IMAGE_URL)
        input_image_coordinates = get_context(message, PUZZLE_INPUT_IMAGE_COORDINATES)
        if input_image_url is None or input_image_coordinates is None:
            # This means input was not by means of an image or url of an image
            output_image_url = generate_matrix_image(input_matrix, filename)
        else:
            # Input came as a texted image or a url to an image
            input_image_bytes = runtime_cache.get(input_image_url)
            image_bytearray = np.asarray(bytearray(input_image_bytes), dtype="uint8")
            input_image = cv2.imdecode(image_bytearray, cv2.IMREAD_COLOR)

            output_image_url = generate_matrix_image(input_matrix, filename, input_image=input_image,
                                                     input_image_coordinates=input_image_coordinates)
        image_response = {'response_type': 'image', 'source': output_image_url}
        message[PAYLOAD][OUTPUT][GENERIC].append(image_response)
        add_response_text(message, ['Here\'s the image I have from you to solve.',
                           'If it\'s not right, you can tell me to start over or tell me to correct specific cells.'])
        # add_response_text(message, [output_image_url])
        # add_response_text(message, vocalize_matrix(input_matrix))
    else:
        text = 'I don\'t yet have a puzzle to solve for you.'
        add_response_text(message, [text])
    return message


def generate_matrix_image(input_matrix, filename, input_image=None, input_image_coordinates=None, solution_matrix=None):
    if solution_matrix is None:
        if input_image is None:
            image = image_utils.generate_matrix_image(input_matrix)
        else:
            image = image_utils.apply_matrix_to_image(input_matrix, input_image, input_image_coordinates)
    else:
        if input_image is None:
            image = image_utils.generate_matrix_image(input_matrix, solution_matrix=solution_matrix)
        else:
            partial_solution_matrix = np.zeros((9, 9), dtype="uint8")
            for row in range(len(solution_matrix)):
                for col in range(len(solution_matrix[row])):
                    if input_matrix[row][col] == 0:
                        partial_solution_matrix[row][col] = solution_matrix[row][col]
            image = image_utils.apply_matrix_to_image(partial_solution_matrix,
                                                      input_image,
                                                      input_image_coordinates,
                                                      show_coordinates=False)
    matrix_filename = urllib.parse.quote("/sudoku/%s" % filename)
    runtime_cache.setex(matrix_filename, REDIS_TTL, image)
    print("Saving matrix_image %s. length: %s: %s" %
          (matrix_filename, len(image), str(image[0:10])))
    matrix_image_url = '%sredis%s' % (request.host_url, matrix_filename)
    return matrix_image_url


def vocalize_matrix(matrix):
    def handle_empty_rows(successive_empty_rows):
        if len(successive_empty_rows) == 1:
            text.append('Row %s. All empty.' % successive_empty_rows[0])
        elif len(successive_empty_rows) == 2:
            text.append('Rows %s and %s. All empty.' % (successive_empty_rows[0], successive_empty_rows[1]))
        elif len(successive_empty_rows) > 2:
            text.append('Rows %s through %s. All empty.' % (successive_empty_rows[0],
                                                            successive_empty_rows[len(successive_empty_rows) - 1]))
        return

    text = []
    successive_empty_rows = []
    row_index = 1
    for row in matrix:
        if sum(row) == 0:
            successive_empty_rows.append(row_index)
        else:
            handle_empty_rows(successive_empty_rows)
            successive_empty_rows = []
            row_string = 'Row %s. ' % row_index
            for column_index in range(0, len(row)):
                if column_index <= len(row) - 2:
                    if sum(row[column_index:len(row)]) == 0:
                        row_string = row_string + 'The rest of the row is empty  '
                        break
                if row[column_index] == 0:
                    row_string = row_string + 'empty, '
                else:
                    row_string = row_string + str(row[column_index]) + ', '
            row_string = row_string[0:len(row_string) - 2] + '.'
            text.append(row_string)
        row_index += 1
    handle_empty_rows(successive_empty_rows)
    return text


def solve_puzzle(message):
    if get_context(message, PUZZLE_INPUT_MATRIX) is None:
        solution_processing_message = 'I don\'t have a matrix to solve.'
    else:
        http_headers = {'Content-Type': 'application/json',
                        'Accept': 'application/json'}
        data = json.dumps({'inputMatrix': get_context(message, PUZZLE_INPUT_MATRIX)})
        response = requests.post(SUDOKU_SOLVER_URL + 'getSolution', headers=http_headers,
                                 data=data)
        solution_processing_message = None
        if response.status_code == 200:
            results = response.json()
            set_context(message, PUZZLE_SOLUTION, results)
            solution_processing_message = 'I have a solution to your puzzle.'

        else:
            solution_processing_message = 'I can\'t solve your puzzle. Make sure it\'s a valid puzzle. And this has nothing to do with the number of margaritas I had last night.'
            if get_context(message, PUZZLE_SOLUTION) is not None:
                delete_context(message, PUZZLE_SOLUTION)

    if solution_processing_message is not None:
        add_response_text(message, [solution_processing_message])

    return message


def send_sms(message, msg):
    #:todo This is going to fail and needs to be updated. First to save the users phone number and then to only text
    # if it is defined
    message = twilio_client.messages.create(
        body=msg,
        from_='+19842144312',
        to=message[CONTEXT][SMS_USER_PHONE_NUMBER]
    )
    return message.sid


def ordinal_to_integer(entity):
    int_value = None
    if entity[ENTITY] == 'ordinal':
        value = entity['value']
        if value == 'first':
            int_value = 1
        elif value == 'second':
            int_value = 2
        elif value == 'third':
            int_value = 3
        elif value == 'fourth':
            int_value = 4
        elif value == 'fifth':
            int_value = 5
        elif value == 'sixth':
            int_value = 6
        elif value == 'seventh':
            int_value = 7
        elif value == 'eighth':
            int_value = 8
        elif value == 'ninth':
            int_value = 9
        elif value == 'last':
            int_value = 9
        elif value == 'penultimate':
            int_value = 8
    return int_value


def check_status(message):
    job_id = get_context(message, PUZZLE_ASYNC_JOB_ID)
    if job_id is None:
        set_response_text(message, ['Sorry, I don\'t have anything for you now. I\'m slacking.'])
    else:
        status_object = json.loads(runtime_cache.get(job_id))
        if status_object['status'] == 'processing':
            set_response_text(message, ['Patience young padawan, I\'m still working.'])
        elif status_object['status'] == 'error':
            set_response_text(message, [status_object['message']])
        elif status_object['status'] == 'finished':
            set_context(message, PUZZLE_INPUT_MATRIX, status_object[PUZZLE_INPUT_MATRIX])
            set_context(message, PUZZLE_INPUT_IMAGE_COORDINATES, status_object[PUZZLE_INPUT_IMAGE_COORDINATES])
            set_context(message, PUZZLE_INPUT_IMAGE_URL, status_object[PUZZLE_INPUT_IMAGE_URL])
            provide_input_matrix(message)
            set_response_text(message, ['Yes, I\'ve finished and I have a matrix.'])
        else:
            set_response_text(message, ['I\'m really confused.'])


def change_name(message):
    new_call_me = None
    for entity in message[PAYLOAD][OUTPUT][ENTITIES]:
        if entity[ENTITY] == ENTITY_SYS_PERSON:
            new_call_me = entity['value']
            break
    if new_call_me is not None:
        set_context(message, PUZZLE_CALL_ME, new_call_me)
        set_response_text(message, ['Fine. I will now call you \'%s\'.' % new_call_me])
    else:
        input_text = get_context(message, PUZZLE_INPUT)
        if input_text is not None and len(input_text) > 0:
            set_response_text(message, ['Fine. I will now call you \'%s\'.' % input_text])
            set_context(message, PUZZLE_CALL_ME, input_text)
        else:
            set_response_text(message, ['Huh, what did you say?'])
    return message


def clean_up(message):
    delete_context(message, PUZZLE_SOLUTION)
    delete_context(message, PUZZLE_INPUT)
    delete_context(message, PUZZLE_INPUT_MATRIX)
    delete_context(message, PUZZLE_INPUT_IMAGE_URL)
    delete_context(message, PUZZLE_INPUT_IMAGE_COORDINATES)
    delete_context(message, INPUT_IMAGE_ID)
    return message


def terminate_session(message):
    # doing nothing currently
    return message


def add_response_text(message, list_of_texts):
    if message[PAYLOAD][OUTPUT].get(GENERIC) is None:
        message[PAYLOAD][OUTPUT][GENERIC] = []
    for text in list_of_texts:
        message[PAYLOAD][OUTPUT][GENERIC].append({'response_type': 'text', 'text': text})
    return message


def set_response_text(message, list_of_texts):
    message[PAYLOAD][OUTPUT][GENERIC] = []
    for text in list_of_texts:
        message[PAYLOAD][OUTPUT][GENERIC].append({'response_type': 'text', 'text': text})
    return message


def add_log_entry(comment, message=None):
    if message is None:
        app.logger.info(comment)
    else:
        log_message = json.dumps(message)
        app.logger.info(comment + " " + log_message)


def get_redis_context_key(message):
    return "/context/%s.json" % message[PAYLOAD][CONTEXT][GLOBAL][SYSTEM][USER_ID]

def get_context_from_redis(message):
    redis_context_key = get_redis_context_key(message)
    context_string = runtime_cache.get(redis_context_key)
    if context_string is None:
        context = {}
    else:
        context = json.loads(context_string)
    return context

def put_context_to_redis(message, context):
    redis_context_key = get_redis_context_key(message)
    runtime_cache.set(redis_context_key, json.dumps(context))
    return context


"""
The get and delete context functions implement a hack that works around a quirk in assistant context
handling between webhooks and dialog turns. Removing a context variable, by removing the key, doesn't always
remove it. So, instead we'll use an empty string to indicate a variable has been deleted. When we see an
empty string on a get, return None. 
"""


def get_context(message, key):
    context = get_context_from_redis(message)
    return context.get(key)


def set_context(message, key, value):
    context = get_context_from_redis(message)
    context[key] = value
    put_context_to_redis(message, context)


def delete_context(message, key):
    context = get_context_from_redis(message)
    value = context.get(key)
    if value is not None:
        context.pop(key)
    put_context_to_redis(message, context)


CONTEXT = 'context'
OUTPUT = 'output'
INPUT = 'input'
TEXT = 'text'
GENERIC = 'generic'
GLOBAL = 'global'
SYSTEM = 'system'
TURN_COUNT = 'turn_count'
SESSION_ID = 'session_id'
USER_ID = 'user_id'
SKILLS = 'skills'
MAIN_SKILL = 'main skill'
USER_DEFINED = 'user_defined'
ENTITY = 'entity'
ENTITIES = 'entities'
ORIGINAL_TEXT = 'original_text'
SMS_USER_PHONE_NUMBER = 'smsUserPhoneNumber'
SMS_RESPONSE_TIMEOUT = 'smsResponseTimeout'
SMS_SESSION_ID = 'smsSessionID'
INPUT_IMAGE_ID = 'input_image_id'
METADATA = 'metadata'
SMSMEDIA = 'smsMedia'
SMSACTION = 'smsAction'
PAYLOAD = 'payload'
NAME = 'name'
EVENT = 'event'
MESSAGE_RECEIVED = 'message_received'
MESSAGE_PROCESSED = 'message_processed'
RESPONSE_TYPE = 'response_type'

PUZZLE_SOLUTION = 'puzzle_solution'
PUZZLE_INPUT = 'puzzle_input'
PUZZLE_INPUT_IMAGE_URL = 'puzzle_input_image_url'
PUZZLE_INPUT_IMAGE_COORDINATES = 'puzzle_input_image_coordinates'
PUZZLE_ACTION = 'puzzle_action'
PUZZLE_INPUT_MATRIX = 'puzzle_input_matrix'
PUZZLE_ASYNC_JOB_ID = 'puzzle_async_job_id'
PRE_WEBHOOK_INPUT = '/sudoku/pre-webhook-input.json'
PRE_WEBHOOK_OUTPUT = '/sudoku/pre-webhook-output.json'
POST_WEBHOOK_INPUT = '/sudoku/post-webhook-input.json'
POST_WEBHOOK_OUTPUT = '/sudoku/post-webhook-output.json'

ECHO_INPUT = 'echo_input'
CHECK_STATUS = 'check_status'
HINT_OR_FIX = 'hint_or_fix'
CONVERSATION_START = 'conversation_start'
START_OVER = 'start_over'
PROVIDE_SOLUTION = 'provide_solution'
CHANGE_NAME = 'change_name'
PUZZLE_CALL_ME = 'puzzle_call_me'
PROCESS_INPUT = 'process_input'
PROCESS_INPUT_IMAGE = 'process_input_image'
SOLVE_PUZZLE = 'solve_puzzle'
CONFIRM_INPUT_IMAGE = 'confirm_input_image'
BYE = 'bye'
ENTITY_SYS_NUMBER = 'sys-number'
ENTITY_SYS_PERSON = 'sys-person'
ENTITY_EMPTY = 'empty'

PORT = os.getenv('PORT', '5020')
SUDOKU_SOLVER_URL = os.environ['SUDOKU_SOLVER_URL']
if SUDOKU_SOLVER_URL[len(SUDOKU_SOLVER_URL) - 1] != '/':
    SUDOKU_SOLVER_URL += '/'
REDIS_HOST = os.environ['REDIS_HOST']
REDIS_PW = os.environ['REDIS_PW']
REDIS_PORT = os.environ['REDIS_PORT']
JWT_SECRET = os.environ['JWT_SECRET']
# Your Account Sid and Auth Token from twilio.com/user/account
account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
REDIS_TTL = 3600
MAX_IMAGE_HEIGHT = 1280

twilio_client = Client(account_sid, auth_token)

runtime_cache = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PW)
# Testing Redis connection.
runtime_cache.set('/test/test', 'test_value')
test_value = runtime_cache.delete('/test/test')

print('Starting %s %s' % (sys.argv[0], app.name))
print('Python: ' + sys.version)
try:
    build_file = open('static/build.txt')
    build_stamp = build_file.readlines()[0]
    build_file.close()
except FileNotFoundError:
    from datetime import date
    build_stamp = generate_build_stamp()
print('Running build: %s' % build_stamp)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
