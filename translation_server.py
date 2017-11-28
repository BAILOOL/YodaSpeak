from __future__ import print_function
from flask import Flask, request, redirect, url_for, jsonify, send_from_directory, render_template
from yoda import YodaSpeak

# global variables
app = Flask(__name__, static_url_path='')
app._static_folder = "templates/"
global translator
port_number = 5010
host_number = '0.0.0.0'

# routes
@app.route('/', methods=['GET'])
def index():
    return render_template('server.html')

@app.route('/api/upload_all', methods=['POST'])
def upload_all():
    #pdb.set_trace()
    content = request.form['input']
    response = ""
    if content == "":
        return jsonify({'error': 'No input was uploaded'})
    else:
        response = translator.decode(content)
    	
    print("Input: {}".format(content))
    print("Output: {}".format(response))
    return jsonify({'output': response})

if __name__ == '__main__':
    translator = YodaSpeak()
    app.run(host=host_number, port=port_number, debug=False)
