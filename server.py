from flask import Flask
# import bsa
# import fine_tuned_model_final
import refined_code

app = Flask(__name__)

@app.route('/')
def display():
    return "Looks like it works!"
@app.route('/alpha')
def alpha():
    return "This is the alpha version"

@app.route('/beta')
def beta():
    return "This is the beta version"
@app.route('/train')
def train_data():
    # return bsa.init_train()
    # return fine_tuned_model_final.init_train()
    return refined_code.init_train()

if __name__=='__main__':
    app.run(host='0.0.0.0', debug=True, port=3000)