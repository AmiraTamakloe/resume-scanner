from flask import Flask, render_template

app = Flask('AMZ-Flask')

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/start')
def start():
    # Code to start using Flask
    return 'Flask is now in use!'

if __name__ == '__main__':
    app.run(debug=True)
