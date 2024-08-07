from flask import Flask, render_template, request
from fileinput import filename 
from distutils.log import debug 
from classes.upload_error import UploadError

app = Flask('AMZ-Flask')

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/upload', methods = ['POST'])
def upload():
    if request.method == 'POST':
        resumeFile = request.files['resume-file']
        print('resume file is: ');
        print(resumeFile);
        jobFile = request.files['job-posting-file']
        print('job file is: ');
        print(jobFile);

        uploadErrors = []
        match resumeFile:
            case None:
                match jobFile:
                    case None:
                        uploadErrors = [UploadError.JOB_POSTING, UploadError.RESUME]
                    case _:
                        uploadErrors = [UploadError.RESUME]
            case _:
                match resumeFile:
                    case None:
                        uploadErrors = [UploadError.RESUME]
        
        if len(uploadErrors) == 0:
            return render_template('./upload-result.html')
        else: 
            return render_template('./index.html', uploadErrors=uploadErrors)



# @app.route('/start')
# def start():
#     # Code to start using Flask
#     return 'Flask is now in use!'

if __name__ == '__main__':
    app.run(debug=True)
