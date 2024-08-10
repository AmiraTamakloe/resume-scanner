from flask import Flask, render_template, request
from fileinput import filename 
from setuptools import distutils
from distutils import debug
from classes.upload_error import UploadError

app = Flask('AMZ-Flask')

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/upload', methods = ['POST'])
def upload():
    if request.method == 'POST':
        resumeFile = request.files['resume-file']
        jobFile = request.files['job-posting-file']

        uploadErrors = []; 

        # FIXME: The match case requires 3.12 python version, dependencies conflict
        # Resolved with a venv
        match resumeFile.filename:
            case '':
                match jobFile.filename:
                    case '':
                        uploadErrors = [UploadError.JOB_POSTING, UploadError.RESUME]
                    case _:
                        uploadErrors = [UploadError.RESUME]
            case _:
                match jobFile.filename:
                    case '':
                        uploadErrors = [UploadError.JOB_POSTING]
                    case _:
                        uploadErrors = []

        if len(uploadErrors) == 0:
            print('no upload errors')
            return render_template('./upload-result.html', resumeFile=resumeFile, jobFile=jobFile)
        else: 
            print(uploadErrors)
            return render_template('./index.html', uploadErrors=uploadErrors)
        
@app.route('/clear-resume', methods=['PUT'])
def clearResume():
    print('clearing')
    if request.method == 'DELETE':
        print('deleting')
        request.form.clear()


if __name__ == '__main__':
    app.run(debug=True)
