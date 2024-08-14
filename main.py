from flask import Flask, render_template, request
from fileinput import filename 
from setuptools import distutils
from distutils import debug
from classes.upload_error import UploadError
from model.model_training import predict_category

app = Flask('AMZ-Flask')

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/resume-upload', methods = ['GET'])
def naviguate_resume_upload():
    print('GET')
    return render_template('./resume-upload.html', uploadErrors=[])

@app.route('/resume-posting-upload', methods = ['GET'])
def naviguate_resume_posting():
    print('GET')
    return render_template('./resume-posting-upload.html', uploadErrors=[])


@app.route('/resume-upload', methods = ['POST'])
def upload_resume():
    if request.method == 'POST':
        resumeFile = request.files['resume-file']

        uploadErrors = []; 
        if(resumeFile.filename == ''):
            uploadErrors = [UploadError.RESUME]

        if len(uploadErrors) == 0:
            print('no upload errors')
            category = predict_category(resumeFile)
            return render_template('./category-result.html', category=category)
        else:
            print(uploadErrors)
            return render_template('./resume-upload.html', uploadErrors=uploadErrors)
        

@app.route('/resume-posting-upload', methods = ['POST'])
def upload_resume_posting():
    if request.method == 'POST':
        resumeFile = request.files['resume-file']
        jobFile = request.files['job-posting-file']

        uploadErrors = []; 

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
            category = predict_category(resumeFile)
            return render_template('./category-result.html', category=category, jobFile=jobFile)
        else:
            print(uploadErrors)
            return render_template('./resume-upload.html', uploadErrors=uploadErrors)
        
@app.route('/clear-resume', methods=['PUT'])
def clearResume():
    print('clearing')
    if request.method == 'DELETE':
        print('deleting')
        request.form.clear()


if __name__ == '__main__':
    app.run(debug=True)
