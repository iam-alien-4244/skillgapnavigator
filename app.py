import os
import uuid
import warnings
import h5py
import pickle
import pandas as pd
from flask import Flask, request, render_template, send_file, jsonify, redirect, url_for
from fuzzywuzzy import fuzz
from config import ProductionConfig
from azure.storage.blob import BlobServiceClient
from flask_login import LoginManager, UserMixin, login_required, login_user, current_user
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import pyodbc

warnings.filterwarnings("ignore", category=UserWarning, message="Workbook contains no default style, apply openpyxl's default")

app = Flask(__name__)
app.config.from_object(ProductionConfig)

# Azure Blob Storage configuration
connect_str = "your_azure_storage_connection_string"
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client("generated-files")

# SendGrid configuration
sendgrid_api_key = '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'
sg = SendGridAPIClient(api_key = sendgrid_api_key)

# Azure SQL Database configuration
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 18 for SQL Server};'
    'SERVER   = skill-gap-navigator.database.windows.net;'
    'DATABASE = SGN Data;'
    'UID      = your_username;'
    'PWD      = your_password'
)
cursor = conn.cursor()

# Flask-Login configuration
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, id, username, role):
        self.id = id
        self.username = username
        self.role = role

    def get_id(self):
        return self.id

    def is_admin(self):
        return self.role == 'admin'

@login_manager.user_loader
def load_user(user_id):
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    if user:
        return User(user['id'], user['username'], user['role'])
    return None

# Load data and model
skills_data = pd.read_csv('../data/skills.csv')
employee_data = pd.read_excel('../data/Skills and Certificates 19th June.xlsx', sheet_name='Sheet1', skiprows=2)

save_dir = 'generated_files'
os.makedirs(save_dir, exist_ok=True)

with h5py.File('../model/skill_classifier.hdf5', 'r') as hdf:
    vectorizer_bytes = hdf['vectorizer'][()].tobytes()
    model_bytes = hdf['model'][()].tobytes()
    vectorizer = pickle.loads(vectorizer_bytes)
    model = pickle.loads(model_bytes)

def fetch_matching_employees(input_skill, column, employee_df, threshold=93):
    normalized_input = ' '.join(input_skill.lower().split())
    input_vector = vectorizer.transform([normalized_input])
    distances, indices = model.kneighbors(input_vector)
    closest_skills = skills_data.iloc[indices[0]].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    employee_df[column] = employee_df[column].fillna('').apply(lambda x: ' '.join(x.lower().split()))
    matching_employees = employee_df[employee_df[column].apply(lambda x: any(fuzz.partial_ratio(normalized_input, x) >= threshold for skill in closest_skills))]
    return matching_employees

def upload_file(file_path, file_name):
    blob_client = container_client.get_blob_client(file_name)
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data)

def download_file(file_name, download_path):
    blob_client = container_client.get_blob_client(file_name)
    with open(download_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

def send_access_request_email():
    email_message = Mail(
        from_email ='admin@gdt.com',
        to_emails  ='admin@gdt.com',
        subject    ='Access Request',
        html_content='<strong>A new access request has been made.</strong>'
    )
    sg.send(email_message)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        if user:
            user_obj = User(user['id'], user['username'], user['role'])
            login_user(user_obj)
            return redirect(url_for('index'))
        else:
            send_access_request_email()
            return "Access requested. Please wait for approval."
    return render_template('login.html')

@app.route('/fetch_employees', methods=['POST'])
@login_required
def fetch_employees():
    skill = request.form.get('skill')
    certification = request.form.get('certification')

    if not skill and not certification:
        return jsonify({'error': "Both fields cannot be empty."})

    skill_employees = fetch_matching_employees(skill, 'Skills', employee_data) if skill else None
    cert_employees = fetch_matching_employees(certification, 'Certification', employee_data) if certification else None

    if skill and certification:
        both_matching_employees = pd.merge(skill_employees, cert_employees, on='Employee ID', how='inner')
    else:
        both_matching_employees = None

    skill_filename = os.path.join(save_dir, f'Employees_with_required_Skill_{uuid.uuid4()}.csv') if skill_employees is not None else None
    cert_filename = os.path.join(save_dir, f'Employees_with_required_Certification_{uuid.uuid4()}.csv') if cert_employees is not None else None
    both_filename = os.path.join(save_dir, f'Employees_with_both_required_Skill_and_Certification_{uuid.uuid4()}.csv') if both_matching_employees is not None else None

    if skill_employees is not None:
        skill_employees.to_csv(skill_filename, index=False)
        upload_file(skill_filename, os.path.basename(skill_filename))
    if cert_employees is not None:
        cert_employees.to_csv(cert_filename, index=False)
        upload_file(cert_filename, os.path.basename(cert_filename))
    if both_matching_employees is not None:
        both_matching_employees.to_csv(both_filename, index=False)
        upload_file(both_filename, os.path.basename(both_filename))

    response = {
        'skill_employees': skill_employees.to_html(index=False) if skill_employees is not None and not skill_employees.empty else '<span class="text-danger">No data found!</span>',
        'cert_employees': cert_employees.to_html(index=False) if cert_employees is not None and not cert_employees.empty else '<span class="text-danger">No data found!</span>',
        'both_matching_employees': both_matching_employees.to_html(index=False) if both_matching_employees is not None and not both_matching_employees.empty else '<span class="text-danger">No data found!</span>',
        'skill_employees_file': os.path.basename(skill_filename) if skill_filename else None,
        'cert_employees_file': os.path.basename(cert_filename) if cert_filename else None,
        'both_matching_employees_file': os.path.basename(both_filename) if both_filename else None
    }

    return jsonify(response)

@app.route('/download/<data_type>', methods=['GET'])
@login_required
def download(data_type):
    filename = request.args.get('data')
    download_path = os.path.join(save_dir, filename)

    if data_type in filename and os.path.exists(download_path):
        return send_file(download_path, as_attachment=True, download_name=os.path.basename(download_path))
    else:
        return "File not found", 404

if __name__ == '__main__':
    os.environ['FLASK_ENV'] = 'production'
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
