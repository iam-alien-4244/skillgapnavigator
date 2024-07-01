import os, uuid, warnings, h5py, pickle, pandas as pd, logging
from fuzzywuzzy import fuzz
from config import ProductionConfig
from flask import Flask, request, render_template, send_file, jsonify
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

warnings.filterwarnings("ignore", category=UserWarning, message="Workbook contains no default style, apply openpyxl's default")

app = Flask(__name__)
app.config.from_object('config.ProductionConfig')

# Azure Storage Configuration
azure_storage_connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
blob_service_client             = BlobServiceClient.from_connection_string(azure_storage_connection_string)
container_name                  = 'generated-files'

try:
    skills_data   = pd.read_csv('../data/skills.csv')
    employee_data = pd.read_excel('../data/June Data.xlsx', sheet_name = 'Sheet1', skiprows = 2)

except:
    app.logger.error(f"Error loading data files: {str(e)}")
    skill_data_data, employees_data = None, None

save_dir = 'generated_files'
os.makedirs(save_dir, exist_ok=True)

# Load the vectorizer and model from an HDF5 file
try:
    with h5py.File('../model/skill_classifier.hdf5', 'r') as hdf:
        vectorizer_bytes = hdf['vectorizer'][()].tobytes()
        model_bytes      = hdf['model'][()].tobytes()
        vectorizer       = pickle.loads(vectorizer_bytes)
        model            = pickle.loads(model_bytes)

except:
    app.logger.error(f"Error loading model: {str(e)}")
    vectorizer, model = None, None

def fetch_matching_employees(input_skill, column, employee_df, threshold = 93):
    normalized_input    = ' '.join(input_skill.lower().split())
    input_vector        = vectorizer.transform([normalized_input])
    distances, indices  = model.kneighbors(input_vector)
    closest_skills      = skills_data.iloc[indices[0]].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    employee_df[column] = employee_df[column].fillna('').apply(lambda x: ' '.join(x.lower().split())) 
    matching_employees  = employee_df[employee_df[column].apply(lambda x: any(fuzz.partial_ratio(normalized_input, x) >= threshold for skill in closest_skills))]
    return matching_employees

def upload_file_to_azure(file_path, file_name):
    try:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data)
            app.logger.info(f"Uploaded file {file_name} to Azure Blob Storage.")

    except Exception as e:
        app.logger.error(f"Error uploading file to Azure Blob Storage: {str(e)}")

logging.basicConfig(level = logging.INFO)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch_employees', methods=['POST'])
def fetch_employees():
    try:
        skill         = request.form.get('skill')
        certification = request.form.get('certification')
    
        if not skill and not certification:
            return jsonify({'error': "Both fields cannot be empty."})
    
        skill_employees = fetch_matching_employees(skill, 'Skills', employee_data) if skill else None
        cert_employees  = fetch_matching_employees(certification, 'Certification', employee_data) if certification else None
    
        if skill and certification:
            both_matching_employees = pd.merge(skill_employees, cert_employees, on='Employee ID', how='inner')
        else:
            both_matching_employees = None
    
        skill_filename = os.path.join(save_dir, f'Employees_with_required_Skill_{uuid.uuid4()}.csv') if skill_employees is not None else None
        cert_filename  = os.path.join(save_dir, f'Employees_with_required_Certification_{uuid.uuid4()}.csv') if cert_employees is not None else None
        both_filename  = os.path.join(save_dir, f'Employees_with_both_required_Skill_Certification_{uuid.uuid4()}.csv') if both_matching_employees is not None else None
    
        if skill_employees is not None:
            skill_employees.to_csv(skill_filename, index = False)
            upload_file_to_azure(skill_filename, os.path.basename(skill_filename))
        
        if cert_employees is not None:
            cert_employees.to_csv(cert_filename, index = False)
            upload_file_to_azure(cert_filename, os.path.basename(cert_filename))
        
        if both_matching_employees is not None:
            both_matching_employees.to_csv(both_filename, index = False)
            upload_file_to_azure(both_filename, os.path.basename(both_filename))
    
        # Prepare response JSON
        response = {
            'skill_employees': skill_employees.to_html(index=False) if skill_employees is not None and not skill_employees.empty else '<span class = "text-danger">No data found!</span>',
            'cert_employees': cert_employees.to_html(index=False) if cert_employees is not None and not cert_employees.empty else '<span class = "text-danger">No data found!</span>',
            'both_matching_employees': both_matching_employees.to_html(index=False) if both_matching_employees is not None and not both_matching_employees.empty else '<span class = "text-danger">No data found!</span>',
            'skill_employees_file': skill_filename,
            'cert_employees_file': cert_filename,
            'both_matching_employees_file': both_filename
        }
    
        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f"Error in fetch_employees: {str(e)}")
        return jsonify({"error": "An error occured during processing. Please try again."})

@app.route('/download/<data_type>', methods = ['GET'])
def download(data_type):
    try:
        filename = request.args.get('data')
    
        if data_type in filename and os.path.exists(filename):
            return send_file(filename, as_attachment = True, download_name = os.path.basename(filename))   
        else:
            return "File not found", 404

    except:
        app.logger.error(f"Error in download: {str(e)}")
        return "An error occurred during file download.", 500

if __name__ == '__main__':
    # Set FLASK_ENV to 'production' to indicate the app is running in production
    os.environ['FLASK_ENV'] = 'production'
    app.run(debug = False, host = '0.0.0.0', port = int(os.environ.get('PORT', 5000)))


