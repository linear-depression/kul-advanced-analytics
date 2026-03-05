# Setting the virtual environment
python -m venv venv

# Repeat this when coming back to proj
source venv/bin/activate

# Installing libraries and saving them to requirements.txt
pip install requests

pip freeze > requirements.txt

# Installing dependencies from existing requirements.txt
pip install -r requirements.txt

pip install python-dotenv

# Exiting venv
deactivate

