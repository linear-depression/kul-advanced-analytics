from dotenv import load_dotenv
my_api_key = os.getenv("API_KEY")
my_db_pass = os.getenv("DB_PASSWORD")

# 1. Import external libraries (installed via pip)
import requests

# 2. Import your VARIABLES from your config file
# Syntax: from <folder>.<filename> import <variable>
from app.config import DEFAULT_TAX_RATE, DATABASE_URL

# 3. Import your FUNCTIONS from your module file
from app.utils import calculate_total, format_username

def main():
    print(f"Starting app with DB: {DATABASE_URL}")

    # Using a variable from config.py
    product_price = 100
    
    # Using a function from utils.py
    final_price = calculate_total(product_price, DEFAULT_TAX_RATE)
    
    print(f"Final Price: ${final_price}")

    # Using an external library
    response = requests.get('https://api.github.com')
    print(f"Github Status: {response.status_code}")

if __name__ == "__main__":
    main()