from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
ASSEMBLY_AI_API_KEY = os.getenv('ASSEMBLY_AI_API_KEY')