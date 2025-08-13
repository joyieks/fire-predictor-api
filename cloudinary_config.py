import cloudinary
import cloudinary.uploader
import cloudinary.api
import os
from dotenv import load_dotenv

# Load environment variables from .env file (local development only)
load_dotenv()

# Use environment variables for security
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME", "dlnq8eaed"),
    api_key=os.environ.get("CLOUDINARY_API_KEY", "495443256383188"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET"),  # Must be set as env var
    secure=True
)