import base64
import os
import sys
from io import BytesIO
from PIL import Image

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import handler
from handler import handler

def run_local_test():
    # Look for an image in the parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_image_name = "final_reconstructed_test01.png" # or any other test image
    test_image_path = os.path.join(parent_dir, test_image_name)
    
    # Fallback to any png/jpg in parent dir if the specific one is not found
    if not os.path.exists(test_image_path):
        candidates = [f for f in os.listdir(parent_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if candidates:
            test_image_path = os.path.join(parent_dir, candidates[0])
            print(f"Using alternative test image: {test_image_path}")
        else:
            print("No test image found in parent directory. Please place an image there.")
            return

    print(f"Reading test image: {test_image_path}")
    with open(test_image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    # Prepare mock job
    job = {
        "input": {
            "image_base64": encoded_string
        }
    }

    print("Running local backend_ai handler...")
    result = handler(job)

    if "error" in result:
        print(f"Error occurred: {result['error']}")
        return

    # Save output
    output_base64 = result.get("output_image_base64")
    if output_base64:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_test_result.jpg")
        with open(output_path, "wb") as out_file:
            out_file.write(base64.b64decode(output_base64))
        print(f"Success! Processed image saved to: {output_path}")
    else:
        print("No output base64 received from handler.")

if __name__ == "__main__":
    run_local_test()
