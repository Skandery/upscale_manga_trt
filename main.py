from upscale import process_upscale

if __name__ == "__main__":
    # Configure paths here for direct execution

    INPUT_DIR = r"C:\testinput"
    OUTPUT_DIR = r"C:\testoutput"

    # --- Set a height to force manga images to before upscaling ---
    #FORCE_HEIGHT = None
    FORCE_HEIGHT = 1200 # Example: Force to 1200p input

    # --- Run the processing ---
    try:
        total_time_elapsed, total_processed_count, total_skipped_count, skip_cls =process_upscale(INPUT_DIR, OUTPUT_DIR, force_image_height=FORCE_HEIGHT)
    except ImportError as e:
         print(f"Import Error: {str(e)}. Please ensure all dependencies including 'trt_utilities' and 'opencv-python' are installed and accessible.")
    except FileNotFoundError as e:
         print(f"File Not Found Error: {str(e)}. Check model paths and input/output directories.")
    except Exception as e:
         print(f"An unexpected error occurred: {str(e)}")
         # Optionally add traceback printing for debugging
         import traceback
         traceback.print_exc()
